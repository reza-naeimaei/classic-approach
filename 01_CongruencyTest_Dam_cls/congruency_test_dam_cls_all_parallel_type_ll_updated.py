import json
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg as la
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.stats import f as f_dist
from tqdm import tqdm

GT_1CM = np.array([110, 109, 118, 127, 137, 147, 157, 167, 177, 188, 199, 200], dtype=int)
GT_2CM = np.array([120, 119, 130, 129, 128, 140, 139, 138, 150, 149, 148,
                   160, 159, 158, 168, 169, 170, 178, 179, 180, 190, 189], dtype=int)
GT_ALL = np.unique(np.r_[GT_1CM, GT_2CM])
GT_ALL_0 = GT_ALL - 1

ALPHAS = (0.05, 0.01)
A_ZERO_TOL = 1e-14


def _is_ref_dtype(ds: h5py.Dataset) -> bool:
    return ds.dtype == h5py.ref_dtype or h5py.check_dtype(ref=ds.dtype) is not None


def _deref_first(f: h5py.File, obj):
    if isinstance(obj, h5py.Reference):
        return f[obj]
    if isinstance(obj, h5py.Dataset) and _is_ref_dtype(obj):
        arr = obj[()]
        ref = np.asarray(arr).reshape(-1)[0]
        return f[ref]
    return obj


def _read_numeric(obj) -> np.ndarray:
    if isinstance(obj, h5py.Dataset):
        return np.asarray(obj[()])
    raise TypeError(f"Expected dataset, got {type(obj)}")


def _read_scalar_numeric(f: h5py.File, parent, field_name: str) -> float:
    obj = _deref_first(f, parent[field_name])
    arr = _read_numeric(obj)
    return float(np.asarray(arr).reshape(-1)[0])


def _read_array_numeric(f: h5py.File, parent, field_name: str) -> np.ndarray:
    obj = _deref_first(f, parent[field_name])
    return np.asarray(_read_numeric(obj))


def _open_realization_struct(file_path: str | Path, epoch_kind: str, realization_idx: int):
    file_path = Path(file_path)
    cell_name = "EPref" if epoch_kind == "ref" else "EPcomp"
    f = h5py.File(file_path, "r")
    cell_ds = f[cell_name]
    shape = cell_ds.shape
    if len(shape) != 2:
        f.close()
        raise ValueError(f"Unexpected shape for {cell_name}: {shape}")
    if shape[0] == 1:
        ref = cell_ds[0, realization_idx]
    elif shape[1] == 1:
        ref = cell_ds[realization_idx, 0]
    else:
        f.close()
        raise ValueError(f"Unsupported cell layout for {cell_name}: {shape}")
    entry = f[ref]
    return f, entry


def _as_points_xyz(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D xyz array, got shape {arr.shape}")
    if arr.shape[0] == 3:
        return arr.T.copy()
    if arr.shape[1] == 3:
        return arr.copy()
    raise ValueError(f"Expected xyz array with one dimension of size 3, got {arr.shape}")


def _as_stacked_xyz(arr: np.ndarray) -> np.ndarray:
    return _as_points_xyz(arr).reshape(-1)


def _as_obs_to_param_map(A_raw: np.ndarray, n_obs: int, n_param: int) -> np.ndarray:
    A_raw = np.asarray(A_raw, dtype=float)
    if A_raw.shape == (n_obs, n_param):
        return A_raw
    if A_raw.shape == (n_param, n_obs):
        return A_raw.T
    raise ValueError(f"Could not interpret A with shape {A_raw.shape}; expected {(n_obs, n_param)} or {(n_param, n_obs)}")


def extract_realization_from_v73(file_path: str | Path, epoch_kind: str, realization_idx: int) -> dict:
    nurbs_key = "nurbs1" if epoch_kind == "ref" else "nurbs2"
    quality_key = "quality1" if epoch_kind == "ref" else "quality2"
    f, entry = _open_realization_struct(file_path, epoch_kind, realization_idx)
    try:
        nurbs = _deref_first(f, entry[nurbs_key])
        qual = _deref_first(f, entry[quality_key])

        cp_xyz = _as_points_xyz(_read_array_numeric(f, nurbs, "coefs"))
        x_stack = cp_xyz.reshape(-1)
        Qxx = _read_array_numeric(f, qual, "Qxx").astype(float)
        addedbias_id = int(round(_read_scalar_numeric(f, entry, "addedbias_id")))
        s0_sq = float(_read_scalar_numeric(f, qual, "sigma0_apost"))

        dof = None
        for k in ("f", "dof", "redundancy", "r"):
            if k in qual.keys():
                try:
                    dof = int(round(_read_scalar_numeric(f, qual, k)))
                    break
                except Exception:
                    pass
        if dof is None and "residuals" in qual.keys():
            residuals = _read_array_numeric(f, qual, "residuals")
            dof = int(residuals.size - x_stack.size)
        if dof is None:
            raise ValueError("Could not determine dof from quality block.")

        lhat_stack = None
        if "lDach" in qual.keys():
            lhat_stack = np.asarray(_read_array_numeric(f, qual, "lDach"), dtype=float).reshape(-1)
        elif "xyz_noised" in entry.keys():
            lhat_stack = _as_stacked_xyz(_read_array_numeric(f, entry, "xyz_noised"))
        if lhat_stack is None:
            raise ValueError("Could not find fitted/surface point coordinates (lDach or xyz_noised).")
        surf_xyz = lhat_stack.reshape(-1, 3)

        if "A" not in entry.keys():
            raise ValueError("Could not find design/evaluation matrix A in realization entry.")
        A_raw = _read_array_numeric(f, entry, "A")
        A_map = _as_obs_to_param_map(A_raw, n_obs=lhat_stack.size, n_param=x_stack.size)

        return {
            "CP_xyz": cp_xyz,
            "CP_stack": x_stack,
            "Qxx": Qxx,
            "s0_sq": s0_sq,
            "dof": dof,
            "addedbias_id": addedbias_id,
            "surface_xyz": surf_xyz,
            "A_map": A_map,
        }
    finally:
        f.close()


def pooled_sigma0_sq(s01_sq: float, f1: int, s02_sq: float, f2: int) -> float:
    return (float(f1) * float(s01_sq) + float(f2) * float(s02_sq)) / (float(f1) + float(f2))


def global_congruency_test_F(d: np.ndarray, Qdd: np.ndarray, s0_pool_sq: float, alpha: float, f_total: int) -> dict:
    d = np.asarray(d, dtype=float).reshape(-1)
    Qdd = np.asarray(Qdd, dtype=float)
    h = int(np.linalg.matrix_rank(Qdd))
    if h <= 0:
        return {"reject": False, "T": np.nan, "R": np.nan, "h": h, "crit": np.nan, "p_value": np.nan}
    x = np.linalg.solve(Qdd, d)
    R = float(d.T @ x)
    T = (R / h) / float(s0_pool_sq)
    crit = float(f_dist.ppf(1 - alpha, dfn=h, dfd=int(f_total)))
    p_value = float(1 - f_dist.cdf(T, dfn=h, dfd=int(f_total)))
    return {"reject": bool(T >= crit), "T": T, "R": R, "h": h, "crit": crit, "p_value": p_value}


def _reduce_by_cp_indices(d3N: np.ndarray, Qdd3N: np.ndarray, cp_idx: np.ndarray):
    cp_idx = np.asarray(cp_idx, dtype=int)
    cols = np.empty(3 * cp_idx.size, dtype=int)
    cols[0::3] = 3 * cp_idx
    cols[1::3] = 3 * cp_idx + 1
    cols[2::3] = 3 * cp_idx + 2
    return d3N[cols], Qdd3N[np.ix_(cols, cols)]


def local_contributions_single_point(d: np.ndarray, Qdd: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float).reshape(-1)
    Qdd = np.asarray(Qdd, dtype=float)
    P = la.inv(Qdd)
    n = d.size
    n_ctrl = n // 3
    q = np.zeros(n_ctrl, float)
    for j in range(n_ctrl):
        sl = slice(3 * j, 3 * j + 3)
        idx_n = np.r_[0:3 * j, 3 * j + 3:n]
        Ppp = P[sl, sl]
        Ppn = P[sl, :][:, idx_n]
        dp = d[sl]
        dn = d[idx_n]
        bar_dp = dp + (np.linalg.inv(Ppp) @ Ppn @ dn)
        q[j] = float(bar_dp.T @ Ppp @ bar_dp)
    return q


def sequential_localization_A(d_full: np.ndarray, Qdd_full: np.ndarray, s0_pool_sq: float, alpha: float, f_total: int) -> dict:
    d_full = np.asarray(d_full, dtype=float).reshape(-1)
    Qdd_full = np.asarray(Qdd_full, dtype=float)
    n_cp = d_full.size // 3
    kept = np.arange(n_cp, dtype=int)
    removed = []
    g0 = global_congruency_test_F(d_full, Qdd_full, s0_pool_sq, alpha=alpha, f_total=f_total)
    while True:
        if kept.size == 0:
            break
        d_red, Q_red = _reduce_by_cp_indices(d_full, Qdd_full, kept)
        g = global_congruency_test_F(d_red, Q_red, s0_pool_sq, alpha=alpha, f_total=f_total)
        if not g["reject"]:
            break
        q = local_contributions_single_point(d_red, Q_red)
        worst_red = int(np.argmax(q))
        worst_cp = int(kept[worst_red])
        removed.append(worst_cp)
        kept = np.delete(kept, worst_red)
    reject_mask = np.zeros(n_cp, dtype=bool)
    if removed:
        reject_mask[np.array(removed, dtype=int)] = True
    return {"removed_idx": np.array(removed, dtype=int), "kept_idx": kept, "reject_mask": reject_mask, "global_initial": g0}


def compare_with_ground_truth(removed_idx_zero_based: np.ndarray) -> dict:
    pred_ids = np.sort(np.asarray(removed_idx_zero_based, dtype=int) + 1)
    pred_set = set(pred_ids.tolist())
    gt_1cm_set = set(GT_1CM.tolist())
    gt_2cm_set = set(GT_2CM.tolist())
    gt_all_set = set(GT_ALL.tolist())
    tp = sorted(pred_set & gt_all_set)
    fp = sorted(pred_set - gt_all_set)
    fn = sorted(gt_all_set - pred_set)
    tp_1cm = sorted(pred_set & gt_1cm_set)
    tp_2cm = sorted(pred_set & gt_2cm_set)
    return {
        "predicted_ids_1based": pred_ids.tolist(),
        "true_positives_all": tp,
        "true_positives_1cm": tp_1cm,
        "true_positives_2cm": tp_2cm,
        "false_positives": fp,
        "false_negatives": fn,
        "n_predicted": int(len(pred_ids)),
        "n_gt_all": int(len(gt_all_set)),
        "n_tp": int(len(tp)),
        "n_fp": int(len(fp)),
        "n_fn": int(len(fn)),
        "recall_all": float(len(tp) / len(gt_all_set)) if gt_all_set else None,
        "recall_1cm": float(len(tp_1cm) / len(gt_1cm_set)) if gt_1cm_set else None,
        "recall_2cm": float(len(tp_2cm) / len(gt_2cm_set)) if gt_2cm_set else None,
        "precision": float(len(tp) / len(pred_set)) if pred_set else None,
    }


def build_surface_pointwise_tests(d_cp: np.ndarray, Qdd: np.ndarray, A_map: np.ndarray, s0_pool_sq: float, alpha: float, f_total: int) -> dict:
    d_cp = np.asarray(d_cp, dtype=float).reshape(-1)
    Qdd = np.asarray(Qdd, dtype=float)
    A_csr = csr_matrix(A_map)
    n_obs, n_param = A_csr.shape
    if n_obs % 3 != 0:
        raise ValueError(f"Observation stack length must be multiple of 3, got {n_obs}")
    if n_param != d_cp.size:
        raise ValueError(f"A_map parameter dimension {n_param} != d_cp length {d_cp.size}")

    n_surf = n_obs // 3
    reject_mask = np.zeros(n_surf, dtype=bool)
    T_vals = np.full(n_surf, np.nan, dtype=float)
    crit_vals = np.full(n_surf, np.nan, dtype=float)
    for j in range(n_surf):
        sl = slice(3 * j, 3 * j + 3)
        B = A_csr[sl, :].toarray()
        nz = np.flatnonzero(np.any(np.abs(B) > A_ZERO_TOL, axis=0))
        if nz.size == 0:
            continue
        B_loc = B[:, nz]
        d_loc = d_cp[nz]
        Q_loc = Qdd[np.ix_(nz, nz)]
        d_s = B_loc @ d_loc
        Q_s = B_loc @ Q_loc @ B_loc.T
        h = int(np.linalg.matrix_rank(Q_s))
        if h <= 0:
            continue
        try:
            R = float(d_s.T @ np.linalg.solve(Q_s, d_s))
        except np.linalg.LinAlgError:
            R = float(d_s.T @ np.linalg.pinv(Q_s) @ d_s)
        T = (R / h) / float(s0_pool_sq)
        crit = float(f_dist.ppf(1 - alpha, dfn=h, dfd=int(f_total)))
        reject_mask[j] = bool(T >= crit)
        T_vals[j] = T
        crit_vals[j] = crit
    return {"reject_mask": reject_mask, "n_points": int(n_surf), "T": T_vals, "crit": crit_vals}


def build_surface_gt_mask(A_map: np.ndarray, gt_cp_zero_based: np.ndarray) -> np.ndarray:
    A = csr_matrix(A_map)
    n_obs, n_param = A.shape
    if n_obs % 3 != 0:
        raise ValueError(f"Observation stack length must be multiple of 3, got {n_obs}")
    n_surf = n_obs // 3
    gt_cols_y = 3 * np.asarray(gt_cp_zero_based, dtype=int) + 1
    gt_mask = np.zeros(n_surf, dtype=bool)
    for j in range(n_surf):
        y_row = 3 * j + 1
        row = A.getrow(y_row)
        if row.nnz == 0:
            continue
        cols = row.indices
        vals = row.data
        active = cols[np.abs(vals) > A_ZERO_TOL]
        gt_mask[j] = bool(np.intersect1d(active, gt_cols_y, assume_unique=False).size > 0)
    return gt_mask


def compare_surface_with_ground_truth(reject_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    reject_mask = np.asarray(reject_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    if reject_mask.shape != gt_mask.shape:
        raise ValueError("reject_mask and gt_mask must have the same shape")
    tp_mask = reject_mask & gt_mask
    fp_mask = reject_mask & (~gt_mask)
    fn_mask = (~reject_mask) & gt_mask
    return {
        "n_points": int(gt_mask.size),
        "n_gt": int(gt_mask.sum()),
        "n_pred": int(reject_mask.sum()),
        "n_tp": int(tp_mask.sum()),
        "n_fp": int(fp_mask.sum()),
        "n_fn": int(fn_mask.sum()),
        "precision": float(tp_mask.sum() / reject_mask.sum()) if reject_mask.any() else None,
        "recall": float(tp_mask.sum() / gt_mask.sum()) if gt_mask.any() else None,
        "miss_rate_over_deformed": float(fn_mask.sum() / gt_mask.sum()) if gt_mask.any() else None,
        "false_alarm_rate_over_undeformed": float(fp_mask.sum() / (~gt_mask).sum()) if (~gt_mask).any() else None,
        "local_any_missed_detection": bool(fn_mask.any()),
        "local_any_false_alarm": bool(fp_mask.any()),
        "miss_mask": fn_mask,
        "false_alarm_mask": fp_mask,
        "gt_mask": gt_mask,
    }


def save_json(output_path: str | Path, payload: dict):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def process_realization(i: int, ref_mat_path: str, comp_mat_path: str, scenario: str):
    ep1 = extract_realization_from_v73(ref_mat_path, "ref", i)
    ep2 = extract_realization_from_v73(comp_mat_path, "comp", i)
    d_cp = ep2["CP_stack"] - ep1["CP_stack"]
    Q_dd = ep1["Qxx"] + ep2["Qxx"]
    s0_pool_sq = pooled_sigma0_sq(ep1["s0_sq"], ep1["dof"], ep2["s0_sq"], ep2["dof"])
    f_total = int(ep1["dof"] + ep2["dof"])
    surface_xyz = ep1["surface_xyz"]
    gt_surface_mask = build_surface_gt_mask(ep1["A_map"], GT_ALL_0)

    rows = []
    per_real_payload = {
        "realization_idx_zero_based": i,
        "realization_idx_one_based": i + 1,
        "scenario": scenario,
        "reference_addedbias_id": ep1["addedbias_id"],
        "comparison_addedbias_id": ep2["addedbias_id"],
        "pooled_sigma0_sq": s0_pool_sq,
        "f_total": f_total,
        "surface_n_points": int(surface_xyz.shape[0]),
        "surface_gt_n_deformed": int(gt_surface_mask.sum()),
        "alphas": {},
    }

    for a in ALPHAS:
        g = global_congruency_test_F(d_cp, Q_dd, s0_pool_sq, alpha=a, f_total=f_total)
        loc = sequential_localization_A(d_cp, Q_dd, s0_pool_sq, alpha=a, f_total=f_total)
        cmp_cp = compare_with_ground_truth(loc["removed_idx"])

        surf_test = build_surface_pointwise_tests(d_cp, Q_dd, ep1["A_map"], s0_pool_sq, alpha=a, f_total=f_total)
        cmp_surf = compare_surface_with_ground_truth(surf_test["reject_mask"], gt_surface_mask)

        if scenario != "type2_defo":
            raise ValueError(f"Unknown scenario: {scenario}")

        row = {
            "realization_idx_one_based": i + 1,
            "scenario": scenario,
            "alpha": a,
            "pooled_sigma0_sq": float(s0_pool_sq),
            "global_reject": bool(g["reject"]),
            "global_T": float(g["T"]) if not np.isnan(g["T"]) else None,
            "global_crit": float(g["crit"]) if not np.isnan(g["crit"]) else None,
            "global_p_value": float(g["p_value"]) if not np.isnan(g["p_value"]) else None,
            "global_type2_error": bool(not g["reject"]),
            "n_removed": int(loc["removed_idx"].size),
            "n_tp": cmp_cp["n_tp"],
            "n_fp": cmp_cp["n_fp"],
            "n_fn": cmp_cp["n_fn"],
            "precision": cmp_cp["precision"],
            "recall_all": cmp_cp["recall_all"],
            "recall_1cm": cmp_cp["recall_1cm"],
            "recall_2cm": cmp_cp["recall_2cm"],
            "local_any_type2_error": bool(cmp_cp["n_fn"] > 0),
            "surface_n_tp": cmp_surf["n_tp"],
            "surface_n_fp": cmp_surf["n_fp"],
            "surface_n_fn": cmp_surf["n_fn"],
            "surface_precision": cmp_surf["precision"],
            "surface_recall": cmp_surf["recall"],
            "surface_miss_rate_over_deformed": cmp_surf["miss_rate_over_deformed"],
            "surface_false_alarm_rate_over_undeformed": cmp_surf["false_alarm_rate_over_undeformed"],
            "surface_any_type2_error": bool(cmp_surf["local_any_missed_detection"]),
        }

        per_real_payload["alphas"][str(a)] = {
            "global": g,
            "cp_localization": {
                "removed_idx_zero_based": loc["removed_idx"].tolist(),
                "removed_idx_one_based": (loc["removed_idx"] + 1).tolist(),
                "reject_mask": loc["reject_mask"].tolist(),
                "n_removed": int(loc["removed_idx"].size),
            },
            "cp_comparison": cmp_cp,
            "surface_pointwise": {
                "reject_mask": surf_test["reject_mask"].tolist(),
                "gt_mask": cmp_surf["gt_mask"].tolist(),
                "miss_mask": cmp_surf["miss_mask"].tolist(),
                "false_alarm_mask": cmp_surf["false_alarm_mask"].tolist(),
                "n_points": cmp_surf["n_points"],
                "n_gt": cmp_surf["n_gt"],
                "n_pred": cmp_surf["n_pred"],
                "n_tp": cmp_surf["n_tp"],
                "n_fp": cmp_surf["n_fp"],
                "n_fn": cmp_surf["n_fn"],
                "precision": cmp_surf["precision"],
                "recall": cmp_surf["recall"],
                "miss_rate_over_deformed": cmp_surf["miss_rate_over_deformed"],
                "false_alarm_rate_over_undeformed": cmp_surf["false_alarm_rate_over_undeformed"],
            },
        }
        rows.append(row)

    return {
        "i": i,
        "payload": per_real_payload,
        "rows": rows,
        "first_cp": ep1["CP_xyz"],
        "first_surface_xyz": surface_xyz,
        "first_surface_gt_mask": gt_surface_mask,
    }


def build_summary(all_rows: list[dict]) -> dict:
    def mean_or_none(values):
        vals = [v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None

    summary = {"alphas": {}}
    for a in ALPHAS:
        rows_a = [r for r in all_rows if float(r["alpha"]) == float(a)]
        global_type2_rate = mean_or_none([r.get("global_type2_error") for r in rows_a])
        global_power = None if global_type2_rate is None else float(1.0 - global_type2_rate)
        summary["alphas"][str(a)] = {
            "type2_defo": {
                "n_realizations": int(len(rows_a)),
                "global_type2_rate": global_type2_rate,
                "global_power": global_power,
                "cp_local_any_type2_rate": mean_or_none([r.get("local_any_type2_error") for r in rows_a]),
                "cp_mean_n_fn": mean_or_none([r.get("n_fn") for r in rows_a]),
                "cp_mean_n_fp": mean_or_none([r.get("n_fp") for r in rows_a]),
                "cp_mean_precision": mean_or_none([r.get("precision") for r in rows_a]),
                "cp_mean_recall_all": mean_or_none([r.get("recall_all") for r in rows_a]),
                "cp_mean_recall_1cm": mean_or_none([r.get("recall_1cm") for r in rows_a]),
                "cp_mean_recall_2cm": mean_or_none([r.get("recall_2cm") for r in rows_a]),
                "surface_any_type2_rate": mean_or_none([r.get("surface_any_type2_error") for r in rows_a]),
                "surface_mean_n_fn": mean_or_none([r.get("surface_n_fn") for r in rows_a]),
                "surface_mean_n_fp": mean_or_none([r.get("surface_n_fp") for r in rows_a]),
                "surface_mean_precision": mean_or_none([r.get("surface_precision") for r in rows_a]),
                "surface_mean_recall": mean_or_none([r.get("surface_recall") for r in rows_a]),
                "surface_mean_miss_rate_over_deformed": mean_or_none([r.get("surface_miss_rate_over_deformed") for r in rows_a]),
                "surface_mean_false_alarm_rate_over_undeformed": mean_or_none([r.get("surface_false_alarm_rate_over_undeformed") for r in rows_a]),
            }
        }
    return summary


if __name__ == "__main__":
    n_jobs = 10
    ref_mat_path = "../00_dam_syntetic/results_v2/EPref_MCS1000_addedbias_id4.mat"
    comp_defo_mat_path = "../00_dam_syntetic/results_v2/EPcomp_MCS1000_addedbias_id1.mat"
    results_dir = Path("./cls_results_all_type2")
    per_real_dir = results_dir / "per_realization"

    results_dir.mkdir(parents=True, exist_ok=True)
    per_real_dir.mkdir(parents=True, exist_ok=True)

    n_rep = 1000
    tasks = [("type2_defo", i, ref_mat_path, comp_defo_mat_path) for i in range(n_rep)]
    print(f"Total tasks: {len(tasks)} with n_jobs={n_jobs}")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_realization)(i, ref_path, comp_path, scenario)
        for scenario, i, ref_path, comp_path in tqdm(tasks, desc="Processing realizations")
    )

    first_res = results[0]
    np.save(results_dir / "first_cp.npy", first_res["first_cp"])
    np.save(results_dir / "first_surface_xyz.npy", first_res["first_surface_xyz"])
    np.save(results_dir / "first_surface_gt_mask.npy", first_res["first_surface_gt_mask"])

    all_rows = []
    for res, task in zip(results, tasks):
        scenario, i, _, _ = task
        save_json(per_real_dir / f"{scenario}_realization_{i+1:04d}.json", res["payload"])
        all_rows.extend(res["rows"])
    save_json(results_dir / "dam_cls_type2_rows.json", {"rows": all_rows})
    save_json(results_dir / "dam_cls_type2_summary.json", build_summary(all_rows))

    csv_path = results_dir / "dam_cls_type2_rows.csv"
    all_keys = sorted({k for row in all_rows for k in row.keys()})
    lines = [",".join(all_keys)]
    for row in all_rows:
        vals = [row.get(k) for k in all_keys]
        lines.append(",".join("" if v is None else str(v) for v in vals))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    print("Done.")
    print(f"Saved outputs under: {results_dir.resolve()}")
