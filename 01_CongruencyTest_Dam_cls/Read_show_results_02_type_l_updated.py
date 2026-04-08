import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_zip_if_needed(results_path: Path) -> Path:
    if results_path.is_file() and results_path.suffix.lower() == ".zip":
        out_dir = results_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(results_path, "r") as zf:
            zf.extractall(out_dir)
        return out_dir
    return results_path


def find_rows_json(base_dir: Path, file_stem: str) -> Path:
    candidates = [
        base_dir / f"{file_stem}.json",
        base_dir / "cls_results_all_type1" / f"{file_stem}.json",
        base_dir / "cls_results_all_realizations" / f"{file_stem}.json",
        base_dir / "cls_results_all_realizations_parallel" / f"{file_stem}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find {file_stem}.json")


def find_per_real_dir(base_dir: Path) -> Path:
    candidates = [
        base_dir / "per_realization",
        base_dir / "cls_results_all_type1" / "per_realization",
        base_dir / "cls_results_all_realizations" / "per_realization",
        base_dir / "cls_results_all_realizations_parallel" / "per_realization",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find per_realization directory")


def list_type1_realization_files(per_real_dir: Path) -> list[Path]:
    files = sorted(per_real_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {per_real_dir}")

    selected = []
    for fp in files:
        try:
            payload = load_json(fp)
            scenario = payload.get("scenario", "")
            if scenario in ("type1_null", "type1"):
                selected.append(fp)
        except Exception:
            pass
    if selected:
        return selected
    return files


def _as_points_xyz(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D xyz array, got shape {arr.shape}")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[0] == 3:
        return arr.T
    raise ValueError(f"Expected xyz array with one dimension of size 3, got {arr.shape}")


def build_reject_matrix(per_real_dir: Path, alpha_str: str, n_items: int, key_path: tuple[str, ...]) -> np.ndarray:
    files = list_type1_realization_files(per_real_dir)
    masks = []
    for fp in files:
        payload = load_json(fp)
        obj = payload["alphas"][alpha_str]
        for k in key_path:
            obj = obj[k]
        mask = np.asarray(obj, dtype=bool)
        if mask.size != n_items:
            raise ValueError(f"Mask size mismatch in {fp.name}: got {mask.size}, expected {n_items}")
        masks.append(mask)
    return np.asarray(masks, dtype=bool)


def plot_hist_two_rows(vals05, vals01, xlabel, title, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].hist(vals05, bins="fd")
    axes[0].set_title(f"{title}, alpha = 0.05")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(vals01, bins="fd")
    axes[1].set_title(f"{title}, alpha = 0.01")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.25)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_hist_two_cols(vals05, vals01, bins, xlabel, title, save_path, show_density=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)

    # Ensure consistent bins across both plots
    all_vals = np.concatenate([vals05, vals01])
    if isinstance(bins, int):
        bins = np.linspace(all_vals.min(), all_vals.max(), bins)

    datasets = [(vals05, 0.05), (vals01, 0.01)]

    for ax, (vals, alpha_nom) in zip(axes, datasets):

        # Histogram
        ax.hist(
            vals,
            bins=bins,
            density=show_density,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.75
        )

        # Mean line
        mean_val = np.mean(vals)
        ax.axvline(mean_val, color='red', linestyle="--", linewidth=1.5, label=f"mean = {mean_val:.3f}")

        # Optional: reference line (e.g., nominal alpha)
        #ax.axvline(alpha_nom, linestyle=":", linewidth=1.5, label=f"α = {alpha_nom}")

        # Labels and title
        ax.set_title(rf"$\alpha = {alpha_nom}$", fontsize=11)
        ax.set_xlabel(xlabel)

        # Grid
        ax.grid(True, alpha=0.3)

        # Legend
        ax.legend(fontsize=9)

    # Shared y-label
    ylabel = "Density" if show_density else "Count"
    axes[0].set_ylabel(ylabel)

    # Global title
    fig.suptitle(title, fontsize=12)

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_frequency_per_points(XYZ, freq05, freq01, point_size, save_path, title_prefix, cbar_label):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)

    #vmax = max(1e-12, float(max(np.max(freq05), np.max(freq01))))
    sc0 = axes[0].scatter(XYZ[:, 0], XYZ[:, 2], c=freq05, s=point_size)
    axes[0].set_title(f"{title_prefix}, alpha = 0.05")
    axes[0].set_ylabel("Z [m]")
    axes[0].grid(True, alpha=0.25)
    cbar0 = fig.colorbar(sc0, ax=axes[0])
    cbar0.set_label(cbar_label)

    sc1 = axes[1].scatter(XYZ[:, 0], XYZ[:, 2], c=freq01, s=point_size)
    axes[1].set_title(f"{title_prefix}, alpha = 0.01")
    axes[1].set_xlabel("X [m]")
    axes[1].set_ylabel("Z [m]")
    axes[1].grid(True, alpha=0.25)
    cbar1 = fig.colorbar(sc1, ax=axes[1])
    cbar1.set_label(cbar_label)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_frequency_per_selected_points(
    XYZ,
    sel_mask,
    freq05,
    freq01,
    save_path,
    title_prefix,
    cbar_label,
    eps=0.005,   # tolerance
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)

    nominal_mask = ~sel_mask
    vmax = max(1e-12, float(max(np.max(freq05), np.max(freq01))))

    # -------- alpha = 0.05 --------
    alpha = 0.05
    highlight_05 = np.abs(freq05 - alpha) <= eps

    sc0 = axes[0].scatter(
        XYZ[sel_mask, 0],
        XYZ[sel_mask, 2],
        c=freq05,
        s=30,
    )

    # background points
    axes[0].scatter(
        XYZ[nominal_mask, 0],
        XYZ[nominal_mask, 2],
        c="lightgray",
        s=8,
        alpha=0.45
    )

    # highlight (red circles)
    axes[0].scatter(
        XYZ[sel_mask][highlight_05, 0],
        XYZ[sel_mask][highlight_05, 2],
        facecolors="none",
        edgecolors="red",
        s=80,
        linewidths=1.5,
        label="≈ ideal Type I"
    )

    axes[0].set_title(f"{title_prefix}, alpha = 0.05")
    axes[0].set_ylabel("Z [m]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    cbar0 = fig.colorbar(sc0, ax=axes[0])
    cbar0.set_label(cbar_label)

    # -------- alpha = 0.01 --------
    alpha = 0.01
    highlight_01 = np.abs(freq01 - alpha) <= eps

    sc1 = axes[1].scatter(
        XYZ[sel_mask, 0],
        XYZ[sel_mask, 2],
        c=freq01,
        s=30,
    )

    axes[1].scatter(
        XYZ[nominal_mask, 0],
        XYZ[nominal_mask, 2],
        c="lightgray",
        s=8,
        alpha=0.45
    )

    axes[1].scatter(
        XYZ[sel_mask][highlight_01, 0],
        XYZ[sel_mask][highlight_01, 2],
        facecolors="none",
        edgecolors="red",
        s=80,
        linewidths=1.5,
        label="≈ ideal Type I"
    )

    axes[1].set_title(f"{title_prefix}, alpha = 0.01")
    axes[1].set_xlabel("X [m]")
    axes[1].set_ylabel("Z [m]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    cbar1 = fig.colorbar(sc1, ax=axes[1])
    cbar1.set_label(cbar_label)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_global_type1_bar(df, save_path):
    alphas = [0.05, 0.01]
    rates = []
    for a in alphas:
        dfa = df[df["alpha"] == a].copy()
        rates.append(float(dfa["global_type1_error"].mean()) if "global_type1_error" in dfa.columns else np.nan)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(["0.05", "0.01"], rates)
    ax.set_ylim(0, 1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("rate")
    ax.set_title("Empirical global Type I error rate")
    ax.grid(True, axis="y", alpha=0.25)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_text_summary(df: pd.DataFrame, out_path: Path):
    lines = []
    for alpha in (0.05, 0.01):
        dfa = df[df["alpha"] == alpha].copy()
        lines.append(f"alpha = {alpha}")
        lines.append(f"  n_realizations                     = {len(dfa)}")
        lines.append(f"  global_type1_error_rate           = {float(dfa['global_type1_error'].mean()) if 'global_type1_error' in dfa.columns else None}")
        lines.append(f"  cp_any_false_alarm_rate           = {float(dfa['local_any_type1_error'].mean()) if 'local_any_type1_error' in dfa.columns else None}")
        lines.append(f"  cp_mean_false_alarm_count         = {float(dfa['n_fp'].mean()) if 'n_fp' in dfa.columns else None}")
        lines.append(f"  cp_mean_false_alarm_rate          = {float(dfa['local_fp_rate_over_cp'].mean()) if 'local_fp_rate_over_cp' in dfa.columns else None}")
        lines.append(f"  surface_any_false_alarm_rate      = {float(dfa['surface_any_type1_error'].mean()) if 'surface_any_type1_error' in dfa.columns else None}")
        lines.append(f"  surface_mean_false_alarm_count    = {float(dfa['surface_n_fp'].mean()) if 'surface_n_fp' in dfa.columns else None}")
        lines.append(f"  surface_mean_false_alarm_rate     = {float(dfa['surface_fp_rate_over_points'].mean()) if 'surface_fp_rate_over_points' in dfa.columns else None}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    results_path = Path("./cls_results_all_type1_update")
    file_name = "dam_cls_type1_rows"

    base = extract_zip_if_needed(results_path)
    rows_json = find_rows_json(base, file_name)
    per_real_dir = find_per_real_dir(base)

    rows_payload = load_json(rows_json)
    df = pd.DataFrame(rows_payload["rows"])
    if "scenario" in df.columns:
        df = df[df["scenario"].isin(["type1_null", "type1"])].copy()

    analysis_dir = base / "analysis_from_saved"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    CP = _as_points_xyz(np.load(base / "first_cp.npy"))
    SURF = _as_points_xyz(np.load(base / "first_surface_xyz.npy"))

    reject_cp_05 = build_reject_matrix(per_real_dir, "0.05", CP.shape[0], ("cp_localization", "reject_mask"))
    reject_cp_01 = build_reject_matrix(per_real_dir, "0.01", CP.shape[0], ("cp_localization", "reject_mask"))
    reject_surf_05 = build_reject_matrix(per_real_dir, "0.05", SURF.shape[0], ("surface_pointwise", "reject_mask"))
    reject_surf_01 = build_reject_matrix(per_real_dir, "0.01", SURF.shape[0], ("surface_pointwise", "reject_mask"))

    freq_cp_05 = reject_cp_05.mean(axis=0)
    freq_cp_01 = reject_cp_01.mean(axis=0)
    freq_surf_05 = reject_surf_05.mean(axis=0)
    freq_surf_01 = reject_surf_01.mean(axis=0)

    plot_global_type1_bar(df, analysis_dir / "global_type1_error_rate.png")

    plot_hist_two_rows(
        df.loc[df["alpha"] == 0.05, "n_fp"].values,
        df.loc[df["alpha"] == 0.01, "n_fp"].values,
        "number of false alarms per realization",
        "CP false alarms over realizations",
        analysis_dir / "hist_cp_false_alarms.png",
    )
    plot_hist_two_rows(
        df.loc[df["alpha"] == 0.05, "surface_n_fp"].values,
        df.loc[df["alpha"] == 0.01, "surface_n_fp"].values,
        "number of false alarming surface points per realization",
        "Surface-point false alarms over realizations",
        analysis_dir / "hist_surface_false_alarms.png",
    )
    plot_hist_two_rows(
        df.loc[df["alpha"] == 0.05, "surface_fp_rate_over_points"].values,
        df.loc[df["alpha"] == 0.01, "surface_fp_rate_over_points"].values,
        "surface-point false alarm rate per realization",
        "Surface-point false alarm rates over realizations",
        analysis_dir / "hist_surface_false_alarm_rate.png",
    )

    plot_hist_two_cols(
        df.loc[df["alpha"] == 0.05, "n_fp"].values,
        df.loc[df["alpha"] == 0.01, "n_fp"].values,
        "fd",
        "number of false alarms per realization",
        "CP false alarms over realizations",
        analysis_dir / "hist_cp_false_alarms1.png",
    )
    plot_hist_two_cols(
        df.loc[df["alpha"] == 0.05, "surface_n_fp"].values,
        df.loc[df["alpha"] == 0.01, "surface_n_fp"].values,
        "fd",
        "number of false alarming surface points per realization",
        "Surface-point false alarms over realizations",
        analysis_dir / "hist_surface_false_alarms1.png",
    )
    plot_hist_two_cols(
        df.loc[df["alpha"] == 0.05, "surface_fp_rate_over_points"].values,
        df.loc[df["alpha"] == 0.01, "surface_fp_rate_over_points"].values,
        "fd",
        "surface-point false alarm rate per realization",
        "Surface-point false alarm rates over realizations",
        analysis_dir / "hist_surface_false_alarm_rate1.png",
    )

    plot_frequency_per_points(
        CP,
        freq_cp_05,
        freq_cp_01,
        30,
        analysis_dir / "local_type1_error_per_cp.png",
        "Empirical local Type I error per control point",
        "false-alarm frequency",
    )
    plot_frequency_per_selected_points(
        CP,
        np.ones(len(CP), dtype=bool),
        freq_cp_05,
        freq_cp_01,
        analysis_dir / "local_type1_error_per_cp__.png",
        "Empirical local Type I error per control point",
        "false-alarm frequency",
    )
    plot_frequency_per_points(
        SURF,
        freq_surf_05,
        freq_surf_01,
        15,
        analysis_dir / "local_type1_error_per_surface_point.png",
        "Empirical local Type I error per surface point",
        "false-alarm frequency",
    )

    summary = {
        "alpha_0.05": {
            "n_realizations": int((df["alpha"] == 0.05).sum()),
            "global_type1_error_rate": float(df.loc[df["alpha"] == 0.05, "global_type1_error"].mean()),
            "cp_any_false_alarm_rate": float(df.loc[df["alpha"] == 0.05, "local_any_type1_error"].mean()),
            "cp_mean_false_alarm_count": float(df.loc[df["alpha"] == 0.05, "n_fp"].mean()),
            "cp_mean_false_alarm_rate": float(df.loc[df["alpha"] == 0.05, "local_fp_rate_over_cp"].mean()),
            "surface_any_false_alarm_rate": float(df.loc[df["alpha"] == 0.05, "surface_any_type1_error"].mean()),
            "surface_mean_false_alarm_count": float(df.loc[df["alpha"] == 0.05, "surface_n_fp"].mean()),
            "surface_mean_false_alarm_rate": float(df.loc[df["alpha"] == 0.05, "surface_fp_rate_over_points"].mean()),
        },
        "alpha_0.01": {
            "n_realizations": int((df["alpha"] == 0.01).sum()),
            "global_type1_error_rate": float(df.loc[df["alpha"] == 0.01, "global_type1_error"].mean()),
            "cp_any_false_alarm_rate": float(df.loc[df["alpha"] == 0.01, "local_any_type1_error"].mean()),
            "cp_mean_false_alarm_count": float(df.loc[df["alpha"] == 0.01, "n_fp"].mean()),
            "cp_mean_false_alarm_rate": float(df.loc[df["alpha"] == 0.01, "local_fp_rate_over_cp"].mean()),
            "surface_any_false_alarm_rate": float(df.loc[df["alpha"] == 0.01, "surface_any_type1_error"].mean()),
            "surface_mean_false_alarm_count": float(df.loc[df["alpha"] == 0.01, "surface_n_fp"].mean()),
            "surface_mean_false_alarm_rate": float(df.loc[df["alpha"] == 0.01, "surface_fp_rate_over_points"].mean()),
        },
    }

    with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_text_summary(df, analysis_dir / "analysis_summary.txt")
    print(f"Saved analysis outputs under: {analysis_dir.resolve()}")
