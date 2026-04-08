import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Ground truth
# -----------------------------
GT_1CM = np.array([110, 109, 118, 127, 137, 147, 157, 167, 177, 188, 199, 200])
GT_2CM = np.array([120, 119, 130, 129, 128, 140, 139, 138, 150, 149, 148,
                   160, 159, 158, 168, 169, 170, 178, 179, 180, 190, 189])

GT_ALL = np.unique(np.r_[GT_1CM, GT_2CM])
GT_ALL_0 = GT_ALL - 1


# -----------------------------
# Utilities
# -----------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_zip_if_needed(results_path: Path) -> Path:
    if results_path.suffix == ".zip":
        out_dir = results_path.with_suffix("")
        out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(results_path, "r") as zf:
            zf.extractall(out_dir)
        return out_dir
    return results_path


def list_files(per_real_dir):
    return sorted(per_real_dir.glob("*.json"))


def _as_points_xyz(arr):
    arr = np.asarray(arr)
    return arr if arr.shape[1] == 3 else arr.T


# -----------------------------
# Core matrices
# -----------------------------
def build_cp_miss_matrix(per_real_dir, alpha, subset_idx, n_cp):
    files = list_files(per_real_dir)
    out = []

    for fp in files:
        data = load_json(fp)
        reject = np.asarray(data["alphas"][alpha]["cp_localization"]["reject_mask"], bool)
        out.append(~reject[subset_idx])

    return np.asarray(out)


def build_cp_detect_matrix(per_real_dir, alpha, subset_idx, n_cp):
    files = list_files(per_real_dir)
    out = []

    for fp in files:
        data = load_json(fp)
        reject = np.asarray(data["alphas"][alpha]["cp_localization"]["reject_mask"], bool)
        out.append(reject[subset_idx])

    return np.asarray(out)


# -----------------------------
# Plot
# -----------------------------
def plot_subset(XYZ, subset_idx, freq05, freq01, title, label, save_path):

    XYZ = np.asarray(XYZ)
    subset_idx = np.asarray(subset_idx)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

    for ax, freq, alpha in zip(axes, [freq05, freq01], ["0.05", "0.01"]):

        sc = ax.scatter(
            XYZ[subset_idx, 0],
            XYZ[subset_idx, 2],
            c=freq,
            cmap="viridis",
            s=35
        )

        other = np.setdiff1d(np.arange(XYZ.shape[0]), subset_idx)

        ax.scatter(
            XYZ[other, 0],
            XYZ[other, 2],
            color="lightgray",
            s=20,
            alpha=0.5
        )

        ax.set_title(f"{title} (alpha={alpha})")
        ax.grid(True, alpha=0.3)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(label)

    axes[1].set_xlabel("X")
    axes[0].set_ylabel("Z")
    axes[1].set_ylabel("Z")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    base = extract_zip_if_needed(Path("./cls_results_all_type2_update"))
    per_real_dir = base / "per_realization"

    CP = _as_points_xyz(np.load(base / "first_cp.npy"))

    remaining_ids = np.setdiff1d(np.arange(CP.shape[0]), GT_ALL_0)

    # -----------------------------
    # GT points (true deformation)
    # -----------------------------
    miss05 = build_cp_miss_matrix(per_real_dir, "0.05", GT_ALL_0, CP.shape[0])
    miss01 = build_cp_miss_matrix(per_real_dir, "0.01", GT_ALL_0, CP.shape[0])

    freq_miss05 = miss05.mean(axis=0)
    freq_miss01 = miss01.mean(axis=0)

    power05 = 1 - freq_miss05
    power01 = 1 - freq_miss01

    # -----------------------------
    # OTHER points
    # -----------------------------
    det05 = build_cp_detect_matrix(per_real_dir, "0.05", remaining_ids, CP.shape[0])
    det01 = build_cp_detect_matrix(per_real_dir, "0.01", remaining_ids, CP.shape[0])

    freq_det05 = det05.mean(axis=0)
    freq_det01 = det01.mean(axis=0)

    out = base / "analysis_clean"
    out.mkdir(exist_ok=True)

    # -----------------------------
    # PLOTS
    # -----------------------------
    plot_subset(
        CP, GT_ALL_0,
        freq_miss05, freq_miss01,
        "Type II error (miss detection) on GT points",
        "miss frequency",
        out / "cp_miss_GT.png"
    )

    plot_subset(
        CP, GT_ALL_0,
        power05, power01,
        "Power on GT points",
        "power",
        out / "cp_power_GT.png"
    )

    plot_subset(
        CP, remaining_ids,
        freq_det05, freq_det01,
        "Detection frequency on other points",
        "detection frequency",
        out / "cp_detection_other.png"
    )

    print("Done. Results in:", out)