"""
phase_a_breakdowns.py: per-curve and per-cohort MAE breakdowns of the multi-seed matched-regressor results.

Inputs:
  - phase_a_per_image_mae.csv (output of phase_a_bootstrap.py) for the
    per-image MAE and per-curve absolute errors averaged across seeds.
  - targets from any of the multi-seed .npz files (used to assign each
    test image to a severity cohort based on its maximum Cobb angle).
    Targets are identical across all the multi-seed files, so only one is loaded.

Computes:
  - Per-curve MAE per configuration (major curve (col 0), col 1, col 2), with bootstrap 95% CIs and paired comparison against the
    baseline within each curve.
  - Per-cohort (mild <25°, moderate 25-50°, severe >50°) MAE per
    configuration, with bootstrap CIs and paired comparison against the
    baseline within each cohort.

Outputs:
  phase_a_per_curve_results.csv     - raw numbers per (config, curve)
  phase_a_per_cohort_results.csv    - raw numbers per (config, cohort)
  phase_a_per_curve_table.tex       - LaTeX table
  phase_a_per_cohort_table.tex      - LaTeX table

Usage:
    python phase_a_breakdowns.py      # run from the repository root
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --- Configuration ---

PER_IMAGE_CSV = Path("phase_a_per_image_mae.csv")
# Any of the multi-seed .npz files can be used for loading targets; they all match.
TARGETS_NPZ   = Path("outputs/multiseed/matched_4x_seed0.npz")
OUT_DIR = Path(".")

# Severity cohort boundaries (max Cobb angle across the three columns):
#   mild     < MILD_MAX
#   moderate in [MILD_MAX, MODERATE_MAX]
#   severe   > MODERATE_MAX
MILD_MAX      = 25.0
MODERATE_MAX  = 50.0

# Display ordering matches the bootstrap table
CONFIGS = [
    ("Pretrained 4x aspect-padded (baseline)", "matched_4x"),
    ("Pretrained 8x aspect-padded",            "matched_8x"),
    ("Pretrained 4x whole-image 384x768",      "matched_whole_768"),
    ("Pretrained 4x whole-image 384x1024",     "matched_whole_1024"),
    ("Pretrained 4x tile-and-stitch",          "matched_tiled_pretrained"),
    ("Patch-finetuned 4x tile-and-stitch",     "matched_tiled"),
    ("Diffusion U-Net features (t=250, level 1)", "matched_diffusion_features"),
]
BASELINE_PREFIX = "matched_4x"

CURVE_NAMES   = ["Major curve (col 0)", "Col 1", "Col 2"]
CURVE_COLUMNS = ["pt_abs_err", "mt_abs_err", "tl_abs_err"]
COHORT_NAMES  = ["mild", "moderate", "severe"]

# Bootstrap settings
N_BOOTSTRAP = 10_000
ALPHA = 0.05
RNG_SEED = 20260606


# --- Helpers (from phase_a_bootstrap.py, duplicated so this script is self-contained) ---

PRED_KEY_CANDIDATES   = ["preds", "predictions", "y_pred", "pred", "test_preds"]
TARGET_KEY_CANDIDATES = ["targets", "labels", "y_true", "true", "test_targets",
                         "y", "gt", "ground_truth"]


def _detect_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def load_targets(path: Path) -> np.ndarray:
    """Load the (n_test, 3) targets array from a multi-seed .npz file."""
    with np.load(path, allow_pickle=True) as npz:
        target_key = _detect_key(npz, TARGET_KEY_CANDIDATES)
        if target_key is None:
            raise KeyError(
                f"Could not find targets in {path}. Keys: {npz.files}."
            )
        targets = np.asarray(npz[target_key]).astype(np.float64)

    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"Expected targets shape (n, 3), got {targets.shape}")
    return targets


def bootstrap_ci_mean(
    per_image_values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for mean of a 1-D array.

    Handles small subgroups gracefully: returns NaN CIs if n < 2.
    """
    n = len(per_image_values)
    if n < 2:
        point = float(per_image_values.mean()) if n == 1 else float("nan")
        return point, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = per_image_values[idx].mean(axis=1)
    return (
        float(per_image_values.mean()),
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


def bootstrap_paired_diff(
    a_per_image: np.ndarray,
    b_per_image: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
) -> Tuple[float, float, float]:
    """Bootstrap CI on paired mean difference (a - b)."""
    if a_per_image.shape != b_per_image.shape:
        raise ValueError(
            f"shape mismatch: a {a_per_image.shape} vs b {b_per_image.shape}"
        )
    n = len(a_per_image)
    if n < 2:
        diff = a_per_image - b_per_image
        return (float(diff.mean()) if n == 1 else float("nan"),
                float("nan"), float("nan"))
    diff = a_per_image - b_per_image
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diff[idx].mean(axis=1)
    return (
        float(diff.mean()),
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


# --- Analyses ---

def compute_per_curve(per_image_df: pd.DataFrame) -> pd.DataFrame:
    """Per-curve MAE breakdown with bootstrap CIs and paired comparisons."""
    baseline_by_curve = {
        c: per_image_df[per_image_df.prefix == BASELINE_PREFIX]
           .sort_values("image_idx")[col].to_numpy()
        for c, col in zip(CURVE_NAMES, CURVE_COLUMNS)
    }

    rows = []
    for cfg_idx, (display, prefix) in enumerate(CONFIGS):
        sub = per_image_df[per_image_df.prefix == prefix].sort_values("image_idx")
        for curve_idx, (curve, col) in enumerate(zip(CURVE_NAMES, CURVE_COLUMNS)):
            vals = sub[col].to_numpy()
            ci_rng = np.random.default_rng(
                RNG_SEED + 1000 * (cfg_idx + 1) + 10 * (curve_idx + 1)
            )
            mae, lo, hi = bootstrap_ci_mean(vals, rng=ci_rng)

            if prefix == BASELINE_PREFIX:
                diff_mean, diff_lo, diff_hi = 0.0, 0.0, 0.0
                sign_imgs = "--"
            else:
                paired_rng = np.random.default_rng(
                    RNG_SEED + 1000 * (cfg_idx + 1) + 10 * (curve_idx + 1) + 5
                )
                diff_mean, diff_lo, diff_hi = bootstrap_paired_diff(
                    vals, baseline_by_curve[curve], rng=paired_rng,
                )
                sign_imgs = f"{int((vals < baseline_by_curve[curve]).sum())}/{len(vals)}"

            rows.append({
                "configuration": display,
                "prefix": prefix,
                "curve": curve,
                "n_images": len(vals),
                "mae": mae,
                "boot_lo": lo,
                "boot_hi": hi,
                "diff_vs_baseline": diff_mean,
                "diff_ci_lo": diff_lo,
                "diff_ci_hi": diff_hi,
                "sign_test_images": sign_imgs,
            })
    return pd.DataFrame(rows)


def assign_cohorts(targets: np.ndarray) -> np.ndarray:
    """Stratify each test image by its maximum Cobb angle.

    Returns an array of cohort labels (str) with length n_images.
    """
    max_cobb = np.max(targets, axis=1)
    cohorts = np.where(
        max_cobb < MILD_MAX, "mild",
        np.where(max_cobb < MODERATE_MAX, "moderate", "severe")
    )
    return cohorts


def compute_per_cohort(
    per_image_df: pd.DataFrame, cohorts: np.ndarray,
) -> pd.DataFrame:
    """Per-cohort MAE breakdown with bootstrap CIs and paired comparisons."""
    baseline_sub = (per_image_df[per_image_df.prefix == BASELINE_PREFIX]
                    .sort_values("image_idx"))
    baseline_mae_by_image = baseline_sub["mean_mae_across_seeds"].to_numpy()

    rows = []
    for cfg_idx, (display, prefix) in enumerate(CONFIGS):
        sub = per_image_df[per_image_df.prefix == prefix].sort_values("image_idx")
        mae_by_image = sub["mean_mae_across_seeds"].to_numpy()
        if len(mae_by_image) != len(cohorts):
            raise ValueError(
                f"Cohort/MAE size mismatch for {prefix}: "
                f"{len(mae_by_image)} vs {len(cohorts)}"
            )

        for cohort_idx, cohort in enumerate(COHORT_NAMES):
            mask = cohorts == cohort
            vals = mae_by_image[mask]
            baseline_vals = baseline_mae_by_image[mask]

            ci_rng = np.random.default_rng(
                RNG_SEED + 2000 * (cfg_idx + 1) + 100 * (cohort_idx + 1)
            )
            mae, lo, hi = bootstrap_ci_mean(vals, rng=ci_rng)

            if prefix == BASELINE_PREFIX:
                diff_mean, diff_lo, diff_hi = 0.0, 0.0, 0.0
                sign_imgs = "--"
            else:
                paired_rng = np.random.default_rng(
                    RNG_SEED + 2000 * (cfg_idx + 1) + 100 * (cohort_idx + 1) + 5
                )
                diff_mean, diff_lo, diff_hi = bootstrap_paired_diff(
                    vals, baseline_vals, rng=paired_rng,
                )
                if len(vals) > 0:
                    sign_imgs = f"{int((vals < baseline_vals).sum())}/{len(vals)}"
                else:
                    sign_imgs = "0/0"

            rows.append({
                "configuration": display,
                "prefix": prefix,
                "cohort": cohort,
                "n_images": int(mask.sum()),
                "mae": mae,
                "boot_lo": lo,
                "boot_hi": hi,
                "diff_vs_baseline": diff_mean,
                "diff_ci_lo": diff_lo,
                "diff_ci_hi": diff_hi,
                "sign_test_images": sign_imgs,
            })
    return pd.DataFrame(rows)


# --- Printing and LaTeX output ---

def print_summary(df: pd.DataFrame, group_col: str, group_names: List[str],
                  n_per_group: Dict[str, int]) -> None:
    """Pretty-print a breakdown table to stdout."""
    print()
    header_groups = "  ".join(
        f"{g} (n={n_per_group[g]:3d})".center(28) for g in group_names
    )
    print(f"{'Configuration':<42} {header_groups}")
    print("-" * (42 + len(header_groups) + 2))
    for _, (display, prefix) in enumerate(CONFIGS):
        sub = df[df.prefix == prefix].set_index(group_col)
        cells = []
        for g in group_names:
            if g in sub.index:
                r = sub.loc[g]
                if np.isnan(r["boot_lo"]):
                    cell = f"{r['mae']:5.2f}    (n<2)        "
                else:
                    cell = f"{r['mae']:5.2f} [{r['boot_lo']:5.2f}, {r['boot_hi']:5.2f}]"
                cells.append(cell.center(28))
            else:
                cells.append(" " * 28)
        print(f"{display:<42} {'  '.join(cells)}")
    print()


def write_latex_table(
    df: pd.DataFrame,
    group_col: str,
    group_names: List[str],
    n_per_group: Dict[str, int],
    title: str,
    label: str,
    out_path: Path,
) -> None:
    """Write a per-(curve|cohort) LaTeX table to disk."""
    import re

    with open(out_path, "w") as f:
        f.write(f"% Auto-generated by phase_a_breakdowns.py\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{title}}}\n")
        f.write(f"\\label{{{label}}}\n")
        col_spec = "l" + "c" * len(group_names)
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        header_cells = [f"{g.title()} (n={n_per_group[g]})"
                        for g in group_names]
        f.write("Configuration & " + " & ".join(header_cells) + " \\\\\n")
        f.write("\\midrule\n")
        for cfg_idx, (display, prefix) in enumerate(CONFIGS):
            sub = df[df.prefix == prefix].set_index(group_col)
            display_tex = re.sub(r"(\d)x ", r"\1$\\times$ ", display)
            if prefix == "matched_tiled":
                display_tex = "\\textbf{" + display_tex + "}"
            cells = []
            for g in group_names:
                if g in sub.index:
                    r = sub.loc[g]
                    if np.isnan(r["boot_lo"]):
                        cells.append(f"${r['mae']:.2f}$ (n$<$2)")
                    else:
                        cells.append(f"${r['mae']:.2f}~[{r['boot_lo']:.2f}, {r['boot_hi']:.2f}]$")
                else:
                    cells.append("--")
            f.write(f"{display_tex} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"    Saved LaTeX table to {out_path}")


# --- Main ---

def main() -> int:
    print("=" * 70)
    print("Per-curve and per-cohort MAE breakdowns")
    print("=" * 70)
    print()

    if not PER_IMAGE_CSV.is_file():
        print(f"Error: {PER_IMAGE_CSV} not found; it is produced by phase_a_bootstrap.py.",
              file=sys.stderr)
        return 1
    if not TARGETS_NPZ.is_file():
        print(f"Error: {TARGETS_NPZ} not found.", file=sys.stderr)
        return 1

    print(f"[1/4] Loading per-image MAE from {PER_IMAGE_CSV}")
    per_image_df = pd.read_csv(PER_IMAGE_CSV)
    n_configs = per_image_df.prefix.nunique()
    n_images  = per_image_df.image_idx.nunique()
    print(f"    {len(per_image_df)} rows ({n_configs} configs * {n_images} images)")
    print()

    print(f"[2/4] Loading targets from {TARGETS_NPZ} for cohort assignment")
    targets = load_targets(TARGETS_NPZ)
    print(f"    Targets shape: {targets.shape}")
    max_cobb = np.max(targets, axis=1)
    print(f"    Max-Cobb-angle range: [{max_cobb.min():.1f}, {max_cobb.max():.1f}] deg")
    print(f"    Severity boundaries: mild < {MILD_MAX} <= moderate <= {MODERATE_MAX} < severe")
    cohorts = assign_cohorts(targets)
    n_per_cohort = {c: int((cohorts == c).sum()) for c in COHORT_NAMES}
    print(f"    Cohort counts: {n_per_cohort}")
    print()

    # --- Per-curve ---
    print("[3/4] Per-curve breakdown")
    per_curve = compute_per_curve(per_image_df)
    per_curve.to_csv(OUT_DIR / "phase_a_per_curve_results.csv", index=False)
    n_per_curve = {c: n_images for c in CURVE_NAMES}
    print_summary(per_curve, "curve", CURVE_NAMES, n_per_curve)
    write_latex_table(
        per_curve, "curve", CURVE_NAMES, n_per_curve,
        title=("Per-curve Cobb regression MAE under the receptive-field-matched "
               "regressor. Each cell shows mean MAE with bootstrap 95\\% CI "
               "(image-level resampling, $B=10{,}000$). Multi-seed predictions "
               "averaged across 5 seeds per image."),
        label="tab:phase-a-per-curve",
        out_path=OUT_DIR / "phase_a_per_curve_table.tex",
    )

    # --- Per-cohort ---
    print("[4/4] Per-cohort breakdown (by max Cobb angle)")
    per_cohort = compute_per_cohort(per_image_df, cohorts)
    per_cohort.to_csv(OUT_DIR / "phase_a_per_cohort_results.csv", index=False)
    print_summary(per_cohort, "cohort", COHORT_NAMES, n_per_cohort)
    write_latex_table(
        per_cohort, "cohort", COHORT_NAMES, n_per_cohort,
        title=(f"Per-severity-cohort Cobb regression MAE under the "
               f"receptive-field-matched regressor. Each test image is "
               f"assigned to a cohort by its maximum Cobb angle across the "
               f"three curves: mild $< {MILD_MAX:.0f}^\\circ$, moderate "
               f"$[{MILD_MAX:.0f}, {MODERATE_MAX:.0f}]^\\circ$, severe $> "
               f"{MODERATE_MAX:.0f}^\\circ$. Bootstrap 95\\% CI in brackets, "
               f"$B=10{{,}}000$."),
        label="tab:phase-a-per-cohort",
        out_path=OUT_DIR / "phase_a_per_cohort_table.tex",
    )

    print()
    print("Breakdowns complete. Questions the output addresses:")
    print("  - Major curve (col 0) vs col 1 vs col 2: are some curves systematically harder under compression?")
    print("  - Is the tile-and-stitch difference concentrated in any one curve?")
    print("  - Cohort balance: the severe cohort may be small, so its CIs should be interpreted with caution.")
    print("  - Is the tile-and-stitch difference concentrated in the moderate or severe cohort?")
    print("    (Native resolution should matter more for larger curves.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
