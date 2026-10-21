"""
phase_a_bootstrap.py: bootstrap CIs on the multi-seed matched-regressor Cobb predictions.

Loads one .npz file per configuration and seed (CONFIGS x SEEDS) from
outputs/multiseed/, validates their structure,
computes per-image MAE, then computes:
  - Per-configuration multi-seed mean MAE
  - Bootstrap 95% CIs (image-level resampling across 128 test images)
  - Paired comparisons against the aspect-padded baseline
  - Sign-test counts at both the per-seed and per-image level
  - LaTeX-ready summary table

The script prints what it finds in each .npz file so that the structure can
be checked.

Inputs:  outputs/multiseed/<prefix>_seed<N>.npz for each configuration in
         CONFIGS and each seed in SEEDS (from run_multiseed_matched.py).

Usage:
    python phase_a_bootstrap.py      # run from the repository root

Outputs:
    phase_a_bootstrap_results.csv          - raw bootstrap numbers
    phase_a_bootstrap_table.tex            - LaTeX table
    phase_a_per_image_mae.csv              - per-image MAE for downstream use
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --- Configuration: edit if file paths differ ---

MULTISEED_DIR = Path("outputs/multiseed")
LABELS_CSV = Path("medvae/data/boostnet_with_angles.csv")
OUT_DIR = Path(".")

# (display_name, file_prefix); file_prefix is the base of the filename
# before _seed{N}.npz
CONFIGS = [
    ("Pretrained 4x aspect-padded (baseline)", "matched_4x"),
    ("Pretrained 8x aspect-padded",            "matched_8x"),
    ("Pretrained 4x whole-image 384x768",      "matched_whole_768"),
    ("Pretrained 4x whole-image 384x1024",     "matched_whole_1024"),
    ("Pretrained 4x tile-and-stitch",          "matched_tiled_pretrained"),
    ("Patch-finetuned 4x tile-and-stitch",     "matched_tiled"),
    ("Diffusion U-Net features (t=250, level 1)", "matched_diffusion_features"),
]
BASELINE_PREFIX = "matched_4x"   # used for paired comparisons
SEEDS = [0, 1, 7, 42, 100]

# Bootstrap settings
N_BOOTSTRAP = 10_000
ALPHA = 0.05                      # for 95% CI
RNG_SEED = 20260606               # deterministic bootstrap

# --- Helpers ---

# Plausible key names; the one each .npz file uses is detected
PRED_KEY_CANDIDATES   = ["preds", "predictions", "y_pred", "pred", "test_preds"]
TARGET_KEY_CANDIDATES = ["targets", "labels", "y_true", "true", "test_targets",
                         "y", "gt", "ground_truth"]
ID_KEY_CANDIDATES     = ["image_uuid", "image_uuids", "uuid", "uuids",
                         "image_id", "image_ids", "id", "ids"]


def _detect_key(npz: np.lib.npyio.NpzFile, candidates: List[str]) -> str | None:
    """Return the first candidate key present in the .npz file, or None."""
    for k in candidates:
        if k in npz.files:
            return k
    return None


def load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load (preds, targets, ids) from a .npz file.

    Tolerant of varying key conventions. preds and targets are expected to
    have shape (n_images, 3) for the three Cobb angles. ids may be None if
    the file does not store image identifiers.
    """
    with np.load(path, allow_pickle=True) as npz:
        pred_key   = _detect_key(npz, PRED_KEY_CANDIDATES)
        target_key = _detect_key(npz, TARGET_KEY_CANDIDATES)
        id_key     = _detect_key(npz, ID_KEY_CANDIDATES)

        if pred_key is None or target_key is None:
            raise KeyError(
                f"Could not find prediction/target arrays in {path}. "
                f"Available keys: {npz.files}. "
                f"Looked for preds in {PRED_KEY_CANDIDATES} and "
                f"targets in {TARGET_KEY_CANDIDATES}."
            )

        preds   = np.asarray(npz[pred_key])
        targets = np.asarray(npz[target_key])
        ids     = np.asarray(npz[id_key]) if id_key is not None else None

    if preds.shape != targets.shape:
        raise ValueError(
            f"{path}: preds shape {preds.shape} != targets shape {targets.shape}"
        )
    if preds.ndim != 2 or preds.shape[1] != 3:
        raise ValueError(
            f"{path}: expected preds shape (n, 3), got {preds.shape}"
        )

    return preds, targets, ids


def per_image_mae(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-image MAE, averaged across the three Cobb angles. Shape: (n,)."""
    return np.mean(np.abs(preds - targets), axis=1)


def bootstrap_ci_mean(
    per_image_values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean of per-image values.

    Resamples images with replacement n_bootstrap times. Returns
    (point_estimate, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(per_image_values)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = per_image_values[idx].mean(axis=1)
    point = float(per_image_values.mean())
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return point, lo, hi


def bootstrap_paired_diff(
    a_per_image: np.ndarray,
    b_per_image: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI on paired mean difference (a - b).

    Both arrays must align on test images (same ordering). Returns
    (point_estimate, ci_low, ci_high) for mean(a - b).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED + 1)
    if a_per_image.shape != b_per_image.shape:
        raise ValueError(
            f"shape mismatch: a {a_per_image.shape} vs b {b_per_image.shape}"
        )
    diff = a_per_image - b_per_image
    n = len(diff)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diff[idx].mean(axis=1)
    point = float(diff.mean())
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return point, lo, hi


# --- Main analysis ---

def main() -> int:
    print("=" * 70)
    print("Bootstrap CIs on multi-seed encoder predictions")
    print("=" * 70)
    print()

    if not MULTISEED_DIR.is_dir():
        print(f"Error: {MULTISEED_DIR} does not exist. The script expects to be "
              f"run from the repository root.", file=sys.stderr)
        return 1

    # --- Load all prediction files and validate structure ---
    print(f"[1/4] Loading predictions from {MULTISEED_DIR}/")
    print()

    # results[prefix][seed] -> {"preds": ..., "targets": ..., "ids": ...,
    #                           "per_image_mae": ..., "per_image_abs_err": ...}
    results: Dict[str, Dict[int, Dict]] = {p: {} for _, p in CONFIGS}

    reference_targets = None    # to verify all files share the same test set
    reference_ids = None
    reference_n = None

    for display_name, prefix in CONFIGS:
        for seed in SEEDS:
            path = MULTISEED_DIR / f"{prefix}_seed{seed}.npz"
            if not path.is_file():
                print(f"    Missing file: {path}", file=sys.stderr)
                return 1

            preds, targets, ids = load_predictions(path)
            piMAE = per_image_mae(preds, targets)
            piAbsErr = np.abs(preds - targets)   # shape (n, 3) for per-curve

            results[prefix][seed] = {
                "preds": preds,
                "targets": targets,
                "ids": ids,
                "per_image_mae": piMAE,
                "per_image_abs_err": piAbsErr,
            }

            # Sanity-check that all files share the same test set
            if reference_targets is None:
                reference_targets = targets
                reference_ids = ids
                reference_n = len(targets)
                print(f"    Test set size: {reference_n} images")
                print(f"    Per-image array shape: {preds.shape}")
                print(f"    Image IDs available: {ids is not None}")
                print()
            else:
                if len(targets) != reference_n:
                    print(f"    WARNING: {path} has {len(targets)} images, "
                          f"expected {reference_n}. Comparisons across files may be invalid.", file=sys.stderr)
                # Targets should match exactly across configs and seeds
                if not np.allclose(targets, reference_targets):
                    print(f"    WARNING: {path} targets do not match the reference targets. "
                          f"The files may use different test sets; paired comparisons are likely invalid.", file=sys.stderr)

    print(f"    Loaded all {len(CONFIGS)} configs * {len(SEEDS)} seeds = "
          f"{len(CONFIGS) * len(SEEDS)} prediction files.")
    print()

    # --- Compute summary stats per configuration ---
    print("[2/4] Computing summary statistics")
    print()

    summary_rows = []
    baseline_seed_means = None      # per-seed mean MAE for the baseline
    baseline_pooled_per_image = None # mean across seeds, per image

    # First pass: compute per-seed means and pooled per-image MAE for each config
    cfg_pooled: Dict[str, np.ndarray] = {}  # pooled per-image MAE across seeds
    cfg_seed_means: Dict[str, np.ndarray] = {}

    for display_name, prefix in CONFIGS:
        seed_means = np.array([
            results[prefix][s]["per_image_mae"].mean() for s in SEEDS
        ])
        # Stack per-image MAE across seeds, then average to a single
        # (n_test,) array per configuration.
        stacked = np.stack(
            [results[prefix][s]["per_image_mae"] for s in SEEDS],
            axis=0,
        )  # shape (n_seeds, n_test)
        pooled_per_image = stacked.mean(axis=0)

        cfg_pooled[prefix] = pooled_per_image
        cfg_seed_means[prefix] = seed_means

        if prefix == BASELINE_PREFIX:
            baseline_seed_means = seed_means
            baseline_pooled_per_image = pooled_per_image

    # Second pass: bootstrap CIs and paired comparisons against baseline
    # Use deterministic per-config RNG offsets (avoid hash() which can vary
    # across runs depending on PYTHONHASHSEED)
    for cfg_idx, (display_name, prefix) in enumerate(CONFIGS):
        seed_means = cfg_seed_means[prefix]
        pooled_per_image = cfg_pooled[prefix]

        seed_mean = float(seed_means.mean())
        seed_std  = float(seed_means.std(ddof=1))   # sample std

        boot_point, boot_lo, boot_hi = bootstrap_ci_mean(
            pooled_per_image,
            rng=np.random.default_rng(RNG_SEED + 100 * (cfg_idx + 1)),
        )

        if prefix == BASELINE_PREFIX:
            # Paired comparison against self is degenerate
            paired_point, paired_lo, paired_hi = 0.0, 0.0, 0.0
            sign_seeds = "--"
            sign_images = "--"
        else:
            # Paired difference: current - baseline (lower = better, so negative = improvement)
            paired_point, paired_lo, paired_hi = bootstrap_paired_diff(
                pooled_per_image, baseline_pooled_per_image,
                rng=np.random.default_rng(RNG_SEED + 100 * (cfg_idx + 1) + 50),
            )
            # Per-seed sign test: how many seeds beat the baseline?
            n_seeds_beat = int(np.sum(seed_means < baseline_seed_means))
            sign_seeds = f"{n_seeds_beat}/{len(SEEDS)}"
            # Per-image sign test on pooled per-image MAE
            n_images_beat = int(np.sum(pooled_per_image < baseline_pooled_per_image))
            sign_images = f"{n_images_beat}/{len(pooled_per_image)}"

        summary_rows.append({
            "display_name": display_name,
            "prefix": prefix,
            "mean_mae": seed_mean,
            "std_mae": seed_std,
            "boot_lo": boot_lo,
            "boot_hi": boot_hi,
            "diff_vs_baseline_mean": paired_point,
            "diff_vs_baseline_ci_lo": paired_lo,
            "diff_vs_baseline_ci_hi": paired_hi,
            "sign_test_seeds": sign_seeds,
            "sign_test_images": sign_images,
            "seed_means_csv": ",".join(f"{m:.4f}" for m in seed_means),
        })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = OUT_DIR / "phase_a_bootstrap_results.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"    Saved raw results to {out_csv}")
    print()

    # --- Print table to stdout ---
    print("[3/4] Summary table")
    print()
    header = (
        f"{'Configuration':<48} "
        f"{'MAE (deg)':<22} "
        f"{'95% CI':<18} "
        f"{'vs baseline':<22} "
        f"{'Seeds':<6} "
        f"{'Images':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        mae_cell = f"{r['mean_mae']:.2f} ± {r['std_mae']:.2f}"
        ci_cell  = f"[{r['boot_lo']:.2f}, {r['boot_hi']:.2f}]"
        if r["prefix"] == BASELINE_PREFIX:
            diff_cell = "--"
        else:
            d = r["diff_vs_baseline_mean"]
            dlo = r["diff_vs_baseline_ci_lo"]
            dhi = r["diff_vs_baseline_ci_hi"]
            diff_cell = f"{d:+.2f} [{dlo:+.2f}, {dhi:+.2f}]"
        print(
            f"{r['display_name']:<48} "
            f"{mae_cell:<22} "
            f"{ci_cell:<18} "
            f"{diff_cell:<22} "
            f"{r['sign_test_seeds']:<6} "
            f"{r['sign_test_images']:<8}"
        )
    print()

    # --- LaTeX table ---
    print("[4/4] Writing LaTeX table")
    tex_path = OUT_DIR / "phase_a_bootstrap_table.tex"
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by phase_a_bootstrap.py\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Multi-seed Cobb regression MAE under the "
                "receptive-field-matched regressor, with bootstrap 95\\% CIs "
                "on the mean (image-level resampling, $B=10{,}000$). "
                "Seeds 0, 1, 7, 42, 100. Test set $n=128$. "
                "``Sign test (seeds)'' counts seeds beating the baseline; "
                "``$\\Delta$ vs baseline'' is the paired difference with bootstrap CI.}\n")
        f.write("\\label{tab:phase-a-bootstrap}\n")
        f.write("\\begin{tabular}{lllll}\n")
        f.write("\\toprule\n")
        f.write("Configuration & MAE (deg) & 95\\% CI & "
                "$\\Delta$ vs baseline (deg) & Sign test (seeds) \\\\\n")
        f.write("\\midrule\n")
        for r in summary_rows:
            mae_cell = f"${r['mean_mae']:.2f} \\pm {r['std_mae']:.2f}$"
            ci_cell  = f"$[{r['boot_lo']:.2f}, {r['boot_hi']:.2f}]$"
            if r["prefix"] == BASELINE_PREFIX:
                diff_cell = "--"
            else:
                d = r["diff_vs_baseline_mean"]
                dlo = r["diff_vs_baseline_ci_lo"]
                dhi = r["diff_vs_baseline_ci_hi"]
                diff_cell = f"${d:+.2f}~[{dlo:+.2f}, {dhi:+.2f}]$"
            # Replace "Nx " (compression ratios) with "N$\times$ " (digits only)
            import re
            display = re.sub(r"(\d)x ", r"\1$\\times$ ", r["display_name"])
            highlight = r["prefix"] == "matched_tiled"
            if highlight:
                display_cell = f"\\textbf{{{display}}}"
            else:
                display_cell = display
            
            row = (
                f"{display_cell} "
                f"& {mae_cell} & {ci_cell} & {diff_cell} & {r['sign_test_seeds']} \\\\\n"
            )

            f.write(row)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"    Saved LaTeX table to {tex_path}")
    print()

    # --- Save per-image MAE for downstream breakdowns ---
    # Include image IDs (if available) and per-curve absolute errors,
    # so phase_a_breakdowns.py (per-curve and per-cohort breakdowns) can join with
    # boostnet_with_angles.csv without reloading the .npz files.
    rows = []
    for display_name, prefix in CONFIGS:
        pooled_mae = cfg_pooled[prefix]
        # Average per-curve absolute errors across seeds; shape (n_test, 3)
        per_curve_stack = np.stack([
            results[prefix][s]["per_image_abs_err"] for s in SEEDS
        ], axis=0)
        pooled_per_curve = per_curve_stack.mean(axis=0)   # (n_test, 3)

        # Image IDs (if any seed has them; they should all match)
        ids = None
        for s in SEEDS:
            if results[prefix][s]["ids"] is not None:
                ids = results[prefix][s]["ids"]
                break

        for img_idx in range(len(pooled_mae)):
            rows.append({
                "prefix": prefix,
                "display_name": display_name,
                "image_idx": img_idx,
                "image_id": str(ids[img_idx]) if ids is not None else "",
                "mean_mae_across_seeds": float(pooled_mae[img_idx]),
                "pt_abs_err": float(pooled_per_curve[img_idx, 0]),
                "mt_abs_err": float(pooled_per_curve[img_idx, 1]),
                "tl_abs_err": float(pooled_per_curve[img_idx, 2]),
            })
    per_img_df = pd.DataFrame(rows)
    per_img_path = OUT_DIR / "phase_a_per_image_mae.csv"
    per_img_df.to_csv(per_img_path, index=False)
    print(f"    Saved per-image MAE (with per-curve breakdown) to {per_img_path}")
    print(f"    Shape: {len(per_img_df)} rows ({len(CONFIGS)} configs * "
          f"{reference_n} images)")
    print()

    print("Bootstrap analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
