"""
compare_cobb_tiled.py

Phase 3 step 3 — paired comparison of Cobb predictions from two regressors:
the new patch-encoder-on-tiled-latents regressor vs the previous baseline
(pretrained encoder + 384x384 spatial regressor, 7.50° test MAE).

Both inputs are .npz files with the schema:
    test_preds        (N, 3) float32   predicted Cobb angles
    test_targets      (N, 3) float32   ground-truth Cobb angles
    test_uuids        (N,)   <U32      image UUIDs (for paired alignment)
    per_image_mae     (N,)   float32   precomputed per-image MAE
    per_angle_mae     (3,)   float32   precomputed per-angle MAE
    best_val_mae      ()     float64   best val MAE seen during training
    mean_baseline_mae ()     float64   predict-train-mean baseline MAE

Usage:
    python compare_cobb_tiled.py \\
        --tiled    ./cobb_regressor_tiled/cobb_tiled.npz \\
        --baseline ./cobb_pretrained.npz \\
        --out-plot ./cobb_tiled_vs_baseline.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_predictions(npz_path: Path) -> pd.DataFrame:
    """Load an npz of Cobb predictions and return as a paired-comparison-friendly DataFrame."""
    data = np.load(npz_path, allow_pickle=True)
    preds = data["test_preds"]
    targets = data["test_targets"]
    uuids = data["test_uuids"].astype(str)
    df = pd.DataFrame({
        "image_uuid": uuids,
        "pred_pt":    preds[:, 0],    "true_pt":    targets[:, 0],
        "pred_mt":    preds[:, 1],    "true_mt":    targets[:, 1],
        "pred_tl":    preds[:, 2],    "true_tl":    targets[:, 2],
    })
    df["per_image_mae"] = np.mean(
        np.abs(preds - targets), axis=1).astype(np.float32)
    return df, float(data["mean_baseline_mae"]), float(data["best_val_mae"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiled",    required=True, type=Path)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--out-plot", required=True, type=Path)
    args = p.parse_args()

    tiled_df,    tiled_baseline,    tiled_best_val    = load_predictions(args.tiled)
    baseline_df, base_baseline_mae, base_best_val     = load_predictions(args.baseline)
    print(f"Tiled regressor:    {len(tiled_df)} test predictions   "
          f"(mean-baseline MAE = {tiled_baseline:.2f}°)")
    print(f"Baseline regressor: {len(baseline_df)} test predictions "
          f"(mean-baseline MAE = {base_baseline_mae:.2f}°)")
    # Sanity: mean-baseline MAEs should match (same test set)
    if abs(tiled_baseline - base_baseline_mae) > 0.05:
        print(f"  WARN: mean-baseline MAEs differ by {abs(tiled_baseline-base_baseline_mae):.3f}°. "
              "Check that both runs used the same train/test split.")

    # Pair by image_uuid
    merged = tiled_df.merge(baseline_df, on="image_uuid",
                            suffixes=("_tiled", "_base"))
    print(f"Joined on uuid:     {len(merged)} paired predictions")
    if len(merged) < len(tiled_df) or len(merged) < len(baseline_df):
        print(f"  WARN: lost predictions in the join "
              f"({len(tiled_df)} + {len(baseline_df)} → {len(merged)}). "
              "Some UUIDs don't match.")

    tiled_mae    = merged["per_image_mae_tiled"].values
    baseline_mae = merged["per_image_mae_base"].values
    diff = tiled_mae - baseline_mae   # negative = tiled is better

    print("\n" + "=" * 60)
    print(f"RESULTS (n = {len(merged)} test images)")
    print("=" * 60)
    print(f"Baseline (pretrained encoder + 384x384 regressor):  "
          f"{baseline_mae.mean():.3f}° ± {baseline_mae.std():.3f}°")
    print(f"Tiled    (patch encoder + 1D-along-spine):          "
          f"{tiled_mae.mean():.3f}° ± {tiled_mae.std():.3f}°")
    print(f"\nPaired Δ MAE (tiled − baseline):  "
          f"{diff.mean():+.3f}° ± {diff.std():.3f}°")
    n_wins = int((diff < 0).sum())
    print(f"Tiled wins (lower MAE):           "
          f"{n_wins}/{len(diff)} ({n_wins/len(diff)*100:.1f}%)")
    t, pval = stats.ttest_rel(tiled_mae, baseline_mae)
    print(f"Paired t-test:                    t = {t:.2f}, p = {pval:.4f}")
    w, wpval = stats.wilcoxon(tiled_mae, baseline_mae)
    print(f"Wilcoxon signed-rank (more robust): W = {w:.1f}, p = {wpval:.4f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(diff, bins=25, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", lw=1, label="no diff")
    axes[0].axvline(diff.mean(), color="green", linestyle="--", lw=1,
                    label=f"mean = {diff.mean():+.2f}°")
    axes[0].set_xlabel("Δ MAE per image (tiled − baseline, °)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Per-image Δ distribution (n={len(diff)})")
    axes[0].legend()

    lo = min(baseline_mae.min(), tiled_mae.min())
    hi = max(baseline_mae.max(), tiled_mae.max())
    axes[1].scatter(baseline_mae, tiled_mae, alpha=0.5, s=20)
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    axes[1].set_xlabel("Baseline per-image MAE (°)")
    axes[1].set_ylabel("Tiled per-image MAE (°)")
    axes[1].set_title("Paired per-image MAE\n(below y=x ⇒ tiled is better)")
    axes[1].legend()
    axes[1].set_aspect("equal")

    plt.tight_layout()
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_plot, dpi=130, bbox_inches="tight")
    print(f"\nSaved {args.out_plot}")


if __name__ == "__main__":
    main()
