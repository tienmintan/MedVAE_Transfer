"""
investigate_joint_dataset_shift.py: per-position (joint) distribution-shift diagnostic between VinDr and BoostNet latents.

Follow-up to investigate_dataset_shift_content_only.py, which tests whether
the marginal per-channel offsets between VinDr and BoostNet latents are
explained by differential padding fractions. That diagnostic motivated the
content-aware cropping pipeline.

This script tests whether a shift remaining after cropping can be explained
by a joint spatial structure that the marginal analysis cannot detect:
padding positioned at different spatial locations in the latent grid
between corpora, even when marginal padding fractions are similar.
Specifically:
  - BoostNet (1:2.8 native aspect) -> after cropping + repad-to-1:1.5
    -> always has padding in vertical bars at the latent grid's left
    and right edges.
  - VinDr (varied aspect, mean ~1:1.8) -> padding pattern varies per
    image, sometimes vertical bars, sometimes top/bottom, sometimes
    both.

The score is the per-(channel, spatial position) effective Z-score between
corpora:
    z(c, i, j) = |μ_BN(c, i, j) - μ_V(c, i, j)|
               / sqrt(σ_BN(c, i, j)² + σ_V(c, i, j)²)

This captures asymmetric spatial structure that the channel-pooled
marginal analysis averages away. It is computed for both pipelines
(original aspect-padded and cropped) and compared.

Inputs:  VinDr and BoostNet SD-VAE latent .npz files from the original and
         the cropped pipelines (four files).
Outputs (in --out-dir, default outputs/figures/joint_shift/):
  joint_shift_padding_structure.png  - per-(i, j) padding probability
  joint_shift_zscore.png             - per-(c, i, j) Z-score heatmaps
  joint_shift_summary.csv            - aggregate statistics

Usage:
    python investigate_joint_dataset_shift.py \\
        --vindr-orig       ./latents_vindr/latents_vindr.npz \\
        --boostnet-orig    ./latents_boostnet_512/latents_boostnet_512.npz \\
        --vindr-cropped    ./latents_vindr_cropped/latents_vindr_cropped.npz \\
        --boostnet-cropped ./latents_boostnet_cropped/latents_boostnet_cropped.npz \\
        --out-dir          ./outputs/figures/joint_shift

Author: Tien Min, June 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse the spike detection + padding classification from the marginal
# analysis. Keeps the two diagnostics methodologically consistent.
from investigate_dataset_shift_content_only import (
    find_spike_location, classify_padding_positions,
)


# --- Configuration ---

DEFAULT_N_VINDR_SUBSAMPLE = 2000     # matches investigate_dataset_shift_content_only.py
DEFAULT_TOLERANCE = 0.3              # matches investigate_dataset_shift_content_only.py
EPS_FOR_ZSCORE = 1e-3                # numerical floor when pooled std is tiny
RNG_SEED = 20260606


# --- Helpers ---

def per_position_mean_std(latents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each (channel, i, j), compute mean and std across the N samples.

    latents: (N, C, H, W).
    Returns: (mean, std) each of shape (C, H, W).
    """
    arr = latents.astype(np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def per_position_z_score(
    mean_a: np.ndarray, std_a: np.ndarray,
    mean_b: np.ndarray, std_b: np.ndarray,
    eps: float = EPS_FOR_ZSCORE,
) -> np.ndarray:
    """Per-position effective Z-score: |μ_a - μ_b| / sqrt(σ_a² + σ_b²)."""
    pooled = np.sqrt(std_a ** 2 + std_b ** 2)
    return np.abs(mean_a - mean_b) / np.maximum(pooled, eps)


def per_position_padding_prob(padding_mask: np.ndarray) -> np.ndarray:
    """For each (i, j), fraction of images where this position is padding.

    padding_mask: (N, H, W).
    Returns: (H, W) padding probability.
    """
    return padding_mask.astype(np.float64).mean(axis=0)


def load_corpus(path: Path, name: str, subsample: int | None = None,
                rng: np.random.Generator | None = None,
                filter_boostnet_test: bool = True) -> np.ndarray:
    """Load a corpus's latents, as in investigate_dataset_shift_content_only.py.

    If `subsample` is provided and the corpus is larger, randomly subsamples.
    If the corpus has a `splits` array containing 'boostnet_test', removes
    those rows (the held-out test set is excluded when diagnosing
    training-distribution shift).
    """
    print(f"  Loading {name} from {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} latents not found at {path}")
    data = np.load(path, allow_pickle=True)
    lats = data["latents"]
    print(f"    Shape: {lats.shape}")

    if filter_boostnet_test and "splits" in data.files:
        splits = data["splits"].astype(str)
        mask = splits != "boostnet_test"
        if not mask.all():
            lats = lats[mask]
            print(f"    Filtered to non-test: {len(lats)} samples")

    if subsample is not None and len(lats) > subsample:
        rng = rng or np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(lats), size=subsample, replace=False)
        lats = lats[idx]
        print(f"    Subsampled to {len(lats)} samples")

    return lats


# --- Per-pipeline analysis ---

def analyse_pipeline(
    vindr_lats: np.ndarray, boost_lats: np.ndarray,
    tolerance: float, pipeline_name: str,
) -> Dict:
    """Run the joint analysis for one pipeline (original or cropped).

    Returns a dict with all numerical results, ready for plotting and
    aggregation.
    """
    print(f"\n  [{pipeline_name}] Detecting padding spikes...")
    v_spike = find_spike_location(vindr_lats)
    b_spike = find_spike_location(boost_lats)
    print(f"    VinDr   spike: {v_spike}")
    print(f"    BoostNet spike: {b_spike}")

    spike_diff = np.abs(v_spike - b_spike)
    if spike_diff.max() > 0.5:
        print(f"    Note: spikes disagree by max {spike_diff.max():.3f}; "
              "using per-corpus spikes.")
        v_pad_mask = classify_padding_positions(vindr_lats, v_spike, tolerance)
        b_pad_mask = classify_padding_positions(boost_lats, b_spike, tolerance)
    else:
        canonical = (v_spike + b_spike) / 2
        v_pad_mask = classify_padding_positions(vindr_lats, canonical, tolerance)
        b_pad_mask = classify_padding_positions(boost_lats, canonical, tolerance)

    v_pad_frac = float(v_pad_mask.mean())
    b_pad_frac = float(b_pad_mask.mean())
    print(f"    Marginal padding fraction: VinDr {v_pad_frac*100:.1f}%, "
          f"BoostNet {b_pad_frac*100:.1f}% "
          f"(gap {abs(b_pad_frac - v_pad_frac)*100:.1f}pp)")

    print(f"  [{pipeline_name}] Computing per-position stats...")
    v_mean, v_std = per_position_mean_std(vindr_lats)
    b_mean, b_std = per_position_mean_std(boost_lats)

    z = per_position_z_score(v_mean, v_std, b_mean, b_std)   # (C, H, W)
    print(f"    Z-score: mean {z.mean():.3f}, median {np.median(z):.3f}, "
          f"max {z.max():.3f}, p95 {np.quantile(z, 0.95):.3f}")

    v_pad_prob = per_position_padding_prob(v_pad_mask)   # (H, W)
    b_pad_prob = per_position_padding_prob(b_pad_mask)   # (H, W)

    return {
        "pipeline":      pipeline_name,
        "v_pad_frac":    v_pad_frac,
        "b_pad_frac":    b_pad_frac,
        "v_pad_prob":    v_pad_prob,    # (H, W)
        "b_pad_prob":    b_pad_prob,    # (H, W)
        "z_score":       z,             # (C, H, W)
        "v_mean":        v_mean,
        "b_mean":        b_mean,
        "v_std":         v_std,
        "b_std":         b_std,
    }


# --- Plotting ---

def plot_padding_structure(orig: Dict, cropped: Dict, out_path: Path) -> None:
    """2x3 grid: padding probability per (i, j) per corpus, per pipeline."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 9),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.30})

    rows = [("Original pipeline", orig), ("Cropped pipeline", cropped)]

    for row_idx, (label, data) in enumerate(rows):
        v_pp = data["v_pad_prob"]
        b_pp = data["b_pad_prob"]
        diff = b_pp - v_pp

        # VinDr padding probability
        ax = axes[row_idx, 0]
        im = ax.imshow(v_pp, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        ax.set_title(f"VinDr padding probability: {label}\n"
                      f"mean = {data['v_pad_frac']*100:.1f}%", fontsize=10)
        ax.set_xlabel("latent col"); ax.set_ylabel("latent row")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # BoostNet padding probability
        ax = axes[row_idx, 1]
        im = ax.imshow(b_pp, vmin=0, vmax=1, cmap="Oranges", interpolation="nearest")
        ax.set_title(f"BoostNet padding probability: {label}\n"
                      f"mean = {data['b_pad_frac']*100:.1f}%", fontsize=10)
        ax.set_xlabel("latent col"); ax.set_ylabel("latent row")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Difference
        ax = axes[row_idx, 2]
        vmax = max(0.5, np.abs(diff).max())
        im = ax.imshow(diff, vmin=-vmax, vmax=vmax, cmap="RdBu_r",
                        interpolation="nearest")
        ax.set_title(f"BoostNet - VinDr padding prob: {label}\n"
                      f"max |Δ| = {np.abs(diff).max():.2f}", fontsize=10)
        ax.set_xlabel("latent col"); ax.set_ylabel("latent row")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "Spatial padding structure: per-(i, j) probability that a latent "
        "position encodes padding.\n"
        "Asymmetric structure (e.g., BoostNet bars at left/right edges, "
        "VinDr distributed differently) is not captured by marginal "
        "(channel-pooled) analyses.",
        fontsize=11, y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_zscore_heatmaps(orig: Dict, cropped: Dict, out_path: Path) -> None:
    """2 rows × 4 cols: per-(c, i, j) effective Z-score per pipeline."""
    n_channels = orig["z_score"].shape[0]
    fig, axes = plt.subplots(2, n_channels, figsize=(4 * n_channels, 8),
                              gridspec_kw={"hspace": 0.30, "wspace": 0.25})

    # Shared colour scale across the whole figure for visual comparability
    z_all = np.concatenate([orig["z_score"].ravel(),
                              cropped["z_score"].ravel()])
    vmax = float(np.quantile(z_all, 0.99))   # robust upper bound

    rows = [("Original pipeline", orig), ("Cropped pipeline", cropped)]
    for row_idx, (label, data) in enumerate(rows):
        z = data["z_score"]
        for c in range(n_channels):
            ax = axes[row_idx, c]
            im = ax.imshow(z[c], vmin=0, vmax=vmax, cmap="magma",
                            interpolation="nearest")
            stats = (f"mean {z[c].mean():.2f}  "
                     f"med {np.median(z[c]):.2f}  "
                     f"p95 {np.quantile(z[c], 0.95):.2f}")
            ax.set_title(f"ch{c} Z-score: {label}\n{stats}", fontsize=9)
            ax.set_xlabel("latent col"); ax.set_ylabel("latent row")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "Per-(channel, spatial position) effective Z-score between corpora.\n"
        "Low values = positions where the two corpora agree on the latent "
        "distribution; high values = positions where they disagree.",
        fontsize=11, y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# --- Summary table ---

def summary_rows(orig: Dict, cropped: Dict) -> pd.DataFrame:
    """Aggregate Z-score statistics per pipeline."""
    rows = []
    for data in (orig, cropped):
        z = data["z_score"]
        rows.append({
            "pipeline":       data["pipeline"],
            "z_mean":         float(z.mean()),
            "z_median":       float(np.median(z)),
            "z_p95":          float(np.quantile(z, 0.95)),
            "z_max":          float(z.max()),
            "frac_z_above_1": float((z > 1.0).mean()),
            "frac_z_above_2": float((z > 2.0).mean()),
            "vindr_pad_frac":    data["v_pad_frac"],
            "boostnet_pad_frac": data["b_pad_frac"],
            "pad_frac_gap":      abs(data["b_pad_frac"] - data["v_pad_frac"]),
            "max_pad_prob_diff": float(np.abs(
                data["b_pad_prob"] - data["v_pad_prob"]).max()),
        })
    return pd.DataFrame(rows)


def print_verdict(df: pd.DataFrame) -> None:
    """Print a short interpretive summary."""
    orig = df.iloc[0]
    crop = df.iloc[1]

    print("\n" + "=" * 78)
    print("Result: did the cropping intervention reduce joint distribution shift?")
    print("=" * 78)

    z_reduction = (1 - crop["z_mean"] / orig["z_mean"]) * 100 \
        if orig["z_mean"] > 1e-6 else 0
    pad_gap_reduction = (1 - crop["pad_frac_gap"] / orig["pad_frac_gap"]) * 100 \
        if orig["pad_frac_gap"] > 1e-6 else 0

    print(f"\nMean per-(c, i, j) Z-score:")
    print(f"  Original pipeline: {orig['z_mean']:.3f}")
    print(f"  Cropped pipeline:  {crop['z_mean']:.3f}")
    print(f"  Reduction: {z_reduction:.0f}%")
    print()
    print(f"Marginal padding-fraction gap (BoostNet - VinDr):")
    print(f"  Original pipeline: {orig['pad_frac_gap']*100:.1f}pp")
    print(f"  Cropped pipeline:  {crop['pad_frac_gap']*100:.1f}pp")
    print(f"  Reduction: {pad_gap_reduction:.0f}%")
    print()
    print(f"Fraction of (c, i, j) positions with Z > 1 (substantial shift):")
    print(f"  Original pipeline: {orig['frac_z_above_1']*100:.1f}%")
    print(f"  Cropped pipeline:  {crop['frac_z_above_1']*100:.1f}%")
    print()
    print(f"Max per-position padding-probability difference (BoostNet - VinDr):")
    print(f"  Original pipeline: {orig['max_pad_prob_diff']:.2f}")
    print(f"  Cropped pipeline:  {crop['max_pad_prob_diff']:.2f}")

    print()
    if z_reduction >= 50 and crop["z_mean"] < 0.30:
        print(f"Result: the cropping intervention reduced joint shift substantially (mean Z reduced by {z_reduction:.0f}%).")
        print("    Residual is small; any remaining gap in downstream recovery is")
        print("    likely limited by other factors (data scale, fine textural")
        print("    differences) rather than padding structure.")
    elif z_reduction >= 30:
        print(f"Result: the cropping intervention partially reduced joint shift (mean Z reduced by {z_reduction:.0f}%).")
        print("    Marginal padding gap was largely closed, but joint spatial")
        print("    structure remains substantially different between corpora.")
        print("    This residual joint shift may explain a partial rather than")
        print("    complete downstream recovery after cropping.")
    else:
        print(f"Result: the cropping intervention reduced the marginal padding gap but")
        print(f"    not the joint spatial structure (mean Z reduced by {z_reduction:.0f}%). The dominant remaining")
        print("    distribution shift is joint, not marginal. This would")
        print("    explain a limited downstream improvement after cropping")
        print("    even when marginal statistics are aligned.")


# --- Main ---

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vindr-orig",       required=True, type=Path)
    p.add_argument("--boostnet-orig",    required=True, type=Path)
    p.add_argument("--vindr-cropped",    required=True, type=Path)
    p.add_argument("--boostnet-cropped", required=True, type=Path)
    p.add_argument("--out-dir",          type=Path,
                    default=Path("outputs/figures/joint_shift"))
    p.add_argument("--n-vindr-subsample", type=int,
                    default=DEFAULT_N_VINDR_SUBSAMPLE)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="Padding classification tolerance in per-channel "
                         "std units. Default 0.3 matches the marginal "
                         "diagnostic.")
    p.add_argument("--seed", type=int, default=RNG_SEED)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("Joint-distribution shift diagnostic")
    print("=" * 78)

    # --- Original pipeline ---
    print("\n[1/3] Original (aspect-padded) pipeline:")
    vindr_orig    = load_corpus(args.vindr_orig,    "VinDr orig",
                                  subsample=args.n_vindr_subsample, rng=rng)
    boostnet_orig = load_corpus(args.boostnet_orig, "BoostNet orig")
    orig_results  = analyse_pipeline(
        vindr_orig, boostnet_orig, args.tolerance, "Original")

    # Free memory before loading the second pipeline
    del vindr_orig, boostnet_orig

    # --- Cropped pipeline ---
    print("\n[2/3] Cropped (content-aware) pipeline:")
    vindr_crop    = load_corpus(args.vindr_cropped,    "VinDr cropped",
                                  subsample=args.n_vindr_subsample, rng=rng)
    boostnet_crop = load_corpus(args.boostnet_cropped, "BoostNet cropped")
    crop_results  = analyse_pipeline(
        vindr_crop, boostnet_crop, args.tolerance, "Cropped")

    del vindr_crop, boostnet_crop

    # --- Aggregate and report ---
    print("\n[3/3] Aggregating + generating figures...")

    summary_df = summary_rows(orig_results, crop_results)
    summary_csv = args.out_dir / "joint_shift_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved {summary_csv}")

    plot_padding_structure(orig_results, crop_results,
        args.out_dir / "joint_shift_padding_structure.png")
    plot_zscore_heatmaps(orig_results, crop_results,
        args.out_dir / "joint_shift_zscore.png")

    print_verdict(summary_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
