"""
investigate_dataset_shift.py

Diagnose the distribution shift between VinDr-SpineXR and BoostNet at three
levels relevant to the diffusion pipeline:

  1. Raw pixel intensities (post-preprocessing, before encoding).
     Tests whether our DICOM and JPEG pipelines yield comparable pixel
     statistics, or whether one corpus dominates the input distribution
     before anything touches the VAE.

  2. SD-VAE latent statistics (per-channel).
     Tests whether SD-VAE produces systematically different latents for
     the two corpora due to anatomical/view differences, even if pixel
     inputs were perfectly matched.

  3. Pooled-standardisation effect.
     Computes per-corpus mean/std versus the pooled mean/std used by
     train_diffusion.py. If pooled stats are essentially VinDr's (due to
     the 22× sample-count imbalance), BoostNet ends up displaced from
     N(0, 1) after standardisation — directly explaining why conditional
     fine-tune learns weak conditioning.

Outputs:
  - Printed numerical summary table
  - A multi-panel figure with histograms at each level

This is a diagnostic script — it loads pre-computed latents from the
extraction stage rather than re-running encoding from scratch.

Usage:
    python investigate_dataset_shift.py \\
        --vindr-latents    ./outputs/latents/latents_vindr/latents_vindr.npz \\
        --boostnet-latents ./outputs/latents/latents_boostnet_512/latents_boostnet_512.npz \\
        --vindr-images     ./outputs/latents/latents_vindr/preprocessed_samples.npz \\
        --boostnet-images  ./outputs/latents/latents_boostnet_512/preprocessed_samples.npz \\
        --out-fig          ./outputs/figures/dataset_shift_diagnosis.png \\
        --n-pixel-samples  500

Pixel-image npz files are optional. If they don't exist, the script skips
the pixel-space comparison and reports just the latent-level findings
(which is the level that actually matters for the diffusion training).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def per_image_pixel_stats(images: np.ndarray) -> dict:
    """Per-image mean, std, min, max, 1st/99th percentile across the batch.
    images: (N, H, W) or (N, 1, H, W) in arbitrary range.
    Returns dict with arrays of shape (N,) each.
    """
    if images.ndim == 4:
        images = images.squeeze(1)
    return {
        "mean":  images.mean(axis=(1, 2)),
        "std":   images.std(axis=(1, 2)),
        "min":   images.min(axis=(1, 2)),
        "max":   images.max(axis=(1, 2)),
        "p01":   np.percentile(images, 1, axis=(1, 2)),
        "p99":   np.percentile(images, 99, axis=(1, 2)),
    }


def per_channel_latent_stats(latents: np.ndarray) -> dict:
    """Per-channel statistics for SD-VAE latents.
    latents: (N, C, H, W).
    Returns dict mapping stat name -> array of shape (C,).

    All statistics computed in fp64 to avoid overflow on fp16-stored latents.
    """
    # Promote to fp64 for stable variance computation
    latents = latents.astype(np.float64)
    C = latents.shape[1]
    out = {}
    for c in range(C):
        ch = latents[:, c]
        out[f"c{c}_mean"]  = float(ch.mean())
        out[f"c{c}_std"]   = float(ch.std())
        out[f"c{c}_p01"]   = float(np.percentile(ch, 1))
        out[f"c{c}_p99"]   = float(np.percentile(ch, 99))
    out["all_mean"] = float(latents.mean())
    out["all_std"]  = float(latents.std())
    out["all_p01"]  = float(np.percentile(latents, 1))
    out["all_p99"]  = float(np.percentile(latents, 99))
    return out


def print_summary_table(vindr_pix, boost_pix, vindr_lat, boost_lat,
                         pooled_mean, pooled_std):
    """Print the human-readable diagnostic table."""
    print("\n" + "=" * 78)
    print("DATASET DISTRIBUTION DIAGNOSIS")
    print("=" * 78)

    if vindr_pix is not None and boost_pix is not None:
        print("\n--- LEVEL 1: pixel-space stats (post-preprocessing, before VAE) ---")
        print(f"{'Statistic':<25} {'VinDr':>14} {'BoostNet':>14} {'Δ (Vin-Boost)':>15}")
        print("-" * 78)
        for stat in ["mean", "std", "min", "max", "p01", "p99"]:
            v_mean = vindr_pix[stat].mean()
            b_mean = boost_pix[stat].mean()
            print(f"{'per-image ' + stat:<25} {v_mean:>14.4f} {b_mean:>14.4f} "
                  f"{v_mean - b_mean:>15.4f}")
        # Spread across images (does VinDr have more variable per-image stats?)
        print()
        for stat in ["mean", "std"]:
            v_std = vindr_pix[stat].std()
            b_std = boost_pix[stat].std()
            print(f"{'std of per-img ' + stat:<25} {v_std:>14.4f} {b_std:>14.4f} "
                  f"{v_std - b_std:>15.4f}")

    print("\n--- LEVEL 2: SD-VAE latent stats (per-channel) ---")
    print(f"{'Statistic':<25} {'VinDr':>14} {'BoostNet':>14} {'Δ (Vin-Boost)':>15}")
    print("-" * 78)
    n_ch = 4
    for c in range(n_ch):
        for k in ["mean", "std", "p01", "p99"]:
            key = f"c{c}_{k}"
            v = vindr_lat[key]; b = boost_lat[key]
            print(f"{f'ch{c} {k}':<25} {v:>14.4f} {b:>14.4f} {v - b:>15.4f}")
        print()
    for k in ["mean", "std", "p01", "p99"]:
        key = f"all_{k}"
        v = vindr_lat[key]; b = boost_lat[key]
        print(f"{f'all-channel {k}':<25} {v:>14.4f} {b:>14.4f} {v - b:>15.4f}")

    print("\n--- LEVEL 3: pooled-standardisation effect ---")
    print("The training pipeline standardises every sample by the *pooled*")
    print("VinDr+BoostNet mean and std (a single scalar mean and std across")
    print("all channels). This is the form of standardisation that")
    print("train_diffusion.py actually uses.\n")

    print(f"  Pooled (used by train_diffusion.py): mean={pooled_mean:.4f}  std={pooled_std:.4f}")
    print(f"  VinDr-only:                          mean={vindr_lat['all_mean']:.4f}  std={vindr_lat['all_std']:.4f}")
    print(f"  BoostNet-only:                       mean={boost_lat['all_mean']:.4f}  std={boost_lat['all_std']:.4f}")

    # What does standardisation by pooled stats do to each corpus?
    vindr_post_mean = (vindr_lat["all_mean"] - pooled_mean) / pooled_std
    vindr_post_std  = vindr_lat["all_std"] / pooled_std
    boost_post_mean = (boost_lat["all_mean"] - pooled_mean) / pooled_std
    boost_post_std  = boost_lat["all_std"] / pooled_std

    print(f"\nAfter scalar standardisation by POOLED stats (all channels collapsed):")
    print(f"  VinDr     post: mean={vindr_post_mean:+.4f}  std={vindr_post_std:.4f}  (ideal: 0, 1)")
    print(f"  BoostNet  post: mean={boost_post_mean:+.4f}  std={boost_post_std:.4f}  (ideal: 0, 1)")

    # Per-channel post-standardisation — the more important diagnostic
    print(f"\n--- LEVEL 3b: PER-CHANNEL post-standardisation ---")
    print(f"Even when the *aggregate* mean and std are normalised, each")
    print(f"individual channel may still be displaced from N(0, 1). The")
    print(f"diffusion model sees the per-channel structure, not the aggregate.\n")
    print(f"{'Channel':<10} {'VinDr (μ, σ)':<22} {'BoostNet (μ, σ)':<22}")
    print("-" * 60)
    for c in range(4):
        v_mean_post = (vindr_lat[f"c{c}_mean"] - pooled_mean) / pooled_std
        v_std_post  = vindr_lat[f"c{c}_std"] / pooled_std
        b_mean_post = (boost_lat[f"c{c}_mean"] - pooled_mean) / pooled_std
        b_std_post  = boost_lat[f"c{c}_std"] / pooled_std
        print(f"ch{c:<7}  ({v_mean_post:+.3f}, {v_std_post:.3f})       "
              f"({b_mean_post:+.3f}, {b_std_post:.3f})")

    # SD-VAE convention diagnostic
    print(f"\n--- LEVEL 3c: SD-VAE scaling-factor convention ---")
    print(f"Stable Diffusion's published convention is to multiply raw SD-VAE")
    print(f"latents by 0.18215 to normalise them to ≈N(0, 1) before diffusion.")
    print(f"If our latents were saved as raw (un-scaled), they'd have native")
    print(f"std much larger than 1.\n")
    print(f"  VinDr   raw std:                {vindr_lat['all_std']:.4f}")
    print(f"  BoostNet raw std:               {boost_lat['all_std']:.4f}")
    print(f"  Expected if SD-VAE scaling applied:  ≈ 1.0")
    print(f"  Expected if raw (un-scaled):         ≈ 5-8")
    sf = 0.18215
    print(f"\n  If we applied 0.18215 retroactively:")
    print(f"    VinDr would have std ≈ {vindr_lat['all_std'] * sf:.4f}")
    print(f"    BoostNet would have std ≈ {boost_lat['all_std'] * sf:.4f}")
    if vindr_lat['all_std'] > 3.0:
        print(f"  ⚠ Latents are RAW (un-scaled by 0.18215).")
        print(f"    Pooled empirical standardisation is masking this but cannot")
        print(f"    correct per-channel offsets that differ between corpora.")

    # Verdict — corrected to use per-channel scores
    print(f"\n--- VERDICT ---")
    max_channel_mean_diff = max(
        abs(vindr_lat[f"c{c}_mean"] - boost_lat[f"c{c}_mean"]) / pooled_std
        for c in range(4)
    )
    print(f"Largest per-channel mean offset between VinDr and BoostNet, in")
    print(f"units of pooled std: {max_channel_mean_diff:.3f}")
    if max_channel_mean_diff > 0.15:
        print(f"  ⚠ Substantial per-channel distribution shift between corpora.")
        print(f"    Scalar pooled standardisation CANNOT correct this — only")
        print(f"    per-channel normalisation can.")
    else:
        print(f"  ✓ Per-channel shifts are small. Pooled scalar standardisation")
        print(f"    is adequate.")
    print("=" * 78)


def make_figure(vindr_pix, boost_pix, vindr_lats, boost_lats,
                 pooled_mean, pooled_std, out_path: Path):
    """Multi-panel histogram figure showing distributions at each level."""

    # Determine layout based on what data is available
    has_pixels = vindr_pix is not None and boost_pix is not None
    n_lat_channels = vindr_lats.shape[1]   # typically 4

    if has_pixels:
        # 3 row layout: pixel stats / per-channel latents / pooled-standardisation
        fig, axes = plt.subplots(3, max(2, n_lat_channels),
                                   figsize=(4 * n_lat_channels, 12),
                                   gridspec_kw={"hspace": 0.4, "wspace": 0.3})
        pixel_row, latent_row, pool_row = 0, 1, 2
    else:
        fig, axes = plt.subplots(2, max(2, n_lat_channels),
                                   figsize=(4 * n_lat_channels, 8),
                                   gridspec_kw={"hspace": 0.4, "wspace": 0.3})
        pixel_row, latent_row, pool_row = None, 0, 1

    # --- Pixel histograms ---
    if has_pixels:
        ax = axes[pixel_row, 0]
        ax.hist(vindr_pix["mean"], bins=50, alpha=0.6, label="VinDr",
                 color="steelblue", density=True)
        ax.hist(boost_pix["mean"], bins=50, alpha=0.6, label="BoostNet",
                 color="orange", density=True)
        ax.set_title("Per-image mean pixel intensity", fontsize=11)
        ax.set_xlabel("mean"); ax.set_ylabel("density")
        ax.legend(fontsize=9)

        ax = axes[pixel_row, 1]
        ax.hist(vindr_pix["std"], bins=50, alpha=0.6, label="VinDr",
                 color="steelblue", density=True)
        ax.hist(boost_pix["std"], bins=50, alpha=0.6, label="BoostNet",
                 color="orange", density=True)
        ax.set_title("Per-image std of pixel intensity", fontsize=11)
        ax.set_xlabel("std"); ax.set_ylabel("density")
        ax.legend(fontsize=9)

        # Hide any extra columns in pixel row
        for j in range(2, axes.shape[1]):
            axes[pixel_row, j].axis("off")

    # --- Per-channel latent histograms ---
    # Promote to fp64 for stable computation, especially for the pooled
    # standardisation visualization below
    vindr_lats_64 = vindr_lats.astype(np.float64)
    boost_lats_64 = boost_lats.astype(np.float64)

    for c in range(n_lat_channels):
        ax = axes[latent_row, c]
        v = vindr_lats_64[:, c].ravel()
        b = boost_lats_64[:, c].ravel()
        # Subsample if too many points (memory)
        if len(v) > 200_000:
            v = np.random.choice(v, size=200_000, replace=False)
        if len(b) > 200_000:
            b = np.random.choice(b, size=200_000, replace=False)
        # Common bin edges from the union
        all_vals = np.concatenate([v, b])
        lo, hi = np.percentile(all_vals, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(v, bins=bins, alpha=0.6, label="VinDr",
                 color="steelblue", density=True)
        ax.hist(b, bins=bins, alpha=0.6, label="BoostNet",
                 color="orange", density=True)
        ax.set_title(f"SD-VAE latent channel {c}", fontsize=11)
        ax.set_xlabel("activation"); ax.set_ylabel("density")
        ax.legend(fontsize=9)

    # Hide any extra cols
    for j in range(n_lat_channels, axes.shape[1]):
        axes[latent_row, j].axis("off")

    # --- Pooled-standardisation visualization ---
    # Show what happens to each corpus after pooled standardisation
    vindr_post = (vindr_lats_64.ravel() - pooled_mean) / pooled_std
    boost_post = (boost_lats_64.ravel() - pooled_mean) / pooled_std
    # Subsample for memory
    if len(vindr_post) > 300_000:
        vindr_post = np.random.choice(vindr_post, size=300_000, replace=False)
    if len(boost_post) > 300_000:
        boost_post = np.random.choice(boost_post, size=300_000, replace=False)

    ax = axes[pool_row, 0]
    bins = np.linspace(-5, 5, 80)
    ax.hist(vindr_post, bins=bins, alpha=0.6, label="VinDr",
             color="steelblue", density=True)
    ax.hist(boost_post, bins=bins, alpha=0.6, label="BoostNet",
             color="orange", density=True)
    # Overlay ideal N(0, 1)
    from scipy.stats import norm
    x = np.linspace(-5, 5, 200)
    ax.plot(x, norm.pdf(x, 0, 1), "k--", lw=1.2, label="ideal N(0, 1)")
    ax.set_title("After pooled standardisation (all channels)", fontsize=11)
    ax.set_xlabel("standardised activation"); ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 5)

    # Side panel: bar chart of post-standardisation summary stats
    ax = axes[pool_row, 1]
    labels = ["VinDr post", "BoostNet post", "Ideal"]
    means = [vindr_post.mean(), boost_post.mean(), 0.0]
    stds = [vindr_post.std(), boost_post.std(), 1.0]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width/2, means, width, label="mean", color="steelblue")
    ax.bar(x_pos + width/2, stds, width, label="std", color="orange")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(1, color="gray", lw=0.5, ls="--")
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Post-standardisation summary", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylabel("value")

    # Hide extras
    for j in range(2, axes.shape[1]):
        axes[pool_row, j].axis("off")

    fig.suptitle("VinDr-SpineXR vs BoostNet — distribution shift diagnosis",
                  fontsize=13, weight="bold", y=0.995)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vindr-latents",    required=True, type=Path,
                    help="latents_vindr.npz (full extraction, ~10k samples)")
    p.add_argument("--boostnet-latents", required=True, type=Path,
                    help="latents_boostnet_512.npz")
    p.add_argument("--vindr-images",     type=Path, default=None,
                    help="Optional: NPZ with sampled preprocessed VinDr "
                         "images. Without it, pixel-level comparison is "
                         "skipped (which is fine — latent-level is what "
                         "matters for diffusion).")
    p.add_argument("--boostnet-images",  type=Path, default=None)
    p.add_argument("--out-fig", required=True, type=Path)
    p.add_argument("--n-vindr-subsample", type=int, default=2000,
                    help="How many VinDr samples to use for stats. "
                         "Random subsample for speed.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- Load latents ----
    print("Loading VinDr latents...")
    vindr_data = np.load(args.vindr_latents, allow_pickle=True)
    vindr_lats = vindr_data["latents"]
    print(f"  Shape: {vindr_lats.shape}")
    # Subsample for speed
    if len(vindr_lats) > args.n_vindr_subsample:
        idx = rng.choice(len(vindr_lats), size=args.n_vindr_subsample,
                          replace=False)
        vindr_lats = vindr_lats[idx]
        print(f"  Subsampled to {len(vindr_lats)} samples")

    print("Loading BoostNet latents...")
    boost_data = np.load(args.boostnet_latents, allow_pickle=True)
    boost_lats = boost_data["latents"]
    print(f"  Shape: {boost_lats.shape}")
    # Use the train+val splits only (excluding test) to match what the
    # diffusion pipeline actually sees
    if "splits" in boost_data:
        splits = boost_data["splits"].astype(str)
        mask = splits != "boostnet_test"
        if not mask.all():
            boost_lats = boost_lats[mask]
            print(f"  Filtered to train+val: {len(boost_lats)} samples")

    # ---- Compute pooled mean/std as train_diffusion.py would ----
    # train_diffusion.py concatenates the lists then computes one scalar
    # mean/std (all channels treated as a single distribution). We compute
    # in fp64 to avoid the fp16 overflow that affects per-sample variance.
    pooled = np.concatenate([vindr_lats, boost_lats], axis=0).astype(np.float64)
    pooled_mean = float(pooled.mean())
    pooled_std  = float(pooled.std())
    print(f"\nPooled stats (as used by train_diffusion.py): "
          f"mean={pooled_mean:.4f}  std={pooled_std:.4f}")
    del pooled

    # ---- Latent-level stats ----
    print("\nComputing per-channel latent statistics...")
    vindr_lat_stats = per_channel_latent_stats(vindr_lats)
    boost_lat_stats = per_channel_latent_stats(boost_lats)

    # ---- Optional pixel-level stats ----
    vindr_pix_stats = boost_pix_stats = None
    if args.vindr_images and args.vindr_images.exists():
        print("\nLoading pixel-space samples for level-1 comparison...")
        vp = np.load(args.vindr_images, allow_pickle=True)
        vindr_imgs = vp["images"] if "images" in vp else vp[list(vp)[0]]
        vindr_pix_stats = per_image_pixel_stats(vindr_imgs)
    if args.boostnet_images and args.boostnet_images.exists():
        bp = np.load(args.boostnet_images, allow_pickle=True)
        boost_imgs = bp["images"] if "images" in bp else bp[list(bp)[0]]
        boost_pix_stats = per_image_pixel_stats(boost_imgs)

    # ---- Print & plot ----
    print_summary_table(vindr_pix_stats, boost_pix_stats,
                         vindr_lat_stats, boost_lat_stats,
                         pooled_mean, pooled_std)

    make_figure(vindr_pix_stats, boost_pix_stats,
                 vindr_lats, boost_lats,
                 pooled_mean, pooled_std, args.out_fig)


if __name__ == "__main__":
    main()
