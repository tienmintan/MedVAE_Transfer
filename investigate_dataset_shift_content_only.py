"""
investigate_dataset_shift_content_only.py: test whether the VinDr/BoostNet latent shift persists with padding excluded.

Follow-up diagnostic to investigate_dataset_shift.py, which measures:
  (1) Per-channel mean offsets between VinDr and BoostNet latents.
  (2) The per-channel histograms of each corpus, which can show a bimodal
      "spike + bulk" structure, with a sharp spike at the SD-VAE encoding
      of black (zero) input pixels. BoostNet is aspect-padded more
      heavily than VinDr (1:2.8 native -> square = ~65% padding by area),
      so its spike is expected to be larger.

This script tests whether the per-channel offsets disappear when the
padding positions are excluded from the statistics.

Interpretation:
  offsets disappear -> the "distribution shift" between corpora is primarily
      an artefact of differing padding fractions, not anatomical/imaging
      differences.

Approach:
  1. Identify each corpus's "padding" latent positions by detecting
     positions whose 4-channel signature matches the empirical spike
     location. The spike is found via the per-channel mode estimation
     described in find_spike_location().
  2. Apply the same content/padding split to both corpora.
  3. Recompute per-channel stats over content positions only.
  4. Compare against the whole-latent stats, computed as in
     investigate_dataset_shift.py.

Inputs:  VinDr and BoostNet SD-VAE latent .npz files (4 channels).
Outputs: a before/after table printed to stdout; a figure of whole-latent
         and content-only histograms per channel (--out-fig).

Usage:
    python investigate_dataset_shift_content_only.py \\
        --vindr-latents    ./latents_vindr/latents_vindr.npz \\
        --boostnet-latents ./latents_boostnet_512/latents_boostnet_512.npz \\
        --out-fig          ./outputs/figures/dataset_shift_content_only.png \\
        --n-vindr-subsample 2000
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_spike_location(latents: np.ndarray, n_bins: int = 200) -> np.ndarray:
    """Find the per-channel mode (most common value), which is where the
    padding spike lies.

    Returns: (C,) array of mode locations.
    """
    C = latents.shape[1]
    flat_by_ch = latents.reshape(latents.shape[0], C, -1)   # (N, C, HW)
    modes = np.zeros(C)
    for c in range(C):
        vals = flat_by_ch[:, c].ravel().astype(np.float64)
        # Estimate via histogram peak (mode)
        hist, bin_edges = np.histogram(vals, bins=n_bins,
                                         range=(np.percentile(vals, 0.5),
                                                np.percentile(vals, 99.5)))
        peak_idx = hist.argmax()
        modes[c] = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
    return modes


def classify_padding_positions(latents: np.ndarray, spike: np.ndarray,
                                 tolerance: float = 0.5) -> np.ndarray:
    """For each latent position in each sample, decide if it is "padding"
    by checking if all four channels match the spike vector within
    tolerance pooled-std units.

    A position is padding iff |latent[c] - spike[c]| < tolerance * std[c]
    for every channel c.

    Returns: (N, H, W) bool array; True where the position is padding.
    """
    N, C, H, W = latents.shape
    # Compute per-channel std for the tolerance calculation
    per_ch_std = np.array([float(latents[:, c].astype(np.float64).std())
                            for c in range(C)])

    # Broadcasting: (N, C, H, W) vs (C,) reshaped to (1, C, 1, 1)
    spike_b = spike.reshape(1, C, 1, 1)
    std_b   = per_ch_std.reshape(1, C, 1, 1)
    within = np.abs(latents.astype(np.float64) - spike_b) < (tolerance * std_b)
    # Padding only if all channels are within the spike
    is_padding = within.all(axis=1)   # (N, H, W)
    return is_padding


def per_channel_stats_masked(latents: np.ndarray, content_mask: np.ndarray) -> dict:
    """Compute per-channel mean and std over content positions only.

    latents: (N, C, H, W)
    content_mask: (N, H, W) bool, True where the position is content
                  (i.e., not padding)

    Returns: dict with c0_mean, c0_std, etc.
    """
    N, C, H, W = latents.shape
    out = {}
    for c in range(C):
        ch_vals = latents[:, c].astype(np.float64)        # (N, H, W)
        content_vals = ch_vals[content_mask]              # 1D, content only
        out[f"c{c}_mean"] = float(content_vals.mean())
        out[f"c{c}_std"]  = float(content_vals.std())
        out[f"c{c}_n"]    = int(content_vals.size)
    # Aggregate stats too
    all_content_vals = latents.astype(np.float64)[
        np.broadcast_to(content_mask[:, None], (N, C, H, W))
    ]
    out["all_mean"] = float(all_content_vals.mean())
    out["all_std"]  = float(all_content_vals.std())
    return out


def print_table(vindr_stats_full, boost_stats_full,
                 vindr_stats_content, boost_stats_content,
                 vindr_padding_frac, boost_padding_frac,
                 pooled_std_full):
    """Print before/after comparison table."""
    print("\n" + "=" * 90)
    print("Content-only diagnostic: do per-channel offsets persist when padding is excluded?")
    print("=" * 90)

    print(f"\nDetected padding fraction:")
    print(f"  VinDr   : {vindr_padding_frac * 100:>5.1f}% of latent positions")
    print(f"  BoostNet: {boost_padding_frac * 100:>5.1f}% of latent positions")
    print(f"  (Expected: BoostNet > VinDr, since BoostNet is more aggressively aspect-padded.)")

    print(f"\n{'Channel':<10} {'Whole-latent stats':<45} {'Content-only stats':<45}")
    print(f"{'':<10} {'VinDr (μ, σ)':<22} {'BoostNet (μ, σ)':<22}"
          f" {'VinDr (μ, σ)':<22} {'BoostNet (μ, σ)':<22}")
    print("-" * 90)
    for c in range(4):
        vfm = vindr_stats_full[f"c{c}_mean"]; vfs = vindr_stats_full[f"c{c}_std"]
        bfm = boost_stats_full[f"c{c}_mean"]; bfs = boost_stats_full[f"c{c}_std"]
        vcm = vindr_stats_content[f"c{c}_mean"]; vcs = vindr_stats_content[f"c{c}_std"]
        bcm = boost_stats_content[f"c{c}_mean"]; bcs = boost_stats_content[f"c{c}_std"]
        print(f"ch{c:<7}  "
              f"({vfm:+.3f}, {vfs:.3f})       "
              f"({bfm:+.3f}, {bfs:.3f})       "
              f"({vcm:+.3f}, {vcs:.3f})       "
              f"({bcm:+.3f}, {bcs:.3f})")

    print(f"\n{'Δ per-channel mean':<28} {'whole-latent':<20} {'content-only':<20} {'ratio':<10}")
    print("-" * 90)
    for c in range(4):
        d_full = abs(vindr_stats_full[f"c{c}_mean"] - boost_stats_full[f"c{c}_mean"])
        d_cont = abs(vindr_stats_content[f"c{c}_mean"] - boost_stats_content[f"c{c}_mean"])
        d_full_norm = d_full / pooled_std_full
        d_cont_norm = d_cont / pooled_std_full
        ratio = d_cont / d_full if d_full > 1e-6 else float("nan")
        print(f"ch{c:<7}                    "
              f"{d_full:.4f} ({d_full_norm:.3f}σ)  "
              f"{d_cont:.4f} ({d_cont_norm:.3f}σ)  "
              f"{ratio:.2f}×")

    # Result
    print(f"\n--- Result ---")
    max_diff_full = max(
        abs(vindr_stats_full[f"c{c}_mean"] - boost_stats_full[f"c{c}_mean"])
        for c in range(4)
    ) / pooled_std_full
    max_diff_cont = max(
        abs(vindr_stats_content[f"c{c}_mean"] - boost_stats_content[f"c{c}_mean"])
        for c in range(4)
    ) / pooled_std_full

    print(f"Max per-channel mean offset (in pooled-σ units):")
    print(f"  Whole-latent:  {max_diff_full:.3f}")
    print(f"  Content-only:  {max_diff_cont:.3f}")
    reduction = (1 - max_diff_cont / max_diff_full) * 100 if max_diff_full > 1e-6 else 0
    print(f"  Reduction:     {reduction:.0f}%")

    if max_diff_cont < 0.10:
        print(f"\n  Result: content-only offsets are small ({max_diff_cont:.3f} pooled SD, threshold 0.10). The whole-latent shift")
        print(f"    is driven by padding fraction differences, not anatomy.")
        print(f"    Remedy: equalise padding fractions during preprocessing.")
    elif reduction > 50:
        print(f"\n  Result: content-only offsets reduced by {reduction:.0f}% (more than 50%), but not eliminated.")
        print(f"    Padding fraction is the dominant cause but not the only one;")
        print(f"    some anatomical/imaging difference remains.")
        print(f"    Remedy: equalise padding + per-channel standardisation.")
    else:
        print(f"\n  Result: content-only offsets remain substantial ({max_diff_cont:.3f} pooled SD; reduction {reduction:.0f}%).")
        print(f"    Padding fraction is not the dominant cause; anatomical")
        print(f"    or imaging differences between corpora persist.")
        print(f"    Remedy: per-channel standardisation is needed; padding")
        print(f"    equalisation is likely insufficient on its own.")
    print("=" * 90)


def make_figure(vindr_lats, boost_lats,
                 vindr_content_mask, boost_content_mask,
                 vindr_padding_frac, boost_padding_frac,
                 out_path: Path):
    """Multi-panel figure showing whole-latent vs content-only distributions."""
    n_channels = vindr_lats.shape[1]
    fig, axes = plt.subplots(2, n_channels,
                              figsize=(4 * n_channels, 8),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.3})

    rng = np.random.default_rng(42)

    for c in range(n_channels):
        # Row 0: whole-latent histograms (for reference)
        ax = axes[0, c]
        v_full = vindr_lats[:, c].astype(np.float64).ravel()
        b_full = boost_lats[:, c].astype(np.float64).ravel()
        if len(v_full) > 200_000:
            v_full = rng.choice(v_full, size=200_000, replace=False)
        if len(b_full) > 200_000:
            b_full = rng.choice(b_full, size=200_000, replace=False)
        all_vals = np.concatenate([v_full, b_full])
        lo, hi = np.percentile(all_vals, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(v_full, bins=bins, alpha=0.6, label="VinDr",
                 color="steelblue", density=True)
        ax.hist(b_full, bins=bins, alpha=0.6, label="BoostNet",
                 color="orange", density=True)
        ax.set_title(f"ch{c}: whole latent (includes padding)", fontsize=10)
        ax.set_xlabel("activation"); ax.set_ylabel("density")
        ax.legend(fontsize=8)

        # Row 1: content-only histograms
        ax = axes[1, c]
        v_content = vindr_lats[:, c].astype(np.float64)[vindr_content_mask].ravel()
        b_content = boost_lats[:, c].astype(np.float64)[boost_content_mask].ravel()
        if len(v_content) > 200_000:
            v_content = rng.choice(v_content, size=200_000, replace=False)
        if len(b_content) > 200_000:
            b_content = rng.choice(b_content, size=200_000, replace=False)
        all_vals = np.concatenate([v_content, b_content])
        lo, hi = np.percentile(all_vals, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(v_content, bins=bins, alpha=0.6, label="VinDr",
                 color="steelblue", density=True)
        ax.hist(b_content, bins=bins, alpha=0.6, label="BoostNet",
                 color="orange", density=True)
        ax.set_title(f"ch{c}: content only (padding excluded)", fontsize=10)
        ax.set_xlabel("activation"); ax.set_ylabel("density")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"VinDr vs BoostNet: content-only distribution comparison\n"
        f"Detected padding: VinDr {vindr_padding_frac*100:.0f}%, "
        f"BoostNet {boost_padding_frac*100:.0f}% of latent positions",
        fontsize=12, weight="bold", y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to {out_path}")


def per_channel_stats_full(latents: np.ndarray) -> dict:
    """Per-channel stats over the whole latent (matches
    investigate_dataset_shift.py)."""
    latents = latents.astype(np.float64)
    C = latents.shape[1]
    out = {}
    for c in range(C):
        ch = latents[:, c]
        out[f"c{c}_mean"] = float(ch.mean())
        out[f"c{c}_std"]  = float(ch.std())
    out["all_mean"] = float(latents.mean())
    out["all_std"]  = float(latents.std())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vindr-latents",    required=True, type=Path)
    p.add_argument("--boostnet-latents", required=True, type=Path)
    p.add_argument("--out-fig",          required=True, type=Path)
    p.add_argument("--n-vindr-subsample", type=int, default=2000)
    p.add_argument("--tolerance", type=float, default=0.3,
                    help="Tolerance (in per-channel std units) for "
                         "classifying a latent position as padding. Smaller "
                         "= stricter; default 0.3 is conservative.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- Load latents ---
    print("Loading VinDr latents...")
    vindr_data = np.load(args.vindr_latents, allow_pickle=True)
    vindr_lats = vindr_data["latents"]
    print(f"  Shape: {vindr_lats.shape}")
    if len(vindr_lats) > args.n_vindr_subsample:
        idx = rng.choice(len(vindr_lats), size=args.n_vindr_subsample,
                          replace=False)
        vindr_lats = vindr_lats[idx]
        print(f"  Subsampled to {len(vindr_lats)} samples")

    print("Loading BoostNet latents...")
    boost_data = np.load(args.boostnet_latents, allow_pickle=True)
    boost_lats = boost_data["latents"]
    print(f"  Shape: {boost_lats.shape}")
    if "splits" in boost_data:
        splits = boost_data["splits"].astype(str)
        mask = splits != "boostnet_test"
        if not mask.all():
            boost_lats = boost_lats[mask]
            print(f"  Filtered to train+val: {len(boost_lats)} samples")

    # --- Find spike locations per corpus ---
    print("\nDetecting padding spike locations (per corpus)...")
    vindr_spike = find_spike_location(vindr_lats)
    boost_spike = find_spike_location(boost_lats)
    print(f"  VinDr   spike (per channel): {vindr_spike}")
    print(f"  BoostNet spike (per channel): {boost_spike}")

    # The average of the two is used as the canonical padding signature,
    # since they should be very close (both encoding "black pixel" through
    # the same SD-VAE).
    canonical_spike = (vindr_spike + boost_spike) / 2
    spike_diff = np.abs(vindr_spike - boost_spike)
    print(f"  Per-channel disagreement: {spike_diff}")
    if spike_diff.max() > 0.5:
        print(f"  WARNING: spike locations differ by up to {spike_diff.max():.2f} between corpora (threshold 0.5).")
        print(f"    Using per-corpus spikes for masking.")
        use_per_corpus_spike = True
    else:
        print(f"  Check passed: spike locations agree within 0.5. Using canonical spike.")
        use_per_corpus_spike = False

    # --- Classify padding positions ---
    print(f"\nClassifying padding positions (tolerance={args.tolerance}σ)...")
    if use_per_corpus_spike:
        vindr_padding_mask = classify_padding_positions(
            vindr_lats, vindr_spike, args.tolerance)
        boost_padding_mask = classify_padding_positions(
            boost_lats, boost_spike, args.tolerance)
    else:
        vindr_padding_mask = classify_padding_positions(
            vindr_lats, canonical_spike, args.tolerance)
        boost_padding_mask = classify_padding_positions(
            boost_lats, canonical_spike, args.tolerance)

    vindr_padding_frac = float(vindr_padding_mask.mean())
    boost_padding_frac = float(boost_padding_mask.mean())
    print(f"  VinDr   padding fraction: {vindr_padding_frac*100:.1f}%")
    print(f"  BoostNet padding fraction: {boost_padding_frac*100:.1f}%")
    print(f"  (Expected: BoostNet > VinDr; if BoostNet ≈ 50-65%, geometry"
          f" matches the aspect-ratio prediction.)")

    vindr_content_mask = ~vindr_padding_mask
    boost_content_mask = ~boost_padding_mask

    # --- Compute stats ---
    print("\nComputing stats over whole latents (for reference)...")
    vindr_stats_full = per_channel_stats_full(vindr_lats)
    boost_stats_full = per_channel_stats_full(boost_lats)

    print("Computing stats over content-only positions...")
    vindr_stats_content = per_channel_stats_masked(
        vindr_lats, vindr_content_mask)
    boost_stats_content = per_channel_stats_masked(
        boost_lats, boost_content_mask)

    # Pooled std for normalising the offsets
    pooled = np.concatenate([vindr_lats, boost_lats], axis=0).astype(np.float64)
    pooled_std = float(pooled.std())
    del pooled

    # --- Print and plot ---
    print_table(vindr_stats_full, boost_stats_full,
                 vindr_stats_content, boost_stats_content,
                 vindr_padding_frac, boost_padding_frac,
                 pooled_std)

    make_figure(vindr_lats, boost_lats,
                 vindr_content_mask, boost_content_mask,
                 vindr_padding_frac, boost_padding_frac,
                 args.out_fig)


if __name__ == "__main__":
    main()
