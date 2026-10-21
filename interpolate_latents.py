"""
interpolate_latents.py: generate synthetic training latents by interpolating between pairs of real latents.

Mixup-style data augmentation in MedVAE's latent space:

    z_synth = (1 - α) * z_a + α * z_b
    y_synth = (1 - α) * y_a + α * y_b

where (z_a, y_a) and (z_b, y_b) are random pairs from the real training set
and α ~ Beta(0.4, 0.4), which is biased toward the endpoints so
interpolations are typically closer to one of the source samples (the
standard mixup paper's recommendation).

This is an alternative to diffusion-based synthesis. The rationale:
  - VAE latent spaces are approximately locally linear; interpolation
    between nearby real samples should produce plausible intermediate
    representations.
  - Cobb angles are continuous and angles between two scoliosis severities
    are semantically meaningful, so label interpolation is well-defined.
  - No training instability to debug: interpolation is a deterministic
    operation on existing data.

Output schema matches extract_latents.py (combined train+val+test in a
single .npz with `latents`, `angles`, `splits`, `uuids`), so
train_cobb_regressor.py can consume it directly. Synthetic samples are
tagged with the "train" split and unique uuids prefixed with "interp_".

Inputs:  --input .npz from extract_latents.py.
Outputs: --output .npz containing the real samples plus --n-synth synthetic
         train samples; summary statistics printed to stdout.

Usage:
    # Generate 500 synthetic training samples, augmenting the 4x baseline
    python interpolate_latents.py \\
        --input    ./latents_pretrained_4x.npz \\
        --output   ./latents_pretrained_4x_augmented.npz \\
        --n-synth  500 \\
        --beta     0.4

    # Then train the standard Cobb regressor on the augmented set
    python train_cobb_regressor.py \\
        --latents  ./latents_pretrained_4x_augmented.npz \\
        --out      ./cobb_pretrained_4x_augmented.npz \\
        --baseline-results ./cobb_pretrained_4x.npz \\
        --seed     42
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True, type=Path,
                   help="Real latents npz from extract_latents.py "
                        "(must have keys: latents, angles, splits, uuids)")
    p.add_argument("--output",   required=True, type=Path)
    p.add_argument("--n-synth",  type=int, default=500,
                   help="Number of synthetic interpolated samples to generate.")
    p.add_argument("--beta",     type=float, default=0.4,
                   help="Beta distribution parameter for mixing coefficient "
                        "α. Values <1 bias α toward endpoints (recommended); "
                        "value 1 is uniform; >1 biases toward midpoint.")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--diverse-pairs", action="store_true",
                   help="If set, bias pair sampling toward pairs with very "
                        "different Cobb angles (more informative interpolations).")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load real latents
    data = np.load(args.input, allow_pickle=True)
    latents = data["latents"]      # (N, C, H, W)
    angles  = data["angles"]       # (N, 3)
    splits  = data["splits"].astype(str)
    uuids   = data["uuids"].astype(str)
    print(f"Loaded {len(latents)} real samples from {args.input}")
    print(f"  Latent shape: {latents.shape[1:]}")
    print(f"  Split counts: " + ", ".join(
        f"{s}={(splits == s).sum()}" for s in ["train", "val", "test"]))

    # Restrict pair sampling to the train split so no val/test data enters synthesis
    train_mask = (splits == "train")
    train_lat = latents[train_mask]
    train_ang = angles[train_mask]
    n_train = len(train_lat)
    print(f"\nSynthesising {args.n_synth} samples from {n_train} train latents")
    print(f"  Mixing strategy: α ~ Beta({args.beta}, {args.beta})")
    print(f"  Pair sampling: "
          f"{'diverse (biased toward different Cobb)' if args.diverse_pairs else 'random uniform'}")

    # Sample pair indices
    if args.diverse_pairs:
        # For each synthetic sample: pick first index uniformly, then pick
        # second index biased toward samples with very different Cobb angle.
        # Distance in angle space is used as the sampling weight.
        idx_a = rng.integers(0, n_train, size=args.n_synth)
        idx_b = np.empty(args.n_synth, dtype=np.int64)
        for i in range(args.n_synth):
            dists = np.linalg.norm(train_ang - train_ang[idx_a[i]], axis=1)
            # Higher prob for more distant samples; ensure self-prob is 0
            weights = dists.copy()
            weights[idx_a[i]] = 0.0
            weights = weights / weights.sum()
            idx_b[i] = rng.choice(n_train, p=weights)
    else:
        idx_a = rng.integers(0, n_train, size=args.n_synth)
        idx_b = rng.integers(0, n_train, size=args.n_synth)
        # Avoid self-pairs (would produce identical copies)
        same = (idx_a == idx_b)
        while same.any():
            idx_b[same] = rng.integers(0, n_train, size=same.sum())
            same = (idx_a == idx_b)

    # Sample mixing coefficients from Beta distribution
    alpha = rng.beta(args.beta, args.beta, size=args.n_synth).astype(np.float32)

    # Interpolate latents and angles
    z_a = train_lat[idx_a]
    z_b = train_lat[idx_b]
    y_a = train_ang[idx_a]
    y_b = train_ang[idx_b]

    a4 = alpha[:, None, None, None]
    a1 = alpha[:, None]
    synth_lat = (1 - a4) * z_a + a4 * z_b
    synth_ang = (1 - a1) * y_a + a1 * y_b

    synth_uuids = np.array([f"interp_{i:06d}" for i in range(args.n_synth)])
    synth_splits = np.array(["train"] * args.n_synth)

    print(f"\nSynthetic latent stats:")
    print(f"  Shape: {synth_lat.shape}")
    print(f"  mean={synth_lat.mean():.3f} std={synth_lat.std():.3f}")
    print(f"  (real train: mean={train_lat.mean():.3f} std={train_lat.std():.3f})")
    print(f"Synthetic angle stats:")
    print(f"  Range, major curve (col 0): [{synth_ang[:,0].min():.1f}, {synth_ang[:,0].max():.1f}]")
    print(f"  Range, col 1:               [{synth_ang[:,1].min():.1f}, {synth_ang[:,1].max():.1f}]")
    print(f"  Range, col 2:               [{synth_ang[:,2].min():.1f}, {synth_ang[:,2].max():.1f}]")
    print(f"  α distribution: mean={alpha.mean():.3f} std={alpha.std():.3f}")

    # Concatenate with real samples
    combined_lat = np.concatenate([latents, synth_lat.astype(np.float32)], axis=0)
    combined_ang = np.concatenate([angles, synth_ang.astype(np.float32)], axis=0)
    combined_splits = np.concatenate([splits, synth_splits])
    combined_uuids = np.concatenate([uuids, synth_uuids])

    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(combined_lat)}")
    print(f"  Train split:   {(combined_splits == 'train').sum()} "
          f"({n_train} real + {args.n_synth} synthetic)")
    print(f"  Val split:     {(combined_splits == 'val').sum()} (unchanged)")
    print(f"  Test split:    {(combined_splits == 'test').sum()} (unchanged)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latents=combined_lat,
        angles=combined_ang,
        splits=combined_splits,
        uuids=combined_uuids,
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
