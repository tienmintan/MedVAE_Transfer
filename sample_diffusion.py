"""
sample_diffusion.py: generate synthetic latents from the trained Cobb-conditioned diffusion model.

Targets a specified distribution of Cobb angles and saves the resulting
latents in the same .npz schema as extract_latents_whole.py, so the
downstream Cobb regressor can consume them directly for augmentation.

Two sampling modes for the Cobb targets:
  - 'match-train': sample Cobb angle vectors from the empirical distribution
    of the real training set (recommended for augmentation experiments, as it
    keeps the synthetic-set distribution close to the real one)
  - 'uniform': sample uniformly from a given range per angle (useful for
    generating severe / underrepresented cases for class-imbalance studies)

Inputs:  a diffusion checkpoint (--checkpoint) with EMA weights and the Cobb
         and latent standardisation statistics; --labels-csv for match-train.
Outputs: --out .npz with latents (un-standardised, MedVAE native scale),
         angles, uuids ("synth_*") and splits ("train").

Usage:
    # Generate 500 synthetic latents matching the training distribution
    python sample_diffusion.py \\
        --checkpoint   ./diffusion_ckpt/final.pt \\
        --out          ./synthetic_latents_500.npz \\
        --n-samples    500 \\
        --cfg-scale    3.0 \\
        --sampler      ddim \\
        --target-mode  match-train \\
        --labels-csv   ./medvae/data/boostnet_with_angles.csv

    # Generate samples uniformly spanning a Cobb range
    python sample_diffusion.py \\
        --checkpoint   ./diffusion_ckpt/final.pt \\
        --out          ./synthetic_latents_severe.npz \\
        --n-samples    200 \\
        --cfg-scale    5.0 \\
        --sampler      ddpm \\
        --target-mode  uniform \\
        --uniform-low   30 10 10 \\
        --uniform-high  60 40 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffusion_model import UNet, GaussianDiffusion, EMA


def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Reconstruct the model from a checkpoint, loaded with EMA weights."""
    # weights_only=False because the checkpoint contains numpy arrays
    # (cobb_mean, cobb_std). Safe because the checkpoint is produced by this
    # project's own training script.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyperparams"]
    model = UNet(in_channels=hp["in_channels"],
                 channels=tuple(hp["channels"]),
                 embed_dim=hp["embed_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    ema = EMA(model)
    ema.load_state_dict(ckpt["ema"])
    ema.copy_to(model)
    model.eval()

    diffusion = GaussianDiffusion(T=hp["T"], device=device,
                                   parameterization=hp.get("parameterization", "eps"))
    cobb_mean = torch.from_numpy(ckpt["cobb_mean"]).to(device)
    cobb_std  = torch.from_numpy(ckpt["cobb_std"]).to(device)
    # Latent standardisation stats, required to un-standardise samples back
    # to MedVAE's native distribution before they can be decoded
    latent_mean = float(ckpt.get("latent_mean", 0.0))
    latent_std  = float(ckpt.get("latent_std", 1.0))
    latent_shape = tuple(hp["latent_shape"])
    return (model, diffusion, cobb_mean, cobb_std,
            latent_mean, latent_std, latent_shape)


def get_target_angles(args, n: int, device: torch.device,
                      cobb_mean: torch.Tensor, cobb_std: torch.Tensor):
    """Return (raw_angles, standardised_angles) tensors of shape (n, 3)."""
    if args.target_mode == "match-train":
        if args.labels_csv is None:
            raise ValueError("--labels-csv required for target-mode match-train")
        labels = pd.read_csv(args.labels_csv)
        angle_cols = ["angle_proximal_thoracic", "angle_main_thoracic",
                      "angle_thoracolumbar"]
        train_angles = labels[labels["split"].isin(["train", "val"])][
            angle_cols].values.astype(np.float32)

        # Bootstrap-sample n vectors from training angles
        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, len(train_angles), size=n)
        raw = train_angles[idx]

    elif args.target_mode == "uniform":
        if args.uniform_low is None or args.uniform_high is None:
            raise ValueError("--uniform-low and --uniform-high required for "
                             "target-mode uniform")
        low = np.array(args.uniform_low, dtype=np.float32)
        high = np.array(args.uniform_high, dtype=np.float32)
        rng = np.random.default_rng(args.seed)
        raw = rng.uniform(low, high, size=(n, 3)).astype(np.float32)

    else:
        raise ValueError(f"Unknown target-mode: {args.target_mode}")

    raw_t = torch.from_numpy(raw).to(device)
    std_t = (raw_t - cobb_mean[None]) / cobb_std[None]
    return raw_t, std_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out",        required=True, type=Path)
    p.add_argument("--n-samples",  type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cfg-scale",  type=float, default=3.0,
                   help="Classifier-free guidance scale. 1=no guidance, "
                        "3-7 typical for conditional generation.")
    p.add_argument("--sampler",    choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--target-mode", choices=["match-train", "uniform"],
                   default="match-train")
    p.add_argument("--labels-csv", type=Path, default=None,
                   help="Required for target-mode match-train.")
    p.add_argument("--uniform-low",  type=float, nargs=3, default=None,
                   help="Lower bounds for uniform target sampling.")
    p.add_argument("--uniform-high", type=float, nargs=3, default=None,
                   help="Upper bounds for uniform target sampling.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}")
    (model, diffusion, cobb_mean, cobb_std,
     latent_mean, latent_std, latent_shape) = load_checkpoint(
        args.checkpoint, device)
    print(f"  Latent shape: {latent_shape}")
    print(f"  Cobb mean: {cobb_mean.cpu().numpy().round(2)}")
    print(f"  Cobb std:  {cobb_std.cpu().numpy().round(2)}")
    print(f"  Latent mean: {latent_mean:.4f}  std: {latent_std:.4f}")

    # Targets
    raw_angles, std_angles = get_target_angles(args, args.n_samples, device,
                                                cobb_mean, cobb_std)
    print(f"Target Cobb angles (raw):")
    print(f"  Range, major curve (col 0): [{raw_angles[:,0].min():.1f}, {raw_angles[:,0].max():.1f}]")
    print(f"  Range, col 1:               [{raw_angles[:,1].min():.1f}, {raw_angles[:,1].max():.1f}]")
    print(f"  Range, col 2:               [{raw_angles[:,2].min():.1f}, {raw_angles[:,2].max():.1f}]")

    # Sample in batches
    all_latents = []
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    print(f"\nSampling {args.n_samples} latents in {n_batches} batches "
          f"({args.sampler}, cfg_scale={args.cfg_scale})...")
    for bi in range(n_batches):
        start = bi * args.batch_size
        end = min(start + args.batch_size, args.n_samples)
        batch_n = end - start
        batch_std = std_angles[start:end]
        batch_shape = (batch_n, *latent_shape)

        if args.sampler == "ddim":
            latents = diffusion.sample_ddim(model, batch_shape, batch_std,
                                            cfg_scale=args.cfg_scale,
                                            n_steps=args.ddim_steps,
                                            device=device)
        else:
            latents = diffusion.sample_ddpm(model, batch_shape, batch_std,
                                            cfg_scale=args.cfg_scale,
                                            device=device)
        all_latents.append(latents.cpu().float().numpy())
        print(f"  Batch {bi+1}/{n_batches}: shape={latents.shape}  "
              f"mean={latents.mean().item():.3f}  std={latents.std().item():.3f}")

    all_latents = np.concatenate(all_latents, axis=0)
    print(f"\nTotal synthetic latents (standardised): {all_latents.shape}")
    print(f"  standardised stats: mean={all_latents.mean():.3f} "
          f"std={all_latents.std():.3f}")

    # Un-standardise back to MedVAE's native distribution so the saved
    # latents can be decoded directly by decode_samples.py
    all_latents = all_latents * latent_std + latent_mean
    print(f"  native (un-standardised) stats: mean={all_latents.mean():.3f} "
          f"std={all_latents.std():.3f}")

    # Synthetic uuids, distinct from real ones so synthetic samples can be identified later
    synth_uuids = np.array([f"synth_{i:06d}" for i in range(args.n_samples)])

    np.savez_compressed(
        args.out,
        latents=all_latents.astype(np.float32),
        angles=raw_angles.cpu().numpy().astype(np.float32),
        uuids=synth_uuids,
        # Splits column so the downstream regressor's CSV loader sees these
        # as "train" examples
        splits=np.array(["train"] * args.n_samples),
    )
    print(f"Wrote {args.out}")
    print(f"\nNext: use these latents to augment Cobb regressor training.")


if __name__ == "__main__":
    main()
