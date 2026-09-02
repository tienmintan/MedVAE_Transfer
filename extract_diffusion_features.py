"""
extract_diffusion_features.py

Phase B: extract intermediate U-Net activations from the trained
unconditional diffusion model and save them as a latents file that the
matched regressor can consume.

Pipeline per image:
  1. Load its precomputed SD-VAE latent from the cropped-pipeline
     extraction (4, 64, 64). This is the same input format the diffusion
     model was trained on.
  2. Standardise the latent using the diffusion checkpoint's stored
     latent_mean/latent_std statistics (the diffusion model operates in
     standardised latent space).
  3. Construct x_t at a fixed timestep by adding a fixed-seed noise
     sample. q_sample: x_t = sqrt(α_cum) * x_0 + sqrt(1 - α_cum) * noise.
  4. Forward through the U-Net's encoder up to a chosen intermediate
     level, with null Cobb conditioning (drop_mask=True → learned null
     embedding regardless of cobb values).
  5. Capture the activation at that level as the image's feature vector.

We bypass the full U-Net forward and inline the encoder up to the
chosen level. This saves compute (no decoder pass needed) and avoids
registering forward hooks that would otherwise persist across calls.

Default extraction settings:
  --level 1 : skips[1], the output of the second ResBlock at level 1.
              Shape (128, 32, 32). Mid-encoder: enough processing for
              anatomical structure but enough spatial resolution to
              preserve curve-localisation information.
  --timestep 250 : moderate noise level under the cosine T=1000 schedule,
                   where sqrt(α_cum) ≈ 0.92.

Output format matches the latents files consumed by
train_cobb_regressor_matched.py (latents, angles, uuids, splits).

Usage:
    python extract_diffusion_features.py \\
        --checkpoint ./diffusion_ckpt_pretrain_cropped/final.pt \\
        --boostnet-latents ./latents_boostnet_cropped/latents_boostnet_cropped.npz \\
        --out ./latents_diffusion_features/latents_diffusion_features.npz

Author: Tien Min, June 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from diffusion_model import UNet, GaussianDiffusion, EMA


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors sample_diffusion.py.load_checkpoint)
# ---------------------------------------------------------------------------

def load_diffusion_checkpoint(ckpt_path: Path, device: torch.device):
    """Load the U-Net + diffusion process + latent-standardisation stats."""
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
    for p in model.parameters():
        p.requires_grad_(False)

    diffusion = GaussianDiffusion(
        T=hp["T"], device=device,
        parameterization=hp.get("parameterization", "eps"),
    )

    latent_mean = float(ckpt.get("latent_mean", 0.0))
    latent_std  = float(ckpt.get("latent_std", 1.0))
    return model, diffusion, latent_mean, latent_std, hp


# ---------------------------------------------------------------------------
# Encoder feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_encoder_features(
    model: UNet,
    x_t: torch.Tensor,           # (B, C_in, H, W) standardised + noisy latents
    t: torch.Tensor,             # (B,) timestep
    cobb: torch.Tensor,          # (B, 3) — ignored when drop_mask=True
    drop_mask: torch.Tensor,     # (B,) bool
    level: int,
) -> torch.Tensor:
    """Inline the U-Net's encoder up to the chosen down_block level.

    Returns the activation that would be appended to `skips[level]` —
    i.e. the output of the second ResBlock at that level, before
    downsampling.

    Mirrors the encoder portion of UNet.forward exactly, just stops early.
    """
    emb = model.time_embed(t) + model.cobb_embed(cobb, drop_mask)

    h = model.conv_in(x_t)
    for i, (res1, res2, down) in enumerate(model.down_blocks):
        h = res1(h, emb)
        h = res2(h, emb)
        if i == level:
            return h
        h = down(h)
    raise ValueError(
        f"level={level} out of range; model has {len(model.down_blocks)} levels"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path,
                    help="Diffusion model checkpoint. Recommended: "
                         "./diffusion_ckpt_pretrain_cropped/final.pt")
    p.add_argument("--boostnet-latents", required=True, type=Path,
                    help="Merged BoostNet SD-VAE latents from "
                         "extract_latents_boostnet_cropped.py. Expects "
                         "(N, 4, 64, 64) latents + angles + splits + uuids.")
    p.add_argument("--out", required=True, type=Path,
                    help="Where to write the diffusion-features latents file.")
    p.add_argument("--level", type=int, default=1,
                    help="Down_block level to extract from. 0=64x64/64ch, "
                         "1=32x32/128ch (default), 2=16x16/256ch, "
                         "3=8x8/256ch (bottleneck input).")
    p.add_argument("--timestep", type=int, default=250,
                    help="Fixed forward-diffusion timestep at which to "
                         "extract features. Default 250 (cosine T=1000, "
                         "moderate noise).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--noise-seed", type=int, default=20260616,
                    help="Seed for the fixed noise vector used at the "
                         "extraction timestep. Same seed → deterministic.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.boostnet_latents.is_file():
        print(f"ERROR: latents not found at {args.boostnet_latents}",
              file=sys.stderr)
        return 1

    device = torch.device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load diffusion model ----
    print(f"Loading diffusion checkpoint from {args.checkpoint}")
    model, diffusion, latent_mean, latent_std, hp = load_diffusion_checkpoint(
        args.checkpoint, device
    )
    print(f"  Architecture: in_channels={hp['in_channels']}, "
          f"channels={hp['channels']}, embed_dim={hp['embed_dim']}")
    print(f"  Diffusion process: T={hp['T']}, "
          f"parameterization={hp.get('parameterization', 'eps')}")
    print(f"  Latent normalisation: mean={latent_mean:.4f}, std={latent_std:.4f}")

    if args.level < 0 or args.level >= len(hp["channels"]):
        print(f"ERROR: --level {args.level} out of range for model with "
              f"{len(hp['channels'])} levels", file=sys.stderr)
        return 1
    if args.timestep < 0 or args.timestep >= hp["T"]:
        print(f"ERROR: --timestep {args.timestep} out of range [0, {hp['T']})",
              file=sys.stderr)
        return 1

    # ---- Load BoostNet SD-VAE latents ----
    print(f"\nLoading BoostNet SD-VAE latents from {args.boostnet_latents}")
    data = np.load(args.boostnet_latents, allow_pickle=True)
    raw_latents = data["latents"]
    angles = data["angles"]
    splits = data["splits"].astype(str)
    uuids = data["uuids"]
    print(f"  Shape: {raw_latents.shape}")
    if raw_latents.shape[1] != hp["in_channels"]:
        print(f"WARNING: latent channels {raw_latents.shape[1]} != model "
              f"in_channels {hp['in_channels']}. Forward pass may error.",
              file=sys.stderr)
    print(f"  Source splits: " + str({s: int((splits == s).sum())
                                        for s in np.unique(splits)}))

    # Translate "boostnet_train" → "train" etc. The cropped-pipeline
    # extraction tags splits with a "boostnet_" prefix (so VinDr and
    # BoostNet can coexist in a shared pool), but train_cobb_regressor_matched.py
    # checks `splits == "train"`. Without this translation the regressor
    # sees zero training samples.
    splits = np.array([s.replace("boostnet_", "") if s.startswith("boostnet_")
                        else s for s in splits])
    print(f"  Translated splits: " + str({s: int((splits == s).sum())
                                            for s in np.unique(splits)}))

    n_images = len(raw_latents)

    # ---- Standardise ----
    x_0_std = ((raw_latents - latent_mean) / latent_std).astype(np.float32)
    x_0_std_t = torch.from_numpy(x_0_std)

    # ---- Fixed noise ----
    rng = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    noise = torch.randn(x_0_std_t.shape, generator=rng).float()

    # ---- Construct x_t coefficients at the fixed timestep ----
    sa = diffusion.sqrt_alphas_cumprod[args.timestep].item()
    soma = diffusion.sqrt_one_minus_alphas_cumprod[args.timestep].item()
    print(f"\nAt t={args.timestep}: sqrt(α_cum)={sa:.4f}, "
          f"sqrt(1-α_cum)={soma:.4f}")
    print(f"Extracting from down_blocks level {args.level}")

    # ---- Forward in batches ----
    print(f"\nForwarding {n_images} latents through the U-Net encoder "
          f"in batches of {args.batch_size}...")
    all_features = []

    for start in tqdm(range(0, n_images, args.batch_size), desc="  batch"):
        end = min(start + args.batch_size, n_images)
        bsz = end - start

        x_0_batch = x_0_std_t[start:end].to(device)
        noise_batch = noise[start:end].to(device)

        # Forward-diffuse to the fixed timestep
        x_t = sa * x_0_batch + soma * noise_batch

        t_tensor = torch.full((bsz,), args.timestep, dtype=torch.long,
                               device=device)
        dummy_cobb = torch.zeros((bsz, 3), device=device)
        drop_mask = torch.ones((bsz,), dtype=torch.bool, device=device)

        features = extract_encoder_features(
            model, x_t, t_tensor, dummy_cobb, drop_mask, args.level
        )   # (bsz, C, H, W)
        all_features.append(features.cpu().float().numpy())

    all_features = np.concatenate(all_features, axis=0)
    print(f"\nExtracted features shape: {all_features.shape}")
    print(f"  Activation stats: mean={all_features.mean():.3f}, "
          f"std={all_features.std():.3f}, "
          f"range=[{all_features.min():.3f}, {all_features.max():.3f}]")

    # ---- Save ----
    np.savez_compressed(
        args.out,
        latents=all_features.astype(np.float32),
        angles=angles.astype(np.float32),
        uuids=uuids,
        splits=splits,
    )
    print(f"\nWrote {args.out}")
    print(f"  Uncompressed size: {all_features.nbytes / 1e6:.1f} MB")

    print(f"\nNext steps:")
    print(f"  1. Add a new condition to run_multiseed_matched.py:")
    print(f"       {{")
    print(f'           "key":     "matched_diffusion_features",')
    print(f'           "label":   "Diffusion U-Net features (level {args.level}, '
          f't={args.timestep})",')
    print(f'           "latents": "{args.out}",')
    print(f"       }},")
    print(f"  2. Run multi-seed:")
    print(f"     python run_multiseed_matched.py "
          f"--out-dir ./outputs/multiseed --seeds 0 1 7 42 100")
    print(f"  3. Add to CONFIGS in phase_a_bootstrap.py and "
          f"phase_a_breakdowns.py, then re-run both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
