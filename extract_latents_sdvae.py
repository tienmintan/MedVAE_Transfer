"""
extract_latents_sdvae.py

Extract latents from BoostNet spine X-rays using Stable Diffusion's VAE
(stabilityai/sd-vae-ft-mse) as the encoder. Parallel to extract_latents_whole.py
but with SD-VAE in place of MedVAE.

The (a) experiment: test whether SD-VAE — which was trained with proper KL
regularisation specifically for use as a diffusion backbone — produces a
better-conditioned latent target than MedVAE for our Cobb-conditioned
diffusion task. The reconstruction sanity check showed mean PSNR ~36 dB
with visually clean recons, so the domain gap is manageable.

Pipeline per image:
  1. Open at native resolution, grayscale.
  2. Resize to 768x256 (H x W; aspect ratio ~1:3, close to BoostNet's
     native 1:2.8; both dims multiple of 32 for clean U-Net downsampling).
  3. Repeat grayscale across 3 RGB channels (SD-VAE expects 3-channel input).
  4. Scale to [-1, 1] (SD-VAE convention).
  5. Single forward pass through encoder. Produces (4, 96, 32) latent.
  6. Save one .npz per split keyed by image_uuid.

Schema matches extract_latents_whole.py: `latents` and `uuids` arrays per
split file, so train_diffusion.py works without modification.

Usage:
    python extract_latents_sdvae.py \\
        --csv     ./medvae/data/boostnet.csv \\
        --img-dir ./medvae/data/boostnet \\
        --out-dir ./latents_sdvae
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL


H_FIX, W_FIX = 768, 256       # (H, W) — multiples of 32, aspect ratio ~1:3
DOWNSAMPLE = 8                 # SD-VAE has 8x spatial compression
LATENT_H, LATENT_W = H_FIX // DOWNSAMPLE, W_FIX // DOWNSAMPLE       # 96, 32
LATENT_CHANNELS = 4


@torch.no_grad()
def encode_one(vae, image_path: Path, device: torch.device,
                dtype: torch.dtype) -> np.ndarray:
    """Encode a single BoostNet image through SD-VAE.

    Returns numpy array of shape (4, 96, 32), float32.
    """
    pil = Image.open(image_path).convert("L").resize((W_FIX, H_FIX), Image.BICUBIC)
    arr = np.array(pil).astype(np.float32) / 255.0       # (H, W) in [0, 1]

    # SD-VAE expects (B, 3, H, W) in [-1, 1].
    inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)              # (1, 1, H, W)
    inp = inp.repeat(1, 3, 1, 1)                                       # (1, 3, H, W)
    inp = (inp * 2.0) - 1.0                                            # [-1, 1]
    inp = inp.to(device=device, dtype=dtype)

    # `.latent_dist.mode()` returns the Gaussian posterior mean — the
    # deterministic latent used at inference, matching what we did for
    # MedVAE via posterior.mode().
    latent = vae.encode(inp).latent_dist.mode()         # (1, 4, 96, 32)
    return latent.float().squeeze(0).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--device",  default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16     # SD-VAE prefers fp16; produces stable latents

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stable Diffusion VAE (stabilityai/sd-vae-ft-mse)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                         torch_dtype=dtype).to(device)
    vae.eval()
    print(f"  Input target: {H_FIX}x{W_FIX}  ->  "
          f"latent {LATENT_CHANNELS}x{LATENT_H}x{LATENT_W}")

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            print(f"No {split} rows, skipping.")
            continue

        latents = np.empty((len(sub), LATENT_CHANNELS, LATENT_H, LATENT_W),
                            dtype=np.float32)
        uuids = []

        for i, row in tqdm(sub.iterrows(), total=len(sub),
                            desc=f"  {split}"):
            img_path = args.img_dir / f"{row['image_uuid']}.jpg"
            latents[i] = encode_one(vae, img_path, device, dtype)
            uuids.append(row["image_uuid"])

        out_path = args.out_dir / f"latents_{split}.npz"
        np.savez_compressed(out_path,
                             latents=latents,
                             uuids=np.array(uuids))
        print(f"  Wrote {out_path}  shape={latents.shape}  "
              f"({latents.nbytes / 1e6:.1f} MB uncompressed)  "
              f"stats: mean={latents.mean():.3f} std={latents.std():.3f}")


if __name__ == "__main__":
    main()
