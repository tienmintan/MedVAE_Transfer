"""
sdvae_reconstruction_check.py

Diagnostic: can Stable Diffusion's VAE reconstruct BoostNet spine X-rays?

Before committing to SD-VAE as a diffusion backbone, we need to verify it
can actually represent our data. SD-VAE was trained on natural images
(LAION dataset) — spine X-rays are visually distinct, so the domain gap
could be severe.

This script:
  1. Loads stabilityai/sd-vae-ft-mse (the SD 1.5 fine-tuned MSE VAE)
  2. Loads a few BoostNet test images
  3. Encodes each at 256x768 (aspect-preserving rectangular shape, multiple
     of 32 for clean U-Net downsampling, latent shape 32x96x4)
  4. Decodes back, computes PSNR/MS-SSIM
  5. Saves a side-by-side comparison figure

If reconstructions look reasonable -> proceed to SD-VAE extraction + diffusion
If reconstructions look bad     -> SD-VAE has too much domain gap, abandon

Usage:
    python sdvae_reconstruction_check.py \\
        --csv     ./medvae/data/boostnet.csv \\
        --img-dir ./medvae/data/boostnet \\
        --out     ./sdvae_recon_check.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import AutoencoderKL


# Target shape for inputs to SD-VAE.
# Multiple of 32 in both dims (for clean U-Net downsampling later).
# Aspect ratio 1:3 — closer to BoostNet's native 1:2.8 than 384x1024 would be,
# while keeping the spatial extent manageable.
W_FIX, H_FIX = 256, 768
LATENT_DOWNSAMPLE = 8     # SD-VAE has 8x spatial compression


@torch.no_grad()
def encode_decode_sdvae(vae, image_pil, device, dtype):
    """Run one BoostNet image through SD-VAE end-to-end.

    Returns (orig_np, recon_np), both numpy float arrays in [0, 1].
    """
    pil = image_pil.convert("L").resize((W_FIX, H_FIX), Image.BICUBIC)
    arr = np.array(pil).astype(np.float32) / 255.0    # (H, W) in [0, 1]

    # SD-VAE expects 3-channel RGB input in [-1, 1].
    # Repeat grayscale across channels.
    inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)              # (1, 1, H, W)
    inp = inp.repeat(1, 3, 1, 1)                                       # (1, 3, H, W)
    inp = (inp * 2.0) - 1.0                                            # [0, 1] -> [-1, 1]
    inp = inp.to(device=device, dtype=dtype)

    # Encode + decode
    latent = vae.encode(inp).latent_dist.mode()      # (1, 4, H/8, W/8)
    recon = vae.decode(latent).sample                # (1, 3, H, W) in [-1, 1]

    # Back to grayscale [0, 1] for comparison
    recon = recon.mean(dim=1).float()                # average RGB -> 1 ch
    recon = (recon + 1.0) / 2.0                      # [-1, 1] -> [0, 1]
    recon = recon.clamp(0.0, 1.0).squeeze(0).cpu().numpy()

    return arr, recon, latent.float().cpu().numpy()


def psnr(a, b):
    """PSNR between two [0, 1] images."""
    mse = ((a - b) ** 2).mean()
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--out",     required=True, type=Path)
    p.add_argument("--n",       type=int, default=4)
    p.add_argument("--device",  default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16    # SD-VAE prefers fp16 on most GPUs

    print("Loading Stable Diffusion VAE (stabilityai/sd-vae-ft-mse)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                        torch_dtype=dtype).to(device)
    vae.eval()
    print(f"  Latent channels: {vae.config.latent_channels}")
    print(f"  Spatial compression: {LATENT_DOWNSAMPLE}x")
    print(f"  Input target shape: {H_FIX}x{W_FIX}  ->  "
          f"latent {H_FIX//LATENT_DOWNSAMPLE}x{W_FIX//LATENT_DOWNSAMPLE}x4")

    # Pick a handful of test images
    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    rng = np.random.default_rng(42)
    selected_idx = rng.choice(len(test_df), size=args.n, replace=False)
    selected = test_df.iloc[selected_idx]

    print(f"\nRunning encode/decode through SD-VAE on {args.n} test images:")
    results = []
    for i, row in selected.iterrows():
        img_path = args.img_dir / f"{row['image_uuid']}.jpg"
        pil = Image.open(img_path)
        orig, recon, latent = encode_decode_sdvae(vae, pil, device, dtype)
        p_val = psnr(orig, recon)
        print(f"  [{i+1}/{args.n}] {row['image_uuid']}  "
              f"PSNR={p_val:.2f} dB  "
              f"latent stats: mean={latent.mean():.3f} std={latent.std():.3f}")
        results.append((row["image_uuid"], orig, recon, p_val))

    # Side-by-side figure with global windowing
    print("\nMaking comparison figure...")
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n),
                              gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    if n == 1:
        axes = axes[None, :]

    # Global windowing across all panels
    all_px = np.concatenate([r[1].ravel() for r in results] +
                            [r[2].ravel() for r in results])
    vmin, vmax = float(np.percentile(all_px, 1)), float(np.percentile(all_px, 99))

    axes[0, 0].set_title("Original (256x768)", fontsize=12)
    axes[0, 1].set_title("SD-VAE reconstruction", fontsize=12)
    last_handle = None
    for i, (uuid, orig, recon, p_val) in enumerate(results):
        last_handle = axes[i, 0].imshow(orig, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(recon, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 1].axis("off")
        axes[i, 1].text(0.02, 0.98, f"PSNR={p_val:.1f} dB",
                        transform=axes[i, 1].transAxes,
                        fontsize=10, color="white", va="top",
                        bbox=dict(boxstyle="round", facecolor="black",
                                  alpha=0.6))

    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.925, 0.10, 0.015, 0.80])
    cb = fig.colorbar(last_handle, cax=cbar_ax)
    cb.set_label("Normalised pixel intensity", rotation=270,
                 labelpad=15, fontsize=10)
    fig.text(0.5, 0.02,
             f"SD-VAE reconstruction sanity check at {H_FIX}x{W_FIX}. "
             f"Display window: 1-99th percentile.",
             ha="center", va="bottom", fontsize=9, style="italic")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {args.out}")

    mean_psnr = np.mean([r[3] for r in results])
    print(f"\nMean PSNR: {mean_psnr:.2f} dB")
    print(f"\nDecision criterion:")
    print(f"  PSNR > 30 dB and recons look sharp        -> proceed to diffusion")
    print(f"  PSNR 25-30 dB but recons look blurry      -> borderline, judgment call")
    print(f"  PSNR < 25 dB or recons have artifacts     -> domain gap too severe")


if __name__ == "__main__":
    main()
