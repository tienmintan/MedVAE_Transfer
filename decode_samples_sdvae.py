"""
decode_samples_sdvae.py: decode synthetic latents to pixel-space spine X-rays with the SD-VAE decoder.

Counterpart of decode_samples.py for Stable Diffusion's VAE
(stabilityai/sd-vae-ft-mse). It reads the same .npz schema but expects
4-channel latents at SD-VAE's spatial extent (e.g., 96x32 for inputs encoded
at 768x256).

Inputs:  a .npz file containing 'latents' (N, 4, H, W) and optionally
         'angles' or 'target_angles'.
Outputs: a grid PNG at --out, titled with the target Cobb angles where present.

Usage:
    python decode_samples_sdvae.py \\
        --latents-npz   ./synthetic_latents_sdvae_500.npz \\
        --out           ./synthetic_sdvae_grid.png \\
        --n-show        16 \\
        --ncols         4
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL


@torch.no_grad()
def decode_one(vae, latent: torch.Tensor, device, dtype) -> np.ndarray:
    """Decode a single (1, 4, H, W) latent to a greyscale image.

    SD-VAE outputs RGB; the channels are averaged for display because the
    inputs were greyscale replicated to 3 channels.
    """
    latent = latent.to(device=device, dtype=dtype)
    decoded = vae.decode(latent).sample                  # (1, 3, H*8, W*8) in [-1, 1]
    gray = decoded.mean(dim=1)                           # avg RGB -> 1ch
    gray = (gray.float() + 1.0) / 2.0                    # [-1, 1] -> [0, 1]
    return gray.clamp(0.0, 1.0).squeeze(0).cpu().numpy()


def compute_bounds(images, mode: str, low: float, high: float):
    if mode == "fixed":
        return 0.0, 1.0
    flat = np.concatenate([im.ravel() for im in images])
    if mode == "minmax":
        return float(flat.min()), float(flat.max())
    return float(np.percentile(flat, low)), float(np.percentile(flat, high))


def make_grid(images, angles, vmin, vmax, out_path, ncols: int = 4):
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4 * nrows),
                              gridspec_kw={"hspace": 0.15, "wspace": 0.05})
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]

    last = None
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off"); continue
        last = ax.imshow(images[i], cmap="gray", vmin=vmin, vmax=vmax)
        if angles is not None:
            a = angles[i]
            ax.set_title(f"PT={a[0]:.0f}°  MT={a[1]:.0f}°  TL={a[2]:.0f}°",
                          fontsize=9)
        ax.axis("off")

    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.925, 0.10, 0.015, 0.80])
    cb = fig.colorbar(last, cax=cbar_ax)
    cb.set_label("Normalised pixel intensity", rotation=270,
                  labelpad=15, fontsize=9)
    fig.text(0.5, 0.02,
              f"Decoded synthetic samples via SD-VAE. "
              f"Display window: [{vmin:.3f}, {vmax:.3f}].",
              ha="center", va="bottom", fontsize=9, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents-npz", required=True, type=Path)
    p.add_argument("--out",         required=True, type=Path)
    p.add_argument("--n-show",      type=int, default=8)
    p.add_argument("--ncols",       type=int, default=4)
    p.add_argument("--window-mode", choices=["percentile", "minmax", "fixed"],
                    default="percentile")
    p.add_argument("--window-low",  type=float, default=1.0)
    p.add_argument("--window-high", type=float, default=99.0)
    p.add_argument("--device",      default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16

    data = np.load(args.latents_npz, allow_pickle=True)
    latents = data["latents"]
    n = min(args.n_show, len(latents))
    latents = latents[:n]
    print(f"Loaded {len(data['latents'])} latents from {args.latents_npz}, "
          f"showing first {n}")
    print(f"  Latent shape: {latents.shape}  "
          f"stats: mean={latents.mean():.3f} std={latents.std():.3f} "
          f"min={latents.min():.3f} max={latents.max():.3f}")

    angles = None
    if "angles" in data.files:
        angles = data["angles"][:n]
    elif "target_angles" in data.files:
        ta = data["target_angles"]
        angles = np.tile(ta, (n // len(ta) + 1, 1))[:n] if len(ta) != n else ta

    print("Loading SD-VAE decoder...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                         torch_dtype=dtype).to(device)
    vae.eval()

    print("Decoding...")
    out = []
    for i in range(n):
        lat = torch.from_numpy(latents[i:i+1].astype(np.float32))
        img = decode_one(vae, lat, device, dtype)
        out.append(img)
        torch.cuda.empty_cache()
    images = np.stack(out, axis=0)
    print(f"  Decoded shape: {images.shape}")

    vmin, vmax = compute_bounds(images, args.window_mode,
                                 args.window_low, args.window_high)
    print(f"  Display window ({args.window_mode}): [{vmin:.3f}, {vmax:.3f}]")

    make_grid(images, angles, vmin, vmax, args.out, ncols=args.ncols)


if __name__ == "__main__":
    main()
