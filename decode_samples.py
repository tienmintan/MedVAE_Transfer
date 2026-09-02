"""
decode_samples.py

Decode synthetic latents back to pixel-space spine X-rays via MedVAE's
decoder, for visual inspection of diffusion model outputs. Reads a .npz
file produced by either sample_diffusion.py (synthetic latents) or
train_diffusion.py's periodic sample save, and writes a grid PNG.

Usage:
    # Visualize a few mid-training samples
    python decode_samples.py \\
        --latents-npz   ./diffusion_ckpt/samples_epoch_0200.npz \\
        --out           ./samples_epoch_0200.png \\
        --window-mode   percentile

    # Visualize first 16 latents from a full sample run
    python decode_samples.py \\
        --latents-npz   ./synthetic_latents_500.npz \\
        --out           ./synthetic_grid.png \\
        --n-show        16
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from medvae.models.autoencoder_kl import AutoencoderKL


def load_medvae_decoder(device: torch.device, dtype: torch.dtype):
    """Load the pretrained medvae_4_1_2d model for decoding."""
    cfg_path = hf_hub_download(repo_id="stanfordmimi/MedVAE",
                               filename="model_weights/medvae_4x1.yaml")
    ckpt_path = hf_hub_download(repo_id="stanfordmimi/MedVAE",
                                filename="model_weights/vae_4x_1c_2D.ckpt")
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    model = AutoencoderKL(**cfg, ckpt_path=ckpt_path).to(device).eval()
    return model


@torch.no_grad()
def decode_latents(model, latents: torch.Tensor, device, dtype):
    """Decode (N, 1, H, W) latents to (N, H*4, W*4) reconstructions.

    Decodes one latent at a time because MedVAE's bottleneck self-attention
    is over H*W positions per sample, and batching multiple together blows
    up the attention matrix memory (16 x 24576^2 = 18 GB at our latent shape).
    """
    out = []
    for i in range(latents.shape[0]):
        x = latents[i:i+1].to(device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype):
            decoded = model.decode(x)
        # MedVAE output is in [-1, 1] convention; bring to [0, 1] for display
        decoded = (decoded.float() + 1.0) / 2.0
        decoded = decoded.clamp(0.0, 1.0)
        out.append(decoded.squeeze(0).squeeze(0).cpu().numpy())
        torch.cuda.empty_cache()
    return np.stack(out, axis=0)                  # (N, H, W)


def compute_bounds(images, mode: str, low: float, high: float):
    """Global windowing bounds across all displayed pixels."""
    if mode == "fixed":
        return 0.0, 1.0
    flat = np.concatenate([im.ravel() for im in images])
    if mode == "minmax":
        return float(flat.min()), float(flat.max())
    return float(np.percentile(flat, low)), float(np.percentile(flat, high))


def make_grid(images, angles, vmin, vmax, out_path: Path,
              ncols: int = 4):
    """Render a grid of decoded images with their target Cobb angles as titles."""
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3 * ncols, 4 * nrows),
                             gridspec_kw={"hspace": 0.15, "wspace": 0.05})
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]

    last_handle = None
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        last_handle = ax.imshow(images[i], cmap="gray", vmin=vmin, vmax=vmax)
        if angles is not None:
            a = angles[i]
            ax.set_title(f"PT={a[0]:.0f}°  MT={a[1]:.0f}°  TL={a[2]:.0f}°",
                         fontsize=9)
        ax.axis("off")

    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.925, 0.10, 0.015, 0.80])
    cb = fig.colorbar(last_handle, cax=cbar_ax)
    cb.set_label("Normalised pixel intensity", rotation=270,
                 labelpad=15, fontsize=9)
    fig.text(0.5, 0.02,
             f"Decoded synthetic samples. Display window: [{vmin:.3f}, {vmax:.3f}].",
             ha="center", va="bottom", fontsize=9, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents-npz", required=True, type=Path,
                   help="NPZ from sample_diffusion.py or train_diffusion's "
                        "periodic samples. Must contain 'latents' array.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-show", type=int, default=8,
                   help="How many latents to decode and show.")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--window-mode", choices=["percentile", "minmax", "fixed"],
                   default="percentile")
    p.add_argument("--window-low",  type=float, default=1.0)
    p.add_argument("--window-high", type=float, default=99.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    data = np.load(args.latents_npz, allow_pickle=True)
    latents = data["latents"]    # (N, 1, H, W)
    n = min(args.n_show, len(latents))
    latents = latents[:n]
    print(f"Loaded {len(data['latents'])} latents from {args.latents_npz}, "
          f"showing first {n}")
    print(f"  Latent shape: {latents.shape}  "
          f"stats: mean={latents.mean():.3f} std={latents.std():.3f} "
          f"min={latents.min():.3f} max={latents.max():.3f}")

    # Some files (synthetic) have 'angles'. Periodic samples have 'target_angles'
    angles = None
    if "angles" in data.files:
        angles = data["angles"][:n]
    elif "target_angles" in data.files:
        angles = data["target_angles"]
        # Periodic-sample file has K targets and K samples; trim/repeat as needed
        if len(angles) != n:
            angles = np.tile(angles, (n // len(angles) + 1, 1))[:n]

    print("Loading MedVAE decoder...")
    model = load_medvae_decoder(device, dtype)

    print("Decoding...")
    latents_t = torch.from_numpy(latents.astype(np.float32))
    images = decode_latents(model, latents_t, device, dtype)
    print(f"  Decoded shape: {images.shape}")

    vmin, vmax = compute_bounds(images, args.window_mode,
                                args.window_low, args.window_high)
    print(f"  Display window ({args.window_mode}): [{vmin:.3f}, {vmax:.3f}]")

    make_grid(images, angles, vmin, vmax, args.out, ncols=args.ncols)


if __name__ == "__main__":
    main()
