"""
extract_latents_tiled.py: encode BoostNet images into per-image latent grids by tile-and-stitch in latent space.

Pre-encodes all BoostNet images into per-image latent grids using the
patch-trained MedVAE encoder via tile-and-stitch in latent space.

Pipeline per image:
  1. Open at native resolution, convert to single-channel float in [0, 1].
  2. Resize to a fixed (H_fix, W_fix) = (2304, 768). This is divisible by
     the encoder's effective stride (4x downsampling * 384 tile = 1536, but
     384-tile / stride-192 with 4x downsampling is used so the latent grid
     also tiles at stride 48 in latent space). The resize is mild for
     BoostNet (mean ~880x2480 -> resized to 768x2304).
  3. Slide 384x384 tiles, encode each, populate a single latent buffer of
     shape (1, H_fix/4, W_fix/4) = (1, 576, 192). Overlaps are blended with
     a cosine window in latent space (matching the reconstruction blending
     in evaluate_reconstruction_tiled.py, applied in the latent domain).
  4. Save the latents to one .npz file per split (latents, uuids), in
     image_uuid order. Per-tile min/max stats and the spine bbox are not
     saved.

Inputs:  --csv (image_uuid, split); JPEG images in --img-dir; the
         patch-finetuned checkpoint (--ckpt).
Outputs: <out-dir>/latents_{train,val,test}.npz (compressed).

Usage:
    python extract_latents_tiled.py \\
        --csv     ./medvae/data/boostnet.csv \\
        --img-dir ./medvae/data/boostnet \\
        --ckpt    /lambda/.../boostnet_finetuning_patch/.../checkpoints/best.pt \\
        --out-dir ./latents_tiled
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm

from evaluate_reconstruction_tiled import (
    make_cosine_window, tile_positions, load_finetuned_medvae,
)

H_FIX, W_FIX  = 2304, 768       # resize target
TILE_SIZE     = 384
STRIDE        = 192
DOWNSAMPLE    = 4               # MedVAE 4x model
LATENT_H, LATENT_W = H_FIX // DOWNSAMPLE, W_FIX // DOWNSAMPLE      # 576, 192
LATENT_TILE   = TILE_SIZE // DOWNSAMPLE                            # 96
LATENT_STRIDE = STRIDE // DOWNSAMPLE                               # 48


@torch.no_grad()
def encode_image_tiled(
    model,
    image: torch.Tensor,             # (1, H_FIX, W_FIX) in [0, 1]
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Slide 384x384 tiles over the resized image, encode each, blend the
    resulting 96x96 latents into a (z_channels, 576, 192) latent grid via
    cosine blending in latent space.

    Returns a tensor of shape (z_channels, 576, 192) on CPU as float32.
    """
    _, H, W = image.shape
    assert (H, W) == (H_FIX, W_FIX), \
        f"encode_image_tiled expects {(H_FIX, W_FIX)}, got {(H, W)}"

    # z_channels is not known up front; buffer creation is deferred until the first tile.
    output  = None
    weights = None
    window  = make_cosine_window(LATENT_TILE, device)   # (96, 96)

    ys = tile_positions(H, TILE_SIZE, STRIDE)
    xs = tile_positions(W, TILE_SIZE, STRIDE)

    for y in ys:
        for x in xs:
            tile = image[:, y:y + TILE_SIZE, x:x + TILE_SIZE]

            # Per-tile min-max normalisation matching training
            tmin, tmax = tile.min(), tile.max()
            if tmax > tmin:
                tile_n = (tile - tmin) / (tmax - tmin)
            else:
                tile_n = torch.zeros_like(tile)
            tile_n = (tile_n - 0.5) / 0.5      # [0, 1] -> [-1, 1]

            inp = tile_n.unsqueeze(0).to(device=device, dtype=dtype)
            with torch.autocast(device_type=device.type, dtype=dtype):
                # encode() returns a DiagonalGaussianDistribution; .mode() gives
                # the deterministic latent (mean, not sampled). This
                # matches what AutoencoderKL.forward does internally when
                # sample_posterior=False, i.e. inference mode.
                posterior = model.encode(inp)
                z = posterior.mode().float()              # (1, Cz, 96, 96)
            z = z.squeeze(0).to(device)                   # (Cz, 96, 96)

            if output is None:
                Cz = z.shape[0]
                output  = torch.zeros((Cz, LATENT_H, LATENT_W), device=device)
                weights = torch.zeros((1,  LATENT_H, LATENT_W), device=device)

            ly, lx = y // DOWNSAMPLE, x // DOWNSAMPLE
            output [:, ly:ly + LATENT_TILE, lx:lx + LATENT_TILE] += z * window
            weights[:, ly:ly + LATENT_TILE, lx:lx + LATENT_TILE] += window

    out = (output / weights.clamp(min=1e-6)).cpu()        # (Cz, 576, 192)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--ckpt",    required=True, type=str,
                   help="Path to patch-finetuned best.pt directory")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--device",  default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading patch-finetuned MedVAE from {args.ckpt}")
    model = load_finetuned_medvae(args.ckpt, device, dtype)

    # Pre-allocate per-split arrays; one .npz is saved per split.
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            print(f"No {split} rows, skipping.")
            continue

        # Discover Cz from the first image
        first_path = args.img_dir / f"{sub.iloc[0]['image_uuid']}.jpg"
        pil = Image.open(first_path).convert("L").resize((W_FIX, H_FIX), Image.BICUBIC)
        first = torch.from_numpy(np.array(pil)).float().unsqueeze(0) / 255.0
        first_z = encode_image_tiled(model, first.to(device), device, dtype)
        Cz = first_z.shape[0]

        latents = np.empty((len(sub), Cz, LATENT_H, LATENT_W), dtype=np.float32)
        uuids = []
        latents[0] = first_z.numpy()
        uuids.append(sub.iloc[0]["image_uuid"])

        for i, row in tqdm(sub.iloc[1:].iterrows(), total=len(sub) - 1,
                           desc=f"  {split}"):
            img_path = args.img_dir / f"{row['image_uuid']}.jpg"
            pil = Image.open(img_path).convert("L").resize(
                (W_FIX, H_FIX), Image.BICUBIC)
            arr = torch.from_numpy(np.array(pil)).float().unsqueeze(0) / 255.0
            z = encode_image_tiled(model, arr.to(device), device, dtype)
            latents[i] = z.numpy()
            uuids.append(row["image_uuid"])

        out_path = args.out_dir / f"latents_{split}.npz"
        np.savez_compressed(out_path,
                            latents=latents,
                            uuids=np.array(uuids))
        print(f"  Wrote {out_path}  shape={latents.shape}  "
              f"({latents.nbytes / 1e9:.2f} GB uncompressed)")


if __name__ == "__main__":
    main()
