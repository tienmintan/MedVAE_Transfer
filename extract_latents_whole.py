"""
extract_latents_whole.py: extract per-image latents by whole-image encoding, without tiling.

Extracts per-image latents via whole-image encoding at W_FIX x H_FIX
(currently 384x1024, set in the module), one forward
pass per image, no tiling. Saves one .npz per split in the format of
extract_latents_tiled.py, so the output fits the downstream regressor
pipeline.

Why this instead of tile-and-stitch latent extraction:
  - One latent per image (96x256 at 384x1024) instead of a tiled grid (576x192).
  - No tile boundaries, no overlapping latents averaged together, no per-tile
    normalisation quirks.
  - Avoids the latent-space cosine blending logic which adds complexity.
  - The whole-image reconstruction check (evaluate_whole_image.py) tests
    whether the patch encoder generalises to whole-image inference; if it
    does, the latents should be coherent.

Trade-offs vs tile-and-stitch:
  - Still requires a resize (~880x2480 -> 384x1024). This is a milder
    interpolation than the 384x384 squash of the aspect-padded pipeline,
    but is not native resolution. Native resolution does not fit in memory
    due to bottleneck self-attention.
  - Whole-image and tile-and-stitch reconstructions are compared in PSNR
    by evaluate_whole_image.py; latent quality may follow the same pattern.

Inputs:  --csv (image_uuid, split); JPEG images in --img-dir; the
         patch-finetuned checkpoint (--ckpt).
Outputs: <out-dir>/latents_{train,val,test}.npz (compressed).

Usage:
    python extract_latents_whole.py \\
        --csv     ./medvae/data/boostnet.csv \\
        --img-dir ./medvae/data/boostnet \\
        --ckpt    $PATCH_CKPT \\
        --out-dir ./latents_whole
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from evaluate_reconstruction_tiled import load_finetuned_medvae

H_FIX, W_FIX = 1024, 384   # PIL uses (W, H); store consistently
DOWNSAMPLE   = 4
LATENT_H, LATENT_W = H_FIX // DOWNSAMPLE, W_FIX // DOWNSAMPLE   # 256, 96


@torch.no_grad()
def encode_whole(model, image: torch.Tensor, device, dtype):
    """One forward pass through the encoder. No tiling.

    image: (1, H_FIX, W_FIX) in [0, 1]
    Returns: (Cz, LATENT_H, LATENT_W) on CPU as float32.
    """
    # Per-image min-max normalisation matching the patch-training scheme,
    # applied at whole-image scale instead of per-patch.
    imin, imax = image.min(), image.max()
    if imax > imin:
        norm = (image - imin) / (imax - imin)
    else:
        norm = torch.zeros_like(image)
    norm = (norm - 0.5) / 0.5     # [0, 1] -> [-1, 1]

    inp = norm.unsqueeze(0).to(device=device, dtype=dtype)
    with torch.autocast(device_type=device.type, dtype=dtype):
        posterior = model.encode(inp)
        z = posterior.mode().float()      # (1, Cz, LATENT_H, LATENT_W)
    return z.squeeze(0).cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--ckpt",    required=True, type=str,
                   help="Path to patch-finetuned best.pt directory.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--device",  default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading patch-finetuned MedVAE from {args.ckpt}")
    model = load_finetuned_medvae(args.ckpt, device, dtype)

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            print(f"No {split} rows, skipping.")
            continue

        # Discover Cz from the first image
        first_path = args.img_dir / f"{sub.iloc[0]['image_uuid']}.jpg"
        pil = Image.open(first_path).convert("L").resize((W_FIX, H_FIX), Image.BICUBIC)
        first = torch.from_numpy(np.array(pil)).float().unsqueeze(0) / 255.0
        first_z = encode_whole(model, first.to(device), device, dtype)
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
            z = encode_whole(model, arr.to(device), device, dtype)
            latents[i] = z.numpy()
            uuids.append(row["image_uuid"])

        out_path = args.out_dir / f"latents_{split}.npz"
        np.savez_compressed(out_path,
                            latents=latents,
                            uuids=np.array(uuids))
        print(f"  Wrote {out_path}  shape={latents.shape}  "
              f"({latents.nbytes / 1e6:.1f} MB uncompressed)")


if __name__ == "__main__":
    main()
