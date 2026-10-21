"""
extract_latents_boostnet_512.py: extract BoostNet SD-VAE latents at 512x512 in the VinDr latent schema.

Output schema, matching extract_latents_vindr.py:

  latents     : (N, 4, 64, 64) float16
  image_ids   : (N,) array of BoostNet image_uuid strings
  splits      : (N,) array of "boostnet_train", "boostnet_val", "boostnet_test"
  angles      : (N, 3) float32; CSV columns angle_proximal_thoracic,
                angle_main_thoracic, angle_thoracolumbar (col 0 holds the major curve)
  aspect_pad  : (N,) array of "horizontal", "vertical", or "none"

The "boostnet_train/val/test" split naming distinguishes these rows from
VinDr's "vindr_train/test", so the two datasets can be pooled downstream
without ambiguity.

This script mirrors the VinDr preprocessing: aspect-padded resize to 512x512,
SD-VAE encode at fp16. Cobb angles come from boostnet_with_angles.csv. Each
input is a JPEG, not a DICOM.

Inputs:  --csv (image_uuid, split, three angle columns); JPEG images in
         --img-dir.
Outputs: <out-dir>/latents_boostnet_512.npz (compressed); summary statistics
         printed to stdout.

Usage:
    python extract_latents_boostnet_512.py \\
        --csv     ./medvae/data/boostnet_with_angles.csv \\
        --img-dir ./medvae/data/boostnet \\
        --out-dir ./latents_boostnet_512 \\
        --batch-size 8
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL


TARGET_SIZE = 512
LATENT_H = LATENT_W = TARGET_SIZE // 8     # 64
LATENT_CHANNELS = 4

ANGLE_COLS = ["angle_proximal_thoracic", "angle_main_thoracic",
              "angle_thoracolumbar"]


# Aspect-padding (same as VinDr extraction, kept inline so this script is
# self-contained; the function is small enough that duplicating it is simpler
# than a shared utils file).

def aspect_pad_to_square(arr: np.ndarray, target_size: int):
    """Resize longer dim to target_size, pad shorter dim with zeros."""
    h, w = arr.shape
    long_side = max(h, w)
    scale = target_size / long_side
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.BICUBIC)
    resized = np.array(pil).astype(np.float32) / 255.0

    out = np.zeros((target_size, target_size), dtype=np.float32)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    out[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    if new_h == target_size and new_w == target_size:
        pad_axis = "none"
    elif new_h == target_size:
        pad_axis = "horizontal"
    else:
        pad_axis = "vertical"
    return out, pad_axis


# JPEG -> normalised array

def load_normalized(jpeg_path: Path) -> np.ndarray:
    pil = Image.open(jpeg_path).convert("L")
    arr = np.array(pil).astype(np.float32) / 255.0      # already [0, 1]
    # Match VinDr per-image min-max norm in case any image is not full-range
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo > 1e-6:
        arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


# Batched SD-VAE encoding

@torch.no_grad()
def encode_batch(vae, arrs: list, device, dtype) -> np.ndarray:
    batch = np.stack(arrs, axis=0)
    t = torch.from_numpy(batch).unsqueeze(1).repeat(1, 3, 1, 1)
    t = (t * 2.0) - 1.0
    t = t.to(device=device, dtype=dtype)
    latents = vae.encode(t).latent_dist.mode()           # (B, 4, 64, 64)
    return latents.to(torch.float16).cpu().numpy()


# Main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        required=True, type=Path)
    p.add_argument("--img-dir",    required=True, type=Path)
    p.add_argument("--out-dir",    required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} BoostNet rows from {args.csv}")

    for col in ANGLE_COLS + ["image_uuid", "split"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}")

    print("Loading Stable Diffusion VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                         torch_dtype=dtype).to(device)
    vae.eval()

    all_latents = []
    all_image_ids = []
    all_splits = []
    all_angles = []
    all_pad_axes = []

    batch_arrs, batch_meta = [], []

    def flush_batch():
        if not batch_arrs:
            return
        latents = encode_batch(vae, batch_arrs, device, dtype)
        for j, (img_id, sp, angles, pad_axis) in enumerate(batch_meta):
            all_latents.append(latents[j])
            all_image_ids.append(img_id)
            all_splits.append(sp)
            all_angles.append(angles)
            all_pad_axes.append(pad_axis)
        batch_arrs.clear()
        batch_meta.clear()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="BoostNet"):
        img_path = args.img_dir / f"{row['image_uuid']}.jpg"
        arr = load_normalized(img_path)
        padded, pad_axis = aspect_pad_to_square(arr, TARGET_SIZE)
        angles = np.array([row[c] for c in ANGLE_COLS], dtype=np.float32)
        # Prefix the split with "boostnet_" to distinguish from VinDr splits
        sp = f"boostnet_{row['split']}"
        batch_arrs.append(padded)
        batch_meta.append((row["image_uuid"], sp, angles, pad_axis))
        if len(batch_arrs) >= args.batch_size:
            flush_batch()

    flush_batch()

    latents_arr = np.stack(all_latents, axis=0).astype(np.float16)
    image_ids_arr = np.array(all_image_ids)
    splits_arr = np.array(all_splits)
    angles_arr = np.stack(all_angles, axis=0).astype(np.float32)
    pad_axes_arr = np.array(all_pad_axes)

    out_path = args.out_dir / "latents_boostnet_512.npz"
    np.savez_compressed(
        out_path,
        latents=latents_arr,
        image_ids=image_ids_arr,
        splits=splits_arr,
        angles=angles_arr,
        aspect_pad=pad_axes_arr,
    )

    # Cast to fp32 for stats reporting (statistics computed in fp16 can overflow)
    lat_fp32 = latents_arr.astype(np.float32)
    print(f"\nWrote {out_path}")
    print(f"  Shape: {latents_arr.shape}  "
          f"(~{latents_arr.nbytes / 1e6:.1f} MB uncompressed)")
    print(f"  Split counts: " +
          ", ".join(f"{s}={(splits_arr == s).sum()}"
                    for s in sorted(set(splits_arr))))
    print(f"  Latent stats (fp32): "
          f"mean={lat_fp32.mean():.3f} std={lat_fp32.std():.3f} "
          f"min={lat_fp32.min():.3f} max={lat_fp32.max():.3f}")
    print(f"  Angle ranges: "
          f"major curve (col 0) [{angles_arr[:,0].min():.1f}, {angles_arr[:,0].max():.1f}], "
          f"col 1 [{angles_arr[:,1].min():.1f}, {angles_arr[:,1].max():.1f}], "
          f"col 2 [{angles_arr[:,2].min():.1f}, {angles_arr[:,2].max():.1f}]")


if __name__ == "__main__":
    main()
