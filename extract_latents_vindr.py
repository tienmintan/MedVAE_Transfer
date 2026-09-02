"""
extract_latents_vindr.py

DICOM -> SD-VAE latent extraction for VinDr-SpineXR.

For each DICOM file in train_images/ and test_images/:
  1. Read DICOM and extract pixel array.
  2. Apply Rescale Slope/Intercept (DICOM tags 0028,1053 and 0028,1052).
  3. Handle MONOCHROME1 inversion (high values = dark vs standard high = bright).
  4. Per-image min-max normalize to [0, 1].
  5. Resize with aspect-padding to 512x512 (preserves anatomy, black borders).
  6. Replicate grayscale to 3 channels and scale to [-1, 1].
  7. Encode through SD-VAE -> (4, 64, 64) latent.

Output: a single .npz at <out>/latents_vindr.npz containing:
  - latents : (N, 4, 64, 64) float16
  - image_ids : (N,) array of image_id strings
  - splits : (N,) array of "vindr_train" or "vindr_test"
  - aspect_pad : (N,) array of "horizontal" or "vertical" (which axis was padded)

Plus a vindr_processing_log.csv recording per-file outcome (ok/skipped/failed)
and a small subset of visualized examples for sanity checking.

This script is resumable: it skips images already in the output .npz, so
if it crashes partway through, re-running picks up where it left off.

Usage:
    python extract_latents_vindr.py \\
        --vindr-dir ~/MedVAE/vindr-spinexr \\
        --out-dir   ~/MedVAE/MedVAE/latents_vindr \\
        --batch-size 8
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import AutoencoderKL


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_SIZE = 512               # 512x512 input to SD-VAE
LATENT_H = TARGET_SIZE // 8      # 64
LATENT_W = TARGET_SIZE // 8      # 64
LATENT_CHANNELS = 4

# How many sanity-check visualizations to save (sampled across the dataset)
N_VISUAL_CHECKS = 24


# ---------------------------------------------------------------------------
# DICOM -> normalized grayscale array
# ---------------------------------------------------------------------------

def dicom_to_array(dicom_path: Path) -> Tuple[Optional[np.ndarray], str]:
    """Read DICOM and return float32 image in [0, 1], or (None, error_msg)
    if the file cannot be processed.

    Handles:
      - Rescale Slope/Intercept (linear mapping to anatomically meaningful units)
      - MONOCHROME1 inversion (DICOM tag 0028,0004)
      - Per-image min-max normalization to [0, 1]
    """
    try:
        ds = pydicom.dcmread(str(dicom_path), force=True)
    except Exception as e:
        return None, f"dcmread failed: {e!r}"

    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        return None, f"pixel_array failed: {e!r}"

    if arr.size == 0:
        return None, "empty pixel array"
    if arr.ndim != 2:
        return None, f"unexpected ndim {arr.ndim} (shape {arr.shape})"

    # Rescale Slope and Intercept: maps stored values to anatomically meaningful
    # units. Default values are slope=1, intercept=0 (identity).
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # MONOCHROME1 vs MONOCHROME2 — handle photometric inversion.
    photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photo == "MONOCHROME1":
        arr = arr.max() - arr     # invert so bone is bright (MONOCHROME2 convention)

    # Per-image min-max normalize to [0, 1] — matches what we do for BoostNet.
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return None, f"degenerate intensity range [{lo}, {hi}]"
    arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32), "ok"


# ---------------------------------------------------------------------------
# Aspect-padding to a fixed square size
# ---------------------------------------------------------------------------

def aspect_pad_to_square(arr: np.ndarray, target_size: int) -> Tuple[np.ndarray, str]:
    """Resize the longer dimension to target_size, pad the shorter dim with
    zeros to make a square. Returns (padded_array, pad_axis_str).

    pad_axis_str is "horizontal" if width was padded, "vertical" if height
    was padded, "none" if the input was already square.
    """
    h, w = arr.shape
    long_side = max(h, w)
    scale = target_size / long_side
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # PIL resize wants (W, H) order
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


# ---------------------------------------------------------------------------
# SD-VAE encoding (batched)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_batch(vae, arrs: list, device, dtype) -> np.ndarray:
    """Encode a list of (H, W) [0, 1] float arrays through SD-VAE.

    Returns (B, 4, 64, 64) float16 array.
    """
    # Stack to (B, H, W), grayscale-to-RGB, scale to [-1, 1]
    batch = np.stack(arrs, axis=0)                       # (B, H, W)
    t = torch.from_numpy(batch).unsqueeze(1)             # (B, 1, H, W)
    t = t.repeat(1, 3, 1, 1)                             # (B, 3, H, W)
    t = (t * 2.0) - 1.0
    t = t.to(device=device, dtype=dtype)

    latents = vae.encode(t).latent_dist.mode()           # (B, 4, 64, 64)
    return latents.to(torch.float16).cpu().numpy()


# ---------------------------------------------------------------------------
# Sanity-check visualization
# ---------------------------------------------------------------------------

def save_visual_check(orig_arrs, recon_arrs, image_ids, out_path: Path):
    """Side-by-side grid of original 512x512 and SD-VAE reconstruction.

    `recon_arrs` is decoded from the latents we just produced. This lets us
    visually verify that the latents we're storing actually encode the
    images we think they do.
    """
    n = len(orig_arrs)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n),
                              gridspec_kw={"hspace": 0.1, "wspace": 0.05})
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        axes[i, 0].imshow(orig_arrs[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(image_ids[i][:12] + "...", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(recon_arrs[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].axis("off")

    axes[0, 0].set_title("Original (512x512)", fontsize=10)
    axes[0, 1].set_title("SD-VAE recon", fontsize=10)
    fig.text(0.5, 0.02,
              "VinDr-SpineXR DICOM -> 512x512 -> SD-VAE encode -> decode "
              "(sanity check)",
              ha="center", fontsize=9, style="italic")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vindr-dir", required=True, type=Path,
                    help="Path to the VinDr-SpineXR directory containing "
                         "train_images/ and test_images/")
    p.add_argument("--out-dir",   required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8,
                    help="Encoding batch size. SD-VAE encoder at 512x512 "
                         "uses ~2 GB attention memory per sample, so keep "
                         "this small on the A10 (24 GB total).")
    p.add_argument("--limit",     type=int, default=None,
                    help="Process only this many files (for debugging).")
    p.add_argument("--device",    default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_dir / "latents_vindr.npz"
    log_csv = args.out_dir / "vindr_processing_log.csv"
    visual_dir = args.out_dir / "visual_checks"

    # Discover DICOM files in train_images and test_images
    train_dir = args.vindr_dir / "train_images"
    test_dir = args.vindr_dir / "test_images"
    train_files = sorted(train_dir.glob("*.dicom"))
    test_files = sorted(test_dir.glob("*.dicom"))
    if args.limit:
        train_files = train_files[: args.limit]
        test_files = test_files[: max(0, args.limit - len(train_files))]
    print(f"Found {len(train_files)} train DICOMs, {len(test_files)} test DICOMs")

    # Build the full processing list: (path, split)
    all_files = [(p, "vindr_train") for p in train_files] + \
                [(p, "vindr_test") for p in test_files]
    print(f"Total to process: {len(all_files)}")

    # Resumability — load any existing latents and skip already-processed IDs
    processed_ids = set()
    existing_latents = []
    existing_ids = []
    existing_splits = []
    existing_pad_axes = []
    if out_npz.exists():
        print(f"Resuming from existing {out_npz}")
        data = np.load(out_npz, allow_pickle=True)
        existing_latents = list(data["latents"])
        existing_ids = list(data["image_ids"].astype(str))
        existing_splits = list(data["splits"].astype(str))
        existing_pad_axes = list(data["aspect_pad"].astype(str))
        processed_ids = set(existing_ids)
        print(f"  {len(processed_ids)} already processed; skipping")

    to_process = [(fp, sp) for fp, sp in all_files if fp.stem not in processed_ids]
    print(f"Remaining to process: {len(to_process)}")

    # Open log in append mode if resuming, otherwise write header
    log_mode = "a" if log_csv.exists() else "w"
    log_f = open(log_csv, log_mode, newline="")
    log_w = csv.writer(log_f)
    if log_mode == "w":
        log_w.writerow(["image_id", "split", "status", "h", "w", "pad_axis"])

    # Load SD-VAE
    print("Loading Stable Diffusion VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                         torch_dtype=dtype).to(device)
    vae.eval()

    # Pick indices to capture visual sanity checks at (evenly spaced)
    visual_check_indices = (set(np.linspace(0, len(to_process) - 1,
                                              N_VISUAL_CHECKS).astype(int))
                              if len(to_process) >= N_VISUAL_CHECKS else set())
    visual_originals, visual_recons, visual_ids = [], [], []

    # Process in batches
    new_latents, new_ids, new_splits, new_pad_axes = [], [], [], []
    batch_arrs, batch_meta = [], []     # accumulators across iterations

    pbar = tqdm(enumerate(to_process), total=len(to_process), desc="DICOMs")
    save_every = 500   # Checkpoint to disk every N successful images

    def flush_batch():
        """Encode the accumulated batch and append to the new_* lists."""
        if not batch_arrs:
            return
        latents = encode_batch(vae, batch_arrs, device, dtype)
        for j, (img_id, sp, pad_axis, want_visual) in enumerate(batch_meta):
            new_latents.append(latents[j])
            new_ids.append(img_id)
            new_splits.append(sp)
            new_pad_axes.append(pad_axis)
            if want_visual:
                # Decode this latent for the visual check
                with torch.no_grad():
                    lt = torch.from_numpy(latents[j:j+1].astype(np.float32))
                    lt = lt.to(device=device, dtype=dtype)
                    recon = vae.decode(lt).sample
                recon = recon.mean(dim=1)                # avg RGB -> 1ch
                recon = ((recon.float() + 1.0) / 2.0).clamp(0, 1)
                recon = recon.squeeze(0).cpu().numpy()
                visual_originals.append(batch_arrs[j])
                visual_recons.append(recon)
                visual_ids.append(img_id)
        batch_arrs.clear()
        batch_meta.clear()

    def checkpoint():
        """Write all accumulated results (existing + new) to the .npz."""
        all_lat = existing_latents + new_latents
        all_ids = existing_ids + new_ids
        all_splits = existing_splits + new_splits
        all_pads = existing_pad_axes + new_pad_axes
        np.savez_compressed(
            out_npz,
            latents=np.stack(all_lat, axis=0).astype(np.float16),
            image_ids=np.array(all_ids),
            splits=np.array(all_splits),
            aspect_pad=np.array(all_pads),
        )

    successes = 0
    failures = 0
    for idx, (dicom_path, sp) in pbar:
        img_id = dicom_path.stem
        arr, status = dicom_to_array(dicom_path)
        if arr is None:
            log_w.writerow([img_id, sp, status, "", "", ""])
            failures += 1
            continue

        h, w = arr.shape
        padded, pad_axis = aspect_pad_to_square(arr, TARGET_SIZE)
        want_visual = (idx in visual_check_indices)

        batch_arrs.append(padded)
        batch_meta.append((img_id, sp, pad_axis, want_visual))
        log_w.writerow([img_id, sp, "ok", h, w, pad_axis])
        successes += 1

        # Flush batch when full
        if len(batch_arrs) >= args.batch_size:
            flush_batch()

        # Periodic checkpoint
        if (successes > 0) and (successes % save_every == 0) and len(new_latents):
            flush_batch()    # ensure batch is encoded before checkpointing
            checkpoint()
            log_f.flush()
            pbar.set_postfix(ok=successes, fail=failures,
                              ckpt=f"@{successes}")

    # Final flush + checkpoint
    flush_batch()
    if new_latents:
        checkpoint()
    log_f.close()

    print(f"\n=== Summary ===")
    print(f"  Newly processed: {successes} ok, {failures} failed")
    print(f"  Total in .npz:   {len(existing_latents) + len(new_latents)}")

    # Latent statistics (informative for diffusion training later)
    if new_latents:
        all_arr = np.stack(new_latents, axis=0)
        print(f"  New latent stats: shape={all_arr.shape}  "
              f"mean={all_arr.mean():.3f} std={all_arr.std():.3f} "
              f"min={all_arr.min():.3f} max={all_arr.max():.3f}")

    # Save the visual checks
    if visual_originals:
        save_visual_check(visual_originals, visual_recons, visual_ids,
                            visual_dir / "vindr_sanity_check.png")
        print(f"  Saved {len(visual_originals)} sanity-check visualizations "
              f"to {visual_dir / 'vindr_sanity_check.png'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Partial progress is saved to disk.", file=sys.stderr)
        sys.exit(130)
