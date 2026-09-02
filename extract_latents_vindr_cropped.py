"""
extract_latents_vindr_cropped.py

Modified version of extract_latents_vindr.py that crops out the aspect-
padding region before SD-VAE encoding, addressing the distribution shift
identified in investigate_dataset_shift_content_only.py.

Pipeline per DICOM:
  1. Load DICOM, apply Rescale Slope/Intercept, handle MONOCHROME1 inversion.
  2. Per-image min-max normalise to [0, 1].
  3. Detect content bbox via detect_content_bbox() — the smallest rectangle
     containing all non-black pixels (i.e., excluding scanner-edge padding,
     DICOM black borders).
  4. Crop to bbox, then resize+pad to a fixed target aspect ratio of 1:1.5
     centred in a 512×512 square. Result: content fills ~67% of the 512×512
     output by area, regardless of the original DICOM aspect ratio.
  5. SD-VAE encode → (4, 64, 64) latent.

Resumable via checkpointing every 500 images (writes intermediate .npz
files; combined into a single output at the end).

Usage:
    python extract_latents_vindr_cropped.py \\
        --dicom-dir   ~/MedVAE/vindr-spinexr \\
        --out-dir     ./latents_vindr_cropped \\
        --batch-size  1 \\
        --device      cuda
"""

import argparse
import sys
from pathlib import Path
from glob import glob

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import AutoencoderKL as SDAutoencoderKL

from detect_content_bbox import detect_content_bbox, crop_and_repad_to_aspect


# ---------------------------------------------------------------------------
# DICOM loading (matches extract_latents_vindr.py)
# ---------------------------------------------------------------------------

def load_dicom_to_array(dcm_path: str) -> np.ndarray:
    """Load a DICOM file, return a normalised [0, 1] grayscale array."""
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)

    # Apply Rescale Slope/Intercept if present
    if hasattr(dcm, "RescaleSlope"):
        img = img * float(dcm.RescaleSlope)
    if hasattr(dcm, "RescaleIntercept"):
        img = img + float(dcm.RescaleIntercept)

    # MONOCHROME1: lower values = brighter (inverted X-ray)
    photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        img = img.max() - img

    # Per-image min-max to [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-6:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    return img


# ---------------------------------------------------------------------------
# Encoding pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_one(img_norm: np.ndarray, vae, device,
               target_aspect: float = 1.5,
               target_size: int = 512) -> np.ndarray:
    """Crop+repad+encode a single image. Returns (4, 64, 64) numpy array."""
    # Step 1: detect content bbox and crop to it
    bbox = detect_content_bbox(img_norm)
    cropped = crop_and_repad_to_aspect(
        img_norm, target_aspect=target_aspect,
        target_size=target_size, bbox=bbox,
    )

    # Step 2: SD-VAE expects [-1, 1] 3-channel input
    tensor = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0   # [0, 1] -> [-1, 1]
    tensor_3ch = tensor.repeat(1, 3, 1, 1).to(device=device,
                                                 dtype=vae.dtype)

    # Step 3: encode
    latent = vae.encode(tensor_3ch).latent_dist.mode()
    return latent.squeeze(0).cpu().float().numpy()


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dicom-dir", required=True, type=Path,
                    help="Directory containing VinDr-SpineXR DICOM files. "
                         "Searched recursively for *.dicom and *.dcm.")
    p.add_argument("--out-dir",   required=True, type=Path,
                    help="Output directory for latents and intermediate "
                         "checkpoints.")
    p.add_argument("--target-aspect", type=float, default=1.5)
    p.add_argument("--target-size",   type=int, default=512)
    p.add_argument("--checkpoint-every", type=int, default=500,
                    help="Write intermediate checkpoint every N images for "
                         "resumability.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-images", type=int, default=None,
                    help="For testing: limit total number of images.")
    args = p.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "intermediate"
    ckpt_dir.mkdir(exist_ok=True)

    # ---- Collect DICOM paths ----
    dicom_paths = sorted(
        list(args.dicom_dir.rglob("*.dicom"))
        + list(args.dicom_dir.rglob("*.dcm"))
    )
    if args.max_images:
        dicom_paths = dicom_paths[:args.max_images]
    print(f"Found {len(dicom_paths)} DICOM files in {args.dicom_dir}")

    # ---- Determine resume point ----
    existing_ckpts = sorted(ckpt_dir.glob("chunk_*.npz"))
    if existing_ckpts:
        n_processed = sum(np.load(c)["latents"].shape[0] for c in existing_ckpts)
        print(f"Resuming: {len(existing_ckpts)} chunks found, "
              f"{n_processed} images already processed")
        dicom_paths = dicom_paths[n_processed:]
    else:
        n_processed = 0

    # ---- Load VAE ----
    print("Loading SD-VAE...")
    vae = SDAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                            torch_dtype=torch.float16).to(device)
    vae.eval()

    # ---- Process in chunks for resumability ----
    chunk_latents = []
    chunk_filenames = []
    chunk_idx = len(existing_ckpts)
    in_chunk_count = 0

    pbar = tqdm(dicom_paths, desc="Encoding")
    for dcm_path in pbar:
        try:
            img = load_dicom_to_array(str(dcm_path))
            latent = encode_one(
                img, vae, device,
                target_aspect=args.target_aspect,
                target_size=args.target_size,
            )
            chunk_latents.append(latent.astype(np.float16))
            chunk_filenames.append(dcm_path.name)
        except Exception as e:
            print(f"\n  Skipping {dcm_path.name}: {e}", file=sys.stderr)
            continue

        in_chunk_count += 1
        if in_chunk_count >= args.checkpoint_every:
            chunk_path = ckpt_dir / f"chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(
                chunk_path,
                latents=np.stack(chunk_latents, axis=0),
                filenames=np.array(chunk_filenames),
            )
            pbar.set_postfix_str(f"saved {chunk_path.name}")
            chunk_latents = []
            chunk_filenames = []
            chunk_idx += 1
            in_chunk_count = 0
            torch.cuda.empty_cache()

    # ---- Save final partial chunk ----
    if chunk_latents:
        chunk_path = ckpt_dir / f"chunk_{chunk_idx:04d}.npz"
        np.savez_compressed(
            chunk_path,
            latents=np.stack(chunk_latents, axis=0),
            filenames=np.array(chunk_filenames),
        )

    # ---- Combine all chunks into final output ----
    print("\nCombining chunks into final output...")
    all_chunks = sorted(ckpt_dir.glob("chunk_*.npz"))
    all_latents = []
    all_filenames = []
    for chunk_path in all_chunks:
        d = np.load(chunk_path, allow_pickle=True)
        all_latents.append(d["latents"])
        all_filenames.extend(d["filenames"].tolist())

    final_latents = np.concatenate(all_latents, axis=0)
    final_filenames = np.array(all_filenames)

    # In the v2 schema used by train_diffusion.py, we add split labels and
    # placeholder Cobb angles (VinDr has no Cobb labels, so all-zeros)
    splits = np.array(["vindr_train"] * len(final_latents))
    angles = np.zeros((len(final_latents), 3), dtype=np.float32)

    final_path = args.out_dir / "latents_vindr_cropped.npz"
    np.savez_compressed(
        final_path,
        latents=final_latents,
        filenames=final_filenames,
        splits=splits,
        angles=angles,
    )
    print(f"\nWrote {final_path}")
    print(f"  Shape: {final_latents.shape}")
    print(f"  Stats: mean={final_latents.astype(np.float64).mean():.4f} "
          f"std={final_latents.astype(np.float64).std():.4f}")


if __name__ == "__main__":
    main()
