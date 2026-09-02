"""
extract_latents_boostnet_cropped.py

Modified version of extract_latents_boostnet_512.py that crops out the
aspect-padding region before SD-VAE encoding, addressing the distribution
shift identified in investigate_dataset_shift_content_only.py.

For BoostNet, we use the spine bbox from prepare_spine_bboxes.py if
available (the bbox columns: xmin, ymin, xmax, ymax in the CSV).
Otherwise we fall back to content-based detection via detect_content_bbox.

Pipeline per image:
  1. Load JPEG, convert to grayscale, normalise to [0, 1].
  2. Use spine bbox from CSV (or detect via threshold).
  3. Crop to bbox, then resize+pad to 1:1.5 aspect centred in 512×512.
  4. SD-VAE encode → (4, 64, 64) latent.

Output schema is v2 (compatible with train_diffusion.py):
  latents, splits, angles, uuids

Usage:
    python extract_latents_boostnet_cropped.py \\
        --csv         ./medvae/data/boostnet_with_angles.csv \\
        --img-dir     ./medvae/data/boostnet \\
        --out         ./latents_boostnet_cropped/latents_boostnet_cropped.npz \\
        --device      cuda
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL as SDAutoencoderKL

from detect_content_bbox import detect_content_bbox, crop_and_repad_to_aspect


@torch.no_grad()
def encode_one(img_norm: np.ndarray, vae, device,
               bbox: tuple = None,
               target_aspect: float = 1.5,
               target_size: int = 512) -> np.ndarray:
    """Crop+repad+encode a single image. Returns (4, 64, 64) numpy array."""
    cropped = crop_and_repad_to_aspect(
        img_norm, target_aspect=target_aspect,
        target_size=target_size, bbox=bbox,
    )
    tensor = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0   # [0, 1] -> [-1, 1]
    tensor_3ch = tensor.repeat(1, 3, 1, 1).to(device=device,
                                                 dtype=vae.dtype)
    latent = vae.encode(tensor_3ch).latent_dist.mode()
    return latent.squeeze(0).cpu().float().numpy()


def load_image(path: Path) -> np.ndarray:
    """Load a JPEG, convert to grayscale, normalise to [0, 1]."""
    pil = Image.open(path).convert("L")
    arr = np.array(pil).astype(np.float32) / 255.0
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path,
                    help="BoostNet labels CSV. Must contain image_uuid, "
                         "split, and three angle columns (auto-detected from "
                         "common naming patterns).")
    p.add_argument("--bbox-csv", type=Path, default=None,
                    help="Optional separate CSV with spine bbox columns "
                         "(image_uuid + xmin/ymin/xmax/ymax). If provided, "
                         "bboxes are joined onto the main CSV by image_uuid. "
                         "If not provided, falls back to threshold-based "
                         "content detection.")
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--target-aspect", type=float, default=1.5)
    p.add_argument("--target-size",   type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)

    # ---- Load CSV ----
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")

    # Auto-detect angle column names
    angle_column_candidates = [
        ("PT", "MT", "TL"),
        ("angle_proximal_thoracic", "angle_main_thoracic", "angle_thoracolumbar"),
        ("proximal_thoracic", "main_thoracic", "thoracolumbar"),
        ("cobb_pt", "cobb_mt", "cobb_tl"),
        ("cobb_angle_1", "cobb_angle_2", "cobb_angle_3"),
    ]
    pt_col, mt_col, tl_col = None, None, None
    for pt, mt, tl in angle_column_candidates:
        if pt in df.columns and mt in df.columns and tl in df.columns:
            pt_col, mt_col, tl_col = pt, mt, tl
            break
    if pt_col is None:
        raise ValueError(
            f"Could not find Cobb angle columns. Looked for one of: "
            f"{angle_column_candidates}. Available columns: {df.columns.tolist()}"
        )
    print(f"  Using angle columns: ({pt_col}, {mt_col}, {tl_col})")

    # Optional bbox join
    if args.bbox_csv is not None:
        bdf = pd.read_csv(args.bbox_csv)
        print(f"\nLoaded {len(bdf)} bbox rows from {args.bbox_csv}")
        print(f"  Columns: {bdf.columns.tolist()}")
        required = {"image_uuid", "xmin", "ymin", "xmax", "ymax"}
        missing = required - set(bdf.columns)
        if missing:
            raise ValueError(
                f"Bbox CSV missing required columns: {missing}. "
                f"Found: {bdf.columns.tolist()}"
            )
        df = df.merge(bdf[list(required)], on="image_uuid", how="left")
        # Report how many got bboxes from the join
        n_with_bbox = df["xmin"].notna().sum()
        print(f"  Joined: {n_with_bbox}/{len(df)} rows have spine bboxes "
              f"(others will fall back to content detection)")
        has_bbox_columns = True
    else:
        has_bbox_columns = all(c in df.columns
                                for c in ["xmin", "ymin", "xmax", "ymax"])
        if has_bbox_columns:
            print("  Spine bbox columns found in main CSV — using them.")
        else:
            print("  No spine bbox columns — using content detection for all images.")

    # Tag splits for the v2 schema. BoostNet uses boostnet_train / boostnet_val
    # / boostnet_test
    def _split_tag(s):
        if s == "test":
            return "boostnet_test"
        elif s == "val":
            return "boostnet_val"
        elif s == "train":
            return "boostnet_train"
        else:
            return s

    df["v2_split"] = df["split"].map(_split_tag)

    # ---- Load SD-VAE ----
    print("\nLoading SD-VAE...")
    vae = SDAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                            torch_dtype=torch.float16).to(device)
    vae.eval()

    # ---- Encode ----
    all_latents = np.empty((len(df), 4, 64, 64), dtype=np.float16)
    all_angles  = np.empty((len(df), 3), dtype=np.float32)
    all_splits  = np.empty(len(df), dtype=object)
    all_uuids   = np.empty(len(df), dtype=object)

    n_bbox_used = 0
    n_fallback = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
        uuid = row["image_uuid"]
        img_path = args.img_dir / f"{uuid}.jpg"

        try:
            img = load_image(img_path)
        except Exception as e:
            print(f"\n  Skipping {uuid}: {e}")
            continue

        # Determine bbox: from CSV if available and non-null, else detect
        bbox = None
        if has_bbox_columns and pd.notna(row.get("xmin", np.nan)):
            bbox = (int(row["ymin"]), int(row["xmin"]),
                    int(row["ymax"]), int(row["xmax"]))
            n_bbox_used += 1
        else:
            bbox = detect_content_bbox(img)
            n_fallback += 1

        latent = encode_one(
            img, vae, device, bbox=bbox,
            target_aspect=args.target_aspect,
            target_size=args.target_size,
        )
        all_latents[i] = latent.astype(np.float16)
        all_angles[i]  = [row[pt_col], row[mt_col], row[tl_col]]
        all_splits[i]  = row["v2_split"]
        all_uuids[i]   = uuid

    print(f"\nBbox source: {n_bbox_used} from CSV, {n_fallback} from content detection")

    # ---- Save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        latents=all_latents,
        angles=all_angles,
        splits=all_splits.astype(str),
        uuids=all_uuids.astype(str),
    )
    print(f"\nWrote {args.out}")
    print(f"  Shape: {all_latents.shape}")
    print(f"  Stats: mean={all_latents.astype(np.float64).mean():.4f} "
          f"std={all_latents.astype(np.float64).std():.4f}")


if __name__ == "__main__":
    main()
