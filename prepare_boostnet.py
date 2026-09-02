"""
prepare_boostnet.py

Prepare the BoostNet (AASCE 2019) spinal AP X-ray dataset for MedVAE finetuning.

What this script does:
  1. Reads images from <SRC>/data/training and <SRC>/data/test.
  2. Cleans filenames (replaces spaces and uppercases extensions, e.g.
     'sunhl-1th-25-Jan-2017-279 A AP.jpg' -> 'sunhl-1th-25-Jan-2017-279_A_AP.jpg').
  3. Copies cleaned images into <DST>/<DATASET_NAME>/<image_uuid>.jpg
     (a flat directory, as required by MedVAE's GenericDataset).
  4. Builds <DST>/<DATASET_NAME>.csv with columns: row_nr, image_uuid, split.
     The original test set (128 images) is preserved as the test split.
     The original training set (481 images) is split 85/15 into train/val.

Why this layout?
  MedVAE's mmgs-style dataloader expects:
    - A CSV at  ${data_dir}/${dataset_name}.csv
    - Images in ${data_dir}/${dataset_name}/<image_uuid><img_suffix>
  We point dataset_name=boostnet and img_suffix=.jpg in our custom configs.

Usage:
  python prepare_boostnet.py \\
      --src /path/to/boostnet_labeldata \\
      --dst /path/to/MedVAE/medvae/data \\
      --dataset-name boostnet \\
      --val-frac 0.15 \\
      --seed 42
"""

import argparse
import os
import random
import re
import shutil
from pathlib import Path

import pandas as pd


def clean_stem(filename: str) -> str:
    """Convert 'sunhl-1th-25-Jan-2017-279 A AP.jpg' -> 'sunhl-1th-25-Jan-2017-279_A_AP'."""
    stem = Path(filename).stem
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def collect_images(src_dir: Path) -> list[Path]:
    return sorted(p for p in src_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path,
                   help="Path to extracted boostnet_labeldata directory.")
    p.add_argument("--dst", required=True, type=Path,
                   help="MedVAE data dir, e.g. /path/to/MedVAE/medvae/data")
    p.add_argument("--dataset-name", default="boostnet")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of the original training images held out as val.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_src = args.src / "data" / "training"
    test_src = args.src / "data" / "test"
    if not train_src.is_dir() or not test_src.is_dir():
        raise FileNotFoundError(f"Expected {train_src} and {test_src} to exist.")

    img_dst = args.dst / args.dataset_name
    img_dst.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    train_imgs = collect_images(train_src)
    test_imgs = collect_images(test_src)
    print(f"Found {len(train_imgs)} training images, {len(test_imgs)} test images.")

    # Make a deterministic train/val split out of the original training set.
    train_imgs_shuffled = train_imgs.copy()
    rng.shuffle(train_imgs_shuffled)
    n_val = int(round(len(train_imgs_shuffled) * args.val_frac))
    val_set = set(train_imgs_shuffled[:n_val])
    train_set = set(train_imgs_shuffled[n_val:])

    rows = []
    seen_stems: dict[str, Path] = {}
    n_copied = 0

    def add(src_path: Path, split: str) -> None:
        nonlocal n_copied
        stem = clean_stem(src_path.name)
        # Defensively handle filename collisions after cleaning.
        if stem in seen_stems and seen_stems[stem] != src_path:
            i = 2
            while f"{stem}__{i}" in seen_stems:
                i += 1
            stem = f"{stem}__{i}"
        seen_stems[stem] = src_path
        dst_path = img_dst / f"{stem}.jpg"
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
            n_copied += 1
        rows.append({"row_nr": len(rows), "image_uuid": stem, "split": split})

    for img in train_imgs:
        add(img, "train" if img in train_set else "val")
    for img in test_imgs:
        add(img, "test")

    df = pd.DataFrame(rows)
    csv_path = args.dst / f"{args.dataset_name}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Copied {n_copied} new images into {img_dst}")
    print(f"Wrote CSV to {csv_path}")
    print("Split counts:")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
