"""
prepare_spine_bboxes.py

For each BoostNet image, compute the bounding box of the spine in normalised
[0,1] coordinates from the landmark file. Saves a CSV with columns:

    image_uuid, bbox_x0, bbox_y0, bbox_x1, bbox_y1

These coordinates are used by the patch loader to bias random crops toward
the spine region rather than wasting training capacity on empty corners.

We deliberately keep the bounding box in normalised coordinates so it
survives any intermediate resize. The patch loader will multiply by the
actual image dimensions at load time.

Usage:
  python prepare_spine_bboxes.py \\
      --boostnet-src /path/to/boostnet_labeldata \\
      --csv-in       $PROJECT_DIR/medvae/data/boostnet.csv \\
      --csv-out      $PROJECT_DIR/medvae/data/spine_bboxes.csv \\
      --margin 0.05
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean_stem(filename: str) -> str:
    """Match the cleaning we did in prepare_boostnet.py."""
    stem = Path(filename).stem
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def load_split_landmarks(split_dir: Path) -> pd.DataFrame:
    """Read filenames.csv + landmarks.csv from a split directory.

    landmarks.csv has 136 columns: 68 landmarks * 2 coords (x, y), normalised
    to [0, 1] of image dimensions. We compute the (x0, y0, x1, y1) bounding
    box that contains all landmarks.
    """
    filenames = pd.read_csv(
        split_dir / "filenames.csv", header=None, names=["filename"]
    )["filename"].str.strip()
    landmarks = pd.read_csv(split_dir / "landmarks.csv", header=None).values
    if len(filenames) != len(landmarks):
        raise RuntimeError(
            f"Mismatch in {split_dir}: {len(filenames)} filenames vs "
            f"{len(landmarks)} landmark rows.")

    # landmarks shape: (N, 136) = (N, 68 landmarks, 2 coords)
    coords = landmarks.reshape(len(landmarks), 68, 2)  # (N, 68, [x, y])
    xs = coords[..., 0]                                 # (N, 68)
    ys = coords[..., 1]
    out = pd.DataFrame({
        "filename":  filenames,
        "image_uuid": filenames.map(clean_stem),
        "bbox_x0":   xs.min(axis=1),
        "bbox_y0":   ys.min(axis=1),
        "bbox_x1":   xs.max(axis=1),
        "bbox_y1":   ys.max(axis=1),
    })
    return out.drop(columns=["filename"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--boostnet-src", required=True, type=Path,
                   help="Path to extracted boostnet_labeldata directory.")
    p.add_argument("--csv-in", required=True, type=Path,
                   help="Existing boostnet.csv (has image_uuid + split columns).")
    p.add_argument("--csv-out", required=True, type=Path)
    p.add_argument("--margin", type=float, default=0.05,
                   help="Expand bbox by this fraction of image dimensions on "
                        "each side, so crops can include some surrounding "
                        "tissue rather than only the spine itself. Default 0.05.")
    args = p.parse_args()

    train_bb = load_split_landmarks(args.boostnet_src / "labels" / "training")
    test_bb = load_split_landmarks(args.boostnet_src / "labels" / "test")
    all_bb = pd.concat([train_bb, test_bb], ignore_index=True)
    print(f"Computed bounding boxes for {len(all_bb)} images "
          f"({len(train_bb)} train, {len(test_bb)} test).")

    # Apply margin and clamp to [0, 1].
    if args.margin > 0:
        all_bb["bbox_x0"] = (all_bb["bbox_x0"] - args.margin).clip(0.0, 1.0)
        all_bb["bbox_y0"] = (all_bb["bbox_y0"] - args.margin).clip(0.0, 1.0)
        all_bb["bbox_x1"] = (all_bb["bbox_x1"] + args.margin).clip(0.0, 1.0)
        all_bb["bbox_y1"] = (all_bb["bbox_y1"] + args.margin).clip(0.0, 1.0)

    # Sanity: report bbox shape statistics
    bb_w = all_bb["bbox_x1"] - all_bb["bbox_x0"]
    bb_h = all_bb["bbox_y1"] - all_bb["bbox_y0"]
    print(f"\nBbox width  (normalised): mean={bb_w.mean():.3f}  "
          f"range=[{bb_w.min():.3f}, {bb_w.max():.3f}]")
    print(f"Bbox height (normalised): mean={bb_h.mean():.3f}  "
          f"range=[{bb_h.min():.3f}, {bb_h.max():.3f}]")

    # Merge with the main split CSV so we have split + bbox in one file
    base = pd.read_csv(args.csv_in)
    merged = base.merge(all_bb, on="image_uuid", how="left")
    missing = merged["bbox_x0"].isna().sum()
    if missing:
        print(f"\nWARNING: {missing} rows have no matching bbox.")
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.csv_out, index=False)
    print(f"\nWrote {args.csv_out}")


if __name__ == "__main__":
    main()
