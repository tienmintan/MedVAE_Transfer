"""
prepare_cobb_labels.py: join BoostNet's per-image Cobb angles to boostnet.csv.

The three angles per image are stored in the columns angle_proximal_thoracic,
angle_main_thoracic and angle_thoracolumbar (dataset order). They are joined
on image_uuid to boostnet.csv, which already has image_uuid + split columns.

The MAT files in labels/{training,test}/*.mat contain landmark data per
image and are not used here. angles.csv + filenames.csv are used instead;
they give the angles in row order matching the image list.

Inputs:  labels/{training,test}/{filenames,angles}.csv under --boostnet-src;
         --csv-in (boostnet.csv).
Outputs: --csv-out (boostnet_with_angles.csv); per-split angle statistics
         printed to stdout.

Usage:
  python prepare_cobb_labels.py \\
      --boostnet-src /path/to/boostnet_labeldata \\
      --csv-in       $PROJECT_DIR/medvae/data/boostnet.csv \\
      --csv-out      $PROJECT_DIR/medvae/data/boostnet_with_angles.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def clean_stem(filename: str) -> str:
    """Apply the same filename cleaning as prepare_boostnet.py."""
    stem = Path(filename).stem
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def load_split_labels(split_dir: Path) -> pd.DataFrame:
    """Read filenames.csv + angles.csv from a split directory and join them."""
    filenames = pd.read_csv(
        split_dir / "filenames.csv", header=None, names=["filename"]
    )["filename"].str.strip()
    angles = pd.read_csv(
        split_dir / "angles.csv",
        header=None,
        names=["angle_proximal_thoracic", "angle_main_thoracic",
               "angle_thoracolumbar"],
    )
    if len(filenames) != len(angles):
        raise RuntimeError(
            f"Mismatch in {split_dir}: {len(filenames)} filenames vs "
            f"{len(angles)} angle rows.")
    df = pd.concat([filenames, angles], axis=1)
    df["image_uuid"] = df["filename"].map(clean_stem)
    return df.drop(columns=["filename"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--boostnet-src", required=True, type=Path,
                   help="Path to extracted boostnet_labeldata directory.")
    p.add_argument("--csv-in", required=True, type=Path,
                   help="Existing boostnet.csv (has image_uuid + split columns).")
    p.add_argument("--csv-out", required=True, type=Path)
    args = p.parse_args()

    train_labels = load_split_labels(args.boostnet_src / "labels" / "training")
    test_labels = load_split_labels(args.boostnet_src / "labels" / "test")
    all_labels = pd.concat([train_labels, test_labels], ignore_index=True)
    print(f"Total labelled images: {len(all_labels)} "
          f"({len(train_labels)} from training, {len(test_labels)} from test)")

    base = pd.read_csv(args.csv_in)
    print(f"boostnet.csv has {len(base)} rows")
    merged = base.merge(all_labels, on="image_uuid", how="left")
    missing = merged["angle_proximal_thoracic"].isna().sum()
    if missing:
        print(f"WARNING: label join found no matching angles for {missing} rows. "
              f"These rows will be dropped from the regressor's training set.")
        print("First few unmatched UUIDs:")
        print(merged.loc[merged["angle_proximal_thoracic"].isna(),
                         "image_uuid"].head().to_string())

    merged.to_csv(args.csv_out, index=False)
    print(f"\nWrote {args.csv_out}")
    print("\nAngle distribution by split:")
    desc = (
        merged.dropna()
              .groupby("split")[["angle_proximal_thoracic",
                                 "angle_main_thoracic",
                                 "angle_thoracolumbar"]]
              .agg(["mean", "std", "min", "max"])
    )
    print(desc.to_string())


if __name__ == "__main__":
    main()
