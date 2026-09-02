"""
verify_landmark_alignment.py — confirm landmarks.csv rows map to the right images.

landmarks.csv is positional (no IDs). The AASCE/BoostNet distribution pairs it
with filenames.csv in the same folder, same row order. This script:

  1. Locates landmarks.csv + filenames.csv for a split.
  2. Builds the row -> filename -> uuid mapping.
  3. Cross-checks those uuids against the uuids your pipeline uses
     (boostnet.csv / boostnet_with_angles.csv) for the same split.
  4. Overlays each sampled image's landmarks (ordering A: all-x then all-y,
     normalised) on THAT specific image, so correspondence is confirmed by eye.
  5. Writes a uuid -> landmark-row-index lookup CSV that Step 2 will consume.

Read-only except for the diagnostic PNG and the lookup CSV.

Run:  python verify_landmark_alignment.py --split test
Then open landmark_alignment_check.png and paste console output back.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HOME = Path.home()
LABELDATA = HOME / "MedVAE" / "data" / "boostnet_labeldata"
PIPELINE_CSV = HOME / "MedVAE" / "MedVAE" / "medvae" / "data" / "boostnet.csv"

IMG_DIRS = [
    LABELDATA / "data" / "training",
    LABELDATA / "data" / "test",
    HOME / "MedVAE" / "MedVAE" / "medvae" / "data" / "boostnet",
]

SPLIT_DIRNAME = {"train": "training", "training": "training",
                 "val": "training", "test": "test"}


def find_image(stem):
    for d in IMG_DIRS:
        if not d.exists():
            continue
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG"]:
            cand = d / f"{stem}{ext}"
            if cand.exists():
                return cand
        hits = list(d.glob(f"{stem}.*"))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test", "val", "training"])
    ap.add_argument("--n", type=int, default=4, help="how many images to overlay")
    args = ap.parse_args()

    sub = SPLIT_DIRNAME[args.split]
    label_dir = LABELDATA / "labels" / sub
    lm_path = label_dir / "landmarks.csv"
    fn_path = label_dir / "filenames.csv"

    print("=" * 66)
    print(f"LANDMARK ALIGNMENT CHECK  (split dir: {sub})")
    print("=" * 66)
    print(f"landmarks.csv: {lm_path}  exists={lm_path.exists()}")
    print(f"filenames.csv: {fn_path}  exists={fn_path.exists()}")

    if not lm_path.exists():
        print("ERROR: landmarks.csv missing. Listing label dir:")
        for p in label_dir.glob("*"):
            print("  ", p.name)
        return 1

    landmarks = pd.read_csv(lm_path, header=None).values.astype(float)
    print(f"\nLandmarks: {landmarks.shape}  (rows = images, 136 = 68 x then 68 y)")

    # --- establish row -> filename mapping ---
    filenames = None
    if fn_path.exists():
        fn_raw = pd.read_csv(fn_path, header=None)
        filenames = fn_raw.iloc[:, 0].astype(str).tolist()
        print(f"filenames.csv: {len(filenames)} entries")
        print(f"  examples: {filenames[:3]}")
        if len(filenames) != len(landmarks):
            print(f"  WARNING: {len(filenames)} filenames vs {len(landmarks)} "
                  f"landmark rows — counts disagree, alignment suspect.")
    else:
        print("\nNo filenames.csv. Trying to infer order from the image directory")
        print("(AASCE images are often numbered in the same order as the labels).")
        img_dir = LABELDATA / "data" / sub
        if img_dir.exists():
            imgs = sorted(img_dir.glob("*.*"))
            print(f"  {len(imgs)} images in {img_dir}")
            filenames = [p.name for p in imgs]
        else:
            print(f"  Could not find image dir {img_dir}; cannot infer order.")

    # --- cross-check against pipeline uuids ---
    def norm_uuid(s):
        return Path(str(s)).stem  # drop extension if present

    pipe_uuids = None
    if PIPELINE_CSV.exists():
        pdf = pd.read_csv(PIPELINE_CSV)
        split_key = "test" if args.split == "test" else (
            "val" if args.split == "val" else "train")
        pipe_uuids = pdf[pdf["split"] == split_key]["image_uuid"].tolist()
        print(f"\nPipeline uuids for split '{split_key}': {len(pipe_uuids)}")
        print(f"  examples: {pipe_uuids[:2]}")

    if filenames is not None:
        lm_uuids = [norm_uuid(f) for f in filenames]
        if pipe_uuids is not None:
            set_lm, set_pipe = set(lm_uuids), set(pipe_uuids)
            overlap = set_lm & set_pipe
            print(f"\nUUID overlap (landmark filenames ∩ pipeline): "
                  f"{len(overlap)}/{len(set_pipe)}")
            if len(overlap) == len(set_pipe):
                print("  -> FULL overlap: same images, just need row mapping. Good.")
            else:
                missing = list(set_pipe - set_lm)[:3]
                print(f"  -> partial. Examples in pipeline but not landmark set: {missing}")
                print("     (May be a naming-format difference; inspect before trusting.)")

    # --- visual overlay on the correctly-matched images ---
    rng = np.random.default_rng(0)
    n = min(args.n, len(landmarks))
    pick = rng.choice(len(landmarks), size=n, replace=False)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]
    any_image = False
    for ax, row_idx in zip(axes, pick):
        row = landmarks[row_idx]
        xs, ys = row[:68], row[68:]
        stem = norm_uuid(filenames[row_idx]) if filenames else None
        img_path = find_image(stem) if stem else None
        if img_path is not None:
            any_image = True
            im = np.array(Image.open(img_path).convert("L"))
            H, W = im.shape
            ax.imshow(im, cmap="gray")
            ax.scatter(xs * W, ys * H, s=8, c="red")
            ax.set_title(f"row {row_idx}\n{stem[:22]}", fontsize=8)
        else:
            ax.scatter(xs, ys, s=8, c="red")
            ax.invert_yaxis()
            ax.set_title(f"row {row_idx}\n(image not found)", fontsize=8)
        ax.axis("off")
    fig.suptitle("Landmarks overlaid on their MATCHED images — dots should sit on vertebrae",
                 fontsize=11)
    out_png = Path.cwd() / "landmark_alignment_check.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out_png}")
    if not any_image:
        print("  (No images found to overlay — set IMG_DIRS so the visual check works.)")

    # --- write the uuid -> row lookup for Step 2 ---
    if filenames is not None:
        lookup = pd.DataFrame({
            "landmark_row": np.arange(len(filenames)),
            "image_uuid": [norm_uuid(f) for f in filenames],
        })
        out_csv = Path.cwd() / f"landmark_row_lookup_{sub}.csv"
        lookup.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}  ({len(lookup)} rows)")
        print("\nStep 2 will use this to fetch the right landmark row per uuid.")

    print("\n-> Confirm the red dots sit on vertebrae in EACH panel, then we")
    print("   build Step 2 (edge vs interior reconstruction error).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
