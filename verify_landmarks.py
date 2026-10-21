"""
verify_landmarks.py: check the BoostNet landmark format (value range, column count, coordinate ordering).

Read-only. Loads a couple of rows from landmarks.csv, reports the value range
(to infer normalised vs pixel coords) and the column count, then tries both
plausible orderings and overlays them on the image so the correct ordering
can be identified visually. Saves one diagnostic PNG.

Inputs:  landmarks.csv under LABELS_DIR and the matching image under IMG_DIRS
         (paths set in the module).
Outputs: landmark_format_check.png in the current directory; the format
         report printed to stdout.

Usage:  python verify_landmarks.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HOME = Path.home()
LABELS_DIR = HOME / "MedVAE" / "data" / "boostnet_labeldata" / "labels"
TRAIN_CSV = LABELS_DIR / "training" / "landmarks.csv"
TEST_CSV  = LABELS_DIR / "test" / "landmarks.csv"

# Image dirs; adjust if the diagnostic cannot find images
IMG_DIRS = [
    HOME / "MedVAE" / "data" / "boostnet_labeldata" / "data" / "training",
    HOME / "MedVAE" / "data" / "boostnet_labeldata" / "data" / "test",
    HOME / "MedVAE" / "MedVAE" / "medvae" / "data" / "boostnet",
]


def find_image(stem):
    for d in IMG_DIRS:
        if not d.exists():
            continue
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            for cand in [d / f"{stem}{ext}", d / f"{stem.lower()}{ext}"]:
                if cand.exists():
                    return cand
        # also try glob in case of naming differences
        hits = list(d.glob(f"{stem}*"))
        if hits:
            return hits[0]
    return None


def main():
    print("=" * 64)
    print("Landmark format verification")
    print("=" * 64)

    csv_path = TRAIN_CSV if TRAIN_CSV.exists() else TEST_CSV
    print(f"Reading: {csv_path}")
    if not csv_path.exists():
        print("Error: landmarks.csv not found at expected path.")
        print("Listing the labels dir to help locate it:")
        for p in LABELS_DIR.rglob("*"):
            print("  ", p)
        return 1

    # Try reading with and without a header; BoostNet's file normally has no header
    raw = pd.read_csv(csv_path, header=None)
    print(f"\nShape (no-header read): {raw.shape}")
    print(f"First row, first 6 values: {raw.iloc[0, :6].tolist()}")
    print(f"First row, last 6 values:  {raw.iloc[0, -6:].tolist()}")

    # Is the first column an image name/id (non-numeric)?
    first_col_numeric = pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna().all()
    print(f"\nFirst column numeric? {first_col_numeric}")
    if not first_col_numeric:
        print(f"  -> first column looks like an ID. Examples: {raw.iloc[:3, 0].tolist()}")
        coords = raw.iloc[:, 1:].astype(float).values
        ids = raw.iloc[:, 0].astype(str).values
    else:
        coords = raw.astype(float).values
        ids = None

    n_vals = coords.shape[1]
    print(f"\nNumeric values per row: {n_vals}  (expect 136 = 68 landmarks x 2)")
    vmin, vmax = np.nanmin(coords), np.nanmax(coords)
    print(f"Value range: [{vmin:.3f}, {vmax:.3f}]  -> "
          f"{'normalised [0,1]' if vmax <= 1.5 else 'pixel coordinates'}")

    if n_vals != 136:
        print(f"  WARNING: value-count check failed (expected 136 values, got {n_vals}). The ordering overlays below may be invalid.")

    # Two candidate orderings for row 0:
    row = coords[0]
    half = n_vals // 2
    # Ordering A: all-x then all-y  (classic BoostNet)
    xs_A, ys_A = row[:half], row[half:]
    # Ordering B: interleaved (x0,y0,x1,y1,...)
    xs_B, ys_B = row[0::2], row[1::2]

    # Overlay both on the image, if it can be found
    stem = ids[0] if ids is not None else None
    img_path = find_image(stem) if stem else None
    if img_path is None and stem is not None:
        print(f"\nCould not auto-find image for id '{stem}'. "
              f"The visual check needs IMG_DIRS to point at the images.")

    fig, axes = plt.subplots(1, 2, figsize=(9, 7))
    for ax, (xs, ys, name) in zip(
        axes, [(xs_A, ys_A, "Ordering A: all-x then all-y"),
               (xs_B, ys_B, "Ordering B: interleaved x,y")]):
        if img_path is not None:
            im = np.array(Image.open(img_path).convert("L"))
            H, W = im.shape
            ax.imshow(im, cmap="gray")
            # scale normalised coords to pixels if needed
            px = xs * W if vmax <= 1.5 else xs
            py = ys * H if vmax <= 1.5 else ys
            ax.scatter(px, py, s=10, c="red")
        else:
            ax.scatter(xs, ys, s=10, c="red")
            ax.invert_yaxis()
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    fig.suptitle("Which ordering puts dots on vertebral corners?", fontsize=12)
    out = Path.cwd() / "landmark_format_check.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved visual check to {out}")
    print("\nVisual check: the correct ordering is the one whose red dots")
    print("   land on vertebral corners (panel A or B in")
    print("   landmark_format_check.png).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
