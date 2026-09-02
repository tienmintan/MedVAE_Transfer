"""
verify_corner_ordering.py — determine how the 68 landmarks group into vertebrae.

68 = 17 vertebrae x 4 corners. We need to know:
  (a) grouping: by-vertebra [v0c0 v0c1 v0c2 v0c3 v1c0 ...] vs
                by-corner-type [all 17 TL, all 17 TR, all 17 BL, all 17 BR]
  (b) within-vertebra order: which two corners are the SUPERIOR endplate and
      which two are the INFERIOR endplate.

This draws several candidate interpretations on a real image. For each, it
connects each vertebra's 4 corners into a quadrilateral and draws the inferred
superior (red) and inferior (blue) endplate lines. The CORRECT interpretation
is the one where:
  - quadrilaterals look like vertebrae (compact, non-self-intersecting boxes),
  - endplates are near-horizontal and stack down the spine,
  - red (superior) sits above blue (inferior) within each box.

Read-only except the diagnostic PNG.

Run:  python verify_corner_ordering.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HOME = Path.home()
LABELDATA = HOME / "MedVAE" / "data" / "boostnet_labeldata"
LM_CSV = LABELDATA / "labels" / "test" / "landmarks.csv"
FN_CSV = LABELDATA / "labels" / "test" / "filenames.csv"
IMG_DIRS = [LABELDATA / "data" / "test", LABELDATA / "data" / "training"]


def find_image(stem):
    for d in IMG_DIRS:
        if d.exists():
            hits = list(d.glob(f"{stem}.*"))
            if hits:
                return hits[0]
    return None


def parse_AB(row):
    """Ordering A confirmed earlier: all-68 x then all-68 y."""
    return np.stack([row[:68], row[68:]], axis=1)   # (68,2) normalised


def vertebrae_by_vertebra(pts, order):
    """pts (68,2). Group consecutive 4s as one vertebra. `order` maps the 4
    slots to roles (sup_left, sup_right, inf_left, inf_right)."""
    verts = []
    for v in range(17):
        four = pts[v * 4:(v + 1) * 4]
        sl, sr, il, ir = (four[order[0]], four[order[1]],
                          four[order[2]], four[order[3]])
        verts.append((sl, sr, il, ir))
    return verts


def vertebrae_by_corner_type(pts):
    """Interpretation: pts[0:17]=TL, [17:34]=TR, [34:51]=BL, [51:68]=BR."""
    TL, TR, BL, BR = pts[0:17], pts[17:34], pts[34:51], pts[51:68]
    return [(TL[i], TR[i], BL[i], BR[i]) for i in range(17)]


def draw(ax, img, W, H, verts, title):
    ax.imshow(img, cmap="gray")
    for (sl, sr, il, ir) in verts:
        poly = np.array([sl, sr, ir, il, sl]) * [W, H]
        ax.plot(poly[:, 0], poly[:, 1], color="lime", lw=0.8, alpha=0.8)
        # superior endplate (sl-sr) red, inferior (il-ir) blue
        ax.plot([sl[0]*W, sr[0]*W], [sl[1]*H, sr[1]*H], color="red", lw=1.6)
        ax.plot([il[0]*W, ir[0]*W], [il[1]*H, ir[1]*H], color="deepskyblue", lw=1.6)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    landmarks = pd.read_csv(LM_CSV, header=None).values.astype(float)
    filenames = pd.read_csv(FN_CSV, header=None).iloc[:, 0].astype(str).tolist()

    # pick one clear image
    idx = 0
    pts = parse_AB(landmarks[idx])
    stem = Path(filenames[idx]).stem
    img_path = find_image(stem)
    if img_path is None:
        print(f"Could not find image for {stem}; drawing on blank canvas.")
        img = np.ones((800, 300)) * 0.5
        H, W = img.shape
    else:
        img = np.array(Image.open(img_path).convert("L"))
        H, W = img.shape

    # candidate interpretations
    candidates = [
        ("by-vertebra [TL,TR,BL,BR]", lambda: vertebrae_by_vertebra(pts, [0, 1, 2, 3])),
        ("by-vertebra [TL,TR,BR,BL]", lambda: vertebrae_by_vertebra(pts, [0, 1, 3, 2])),
        ("by-vertebra [TL,BL,TR,BR]", lambda: vertebrae_by_vertebra(pts, [0, 2, 1, 3])),
        ("by-corner-type [TL|TR|BL|BR]", vertebrae_by_corner_type),
    ]

    fig, axes = plt.subplots(1, len(candidates), figsize=(4.2 * len(candidates), 7))
    for ax, (name, fn) in zip(axes, candidates):
        try:
            verts = fn()
            draw(ax, img, W, H, verts, name)
        except Exception as e:
            ax.set_title(f"{name}\n[error: {e}]", fontsize=7)
            ax.axis("off")
    fig.suptitle("Which ordering gives clean vertebrae?  "
                 "(green=box, red=superior endplate, blue=inferior)", fontsize=11)
    out = Path.cwd() / "corner_ordering_check.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")
    print("\nThe CORRECT panel has: compact non-crossing green boxes, red lines")
    print("above blue within each box, and endplates roughly horizontal,")
    print("stacking down the spine. Tell me which panel (1-4) is correct.")


if __name__ == "__main__":
    main()
