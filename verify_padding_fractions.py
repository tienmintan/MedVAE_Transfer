"""
verify_padding_fractions.py: measure padding fractions at each stage of the original and cropped preprocessing pipelines.

Measures, image by image, how much padding the tensors fed to SD-VAE contain
under the original (aspect-padded) and the content-aware (cropped) pipelines.

Stages measured (per image):

    A. Original pipeline: aspect-pad image into a 512x512 square.
       This is the input of the original (aspect-padded) diffusion training data.

    B. Content-bbox crop: crop image to its content bounding box.
       (No padding added; just isolates the non-padding region.)

    C. Pad to 1:1.5 aspect: extend short side with zero padding.
       (Step 2 of crop_and_repad_to_aspect.)

    D. Final 512x512 (new pipeline): full output of
       crop_and_repad_to_aspect. This is the input of the cropped-pipeline
       diffusion training data.

For each stage the script computes the "near-zero pixel fraction": the
fraction of pixels below a small threshold, used as a proxy for padding. It
also records the content bounding box (coordinates, height, width and
aspect ratio) of each input image.

Inputs:  the BoostNet CSV and JPEG directory; optionally a bbox CSV; the
         VinDr-SpineXR DICOM directory (unless --skip-vindr).
Outputs: --out-csv (per-image measurements), --out-fig (stage A vs stage D
         examples); a per-corpus summary printed to stdout.

Usage:
    # run from the repository root
    python verify_padding_fractions.py \\
        --boostnet-csv      ./medvae/data/boostnet_with_angles.csv \\
        --boostnet-img-dir  ./medvae/data/boostnet \\
        --boostnet-bbox-csv ./medvae/data/boostnet.csv \\
        --vindr-dicom-dir   ./vindr-spinexr \\
        --n-samples         30 \\
        --out-csv           verify_padding_fractions.csv \\
        --out-fig           verify_padding_fractions.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Import from the project's content-detection module
from detect_content_bbox import detect_content_bbox


# --- Configuration ---

TARGET_ASPECT = 1.5     # height:width for the intermediate stage
TARGET_SIZE   = 512     # final output side length
NEAR_ZERO_THRESHOLD = 1e-3   # pixels below this are "padding"
                              # (input images are in [0, 1])
SAMPLE_RNG_SEED = 20260606


# --- Helpers: image loading (matching extract_latents_*_cropped.py) ---

def load_boostnet_image(path: Path) -> np.ndarray:
    """Load a BoostNet JPEG and normalise to [0, 1]."""
    pil = Image.open(path).convert("L")
    return np.array(pil).astype(np.float32) / 255.0


def load_vindr_dicom(path: Path) -> np.ndarray:
    """Load a VinDr DICOM, apply MONOCHROME inversion if needed, normalise."""
    import pydicom
    dcm = pydicom.dcmread(str(path))
    img = dcm.pixel_array.astype(np.float32)

    if hasattr(dcm, "RescaleSlope"):
        img = img * float(dcm.RescaleSlope)
    if hasattr(dcm, "RescaleIntercept"):
        img = img + float(dcm.RescaleIntercept)

    photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        img = img.max() - img

    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-6:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    return img


# --- Pipeline stages (instrumented versions) ---

def stage_a_original_pipeline(img: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """The original pre-cropping pipeline: aspect-pad to a square then resize.

    Equivalent to the per-image preprocessing used for the original
    (aspect-padded) diffusion training corpora.
    """
    H, W = img.shape
    # Resize so longer side = target_size, preserving aspect
    if H >= W:
        new_h = target_size
        new_w = max(1, int(round(W * target_size / H)))
    else:
        new_w = target_size
        new_h = max(1, int(round(H * target_size / W)))
    pil = Image.fromarray(img.astype(np.float32))
    pil = pil.resize((new_w, new_h), Image.BICUBIC)
    resized = np.array(pil)

    # Pad to (target_size, target_size) with zeros
    out = np.zeros((target_size, target_size), dtype=resized.dtype)
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return out


def stage_b_cropped(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Stage B: crop image to its content bbox. No padding added."""
    y0, x0, y1, x1 = bbox
    return img[y0:y1, x0:x1]


def stage_c_padded_to_1p5(content: np.ndarray, target_aspect: float = TARGET_ASPECT) -> np.ndarray:
    """Stage C: pad the short side of content to reach target_aspect.

    This is step 2 of crop_and_repad_to_aspect. Returns an image
    whose aspect ratio is target_aspect.
    """
    ch, cw = content.shape
    content_aspect = ch / cw

    if content_aspect > target_aspect:
        new_w = int(round(ch / target_aspect))
        pad_w = (new_w - cw) // 2
        out = np.zeros((ch, new_w), dtype=content.dtype)
        out[:, pad_w:pad_w + cw] = content
        return out
    elif content_aspect < target_aspect:
        new_h = int(round(cw * target_aspect))
        pad_h = (new_h - ch) // 2
        out = np.zeros((new_h, cw), dtype=content.dtype)
        out[pad_h:pad_h + ch, :] = content
        return out
    else:
        return content.copy()


def stage_d_final_512(padded: np.ndarray, target_aspect: float = TARGET_ASPECT,
                        target_size: int = TARGET_SIZE) -> np.ndarray:
    """Stage D: resize the 1:1.5 padded image to a 341x512 strip,
    then pad horizontally to 512x512.

    Equivalent to steps 3-4 of crop_and_repad_to_aspect.
    """
    new_h = target_size
    new_w = int(round(target_size / target_aspect))    # 512/1.5 = 341

    pil = Image.fromarray(padded.astype(np.float32))
    pil = pil.resize((new_w, new_h), Image.BICUBIC)
    resized = np.array(pil)

    out = np.zeros((target_size, target_size), dtype=resized.dtype)
    pad_left = (target_size - new_w) // 2
    out[:, pad_left:pad_left + new_w] = resized
    return out


# --- Measurement ---

def near_zero_fraction(arr: np.ndarray, threshold: float = NEAR_ZERO_THRESHOLD) -> float:
    """Fraction of pixels at or below the threshold (treated as 'padding')."""
    return float((arr <= threshold).mean())


def process_image(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int] | None = None,
) -> Dict[str, float | int | tuple]:
    """Run an image through all four pipeline stages and measure each."""
    if bbox is None:
        bbox = detect_content_bbox(img)

    H, W = img.shape
    y0, x0, y1, x1 = bbox
    bbox_h, bbox_w = y1 - y0, x1 - x0
    if bbox_h <= 0 or bbox_w <= 0:
        raise ValueError(
            f"Degenerate bbox (h={bbox_h}, w={bbox_w}); "
            f"input ({H}x{W}), bbox ({y0},{x0},{y1},{x1})."
        )

    # A. Original pipeline
    stage_a = stage_a_original_pipeline(img)

    # B. Cropped to content bbox
    stage_b = stage_b_cropped(img, bbox)

    # C. Padded to 1:1.5
    stage_c = stage_c_padded_to_1p5(stage_b)

    # D. Final 512x512
    stage_d = stage_d_final_512(stage_c)

    return {
        "input_h": H,
        "input_w": W,
        "input_aspect_hw": H / W,
        "bbox_y0": y0, "bbox_x0": x0, "bbox_y1": y1, "bbox_x1": x1,
        "bbox_h": bbox_h, "bbox_w": bbox_w,
        "bbox_aspect_hw": bbox_h / bbox_w,
        "input_near_zero_frac":   near_zero_fraction(img),
        "stage_a_near_zero_frac": near_zero_fraction(stage_a),
        "stage_b_near_zero_frac": near_zero_fraction(stage_b),
        "stage_c_near_zero_frac": near_zero_fraction(stage_c),
        "stage_d_near_zero_frac": near_zero_fraction(stage_d),
        "stage_a_array": stage_a,    # for visualisation
        "stage_d_array": stage_d,    # for visualisation
    }


# --- Main ---

def gather_boostnet_samples(args, rng: np.random.Generator) -> List[Dict]:
    """Sample BoostNet images and process them through all stages.

    By default uses content detection (matching what
    extract_latents_boostnet_cropped.py does in practice). If a bbox CSV
    is explicitly provided, attempts to use it, auto-detecting between
    pixel and [0, 1]-normalised coordinate conventions.
    """
    df = pd.read_csv(args.boostnet_csv)
    print(f"BoostNet: {len(df)} total images")

    # If bbox CSV provided AND its values are usable, join it.
    # Otherwise detect_content_bbox() is used for every image, which is what
    # the cropped-extraction pipeline did in practice.
    use_csv_bboxes = False
    if args.boostnet_bbox_csv is not None and args.boostnet_bbox_csv.is_file():
        bdf = pd.read_csv(args.boostnet_bbox_csv)
        # Recognise common column conventions
        if all(c in bdf.columns for c in ["xmin", "ymin", "xmax", "ymax", "image_uuid"]):
            keep_cols = ["image_uuid", "xmin", "ymin", "xmax", "ymax"]
            rename_map = None
        elif all(c in bdf.columns for c in
                 ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "image_uuid"]):
            keep_cols = ["image_uuid", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]
            rename_map = {"bbox_x0": "xmin", "bbox_y0": "ymin",
                          "bbox_x1": "xmax", "bbox_y1": "ymax"}
        else:
            print(f"BoostNet: bbox CSV columns are {bdf.columns.tolist()}; "
                  "no recognised convention, using content detection.")
            keep_cols = None

        if keep_cols is not None:
            bdf2 = bdf[keep_cols].copy()
            if rename_map is not None:
                bdf2 = bdf2.rename(columns=rename_map)

            # Check whether bbox values look like pixel coords or normalised
            max_val = max(bdf2["xmax"].max(), bdf2["ymax"].max())
            if max_val <= 2.0:
                print(f"BoostNet: bbox CSV max coord is {max_val:.3f}, "
                      "looks like normalised [0,1]. Skipping CSV "
                      "(would need image dims to denormalise per row); "
                      "using content detection instead.")
            else:
                df = df.merge(bdf2, on="image_uuid", how="left")
                n_with_bbox = df["xmin"].notna().sum()
                print(f"BoostNet: joined {n_with_bbox} pixel-coord bboxes")
                use_csv_bboxes = True

    if not use_csv_bboxes:
        print("BoostNet: using detect_content_bbox() for all images "
              "(matches the cropped extraction pipeline).")

    sample_idx = rng.choice(len(df), size=min(args.n_samples, len(df)),
                              replace=False)
    sample_df = df.iloc[sample_idx].copy()

    results = []
    skipped = 0
    for _, row in sample_df.iterrows():
        uuid = row["image_uuid"]
        path = args.boostnet_img_dir / f"{uuid}.jpg"
        if not path.is_file():
            skipped += 1
            continue
        try:
            img = load_boostnet_image(path)
            if use_csv_bboxes and "xmin" in row.index \
                    and pd.notna(row.get("xmin", np.nan)):
                bbox = (int(row["ymin"]), int(row["xmin"]),
                        int(row["ymax"]), int(row["xmax"]))
                bbox_source = "csv"
            else:
                bbox = detect_content_bbox(img)
                bbox_source = "detected"
            res = process_image(img, bbox=bbox)
            res["corpus"] = "BoostNet"
            res["image_id"] = uuid
            res["bbox_source"] = bbox_source
            results.append(res)
        except Exception as e:
            print(f"  Skipping {uuid}: {e}", file=sys.stderr)
            skipped += 1

    print(f"BoostNet: processed {len(results)} samples ({skipped} skipped)")
    return results


def gather_vindr_samples(args, rng: np.random.Generator) -> List[Dict]:
    """Sample VinDr DICOMs and process them through all stages."""
    dicom_paths = sorted(
        list(args.vindr_dicom_dir.rglob("*.dicom"))
        + list(args.vindr_dicom_dir.rglob("*.dcm"))
    )
    print(f"VinDr: {len(dicom_paths)} DICOMs found in {args.vindr_dicom_dir}")

    if len(dicom_paths) == 0:
        return []

    sample_idx = rng.choice(len(dicom_paths),
                              size=min(args.n_samples, len(dicom_paths)),
                              replace=False)
    sample_paths = [dicom_paths[i] for i in sample_idx]

    results = []
    skipped = 0
    for path in sample_paths:
        try:
            img = load_vindr_dicom(path)
            res = process_image(img)   # content detection for all VinDr
            res["corpus"] = "VinDr-SpineXR"
            res["image_id"] = path.name
            res["bbox_source"] = "detected"
            results.append(res)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}", file=sys.stderr)
            skipped += 1

    print(f"VinDr: processed {len(results)} samples ({skipped} skipped)")
    return results


def summarise(results: List[Dict]) -> pd.DataFrame:
    """Aggregate per-image results into a summary DataFrame."""
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items()
                     if not isinstance(v, np.ndarray)})
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    """Print a per-corpus summary of padding fractions at each stage."""
    print()
    print("=" * 78)
    print("Padding fraction summary (mean ± std across sampled images)")
    print(f"Near-zero threshold: pixels ≤ {NEAR_ZERO_THRESHOLD}")
    print("=" * 78)
    print()

    for corpus, sub in df.groupby("corpus"):
        n = len(sub)
        print(f"{corpus} (n={n} samples)")
        print(f"  Input image aspect (H/W):     mean={sub['input_aspect_hw'].mean():.2f}  "
              f"min={sub['input_aspect_hw'].min():.2f}  max={sub['input_aspect_hw'].max():.2f}")
        print(f"  Content bbox aspect (H/W):    mean={sub['bbox_aspect_hw'].mean():.2f}  "
              f"min={sub['bbox_aspect_hw'].min():.2f}  max={sub['bbox_aspect_hw'].max():.2f}")
        print()
        print(f"  Stage A (original 512x512):   "
              f"mean={sub['stage_a_near_zero_frac'].mean()*100:5.1f}%  "
              f"std={sub['stage_a_near_zero_frac'].std()*100:4.1f}%")
        print(f"  Stage B (just cropped):       "
              f"mean={sub['stage_b_near_zero_frac'].mean()*100:5.1f}%  "
              f"std={sub['stage_b_near_zero_frac'].std()*100:4.1f}%")
        print(f"  Stage C (padded to 1:1.5):    "
              f"mean={sub['stage_c_near_zero_frac'].mean()*100:5.1f}%  "
              f"std={sub['stage_c_near_zero_frac'].std()*100:4.1f}%")
        print(f"  Stage D (final 512x512):      "
              f"mean={sub['stage_d_near_zero_frac'].mean()*100:5.1f}%  "
              f"std={sub['stage_d_near_zero_frac'].std()*100:4.1f}%")
        print()

    # Compare BoostNet vs VinDr at each stage
    print("Inter-corpus padding-fraction gaps:")
    print("  Stage           BoostNet   VinDr     Gap")
    for stage in ["stage_a", "stage_b", "stage_c", "stage_d"]:
        bn = df[df.corpus == "BoostNet"][f"{stage}_near_zero_frac"].mean() * 100
        vd = df[df.corpus == "VinDr-SpineXR"][f"{stage}_near_zero_frac"].mean() * 100
        gap = bn - vd
        print(f"  {stage:12s}    {bn:5.1f}%   {vd:5.1f}%   {gap:+5.1f}pp")
    print()


def make_figure(results: List[Dict], out_path: Path,
                n_examples_per_corpus: int = 3) -> None:
    """Visualisation: stage A (original) vs stage D (cropped) for a few examples."""
    import matplotlib.pyplot as plt

    boost = [r for r in results if r["corpus"] == "BoostNet"][:n_examples_per_corpus]
    vindr = [r for r in results if r["corpus"] == "VinDr-SpineXR"][:n_examples_per_corpus]

    n_examples = len(boost) + len(vindr)
    if n_examples == 0:
        print("No examples to plot, skipping figure.")
        return

    fig, axes = plt.subplots(n_examples, 2, figsize=(8, 3.0 * n_examples),
                              squeeze=False)

    for i, r in enumerate(boost + vindr):
        ax_a = axes[i, 0]
        ax_d = axes[i, 1]

        ax_a.imshow(r["stage_a_array"], cmap="gray", vmin=0, vmax=1)
        ax_a.set_title(
            f"{r['corpus']}: Stage A (original)\n"
            f"padding = {r['stage_a_near_zero_frac']*100:.1f}%",
            fontsize=9,
        )
        ax_a.axis("off")

        ax_d.imshow(r["stage_d_array"], cmap="gray", vmin=0, vmax=1)
        ax_d.set_title(
            f"{r['corpus']}: Stage D (final cropped)\n"
            f"padding = {r['stage_d_near_zero_frac']*100:.1f}%",
            fontsize=9,
        )
        ax_d.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualisation to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boostnet-csv", type=Path,
                    default=Path("medvae/data/boostnet_with_angles.csv"))
    p.add_argument("--boostnet-img-dir", type=Path,
                    default=Path("medvae/data/boostnet"))
    p.add_argument("--boostnet-bbox-csv", type=Path,
                    default=None,
                    help="Optional bbox CSV. If absent, content detection is "
                         "used (this matches what extract_latents_boostnet_cropped.py "
                         "does in practice: its bbox CSV code path requires "
                         "columns named xmin/ymin/xmax/ymax, which "
                         "medvae/data/boostnet.csv does not have, so it falls "
                         "back to detection anyway).")
    p.add_argument("--vindr-dicom-dir", type=Path,
                    default=Path.home() / "MedVAE" / "vindr-spinexr")
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--out-csv", type=Path, default=Path("verify_padding_fractions.csv"))
    p.add_argument("--out-fig", type=Path, default=Path("verify_padding_fractions.png"))
    p.add_argument("--skip-vindr", action="store_true",
                    help="Skip VinDr (e.g., if DICOMs aren't available locally).")
    p.add_argument("--skip-boostnet", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(SAMPLE_RNG_SEED)

    all_results: List[Dict] = []

    if not args.skip_boostnet:
        if not args.boostnet_csv.is_file():
            print(f"Error: BoostNet CSV not found at {args.boostnet_csv}",
                  file=sys.stderr)
            return 1
        all_results.extend(gather_boostnet_samples(args, rng))

    if not args.skip_vindr:
        if not args.vindr_dicom_dir.is_dir():
            print(f"WARNING: VinDr directory not found at {args.vindr_dicom_dir}. VinDr samples are skipped, so inter-corpus gaps cannot be computed.",
                  file=sys.stderr)
        else:
            all_results.extend(gather_vindr_samples(args, rng))

    if not all_results:
        print("No samples processed.", file=sys.stderr)
        return 1

    df = summarise(all_results)
    print_summary(df)

    df.to_csv(args.out_csv, index=False)
    print(f"Saved per-image CSV to {args.out_csv}")

    make_figure(all_results, args.out_fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
