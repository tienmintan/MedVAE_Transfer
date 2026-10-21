"""
landmark_pixel_audit.py: landmark error in pixels (and mm where spacing is known), plus a quality audit of the landmark detector.

Reports:
  (1) pixel error instead of percent/normalised error;
  (2) landmark-detector quality (worst cases, per-region error, coordinate
      correlation, mean-shape baseline).

Works on any saved prediction file in the shared format (uuids, pred_landmarks,
gt_landmarks, all (N,136) ordering A normalised [0,1]); it can therefore audit
both the direct-regressor preds (preds_{config}.npz) and the heatmap-detector
preds (preds_heatmap_{config}.npz).

Image size is auto-detected from --img-dir:
  * WxH per image via PIL (jpg/png/...);
  * mm pixel-spacing via pydicom if DICOMs are present;
  * if no img-dir, falls back to normalised-only error with a printed warning.
Landmarks are normalised to the original image, not a padded canvas, so the
original-pixel error is (Δx*W, Δy*H) per point, with no padding or bbox
involved.

Inputs:  --preds (.npz in the shared format); optionally --img-dir.
Outputs: <out-dir>/<preds stem>_per_image.csv, _per_landmark.csv and, with
         --img-dir, _overlays.png; the audit printed to stdout.

Usage:
  python landmark_pixel_audit.py --preds ./outputs/frontier/preds_medvae_4_1_2d.npz \
      --img-dir ./medvae/data/boostnet --out-dir ./outputs/landmark_audit
  python landmark_pixel_audit.py --smoke      # synthetic-data self-test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_LM = 68
N_VERT = 17
# 17 vertebrae T1..T12, L1..L5 -> coarse regions (by vertebra index, vmajor)
REGIONS = {"upper-thoracic (T1-5)": range(0, 5),
           "mid-thoracic (T6-9)": range(5, 9),
           "lower-thoracic (T10-12)": range(9, 12),
           "lumbar (L1-5)": range(12, 17)}


# Image size / spacing auto-detection
def find_image(img_dir, uuid):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"):
        p = Path(img_dir) / f"{uuid}{ext}"
        if p.exists():
            return p
    return None


def image_geometry(path):
    """Return (W, H, sx_mm, sy_mm); spacing is None if unknown."""
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            W, H = int(ds.Columns), int(ds.Rows)
            sp = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
            if sp is not None:
                return W, H, float(sp[1]), float(sp[0])
            return W, H, None, None
        except Exception:
            return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.width, im.height, None, None
    except Exception:
        return None


# Error helpers
def per_landmark_px(pred, gt, W, H):
    """(136,),(136,) normalised -> (68,) per-landmark pixel error."""
    dx = (pred[:N_LM] - gt[:N_LM]) * W
    dy = (pred[N_LM:] - gt[N_LM:]) * H
    return np.sqrt(dx ** 2 + dy ** 2)


def spine_extent_px(gt, W, H):
    """Vertical span of the ground-truth spine (T1 -> L5) in pixels.

    Normalising by spine size rather than image size makes patients comparable
    regardless of how much of the frame the spine occupies (spines differ in
    height)."""
    ys = gt[N_LM:] * H
    return float(ys.max() - ys.min())


def spine_length_px(gt, W, H):
    """Path length along the vertebra centroids (T1->L5) in pixels; a
    curvature-aware alternative to the vertical span."""
    xs = gt[:N_LM] * W
    ys = gt[N_LM:] * H
    c = np.stack([xs.reshape(N_VERT, 4).mean(1), ys.reshape(N_VERT, 4).mean(1)], 1)
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def per_landmark_norm(pred, gt):
    dx = pred[:N_LM] - gt[:N_LM]; dy = pred[N_LM:] - gt[N_LM:]
    return np.sqrt(dx ** 2 + dy ** 2)


def boot_ci(v, seed=0, nb=10000):
    rng = np.random.default_rng(seed); v = np.asarray(v); n = len(v)
    b = np.array([v[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# Audit
def audit(preds_path, img_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(preds_path, allow_pickle=True)
    pred, gt, uuids = d["pred_landmarks"], d["gt_landmarks"], d["uuids"]
    N = len(pred)
    tag = Path(preds_path).stem

    # Geometry per image
    geoms, have_px, have_mm = [], 0, 0
    for u in uuids:
        g = None
        if img_dir:
            p = find_image(img_dir, str(u))
            if p is not None:
                g = image_geometry(p)
        geoms.append(g)
        if g:
            have_px += 1
            if g[2] is not None:
                have_mm += 1
    print(f"=== {tag}  (N={N}) ===")
    print(f"image geometry found: {have_px}/{N} sized"
          + (f", {have_mm}/{N} with mm spacing" if have_mm else " (no mm spacing)"))
    if have_px == 0:
        print(f"  WARNING: no image sizes available (0/{N} images sized), so only normalised error is reported. "
              "Pixel error requires --img-dir.\n")

    # Per-image errors
    rows, pl_px_all, pl_norm_all = [], [], []
    px_img, mm_img, norm_img, spine_img = [], [], [], []
    for i in range(N):
        pln = per_landmark_norm(pred[i], gt[i])
        pl_norm_all.append(pln); norm_img.append(pln.mean())
        g = geoms[i]
        spine_pct = np.nan
        if g:
            W, H, sx, sy = g
            plp = per_landmark_px(pred[i], gt[i], W, H)
            pl_px_all.append(plp); px_img.append(plp.mean())
            ext = spine_extent_px(gt[i], W, H)
            if ext > 1:
                spine_pct = 100.0 * plp.mean() / ext
                spine_img.append(spine_pct)
            if sx is not None:
                dxmm = (pred[i][:N_LM] - gt[i][:N_LM]) * W * sx
                dymm = (pred[i][N_LM:] - gt[i][N_LM:]) * H * sy
                mm_img.append(np.sqrt(dxmm ** 2 + dymm ** 2).mean())
        rows.append({"uuid": str(uuids[i]), "err_norm": norm_img[-1],
                     "err_px": (px_img[-1] if g else np.nan),
                     "err_pct_spine": spine_pct})
    norm_img = np.array(norm_img)

    # Summary
    m, lo, hi = boot_ci(norm_img)
    print(f"\nlandmark error (mean over 68 pts, per image):")
    print(f"  normalised (image) : {m:.4f}  CI[{lo:.4f},{hi:.4f}]")
    if px_img:
        m, lo, hi = boot_ci(np.array(px_img))
        print(f"  pixels             : {m:.1f}px  CI[{lo:.1f},{hi:.1f}]  "
              f"(median {np.median(px_img):.1f}, 90th pct {np.percentile(px_img,90):.1f}, "
              f"worst {np.max(px_img):.1f})")
    if spine_img:
        m, lo, hi = boot_ci(np.array(spine_img))
        print(f"  % of spine extent  : {m:.2f}%  CI[{lo:.2f},{hi:.2f}]  "
              f"(median {np.median(spine_img):.2f}%)   <- patient-comparable")
    if mm_img:
        m, lo, hi = boot_ci(np.array(mm_img))
        print(f"  millimetres        : {m:.2f}mm CI[{lo:.2f},{hi:.2f}]")

    # Worst-case images
    metric_img = np.array(px_img) if px_img else norm_img
    unit = "px" if px_img else "norm"
    order = np.argsort(metric_img)[::-1]
    fmt = "{:.1f}" if px_img else "{:.4f}"
    print(f"\nworst 5 images ({unit}):")
    for j in order[:5]:
        print(f"  {str(uuids[j]):<38} " + fmt.format(metric_img[j]))

    # Per-region
    pl = np.array(pl_px_all) if pl_px_all else np.array(pl_norm_all)   # (N,68)
    ru = "px" if pl_px_all else "norm"
    print(f"\nper-region error ({ru}):")
    for name, verts in REGIONS.items():
        idx = [v * 4 + c for v in verts for c in range(4)]
        print(f"  {name:<24} {pl[:, idx].mean():.2f}")

    # Coordinate correlation (predicted vs true)
    px_all = pred[:, :N_LM].ravel(); gx_all = gt[:, :N_LM].ravel()
    py_all = pred[:, N_LM:].ravel(); gy_all = gt[:, N_LM:].ravel()
    rx = np.corrcoef(px_all, gx_all)[0, 1]; ry = np.corrcoef(py_all, gy_all)[0, 1]
    print(f"\npredicted-vs-true coordinate correlation: x r={rx:.3f}, y r={ry:.3f}")

    # Mean-shape baseline (context)
    base = gt.mean(0, keepdims=True)
    base_norm = np.array([per_landmark_norm(base[0], gt[i]).mean() for i in range(N)]).mean()
    print(f"mean-shape baseline (norm): {base_norm:.4f}  -> detector is "
          f"{100*(1-norm_img.mean()/base_norm):.0f}% better")

    pd.DataFrame(rows).to_csv(out_dir / f"{tag}_per_image.csv", index=False)
    per_lm = pd.DataFrame({"landmark": range(N_LM),
                           "vertebra": [k // 4 for k in range(N_LM)],
                           "corner": [k % 4 for k in range(N_LM)],
                           f"mean_err_{ru}": pl.mean(0)})
    per_lm.to_csv(out_dir / f"{tag}_per_landmark.csv", index=False)

    if img_dir and have_px:
        overlay_figure(preds_path, img_dir, geoms, metric_img, uuids, pred, gt,
                       out_dir / f"{tag}_overlays.png")
        print(f"\nWrote overlays + CSVs to {out_dir}")
    else:
        print(f"\nWrote CSVs to {out_dir} (overlays need --img-dir)")


def overlay_figure(preds_path, img_dir, geoms, metric, uuids, pred, gt, path):
    """Best / median / worst example: GT (green) vs predicted (red) on the image."""
    from PIL import Image
    order = np.argsort(metric)
    picks = [("best", order[0]), ("median", order[len(order) // 2]), ("worst", order[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    for ax, (label, i) in zip(axes, picks):
        p = find_image(img_dir, str(uuids[i]))
        g = geoms[i]
        if p is None or g is None:
            ax.axis("off"); continue
        W, H = g[0], g[1]
        try:
            im = Image.open(p).convert("L")
        except Exception:
            ax.axis("off"); continue
        ax.imshow(im, cmap="gray")
        ax.scatter(gt[i][:N_LM] * W, gt[i][N_LM:] * H, s=14, c="lime",
                   label="ground truth", edgecolors="black", linewidths=0.3)
        ax.scatter(pred[i][:N_LM] * W, pred[i][N_LM:] * H, s=14, c="red",
                   label="predicted", edgecolors="black", linewidths=0.3)
        ax.set_title(f"{label}: {metric[i]:.1f}px", fontsize=10)
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Landmark detector: predicted vs ground truth", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


# Synthetic-data self-test
def smoke():
    rng = np.random.default_rng(0); N = 20
    gt = np.clip(0.5 + 0.15 * rng.standard_normal((N, 136)), 0.02, 0.98).astype(np.float32)
    pred = gt + rng.normal(0, 0.02, gt.shape).astype(np.float32)     # ~2% error
    np.savez("/tmp/preds_smoke.npz", pred_landmarks=pred, gt_landmarks=gt,
             uuids=np.array([f"img{i}" for i in range(N)]))
    audit(Path("/tmp/preds_smoke.npz"), None, Path("/tmp/lm_audit"))
    print("\nSelf-test: PASS (no img-dir -> normalised-only path)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, help="preds_*.npz to audit")
    ap.add_argument("--img-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/landmark_audit"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.preds:
        audit(args.preds, args.img_dir, args.out_dir)
    else:
        raise SystemExit("--preds is required (or --smoke for the synthetic-data self-test)")


if __name__ == "__main__":
    main()
