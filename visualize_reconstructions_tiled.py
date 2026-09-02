"""
visualize_reconstructions_tiled.py

Phase 2 qualitative comparison: side-by-side figures of full-image
reconstructions from three MedVAE checkpoints, with zoomed crops centred on
a vertebra-sized region to make high-frequency differences visible.

For each selected image we produce two rows:
    Row 1: full image — original, pretrained, aspect-ft, patch-ft
    Row 2: zoomed ~256x256 crop centred at the bbox centre

All panels (full images and zoom crops, every row, every column) share the
SAME intensity windowing — vmin/vmax computed globally across every pixel
we display. This is the standard convention in medical-image viewing
(window/level): radiologists fix the intensity bounds deliberately, never
letting the display auto-scale per image, so visual differences reflect
real differences in the data rather than display artefacts. A colorbar on
the right shows the active scale.

Usage:
    python visualize_reconstructions_tiled.py \\
        --csv         $PROJECT_DIR/medvae/data/boostnet.csv \\
        --img-dir     $PROJECT_DIR/medvae/data/boostnet \\
        --aspect-ckpt /lambda/.../boostnet_finetuning_aspect/.../checkpoints/best.pt \\
        --patch-ckpt  /lambda/.../boostnet_finetuning_patch/.../checkpoints/best.pt \\
        --n-images 6 \\
        --window-mode percentile \\
        --out         ./recon_tiled_zoomed.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from evaluate_reconstruction_tiled import (
    tile_and_stitch,
    load_pretrained_medvae,
    load_finetuned_medvae,
)


@torch.no_grad()
def reconstruct(model, image, tile_size, stride, device, dtype):
    """Wrapper that returns a numpy array in [0, 1]."""
    out = tile_and_stitch(model, image, tile_size, stride, device, dtype)
    return out.squeeze(0).cpu().float().numpy()


def compute_global_bounds(pixels_iter, mode: str, low_pct: float, high_pct: float):
    """Compute (vmin, vmax) for windowing across all displayed pixels.

    Three modes:
      - 'percentile': use low_pct/high_pct percentiles (default 1/99).
        Robust to bright outliers (e.g. text overlays) that would otherwise
        wash out everything else.
      - 'minmax': use the raw global min and max. Honest but sensitive to
        outliers — one bright image compresses everyone else's dynamic range.
      - 'fixed': always return (0, 1). Useful when the data is already
        clipped to that range and you don't want any clipping.

    pixels_iter is an iterable of numpy arrays; we sample a flattened
    concatenation to avoid loading all pixels at once for huge image sets.
    """
    if mode == "fixed":
        return 0.0, 1.0

    # Cap memory by sampling at most 5M pixels across all images
    chunks = []
    budget = 5_000_000
    for arr in pixels_iter:
        flat = arr.ravel()
        if len(flat) > budget:
            idx = np.random.default_rng(0).choice(len(flat), budget, replace=False)
            chunks.append(flat[idx])
            break
        chunks.append(flat)
        budget -= len(flat)
    pool = np.concatenate(chunks)

    if mode == "minmax":
        return float(pool.min()), float(pool.max())
    elif mode == "percentile":
        return (float(np.percentile(pool, low_pct)),
                float(np.percentile(pool, high_pct)))
    else:
        raise ValueError(f"Unknown window-mode: {mode}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--aspect-ckpt", required=True, type=str)
    p.add_argument("--patch-ckpt", required=True, type=str)
    p.add_argument("--n-images", type=int, default=6,
                   help="How many test images to visualize.")
    p.add_argument("--tile-size", type=int, default=384)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--zoom-size", type=int, default=256,
                   help="Side length of zoomed crop in original pixels.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for image selection (for reproducibility).")
    p.add_argument("--window-mode", choices=["percentile", "minmax", "fixed"],
                   default="percentile",
                   help="How to pick the global vmin/vmax for windowing. "
                        "'percentile' (default) is robust to bright outliers; "
                        "'minmax' uses raw global bounds; 'fixed' uses [0, 1].")
    p.add_argument("--window-low",  type=float, default=1.0,
                   help="Lower percentile for 'percentile' mode (default 1).")
    p.add_argument("--window-high", type=float, default=99.0,
                   help="Upper percentile for 'percentile' mode (default 99).")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    selected_idx = rng.choice(len(test_df), size=args.n_images, replace=False)
    selected = test_df.iloc[selected_idx].reset_index(drop=True)
    print(f"Selected {len(selected)} test images.")

    print("Loading pretrained MedVAE...")
    m_pre = load_pretrained_medvae(device, dtype)
    print("Loading aspect-finetuned MedVAE...")
    m_asp = load_finetuned_medvae(args.aspect_ckpt, device, dtype)
    print("Loading patch-finetuned MedVAE...")
    m_pat = load_finetuned_medvae(args.patch_ckpt, device, dtype)

    # =========================================================================
    # PASS 1: compute everything, store in memory
    # =========================================================================
    print("\nPass 1: computing reconstructions and zoom regions...")
    records = []
    for i, row in selected.iterrows():
        img_path = args.img_dir / f"{row['image_uuid']}.jpg"
        pil = Image.open(img_path).convert("L")
        W, H = pil.size
        orig = torch.from_numpy(np.array(pil)).float().unsqueeze(0) / 255.0
        orig = orig.to(device)

        print(f"  [{i+1}/{len(selected)}] {row['image_uuid']} ({W}x{H})")
        recon_pre = reconstruct(m_pre, orig, args.tile_size, args.stride, device, dtype)
        recon_asp = reconstruct(m_asp, orig, args.tile_size, args.stride, device, dtype)
        recon_pat = reconstruct(m_pat, orig, args.tile_size, args.stride, device, dtype)
        orig_np = orig.squeeze(0).cpu().float().numpy()

        cx = int(round((row["bbox_x0"] + row["bbox_x1"]) / 2 * W))
        cy = int(round((row["bbox_y0"] + row["bbox_y1"]) / 2 * H))
        z = args.zoom_size // 2
        x0 = max(0, cx - z); x1 = min(W, cx + z)
        y0 = max(0, cy - z); y1 = min(H, cy + z)

        records.append({
            "uuid":  row["image_uuid"],
            "full":  [orig_np, recon_pre, recon_asp, recon_pat],
            "zoom":  [im[y0:y1, x0:x1] for im in [orig_np, recon_pre, recon_asp, recon_pat]],
            "zoom_box": (x0, y0, x1 - x0, y1 - y0),
        })

    # =========================================================================
    # PASS 2: compute global intensity bounds across everything
    # =========================================================================
    def iter_all_pixels():
        for r in records:
            for im in r["full"]:
                yield im
            for im in r["zoom"]:
                yield im

    vmin, vmax = compute_global_bounds(
        iter_all_pixels(), args.window_mode,
        args.window_low, args.window_high)
    print(f"\nGlobal intensity window ({args.window_mode}): "
          f"[{vmin:.4f}, {vmax:.4f}]")

    # =========================================================================
    # PASS 3: render with fixed bounds + colorbar
    # =========================================================================
    fig, axes = plt.subplots(
        nrows=args.n_images * 2, ncols=4,
        figsize=(16, 4 * args.n_images),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    col_titles = ["Original", "Pretrained", "Aspect-finetuned", "Patch-finetuned"]
    last_im_handle = None

    for i, r in enumerate(records):
        x0, y0, w, h = r["zoom_box"]
        for j in range(4):
            # Full image row
            ax = axes[i * 2, j]
            mappable = ax.imshow(r["full"][j], cmap="gray",
                                 vmin=vmin, vmax=vmax)
            ax.axis("off")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=14)
            if j == 0:
                ax.add_patch(Rectangle(
                    (x0, y0), w, h,
                    edgecolor="red", facecolor="none", lw=1.5,
                ))
            # Zoom row
            ax = axes[i * 2 + 1, j]
            mappable = ax.imshow(r["zoom"][j], cmap="gray",
                                 vmin=vmin, vmax=vmax)
            ax.axis("off")
            last_im_handle = mappable

        # Image-UUID label at the far left of the row, via figure-level text
        # (axes-level ylabels don't render when axis is off)
        row_top_ax = axes[i * 2, 0]
        bbox = row_top_ax.get_position()
        fig.text(bbox.x0 - 0.005, (bbox.y0 + bbox.y1) / 2,
                 r["uuid"][:25],
                 ha="right", va="center", fontsize=8,
                 family="monospace")

    # Single colorbar on the right of the entire figure
    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.925, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_im_handle, cax=cbar_ax)
    cbar.set_label("Normalised pixel intensity", rotation=270,
                   labelpad=15, fontsize=10)

    # Caption explaining the windowing — readable evidence the comparison is fair
    if args.window_mode == "percentile":
        caption = (f"Display windowing: {args.window_low:.0f}th–"
                   f"{args.window_high:.0f}th percentile of all panel pixels "
                   f"(vmin={vmin:.3f}, vmax={vmax:.3f}). Same scale for "
                   f"every panel.")
    elif args.window_mode == "minmax":
        caption = (f"Display windowing: global min/max of all panel pixels "
                   f"(vmin={vmin:.3f}, vmax={vmax:.3f}). Same scale for "
                   f"every panel.")
    else:
        caption = (f"Display windowing: fixed [0, 1]. Same scale for "
                   f"every panel.")
    fig.text(0.5, 0.02, caption, ha="center", va="bottom",
             fontsize=9, style="italic")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
