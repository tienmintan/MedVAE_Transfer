"""
evaluate_whole_image.py

Test whether the patch-trained MedVAE encoder generalizes to whole-image
inference, despite never having seen whole images during training.

The patch model was trained on 384x384 native-resolution crops. At inference,
we encode + decode entire images in a single forward pass — no tiling, no
stitching. This tests whether the convolutional features generalize to
larger spatial extents (CNNs would, by translation equivariance) and
whether the bottleneck self-attention layer can handle the much larger
spatial grid it never saw at training time.

Two things make this a real test, not a foregone conclusion:
  1. MedVAE has bottleneck self-attention (not fully convolutional). The
     attention matrix at 384 input is 96x96 = 9k positions. At 768x2304
     input it's 192x576 = 110k positions, ~12x more. Attention weights
     learned on small grids may not transfer cleanly.
  2. Per-image normalization replaces per-tile normalization. Each image
     gets its own min-max applied once globally, which is a different
     normalization scheme than training saw.

The script evaluates all three models (pretrained / aspect-ft / patch-ft)
through the same whole-image pipeline so the comparison is apples-to-apples,
and produces metrics comparable to the existing tile-and-stitch numbers in
tiled_metrics_per_image.csv.

Usage:
    python evaluate_whole_image.py \\
        --csv         ./medvae/data/boostnet.csv \\
        --img-dir     ./medvae/data/boostnet \\
        --aspect-ckpt $ASPECT_CKPT \\
        --patch-ckpt  $PATCH_CKPT \\
        --resolution  768x2304 \\
        --out-csv     ./whole_image_metrics.csv \\
        --out-vis     ./whole_image_samples.png \\
        --tiled-csv   ./tiled_metrics_per_image.csv  # optional: for comparison
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

try:
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
except ImportError:
    from torchmetrics import PeakSignalNoiseRatio
    from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM

from evaluate_reconstruction_tiled import (
    load_pretrained_medvae, load_finetuned_medvae,
)


def parse_resolution(s: str) -> tuple:
    """'768x2304' -> (768, 2304). Order is (W, H) for PIL resize compatibility."""
    m = re.match(r"^(\d+)x(\d+)$", s.strip())
    if not m:
        raise ValueError(f"--resolution should look like '768x2304', got {s!r}")
    return int(m.group(1)), int(m.group(2))


@torch.no_grad()
def encode_decode_whole(
    model,
    image_tensor: torch.Tensor,    # (1, H, W) in [0, 1]
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One forward pass through the autoencoder. No tiling.

    Returns reconstructed image (1, H, W) in [0, 1].
    """
    # Per-image min-max normalize to [-1, 1] (matches the per-patch
    # normalization used at training, just applied at whole-image scale)
    imin, imax = image_tensor.min(), image_tensor.max()
    if imax > imin:
        norm = (image_tensor - imin) / (imax - imin)
    else:
        norm = torch.zeros_like(image_tensor)
    norm = (norm - 0.5) / 0.5     # [0, 1] -> [-1, 1]

    inp = norm.unsqueeze(0).to(device=device, dtype=dtype)
    with torch.autocast(device_type=device.type, dtype=dtype):
        recon, _, _ = model(inp)
    recon = recon.squeeze(0).float()             # (1, H, W) in [-1, 1] (ish)

    # De-normalize back to original intensity range
    recon = (recon * 0.5) + 0.5
    recon = recon.clamp(0.0, 1.0)
    if imax > imin:
        recon = recon * (imax - imin) + imin
    else:
        recon = torch.full_like(recon, float(imin))
    return recon


def evaluate_model(
    name: str, model, test_df: pd.DataFrame, img_dir: Path,
    target_size: tuple, device: torch.device, dtype: torch.dtype,
) -> tuple:
    """Returns (DataFrame of per-image metrics, dict of {uuid: (orig, recon)} for a few images)."""
    W_tgt, H_tgt = target_size

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    msssim_fn = MSSSIM(data_range=1.0).to(device)

    rows = []
    sample_images = {}
    fails = 0

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc=name):
        img_path = img_dir / f"{row['image_uuid']}.jpg"
        pil = Image.open(img_path).convert("L").resize((W_tgt, H_tgt), Image.BICUBIC)
        orig = torch.from_numpy(np.array(pil)).float().unsqueeze(0) / 255.0
        orig = orig.to(device)

        try:
            recon = encode_decode_whole(model, orig, device, dtype)
        except torch.cuda.OutOfMemoryError:
            fails += 1
            torch.cuda.empty_cache()
            rows.append({
                "model": name, "image_uuid": row["image_uuid"],
                "psnr_unmasked": float("nan"), "psnr_masked": float("nan"),
                "msssim_unmasked": float("nan"), "msssim_masked": float("nan"),
                "oom": True,
            })
            continue

        # Bbox in pixel coords at the target resolution
        bx0 = int(round(row["bbox_x0"] * W_tgt))
        by0 = int(round(row["bbox_y0"] * H_tgt))
        bx1 = int(round(row["bbox_x1"] * W_tgt))
        by1 = int(round(row["bbox_y1"] * H_tgt))
        bx1 = max(bx1, bx0 + 1);   by1 = max(by1, by0 + 1)

        bbox_h, bbox_w = by1 - by0, bx1 - bx0
        if bbox_h < 161 or bbox_w < 161:
            o_m, r_m = orig, recon
        else:
            o_m = orig [:, by0:by1, bx0:bx1]
            r_m = recon[:, by0:by1, bx0:bx1]

        psnr_fn.reset();   msssim_fn.reset()
        psnr_unmasked = psnr_fn(recon.unsqueeze(0), orig.unsqueeze(0)).item()
        psnr_fn.reset()
        psnr_masked   = psnr_fn(r_m.unsqueeze(0),  o_m.unsqueeze(0)).item()
        msssim_fn.reset()
        msssim_unmasked = msssim_fn(recon.unsqueeze(0), orig.unsqueeze(0)).item()
        msssim_fn.reset()
        msssim_masked   = msssim_fn(r_m.unsqueeze(0),  o_m.unsqueeze(0)).item()

        rows.append({
            "model": name, "image_uuid": row["image_uuid"],
            "psnr_unmasked": psnr_unmasked, "psnr_masked": psnr_masked,
            "msssim_unmasked": msssim_unmasked, "msssim_masked": msssim_masked,
            "oom": False,
        })

        # Save a few images for qualitative inspection
        if i < 4:
            sample_images[row["image_uuid"]] = (
                orig.squeeze(0).cpu().float().numpy(),
                recon.squeeze(0).cpu().float().numpy(),
            )

    if fails:
        print(f"  {name}: {fails}/{len(test_df)} images OOM'd")
    return pd.DataFrame(rows), sample_images


def summarise(df: pd.DataFrame, tiled_csv: Path = None) -> None:
    print("\n" + "=" * 78)
    print("WHOLE-IMAGE RECONSTRUCTION METRICS (n test images per model shown)")
    print("=" * 78)

    n_oom = df.groupby("model")["oom"].sum()
    valid = df[~df["oom"]]
    metrics = ["psnr_unmasked", "psnr_masked", "msssim_unmasked", "msssim_masked"]
    summary = valid.groupby("model")[metrics].agg(["mean", "std"])
    print(summary.round(4).to_string())
    for m, n in n_oom.items():
        if n > 0:
            print(f"  {m}: {n} images OOM'd, excluded from summary")

    if tiled_csv and Path(tiled_csv).exists():
        print("\n" + "=" * 78)
        print("WHOLE-IMAGE vs TILE-AND-STITCH (paired, masked metrics)")
        print("=" * 78)
        tiled = pd.read_csv(tiled_csv)
        for model_name in valid["model"].unique():
            w = valid[valid["model"] == model_name]
            t = tiled[tiled["model"] == model_name]
            if len(t) == 0:
                print(f"\n  {model_name}: no tile-and-stitch baseline in {tiled_csv}")
                continue
            merged = w.merge(t, on="image_uuid", suffixes=("_whole", "_tiled"))
            d_psnr = merged["psnr_masked_whole"] - merged["psnr_masked_tiled"]
            d_msssim = merged["msssim_masked_whole"] - merged["msssim_masked_tiled"]
            wins_psnr = (d_psnr > 0).sum()
            wins_msssim = (d_msssim > 0).sum()
            n = len(merged)
            print(f"\n  {model_name}  (whole − tiled):")
            print(f"    PSNR (masked):    Δ = {d_psnr.mean():+.3f} ± {d_psnr.std():.3f} dB  "
                  f"(whole wins {wins_psnr}/{n})")
            print(f"    MS-SSIM (masked): Δ = {d_msssim.mean():+.4f} ± {d_msssim.std():.4f}  "
                  f"(whole wins {wins_msssim}/{n})")


def make_visualization(samples_by_model: dict, out_path: Path, target_size: tuple,
                       window_low: float = 1.0, window_high: float = 99.0) -> None:
    """Side-by-side: original vs each model's whole-image reconstruction,
    for the first 3 test images. All panels share the same intensity windowing.
    """
    # Pick the same 3 UUIDs across all models (use whatever the first model saved)
    first_model = next(iter(samples_by_model))
    uuids = list(samples_by_model[first_model].keys())[:3]
    if not uuids:
        print("No samples to visualize.")
        return

    model_names = list(samples_by_model.keys())
    n_rows, n_cols = len(uuids), 1 + len(model_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    # Compute global windowing across all panels
    all_pixels = []
    for u in uuids:
        for m in model_names:
            if u in samples_by_model[m]:
                orig, recon = samples_by_model[m][u]
                all_pixels.append(orig); all_pixels.append(recon)
    flat = np.concatenate([a.ravel() for a in all_pixels])
    vmin = float(np.percentile(flat, window_low))
    vmax = float(np.percentile(flat, window_high))
    print(f"Visualization windowing: [{vmin:.3f}, {vmax:.3f}]")

    last_handle = None
    for i, u in enumerate(uuids):
        # Original (from any model — they all see the same input)
        orig, _ = samples_by_model[first_model][u]
        last_handle = axes[i, 0].imshow(orig, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=12)
        for j, m in enumerate(model_names, start=1):
            if u not in samples_by_model[m]:
                axes[i, j].axis("off")
                continue
            _, recon = samples_by_model[m][u]
            last_handle = axes[i, j].imshow(recon, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(m, fontsize=12)

    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.925, 0.15, 0.015, 0.7])
    cb = fig.colorbar(last_handle, cax=cbar_ax)
    cb.set_label("Normalised pixel intensity", rotation=270, labelpad=15, fontsize=10)
    fig.text(0.5, 0.02,
             f"Whole-image encoding at {target_size[0]}x{target_size[1]}. "
             f"Display window: {window_low:.0f}–{window_high:.0f}th percentile.",
             ha="center", va="bottom", fontsize=9, style="italic")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--aspect-ckpt", required=True, type=str)
    p.add_argument("--patch-ckpt", required=True, type=str)
    p.add_argument("--resolution", default="768x2304",
                   help="WxH for the whole-image input. Default 768x2304 "
                        "matches the tile-and-stitch latent extraction pipeline.")
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--out-vis", required=True, type=Path)
    p.add_argument("--tiled-csv", type=Path, default=None,
                   help="Optional: tile-and-stitch metrics CSV from "
                        "evaluate_reconstruction_tiled.py, for paired comparison.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    target_size = parse_resolution(args.resolution)
    print(f"Target resolution (WxH): {target_size}")

    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    for col in ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]:
        if col not in test_df.columns:
            raise ValueError(f"Missing bbox column {col!r} — run prepare_spine_bboxes.py")
    print(f"Loaded {len(test_df)} test images.")

    all_results = []
    samples_by_model = {}
    for name, ckpt in [
        ("pretrained", None),
        ("aspect",     args.aspect_ckpt),
        ("patch",      args.patch_ckpt),
    ]:
        print(f"\n[{name}] Loading model...")
        if ckpt is None:
            model = load_pretrained_medvae(device, dtype)
        else:
            model = load_finetuned_medvae(ckpt, device, dtype)

        results, samples = evaluate_model(
            name, model, test_df, args.img_dir, target_size, device, dtype)
        all_results.append(results)
        samples_by_model[name] = samples
        del model
        torch.cuda.empty_cache()

    combined = pd.concat(all_results, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False)
    print(f"\nWrote per-image results to {args.out_csv}")
    summarise(combined, args.tiled_csv)
    make_visualization(samples_by_model, args.out_vis, target_size)


if __name__ == "__main__":
    main()
