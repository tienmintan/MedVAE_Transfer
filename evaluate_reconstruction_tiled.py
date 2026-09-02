"""
evaluate_reconstruction_tiled.py

Phase 2 evaluation: tile-and-stitch reconstruction of full-resolution BoostNet
test images through three MedVAE checkpoints (pretrained, aspect-finetuned,
patch-finetuned), with masked metrics computed over the spine bounding box.

This is the fair comparison for Phase 1's patch-training experiment: all three
models go through the same inference pipeline (slide window, encode/decode each
tile at native resolution, blend back to full image), so the only difference
between conditions is the encoder/decoder weights themselves.

Usage:
    python evaluate_reconstruction_tiled.py \\
        --csv         $PROJECT_DIR/medvae/data/boostnet.csv \\
        --img-dir     $PROJECT_DIR/medvae/data/boostnet \\
        --pretrained-ckpt  (omit; auto-downloads from HF hub) \\
        --aspect-ckpt /lambda/.../boostnet_finetuning_aspect/.../checkpoints/best.pt \\
        --patch-ckpt  /lambda/.../boostnet_finetuning_patch/.../checkpoints/best.pt \\
        --tile-size 384 \\
        --stride 192 \\
        --out-csv     ./tiled_metrics_per_image.csv

Notes
-----
- Per-tile min-max normalisation matches the patch training pipeline. This
  also matches what _PadToSquare + ScaleIntensity did during aspect-preserving
  training, so the comparison is consistent.
- For each tile we record (min, max) and use them to de-normalise the
  reconstruction back to its original intensity range before blending.
  Without this step, tiles with low natural contrast would have their noise
  amplified visibly in the stitched output.
- Masking uses the bbox columns added by prepare_spine_bboxes.py. We compute
  PSNR/MS-SSIM both unmasked (over the whole image) and masked (over the
  spine bbox region only). Unmasked metrics include near-background regions
  outside the spine — they're easier to reconstruct, so the unmasked numbers
  are systematically higher than the masked ones.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm import tqdm

# torchmetrics location varies across versions; tolerate both.
try:
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
except ImportError:
    from torchmetrics import PeakSignalNoiseRatio
    from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM

from medvae.models.autoencoder_kl import AutoencoderKL
from huggingface_hub import hf_hub_download


# ----------------------------------------------------------------------------
# Tile-and-stitch reconstruction
# ----------------------------------------------------------------------------

def make_cosine_window(tile_size: int, device) -> torch.Tensor:
    """2D cosine window: 1.0 at centre, ~0 at edges. For blending overlapping
    tile reconstructions without seams.

    Shape: (tile_size, tile_size). Use as a per-tile weight when accumulating.
    """
    x = torch.linspace(-np.pi / 2, np.pi / 2, tile_size, device=device)
    w_1d = torch.cos(x)               # peak 1.0 at centre, 0 at edges
    w_2d = torch.outer(w_1d, w_1d)    # tile-shaped weight
    # Avoid exact zeros at the corners (division stability later)
    return w_2d.clamp(min=1e-6)


def tile_positions(image_size: int, tile_size: int, stride: int) -> list:
    """Sliding-window top-left positions along one dimension. Always includes
    the last valid position so the image is fully covered."""
    if image_size <= tile_size:
        return [0]
    positions = list(range(0, image_size - tile_size + 1, stride))
    last = image_size - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.no_grad()
def tile_and_stitch(
    model: AutoencoderKL,
    image: torch.Tensor,        # (1, H, W) at native resolution, [0, 1]
    tile_size: int,
    stride: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct a full image by sliding window over tiles and blending.

    The image is expected in [0, 1] (single-channel). For each tile we
    record (min, max), normalise to [-1, 1], run through the model,
    de-normalise the reconstruction, and accumulate with a cosine window.

    Returns: (1, H, W) in [0, 1], same shape as the input.
    """
    _, H, W = image.shape

    # Pad image to a multiple of stride so the last tile lands cleanly.
    # The last-position trick in tile_positions makes this unnecessary for
    # coverage, but having the canvas exactly multiple-of-stride simplifies
    # bookkeeping. We crop back at the end.
    pad_h = (stride - (H - tile_size) % stride) % stride if H > tile_size else 0
    pad_w = (stride - (W - tile_size) % stride) % stride if W > tile_size else 0
    if pad_h > 0 or pad_w > 0:
        # Reflect padding so the model doesn't see hard zero edges
        image_padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h),
                             mode="reflect").squeeze(0)
    else:
        image_padded = image
    _, Hp, Wp = image_padded.shape

    # Buffers
    output  = torch.zeros((1, Hp, Wp), device=device)
    weights = torch.zeros((1, Hp, Wp), device=device)
    window  = make_cosine_window(tile_size, device)  # (T, T)

    ys = tile_positions(Hp, tile_size, stride)
    xs = tile_positions(Wp, tile_size, stride)

    for y in ys:
        for x in xs:
            tile = image_padded[:, y:y + tile_size, x:x + tile_size]  # (1, T, T)

            # Per-tile min-max normalisation, matching patch training
            tmin, tmax = tile.min(), tile.max()
            if tmax > tmin:
                tile_norm = (tile - tmin) / (tmax - tmin)
            else:
                tile_norm = torch.zeros_like(tile)
            tile_norm = (tile_norm - 0.5) / 0.5    # [0, 1] -> [-1, 1]

            # Encode + decode
            inp = tile_norm.unsqueeze(0).to(device=device, dtype=dtype)
            with torch.autocast(device_type=device.type, dtype=dtype):
                recon, _, _ = model(inp)
            recon = recon.squeeze(0).float()      # (1, T, T) in [-1, 1] (ish)

            # De-normalise back to original tile intensity range
            recon = (recon * 0.5) + 0.5            # [-1, 1] -> [0, 1]
            recon = recon.clamp(0.0, 1.0)
            if tmax > tmin:
                recon = recon * (tmax - tmin) + tmin
            else:
                recon = torch.full_like(recon, float(tmin))

            # Blend with cosine window
            output [:, y:y + tile_size, x:x + tile_size] += recon * window
            weights[:, y:y + tile_size, x:x + tile_size] += window

    # Normalise by the accumulated weight map
    output = output / weights.clamp(min=1e-6)

    # Crop padding back off
    output = output[:, :H, :W]
    return output


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def load_pretrained_medvae(device: torch.device, dtype: torch.dtype) -> AutoencoderKL:
    """Construct medvae_4_1_2d with original Stanford weights.

    The HuggingFace yaml is flat — it has `embed_dim` and `ddconfig` at the
    top level, matching AutoencoderKL.__init__'s signature directly. (Not
    nested under `model.params` like ldm/stable-diffusion configs.)
    """
    from omegaconf import OmegaConf
    cfg_path = hf_hub_download(
        repo_id="stanfordmimi/MedVAE",
        filename="model_weights/medvae_4x1.yaml",
    )
    ckpt_path = hf_hub_download(
        repo_id="stanfordmimi/MedVAE",
        filename="model_weights/vae_4x_1c_2D.ckpt",
    )
    cfg = OmegaConf.load(cfg_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model = AutoencoderKL(**cfg_dict, ckpt_path=ckpt_path).to(device).eval()
    return model


def load_finetuned_medvae(ckpt_dir: str, device: torch.device, dtype: torch.dtype) -> AutoencoderKL:
    """Load a MedVAE finetuned by the accelerate training loop.

    ckpt_dir should be a directory like .../checkpoints/best.pt/ that
    contains pytorch_model.bin (the accelerate-saved weights).
    """
    # Start from the same architecture as pretrained, then overwrite weights.
    model = load_pretrained_medvae(device, dtype)

    # Accelerate saves the AE weights as pytorch_model.bin (and shards the
    # rest as pytorch_model_N.bin for the optimizer / discriminator). We only
    # need the first file for the autoencoder.
    weight_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Expected {weight_path}")
    state = torch.load(weight_path, map_location=device)
    # Strip any 'module.' prefix from DDP wrapping
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] {len(missing)} missing keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"  [load] {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")
    return model.eval()


# ----------------------------------------------------------------------------
# Per-image evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model_name: str,
    model: AutoencoderKL,
    test_df: pd.DataFrame,
    img_dir: Path,
    tile_size: int,
    stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    """Loop over all test images, reconstruct each via tile-and-stitch,
    compute PSNR and MS-SSIM both unmasked and masked-to-bbox."""

    # Metric objects are stateful in torchmetrics; create per-image instances
    # so we don't accumulate across images (we want per-image numbers).
    psnr_fn  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    msssim_fn = MSSSIM(data_range=1.0).to(device)

    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=model_name):
        img_path = img_dir / f"{row['image_uuid']}.jpg"
        pil = Image.open(img_path).convert("L")
        W, H = pil.size

        # Original image as tensor in [0, 1]
        orig = torch.from_numpy(np.array(pil)).float().unsqueeze(0) / 255.0
        orig = orig.to(device)

        # Tile-and-stitch reconstruction
        recon = tile_and_stitch(model, orig, tile_size, stride, device, dtype)

        # Bbox in pixel coords
        bx0 = int(round(row["bbox_x0"] * W));  by0 = int(round(row["bbox_y0"] * H))
        bx1 = int(round(row["bbox_x1"] * W));  by1 = int(round(row["bbox_y1"] * H))
        bx1 = max(bx1, bx0 + 1);   by1 = max(by1, by0 + 1)

        # MS-SSIM needs at least 161x161 for the default 5-scale pyramid.
        # If a bbox is tiny (shouldn't happen with BoostNet but guard anyway)
        # we fall back to the whole image for masked metrics.
        bbox_h, bbox_w = by1 - by0, bx1 - bx0
        if bbox_h < 161 or bbox_w < 161:
            print(f"  WARN: bbox too small for {row['image_uuid']} "
                  f"({bbox_h}x{bbox_w}); using full image for masked metrics")
            o_m, r_m = orig, recon
        else:
            o_m = orig [:, by0:by1, bx0:bx1]
            r_m = recon[:, by0:by1, bx0:bx1]

        # Both metrics expect (B, C, H, W)
        psnr_fn.reset();   msssim_fn.reset()
        psnr_unmasked = psnr_fn(recon.unsqueeze(0), orig.unsqueeze(0)).item()
        psnr_fn.reset()
        psnr_masked   = psnr_fn(r_m.unsqueeze(0),  o_m.unsqueeze(0)).item()
        msssim_fn.reset()
        msssim_unmasked = msssim_fn(recon.unsqueeze(0), orig.unsqueeze(0)).item()
        msssim_fn.reset()
        msssim_masked   = msssim_fn(r_m.unsqueeze(0),  o_m.unsqueeze(0)).item()

        rows.append({
            "model":          model_name,
            "image_uuid":     row["image_uuid"],
            "H":              H,
            "W":              W,
            "psnr_unmasked":  psnr_unmasked,
            "psnr_masked":    psnr_masked,
            "msssim_unmasked": msssim_unmasked,
            "msssim_masked":  msssim_masked,
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Aggregation + paired comparison
# ----------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> None:
    """Print mean ± std for each model and each metric, plus paired
    comparisons between models."""
    print("\n" + "=" * 78)
    print("PER-MODEL MEANS (n = {} test images)".format(df["image_uuid"].nunique()))
    print("=" * 78)
    metrics = ["psnr_unmasked", "psnr_masked", "msssim_unmasked", "msssim_masked"]
    summary = df.groupby("model")[metrics].agg(["mean", "std"])
    print(summary.round(4).to_string())

    print("\n" + "=" * 78)
    print("PAIRED COMPARISONS (Δ = row − col, masked metrics)")
    print("=" * 78)
    models = df["model"].unique()
    pivot_psnr   = df.pivot(index="image_uuid", columns="model", values="psnr_masked")
    pivot_msssim = df.pivot(index="image_uuid", columns="model", values="msssim_masked")
    for a, b in [(x, y) for x in models for y in models if x != y]:
        if a not in pivot_psnr.columns or b not in pivot_psnr.columns:
            continue
        d_psnr   = pivot_psnr[a]   - pivot_psnr[b]
        d_msssim = pivot_msssim[a] - pivot_msssim[b]
        wins_psnr   = (d_psnr   > 0).sum()
        wins_msssim = (d_msssim > 0).sum()
        n = len(d_psnr.dropna())
        print(f"\n  {a}  vs  {b}:")
        print(f"    PSNR (masked):    Δ = {d_psnr.mean():+.3f} ± {d_psnr.std():.3f} dB  "
              f"({a} wins {wins_psnr}/{n})")
        print(f"    MS-SSIM (masked): Δ = {d_msssim.mean():+.4f} ± {d_msssim.std():.4f}  "
              f"({a} wins {wins_msssim}/{n})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--aspect-ckpt", required=True, type=str,
                   help="Path to aspect-finetuned best.pt directory")
    p.add_argument("--patch-ckpt", required=True, type=str,
                   help="Path to patch-finetuned best.pt directory")
    p.add_argument("--tile-size", type=int, default=384)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Load test split
    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Loaded {len(test_df)} test images.")
    for col in ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]:
        if col not in test_df.columns:
            raise ValueError(
                f"Column {col!r} missing — run prepare_spine_bboxes.py first.")

    all_results = []
    for name, ckpt_path in [
        ("pretrained", None),
        ("aspect",     args.aspect_ckpt),
        ("patch",      args.patch_ckpt),
    ]:
        print(f"\n[{name}] Loading model...")
        if ckpt_path is None:
            model = load_pretrained_medvae(device, dtype)
        else:
            model = load_finetuned_medvae(ckpt_path, device, dtype)

        results = evaluate_model(
            name, model, test_df, args.img_dir,
            args.tile_size, args.stride, device, dtype,
        )
        all_results.append(results)
        # Free GPU memory before the next model
        del model
        torch.cuda.empty_cache()

    combined = pd.concat(all_results, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False)
    print(f"\nWrote per-image results to {args.out_csv}")
    summarise(combined)


if __name__ == "__main__":
    main()
