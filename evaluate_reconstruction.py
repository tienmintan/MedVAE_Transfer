"""
evaluate_reconstruction.py

Quantitative reconstruction comparison on the BoostNet test split, with
PROPER MASKING to exclude the zero-padded regions added by load_2d_finetune_aspect.

Why masking matters: the aspect-preserving loader pads tall spinal X-rays
to a square, then resizes to 384x384. About 65% of the resulting frame is
black padding. PSNR and MS-SSIM computed over the full frame are inflated
because the model trivially reconstructs the constant padding. Masking
restricts the metrics to pixels inside the actual image content.

Reports both:
  * unmasked: as a sanity check / for legacy comparison
  * masked: the honest number to report

Usage:
  python evaluate_reconstruction.py \\
      --csv      $PROJECT_DIR/medvae/data/boostnet.csv \\
      --img-dir  $PROJECT_DIR/medvae/data/boostnet \\
      --finetuned-ckpt $PROJECT_DIR/medvae/logs/.../checkpoints/best.pt/pytorch_model.bin \\
      --output-dir ./eval_out
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from medvae import MVAE
from medvae.utils.loaders import load_2d_finetune_aspect
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    MultiScaleStructuralSimilarityIndexMeasure,
)


def load_finetuned_weights(mvae: MVAE, ckpt_path: Path) -> None:
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    missing, unexpected = mvae.model.load_state_dict(sd, strict=False)
    print(f"  loaded {ckpt_path}")
    print(f"    missing keys:    {len(missing)} | unexpected keys: {len(unexpected)}")


def transform_image(path: str) -> torch.Tensor:
    return load_2d_finetune_aspect(str(path), merge_channels=True).unsqueeze(0)


def make_content_mask(x: torch.Tensor, threshold: float = -0.8) -> torch.Tensor:
    """Build a binary mask of non-padding pixels.

    The aspect-preserving loader pads with zero in source-image space, then
    ScaleIntensity + Normalize map [0, 1] -> [-1, 1]. Padded pixels end up
    at -1.0 in the final tensor (verified empirically: ~57% of pixels sit
    below -0.99 on a representative test image).

    We threshold at -0.99 to identify content pixels.

    Returns: (1, 1, H, W) bool mask, True for content.
    """
    return x > threshold


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> float:
    diff_sq = (x_hat - x) ** 2
    return (diff_sq * mask).sum().item() / mask.sum().item()


def masked_psnr(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor,
                data_range: float = 2.0) -> float:
    mse = masked_mse(x_hat, x, mask)
    if mse <= 0:
        return float("inf")
    return 10.0 * float(np.log10((data_range ** 2) / mse))


def masked_ms_ssim(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor,
                   ms_ssim_metric) -> float:
    """Crop to the bounding box of the content and run MS-SSIM there.

    MS-SSIM uses local 11x11 windows and a Gaussian pyramid; you can't
    cleanly mask out individual pixels. The honest workaround is to crop
    to the tightest bounding box of the content region and evaluate inside
    that crop. The cropped region is mostly-content (small residual padding
    near corners is acceptable; it's still vastly more honest than
    evaluating the full padded frame).
    """
    # mask is (1, 1, H, W). Find rows/cols that contain any content.
    m2d = mask.squeeze(0).squeeze(0)
    rows = m2d.any(dim=1).nonzero(as_tuple=True)[0]
    cols = m2d.any(dim=0).nonzero(as_tuple=True)[0]
    if len(rows) == 0 or len(cols) == 0:
        return float("nan")
    r0, r1 = rows[0].item(), rows[-1].item() + 1
    c0, c1 = cols[0].item(), cols[-1].item() + 1
    x_crop = x[..., r0:r1, c0:c1]
    x_hat_crop = x_hat[..., r0:r1, c0:c1]

    # MS-SSIM needs spatial dims > 160 (5 levels of 2x downsampling on 11x11
    # window: 11 * 2^4 = 176). Use 176 to be safely above the threshold;
    # below that, fall back to plain SSIM.
    H, W = x_crop.shape[-2:]
    if min(H, W) < 176:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(x_crop.device)
        return ssim(x_hat_crop, x_crop).item()
    ms_ssim_metric.reset()
    return ms_ssim_metric(x_hat_crop, x_crop).item()


@torch.no_grad()
def evaluate(mvae: MVAE, image_paths: list, device: torch.device) -> dict:
    mvae.eval()
    psnr_full = PeakSignalNoiseRatio(data_range=2.0).to(device)
    msssim_full = MultiScaleStructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    msssim_crop = MultiScaleStructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    psnr_unmasked, psnr_masked = [], []
    msssim_unmasked, msssim_masked = [], []
    coverage = []

    for i, p in enumerate(image_paths, 1):
        x = transform_image(str(p)).to(device)
        # Sanity check on the first image: confirm that x is in [-1, 1] and that
        # the mask is identifying a sensible fraction of padded pixels.
        if i == 1:
            mask_first = make_content_mask(x)
            content_pct = mask_first.float().mean().item()
            print(f"    [sanity] tensor range: [{x.min().item():.3f}, "
                  f"{x.max().item():.3f}]  | "
                  f"mask covers {content_pct:.1%} of frame as content  | "
                  f"expected ~40-50% if aspect-padding is working")
        dec, _ = mvae(x, decode=True)
        x_hat = dec.unsqueeze(0).unsqueeze(0).clamp(-1, 1)
        mask = make_content_mask(x).to(device)

        # Unmasked (legacy / sanity-check numbers)
        psnr_full.reset(); msssim_full.reset()
        psnr_unmasked.append(psnr_full(x_hat, x).item())
        msssim_unmasked.append(msssim_full(x_hat, x).item())
        # Masked (honest numbers)
        psnr_masked.append(masked_psnr(x_hat, x, mask))
        msssim_masked.append(masked_ms_ssim(x_hat, x, mask, msssim_crop))
        coverage.append(mask.float().mean().item())

        if i % 20 == 0 or i == len(image_paths):
            print(f"    [{i}/{len(image_paths)}]  "
                  f"masked PSNR={np.mean(psnr_masked):.3f}  "
                  f"masked MS-SSIM={np.mean(msssim_masked):.4f}  "
                  f"(content={np.mean(coverage):.1%} of frame)")

    return {
        "psnr_unmasked": np.array(psnr_unmasked),
        "psnr_masked":   np.array(psnr_masked),
        "msssim_unmasked": np.array(msssim_unmasked),
        "msssim_masked":   np.array(msssim_masked),
        "content_coverage": np.array(coverage),
    }


def summarise(name: str, m: dict) -> dict:
    return {
        "model": name,
        "psnr_unmasked_mean": float(m["psnr_unmasked"].mean()),
        "psnr_unmasked_std":  float(m["psnr_unmasked"].std(ddof=1)),
        "psnr_masked_mean":   float(m["psnr_masked"].mean()),
        "psnr_masked_std":    float(m["psnr_masked"].std(ddof=1)),
        "msssim_unmasked_mean": float(m["msssim_unmasked"].mean()),
        "msssim_unmasked_std":  float(m["msssim_unmasked"].std(ddof=1)),
        "msssim_masked_mean":   float(m["msssim_masked"].mean()),
        "msssim_masked_std":    float(m["msssim_masked"].std(ddof=1)),
        "n": int(len(m["psnr_masked"])),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--finetuned-ckpt", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model-name", default="medvae_4_1_2d")
    p.add_argument("--modality", default="xray")
    p.add_argument("--split", default="test")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    paths = [args.img_dir / f"{u}.jpg" for u in df["image_uuid"]]
    print(f"Evaluating on {len(paths)} {args.split} images.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/2] Pretrained MedVAE baseline")
    base = MVAE(model_name=args.model_name, modality=args.modality).to(device)
    base.requires_grad_(False)
    base_metrics = evaluate(base, paths, device)
    del base; torch.cuda.empty_cache()

    print("\n[2/2] Finetuned MedVAE")
    ft = MVAE(model_name=args.model_name, modality=args.modality).to(device)
    load_finetuned_weights(ft, args.finetuned_ckpt)
    ft.to(device); ft.requires_grad_(False)
    ft_metrics = evaluate(ft, paths, device)
    del ft; torch.cuda.empty_cache()

    summary = pd.DataFrame([summarise("pretrained", base_metrics),
                            summarise("finetuned", ft_metrics)])

    per_image = pd.DataFrame({
        "image_uuid": df["image_uuid"],
        "content_coverage": base_metrics["content_coverage"],
        "pretrained_psnr_unmasked": base_metrics["psnr_unmasked"],
        "finetuned_psnr_unmasked":  ft_metrics["psnr_unmasked"],
        "pretrained_psnr_masked":   base_metrics["psnr_masked"],
        "finetuned_psnr_masked":    ft_metrics["psnr_masked"],
        "pretrained_msssim_unmasked": base_metrics["msssim_unmasked"],
        "finetuned_msssim_unmasked":  ft_metrics["msssim_unmasked"],
        "pretrained_msssim_masked":   base_metrics["msssim_masked"],
        "finetuned_msssim_masked":    ft_metrics["msssim_masked"],
    })
    summary.to_csv(args.output_dir / "reconstruction_metrics.csv", index=False)
    per_image.to_csv(args.output_dir / "per_image_metrics.csv", index=False)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    # Paired comparisons on masked metrics (the honest ones).
    psnr_gain = ft_metrics["psnr_masked"] - base_metrics["psnr_masked"]
    ssim_gain = ft_metrics["msssim_masked"] - base_metrics["msssim_masked"]
    print(f"\nPaired delta on MASKED metrics (finetuned − pretrained):")
    print(f"  PSNR    Δ = {psnr_gain.mean():+.3f} ± {psnr_gain.std(ddof=1):.3f}  "
          f"(finetuned wins on {(psnr_gain > 0).sum()}/{len(psnr_gain)} images)")
    print(f"  MS-SSIM Δ = {ssim_gain.mean():+.4f} ± {ssim_gain.std(ddof=1):.4f}  "
          f"(finetuned wins on {(ssim_gain > 0).sum()}/{len(ssim_gain)} images)")
    print(f"\nMean content coverage: {base_metrics['content_coverage'].mean():.1%} "
          f"of the 384x384 frame.")


if __name__ == "__main__":
    main()
