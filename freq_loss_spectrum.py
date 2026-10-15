"""
freq_loss_spectrum.py: reconstruction-error power spectrum by spatial frequency for MedVAE compression configurations.

First of the freq_loss_*.py frequency analyses (followed by
freq_loss_edge_interior.py and freq_loss_endplate.py).

Question: which spatial frequencies does MedVAE fail to preserve, and does
that loss grow with compression?

Method, per test image, per compression config:
  1. Aspect-pad + resize to 384x384 and normalise to [-1, 1] using the same
     loader as evaluate_reconstruction.py (load_2d_finetune_aspect).
  2. Encode -> decode with MedVAE (the variant sets the compression).
  3. Residual r = original - reconstruction (both in [-1, 1]).
  4. Crop original, recon, and residual to the content bounding box (the
     x > -0.8 mask), so neither the padding's low-frequency mass nor a
     zeroed-boundary's high-frequency artefact contaminates the spectrum.
  5. 2D FFT of the residual; radially average |FFT|^2 into frequency bins.
  6. Resample each image's radial profile onto a common normalised frequency
     axis (0 = DC, 1 = Nyquist) so images with different crop sizes can be
     averaged.

The script reports the residual power spectrum (absolute loss) and the
residual spectrum normalised by the original image's own spectrum (fraction
of content lost at each frequency); the latter controls for images having
different amounts of high-frequency content to begin with.

Interpretation:
  residual power rising toward high frequency, with the 8x curve above the
  4x curve at high frequencies -> compression preferentially discards
  high-frequency detail, more so at 8x.

Inputs:  --csv (image_uuid, split); JPEG images in --img-dir; MedVAE
         configurations in --configs.
Outputs: <out-dir>/freq_loss_spectrum.png and freq_loss_spectrum.csv; a
         per-band summary printed to stdout.

Usage:
  python freq_loss_spectrum.py \\
      --csv      ./medvae/data/boostnet.csv \\
      --img-dir  ./medvae/data/boostnet \\
      --out-dir  ./outputs/freq_loss \\
      --configs  medvae_4_1_2d medvae_8_1_2d \\
      --split    test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medvae import MVAE
from medvae.utils.loaders import load_2d_finetune_aspect


CONTENT_THRESHOLD = -0.8       # padding sits at -1.0; matches evaluate_reconstruction.py
N_FREQ_BINS = 64               # bins along the common normalised-frequency axis
CONFIG_LABELS = {
    "medvae_4_1_2d": "4x (latent 96x96)",
    "medvae_8_1_2d": "8x (latent 48x48)",
}
CONFIG_COLORS = {
    "medvae_4_1_2d": "#1C7293",
    "medvae_8_1_2d": "#B85042",
}


# Reconstruction (mirrors evaluate_reconstruction.py)

def transform_image(path: str) -> torch.Tensor:
    # (1, 1, 384, 384) in [-1, 1], padding at -1.0
    return load_2d_finetune_aspect(str(path), merge_channels=True).unsqueeze(0)


@torch.no_grad()
def reconstruct(mvae, x):
    dec, _ = mvae(x, decode=True)
    return dec.unsqueeze(0).unsqueeze(0).clamp(-1, 1)


def content_bbox(x, threshold=CONTENT_THRESHOLD):
    """Bounding box (r0, r1, c0, c1) of content pixels in a (1,1,H,W) tensor."""
    m = (x > threshold).squeeze(0).squeeze(0)
    rows = m.any(dim=1).nonzero(as_tuple=True)[0]
    cols = m.any(dim=0).nonzero(as_tuple=True)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (rows[0].item(), rows[-1].item() + 1,
            cols[0].item(), cols[-1].item() + 1)


# Radial power spectrum

def radial_power_spectrum(img2d: np.ndarray, n_bins: int = N_FREQ_BINS):
    """Return (norm_freq_bin_centres, radially-averaged power) for a 2D array.

    The frequency axis is normalised so 1.0 = Nyquist, making spectra from
    differently-sized crops comparable.
    """
    # Hann window to avoid edge-wrap artefacts from the crop boundary
    H, W = img2d.shape
    wy = np.hanning(H)[:, None]
    wx = np.hanning(W)[None, :]
    windowed = img2d * (wy * wx)

    F = np.fft.fftshift(np.fft.fft2(windowed))
    power = np.abs(F) ** 2

    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.indices((H, W))
    # normalised radius: distance from centre / max possible (Nyquist) along each axis
    ry = (yy - cy) / (H / 2.0)
    rx = (xx - cx) / (W / 2.0)
    r = np.sqrt(ry ** 2 + rx ** 2) / np.sqrt(2)   # in [0, 1], 1 = corner
    r = np.clip(r, 0, 1)

    bins = np.linspace(0, 1, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    prof = np.zeros(n_bins)
    idx = np.digitize(r.ravel(), bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    p_flat = power.ravel()
    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=p_flat, minlength=n_bins)
    prof = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return centres, prof


# Main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--configs", nargs="+", default=["medvae_4_1_2d", "medvae_8_1_2d"])
    p.add_argument("--split", default="test")
    p.add_argument("--modality", default="xray")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on #images for a quick dry run.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit]
    paths = [args.img_dir / f"{u}.jpg" for u in df["image_uuid"]]
    print(f"Frequency-loss spectrum on {len(paths)} {args.split} images, "
          f"configs={args.configs}")

    # results[config] = list of (residual_profile, original_profile)
    results = {c: {"resid": [], "orig": []} for c in args.configs}
    freq_axis = None

    for ci, cfg in enumerate(args.configs):
        print(f"\n[{ci+1}/{len(args.configs)}] Loading MedVAE variant {cfg}")
        mvae = MVAE(model_name=cfg, modality=args.modality).to(device)
        mvae.requires_grad_(False)
        mvae.eval()

        for i, path in enumerate(paths, 1):
            x = transform_image(str(path)).to(device)
            x_hat = reconstruct(mvae, x)
            bbox = content_bbox(x)
            if bbox is None:
                continue
            r0, r1, c0, c1 = bbox

            orig_c = x[..., r0:r1, c0:c1].squeeze().cpu().numpy()
            recon_c = x_hat[..., r0:r1, c0:c1].squeeze().cpu().numpy()
            resid_c = orig_c - recon_c

            fbins, resid_prof = radial_power_spectrum(resid_c)
            _, orig_prof = radial_power_spectrum(orig_c)
            if freq_axis is None:
                freq_axis = fbins

            results[cfg]["resid"].append(resid_prof)
            results[cfg]["orig"].append(orig_prof)

            if i % 20 == 0 or i == len(paths):
                print(f"    [{i}/{len(paths)}] processed")

        del mvae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for cfg in args.configs:
        resid = np.stack(results[cfg]["resid"])      # (N, n_bins)
        orig  = np.stack(results[cfg]["orig"])
        resid_mean = resid.mean(0)
        resid_sd = resid.std(0)
        # fraction of content lost at each frequency: residual power / original power
        frac_lost = (resid / np.maximum(orig, 1e-12)).mean(0)

        label = CONFIG_LABELS.get(cfg, cfg)
        color = CONFIG_COLORS.get(cfg, None)

        # Left panel: absolute residual power (log scale)
        axes[0].plot(freq_axis, resid_mean, label=label, color=color, lw=2)
        axes[0].fill_between(freq_axis, resid_mean - resid_sd / np.sqrt(len(resid)),
                             resid_mean + resid_sd / np.sqrt(len(resid)),
                             color=color, alpha=0.15)

        # Right panel: fraction of content lost
        axes[1].plot(freq_axis, frac_lost, label=label, color=color, lw=2)

        for f, rm, fl in zip(freq_axis, resid_mean, frac_lost):
            rows.append({"config": cfg, "norm_freq": f,
                         "resid_power_mean": rm, "frac_content_lost": fl})

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Normalised spatial frequency (1 = Nyquist)")
    axes[0].set_ylabel("Residual error power (log)")
    axes[0].set_title("Reconstruction-error power spectrum")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Normalised spatial frequency (1 = Nyquist)")
    axes[1].set_ylabel("Fraction of content power lost")
    axes[1].set_title("Relative information loss by frequency")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, None)

    fig.suptitle("What spatial frequencies does MedVAE compression discard?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out_dir / "freq_loss_spectrum.png", dpi=140, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(args.out_dir / "freq_loss_spectrum.csv", index=False)
    print(f"\nSaved {args.out_dir/'freq_loss_spectrum.png'}")
    print(f"Saved {args.out_dir/'freq_loss_spectrum.csv'}")

    # Quick summary to stdout.
    # The relevant quantity is the fraction-lost spectrum (residual power /
    # original power), not the absolute residual spectrum, which is
    # low-frequency dominated for every config and so hides the effect.
    print("\n--- Quick read (fraction of content power lost) ---")
    lo_band = freq_axis <= 0.33
    mid_band = (freq_axis > 0.33) & (freq_axis <= 0.66)
    hi_band = freq_axis > 0.66

    frac_by_cfg = {}
    for cfg in args.configs:
        resid = np.stack(results[cfg]["resid"])
        orig = np.stack(results[cfg]["orig"])
        frac = (resid / np.maximum(orig, 1e-12)).mean(0)   # per-bin fraction lost
        frac_by_cfg[cfg] = frac
        print(f"  {CONFIG_LABELS.get(cfg, cfg):22s}  "
              f"low={frac[lo_band].mean():.2f}  "
              f"mid={frac[mid_band].mean():.2f}  "
              f"high={frac[hi_band].mean():.2f}  "
              f"(slope low->high = {frac[hi_band].mean()-frac[lo_band].mean():+.2f})")

    if len(args.configs) >= 2:
        c4, c8 = args.configs[0], args.configs[1]
        hi_gap = frac_by_cfg[c8][hi_band].mean() - frac_by_cfg[c4][hi_band].mean()
        print(f"\n  High-frequency gap ({CONFIG_LABELS.get(c8,c8)} - "
              f"{CONFIG_LABELS.get(c4,c4)}): {hi_gap:+.2f}")
        print("  Interpretation: positive gap + rising-with-frequency fraction = compression")
        print("  preferentially discards high-frequency detail, more so at 8x.")
    print("\n(Numbers from a --limit run cover only part of the test split.)")


if __name__ == "__main__":
    main()
