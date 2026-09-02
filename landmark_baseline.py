"""
landmark_baseline.py — sanity control for the frontier.

The frontier shows landmark error is FLAT across compression while Cobb degrades.
That's only meaningful if the landmark regressor is actually localising from the
latent rather than predicting a canonical mean spine (which would be flat under
compression trivially, because it ignores the image).

This computes the trivial baselines from the extracted labels:
  - landmark: predict the TRAIN-mean 68-point shape for every test image
  - cobb:     predict the TRAIN-mean 3 angles for every test image
and reports the same normalised metrics, so you can compare against the trained
numbers (landmark 0.0418 / cobb ~7.7-8.2 deg). If trained landmark error is far
below the mean-shape baseline, the regressor is genuinely localising and the
null is real. If it's close, the flatness is an artifact.

Run:
  python landmark_baseline.py --latents-dir ./latents_aspect
"""
import argparse
from pathlib import Path
import numpy as np

N = 68
CONFIGS = ["medvae_4_1_2d", "medvae_8_1_2d"]
LABEL = {"medvae_4_1_2d": "4x", "medvae_8_1_2d": "8x"}


def lmk_err(pred, true):
    px, py = pred[:, :N], pred[:, N:]
    tx, ty = true[:, :N], true[:, N:]
    return np.sqrt((px - tx) ** 2 + (py - ty) ** 2).mean(1)


def boot(v, seed=0, nb=10000):
    rng = np.random.default_rng(seed); v = np.asarray(v); n = len(v)
    b = [v[rng.integers(0, n, n)].mean() for _ in range(nb)]
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def load(d, c, s):
    z = np.load(Path(d) / f"latents_{c}_{s}.npz", allow_pickle=True)
    return z["landmarks"].astype(float), z["cobb"].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path, required=True)
    args = ap.parse_args()

    # trained numbers for reference (from your frontier run)
    trained = {"medvae_4_1_2d": (0.04179, 7.709), "medvae_8_1_2d": (0.04178, 8.237)}

    for c in CONFIGS:
        ltr, ctr = load(args.latents_dir, c, "train")
        lte, cte = load(args.latents_dir, c, "test")
        mean_shape = ltr.mean(0, keepdims=True)
        lb = lmk_err(np.repeat(mean_shape, len(lte), 0), lte)
        mean_ang = ctr.mean(0, keepdims=True)
        cb = np.abs(np.repeat(mean_ang, len(cte), 0) - cte).mean(1)
        lm_m, lm_lo, lm_hi = boot(lb)
        cb_m, cb_lo, cb_hi = boot(cb)
        tl, tc = trained[c]
        print(f"=== {LABEL[c]} ===")
        print(f"  landmark  mean-shape baseline = {lm_m:.5f}  CI[{lm_lo:.5f},{lm_hi:.5f}]"
              f"   | trained = {tl:.5f}   -> trained is {100*(1-tl/lm_m):.1f}% better")
        print(f"  cobb      mean-angle baseline = {cb_m:.3f} deg CI[{cb_lo:.3f},{cb_hi:.3f}]"
              f" | trained = {tc:.3f} deg -> trained is {100*(1-tc/cb_m):.1f}% better")


if __name__ == "__main__":
    main()
