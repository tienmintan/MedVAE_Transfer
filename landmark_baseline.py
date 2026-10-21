"""
landmark_baseline.py: trivial-predictor baselines for the landmark and Cobb tasks, as a sanity check on the compression comparison.

A landmark error that does not change across compression levels is
informative only if the landmark regressor localises from the latent rather
than predicting a canonical mean spine (which would be insensitive to
compression trivially, because it ignores the image).

This computes the trivial baselines from the extracted labels:
  - landmark: predict the train-split mean 68-point shape for every test image
  - cobb:     predict the train-split mean 3 angles for every test image
and reports the same normalised metrics, with bootstrap 95% CIs, alongside the
trained-regressor errors from train_frontier.py (set in `trained` in main()).

Interpretation:
  trained landmark error well below the mean-shape baseline -> the regressor
      localises from the latent, so a flat landmark error across compression
      is not explained by mean-shape prediction.
  trained landmark error close to the baseline -> a flat landmark error is
      an artefact of mean-shape prediction.

Inputs:  <latents-dir>/latents_<config>_{train,test}.npz with `landmarks`
         and `cobb` arrays, for the configurations in CONFIGS.
Outputs: none; results are printed to stdout.

Usage:
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

    # trained-regressor errors for reference (landmark, Cobb deg), from train_frontier.py
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
