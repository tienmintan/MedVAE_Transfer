"""
cobb_distributions.py: characterise the ground-truth and inferred Cobb angle values and their errors, per angle.

Reports:
  * level and spread of the ground-truth and inferred values for each angle;
  * the distribution of angle values and of absolute error for each of the
    three angles;
  * a regression of inferred on ground truth, to test whether the model
    compresses the range (regresses to the mean);
  * the range each Cobb measurement takes.

All errors are in degrees, not percentages. Columns follow the dataset order
[PT, MT, TL]; column 0 holds the major curve.

Inputs:  --preds, a saved prediction file preds_{config}.npz with pred_cobb and
         gt_cobb, each of shape (N, 3).
Outputs: in --out-dir, {tag}_cobb_distributions.csv, {tag}_scatter.png and
         {tag}_boxplots.png, where tag is the prediction file stem.

Usage:
  python cobb_distributions.py --preds ./outputs/frontier/preds_medvae_4_1_2d.npz \
      --out-dir ./outputs/cobb_dist
  python cobb_distributions.py --smoke      # synthetic-data self-test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANGLES = ["Major curve (col 0)", "Col 1", "Col 2"]
SHORT = ["PT", "MT", "TL"]
GT_COLOR = "#1C7293"
INF_COLOR = "#B85042"


def summarise(v):
    return dict(mean=v.mean(), sd=v.std(), min=v.min(),
                q1=np.percentile(v, 25), median=np.median(v),
                q3=np.percentile(v, 75), max=v.max())


def characterise(preds_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(preds_path, allow_pickle=True)
    gt, pred = d["gt_cobb"], d["pred_cobb"]        # (N,3)
    tag = Path(preds_path).stem
    N = len(gt)
    print(f"=== Cobb angle characterisation: {tag} (N={N}) ===\n")

    rows = []
    for i, name in enumerate(SHORT):
        g, p = gt[:, i], pred[:, i]
        err = np.abs(p - g)
        sg, sp = summarise(g), summarise(p)
        lr = stats.linregress(g, p)          # inferred = slope*gt + b
        rows.append({
            "angle": name,
            "gt_mean": sg["mean"], "gt_sd": sg["sd"],
            "gt_min": sg["min"], "gt_median": sg["median"], "gt_max": sg["max"],
            "inf_mean": sp["mean"], "inf_sd": sp["sd"],
            "inf_min": sp["min"], "inf_median": sp["median"], "inf_max": sp["max"],
            "MAE_deg": err.mean(), "err_median": np.median(err),
            "err_90pct": np.percentile(err, 90), "err_max": err.max(),
            "r": lr.rvalue, "slope": lr.slope,
        })
    tab = pd.DataFrame(rows)

    # console: level + spread + range (GT vs inferred), error, regression
    pd.set_option("display.float_format", lambda x: f"{x:6.2f}")
    print("Value distribution: level, spread, range (degrees):")
    print(tab[["angle", "gt_mean", "gt_sd", "gt_min", "gt_median", "gt_max",
               "inf_mean", "inf_sd", "inf_min", "inf_median", "inf_max"]].to_string(index=False))
    print("\nError distribution (degrees) and GT-vs-inferred regression:")
    print(tab[["angle", "MAE_deg", "err_median", "err_90pct", "err_max", "r", "slope"]].to_string(index=False))
    print("\nNote: slope < 1 indicates that the model compresses the range (under-predicts "
          "large angles, regressing to the mean); this is worth watching for the large curves in col 1 (MT).")
    tab.to_csv(out_dir / f"{tag}_cobb_distributions.csv", index=False)

    make_figures(gt, pred, tag, out_dir)
    print(f"\nWrote table and figures to {out_dir}")
    return tab


def make_figures(gt, pred, tag, out_dir):
    # 1) GT vs inferred scatter, per angle, with y=x and regression line
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for i, (ax, name) in enumerate(zip(axes, ANGLES)):
        g, p = gt[:, i], pred[:, i]
        lr = stats.linregress(g, p)
        lim = [min(g.min(), p.min()) - 2, max(g.max(), p.max()) + 2]
        ax.plot(lim, lim, "--", color="0.6", lw=1, label="y = x (perfect)")
        xs = np.array(lim)
        ax.plot(xs, lr.slope * xs + lr.intercept, "-", color=INF_COLOR, lw=1.8,
                label=f"fit (slope={lr.slope:.2f})")
        ax.scatter(g, p, s=16, color=GT_COLOR, alpha=0.6, edgecolors="none")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel("ground truth (°)"); ax.set_ylabel("inferred (°)")
        ax.set_title(f"{name}\nr={lr.rvalue:.3f}", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"Ground truth vs inferred Cobb angle: {tag}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / f"{tag}_scatter.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # 2) value ranges (GT vs inferred) + error, as box plots
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    data, positions, colors, ticks = [], [], [], []
    for i in range(3):
        data += [gt[:, i], pred[:, i]]
        positions += [i * 3 + 1, i * 3 + 2]
        colors += [GT_COLOR, INF_COLOR]
        ticks.append(i * 3 + 1.5)
    bp = axL.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    axL.set_xticks(ticks); axL.set_xticklabels(SHORT)
    axL.set_ylabel("Cobb angle (°)")
    axL.set_title("Value range per angle: GT (blue) vs inferred (red)")
    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(facecolor=GT_COLOR, alpha=0.6, label="ground truth"),
                        Patch(facecolor=INF_COLOR, alpha=0.6, label="inferred")],
               frameon=False, fontsize=9)

    errs = [np.abs(pred[:, i] - gt[:, i]) for i in range(3)]
    bp2 = axR.boxplot(errs, positions=[1, 2, 3], widths=0.6, patch_artist=True, showfliers=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor(INF_COLOR); patch.set_alpha(0.6)
    axR.set_xticks([1, 2, 3]); axR.set_xticklabels(SHORT)
    axR.set_ylabel("absolute error (°)")
    axR.set_title("Error distribution per angle (degrees)")
    fig.suptitle(f"Cobb angle value & error distributions: {tag}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / f"{tag}_boxplots.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def smoke():
    rng = np.random.default_rng(0); N = 128
    # synthetic data: col 1 (MT) larger and wider than cols 0 and 2;
    # inferred = regress-to-mean version of GT plus noise
    gt = np.stack([rng.normal(18, 8, N), rng.normal(30, 14, N), rng.normal(15, 9, N)], 1).clip(0)
    pred = (0.8 * gt + 0.2 * gt.mean(0) + rng.normal(0, 3, gt.shape)).clip(0)
    np.savez("/tmp/preds_cobb_smoke.npz", gt_cobb=gt.astype(np.float32),
             pred_cobb=pred.astype(np.float32),
             uuids=np.array([f"i{k}" for k in range(N)]))
    characterise(Path("/tmp/preds_cobb_smoke.npz"), Path("/tmp/cobb_dist"))
    print("\nSelf-test: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/cobb_dist"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.preds:
        characterise(args.preds, args.out_dir)
    else:
        raise SystemExit("--preds is required (or use --smoke for the synthetic-data self-test)")


if __name__ == "__main__":
    main()
