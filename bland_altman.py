"""
bland_altman.py

Generate Bland-Altman plots comparing the regressor's predicted Cobb angles
against the ground-truth labels on the test set. Reads the .npz files
produced by train_cobb_regressor.py.

What a Bland-Altman plot shows:
  * X-axis: mean of (prediction, ground truth) per image — the magnitude
    of the underlying angle.
  * Y-axis: difference (prediction − ground truth) per image.
  * Horizontal lines: bias (mean Δ) and limits of agreement (bias ± 1.96·SD).
  * 95% of differences should fall within the limits of agreement if errors
    are roughly normal.

What it tells you that MAE doesn't:
  * Whether errors are biased (systematic over/under-prediction).
  * Whether bias varies with angle magnitude (e.g. underestimating severe
    cases — a common failure mode for regression models trained on data
    skewed toward mild cases).
  * Where the outliers are.

Usage:
  python bland_altman.py \\
      --pretrained-results ./cobb_pretrained.npz \\
      --finetuned-results  ./cobb_finetuned.npz \\
      --out ./bland_altman.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANGLE_NAMES = ["Proximal-thoracic", "Main-thoracic", "Thoracolumbar"]


def bland_altman_panel(ax, preds, targets, title: str, units: str = "°"):
    """Draw a single Bland-Altman panel."""
    mean = (preds + targets) / 2
    diff = preds - targets
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    loa_lo = bias - 1.96 * sd
    loa_hi = bias + 1.96 * sd

    ax.scatter(mean, diff, alpha=0.55, s=24, edgecolor="none")
    ax.axhline(bias, color="C3", linestyle="-", linewidth=1.4,
               label=f"Bias = {bias:+.2f}{units}")
    ax.axhline(loa_hi, color="C3", linestyle="--", linewidth=1.0,
               label=f"+1.96 SD = {loa_hi:+.2f}{units}")
    ax.axhline(loa_lo, color="C3", linestyle="--", linewidth=1.0,
               label=f"−1.96 SD = {loa_lo:+.2f}{units}")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel(f"Mean of prediction and ground truth ({units})")
    ax.set_ylabel(f"Prediction − ground truth ({units})")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    return {"bias": bias, "sd": sd, "loa_lo": loa_lo, "loa_hi": loa_hi}


def load(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    return d["test_preds"], d["test_targets"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-results", required=True, type=Path)
    p.add_argument("--finetuned-results", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    pre_preds, pre_targets = load(args.pretrained_results)
    ft_preds, ft_targets = load(args.finetuned_results)

    # Sanity check: the two runs should share the same test set in the same order.
    if not np.allclose(pre_targets, ft_targets):
        raise RuntimeError("Test targets differ between the two runs — "
                           "did you re-run with the same seed and CSV?")

    # 4 columns x 2 rows: per-angle (3) + overall (1) for each encoder
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    summary_rows = []

    for row, (name, preds, targets) in enumerate([
        ("Pretrained encoder", pre_preds, pre_targets),
        ("Finetuned encoder", ft_preds, ft_targets),
    ]):
        # Per-angle panels
        for j in range(3):
            stats = bland_altman_panel(
                axes[row, j], preds[:, j], targets[:, j],
                title=f"{name} — {ANGLE_NAMES[j]}")
            summary_rows.append({"encoder": name, "angle": ANGLE_NAMES[j], **stats})
        # Overall panel: flatten all 3 angles together
        stats = bland_altman_panel(
            axes[row, 3], preds.flatten(), targets.flatten(),
            title=f"{name} — All angles combined")
        summary_rows.append({"encoder": name, "angle": "All", **stats})

    fig.suptitle(
        "Bland-Altman: predicted Cobb angle vs ground truth (test set)",
        fontsize=12, y=1.00)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")

    # Also print the numerical summary so you have it for the writeup
    print("\n=== Bland-Altman summary ===")
    print(f"{'encoder':<20} {'angle':<22} {'bias':>8} {'SD':>8} "
          f"{'LoA lower':>11} {'LoA upper':>11}")
    for r in summary_rows:
        print(f"{r['encoder']:<20} {r['angle']:<22} "
              f"{r['bias']:>+7.2f}° {r['sd']:>7.2f}° "
              f"{r['loa_lo']:>+10.2f}° {r['loa_hi']:>+10.2f}°")


if __name__ == "__main__":
    main()
