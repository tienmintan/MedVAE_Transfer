"""
inspect_label_distribution.py

Look at the distribution of Cobb angle labels in BoostNet:
  - Per-angle histograms (proximal-thoracic, main-thoracic, thoracolumbar)
  - Per-split breakdown (train, val, test)
  - Clinical severity classification (none / mild / moderate / severe)
  - Correlations between the three angles

Useful for understanding why a regressor performs the way it does — class
imbalance, range coverage, joint dependencies.

Usage:
  python inspect_label_distribution.py \\
      --csv $PROJECT_DIR/medvae/data/boostnet_with_angles.csv \\
      --out ./label_distribution.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANGLE_COLS = ["angle_proximal_thoracic", "angle_main_thoracic", "angle_thoracolumbar"]
ANGLE_NAMES = ["Proximal-thoracic", "Main-thoracic", "Thoracolumbar"]
# Standard Cobb angle severity bins
SEVERITY_BINS = [-0.01, 10, 25, 40, 999]
SEVERITY_LABELS = ["normal (<10°)", "mild (10–25°)", "moderate (25–40°)", "severe (>40°)"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=ANGLE_COLS)
    print(f"Loaded {len(df)} labelled images "
          f"(splits: {df['split'].value_counts().to_dict()})\n")

    # === Numerical summary ===
    print("Per-angle summary (all splits):")
    print(df[ANGLE_COLS].describe().round(2).to_string())
    print()

    print("Per-angle summary, per split:")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        print(f"\n  {split} (n={len(sub)}):")
        print(sub[ANGLE_COLS].describe().round(2).to_string())
    print()

    # === Severity classification (using the largest of the three angles per image) ===
    df["max_angle"] = df[ANGLE_COLS].max(axis=1)
    df["severity"] = pd.cut(df["max_angle"], bins=SEVERITY_BINS,
                            labels=SEVERITY_LABELS, include_lowest=True)
    print("Severity breakdown (using max angle per image):")
    sev_table = (df.groupby(["split", "severity"], observed=True)
                   .size().unstack(fill_value=0))
    print(sev_table.to_string())
    print()
    print("Severity proportions per split (%):")
    sev_pct = sev_table.div(sev_table.sum(axis=1), axis=0) * 100
    print(sev_pct.round(1).to_string())
    print()

    # === Inter-angle correlation ===
    print("Correlation between angles (all splits):")
    print(df[ANGLE_COLS].corr().round(2).to_string())

    # === Plots ===
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # Row 1: per-angle histograms split-coloured
    for j, (col, name) in enumerate(zip(ANGLE_COLS, ANGLE_NAMES)):
        ax = fig.add_subplot(gs[0, j])
        for split, colour in [("train", "C0"), ("val", "C1"), ("test", "C2")]:
            ax.hist(df.loc[df["split"] == split, col], bins=30,
                    alpha=0.55, label=f"{split} (n={(df['split']==split).sum()})",
                    color=colour)
        # Severity boundaries
        for thresh in [10, 25, 40]:
            ax.axvline(thresh, color="grey", linestyle=":", alpha=0.6)
        ax.set_xlabel(f"{name} (°)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} angle distribution")
        ax.legend(fontsize=8)

    # Top right: severity distribution stacked bar
    ax = fig.add_subplot(gs[0, 3])
    sev_pct.plot(kind="bar", stacked=True, ax=ax,
                 color=["C2", "C0", "C1", "C3"])
    ax.set_ylabel("% of images")
    ax.set_xlabel("Split")
    ax.set_title("Severity distribution by split")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Row 2: pairwise scatter of angles (just train, to keep clean)
    train = df[df["split"] == "train"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    for j, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, j])
        ax.scatter(train[ANGLE_COLS[a]], train[ANGLE_COLS[b]],
                   alpha=0.4, s=15, color="C0")
        ax.set_xlabel(f"{ANGLE_NAMES[a]} (°)")
        ax.set_ylabel(f"{ANGLE_NAMES[b]} (°)")
        r = train[[ANGLE_COLS[a], ANGLE_COLS[b]]].corr().iloc[0, 1]
        ax.set_title(f"{ANGLE_NAMES[a]} vs {ANGLE_NAMES[b]}\nr = {r:.2f}")

    # Bottom-right: max angle distribution per split
    ax = fig.add_subplot(gs[1, 3])
    for split, colour in [("train", "C0"), ("val", "C1"), ("test", "C2")]:
        ax.hist(df.loc[df["split"] == split, "max_angle"], bins=30,
                alpha=0.55, label=split, color=colour)
    for thresh, lab in zip([10, 25, 40], ["mild", "moderate", "severe"]):
        ax.axvline(thresh, color="grey", linestyle=":", alpha=0.6)
        ax.text(thresh, ax.get_ylim()[1] * 0.95, f"  {lab}",
                fontsize=7, color="grey", verticalalignment="top")
    ax.set_xlabel("Max angle per image (°)")
    ax.set_ylabel("Count")
    ax.set_title("Max angle (severity proxy) per split")
    ax.legend(fontsize=8)

    # Row 3: cumulative distribution per angle (all splits combined)
    for j, (col, name) in enumerate(zip(ANGLE_COLS, ANGLE_NAMES)):
        ax = fig.add_subplot(gs[2, j])
        sorted_vals = np.sort(df[col].values)
        cumprop = np.linspace(0, 1, len(sorted_vals))
        ax.plot(sorted_vals, cumprop * 100, lw=1.5)
        for thresh in [10, 25, 40]:
            ax.axvline(thresh, color="grey", linestyle=":", alpha=0.6)
        ax.set_xlabel(f"{name} (°)")
        ax.set_ylabel("Cumulative % of images")
        ax.set_title(f"{name}: cumulative distribution")
        ax.grid(alpha=0.3)

    # Bottom right: training set sample-weighting recommendation
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    text = ["Training-set severity counts:", ""]
    train_sev = (df[df["split"] == "train"]
                   .groupby("severity", observed=True).size())
    for sev_label in SEVERITY_LABELS:
        n = train_sev.get(sev_label, 0)
        text.append(f"  {sev_label}: {n}")
    text.append("")
    text.append("If 'severe' is underrepresented,")
    text.append("regressor will under-predict severe cases.")
    text.append("Consider sample weighting or stratified")
    text.append("sampling in future regressor runs.")
    ax.text(0, 1, "\n".join(text), fontsize=9, family="monospace",
            verticalalignment="top")

    fig.suptitle("BoostNet Cobb angle label distribution", fontsize=13, y=0.99)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
