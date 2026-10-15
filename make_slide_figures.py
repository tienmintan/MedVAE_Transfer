"""
make_slide_figures.py: generate presentation figures from the multi-seed result CSVs.

Produces three slide-ready PNGs (transparent-friendly white background,
large fonts, palette-matched).

All numbers in the figures are read from the CSVs or computed from them,
including the cohort sizes in fig_cohort.png and the shares of the total
MAE change in fig_ablation.png.

Inputs:  phase_a_bootstrap_results.csv (from phase_a_bootstrap.py) and
         phase_a_per_cohort_results.csv (from phase_a_breakdowns.py) in the
         current directory.
Outputs (in ./slide_figs/):
  fig_forest.png        paired Δ-vs-baseline with bootstrap CIs (forest plot)
  fig_cohort.png        per-cohort MAE, baseline vs key configs (grouped bars)
  fig_ablation.png      decomposition of the MAE change from baseline (waterfall-style bars)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from pathlib import Path

# Palette (Ocean Gradient, matches the slide deck)
DEEP   = "#065A82"   # deep blue (primary)
TEAL   = "#1C7293"   # teal (secondary)
MID    = "#21295C"   # midnight (accent)
GREY   = "#64748B"   # muted labels
LIGHT  = "#E2E8F0"   # grid
GOOD   = "#1C7293"   # improvement
BAD    = "#B85042"   # degradation
BASE   = "#94A3B8"   # baseline / neutral

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

SRC = Path("./")
OUT = Path("./slide_figs/")
OUT.mkdir(exist_ok=True)


# Figure 1: forest plot of paired Δ vs baseline
def make_forest():
    df = pd.read_csv(SRC / "phase_a_bootstrap_results.csv")
    df = df[df.prefix != "matched_4x"].copy()   # drop baseline (Δ=0)

    order = ["matched_8x", "matched_whole_768", "matched_whole_1024",
             "matched_tiled_pretrained", "matched_tiled",
             "matched_diffusion_features"]
    labels = {
        "matched_8x": "8× aspect-padded",
        "matched_whole_768": "Whole-image 768",
        "matched_whole_1024": "Whole-image 1024",
        "matched_tiled_pretrained": "Tile-and-stitch (pretrained)",
        "matched_tiled": "Tile-and-stitch (patch-finetuned)",
        "matched_diffusion_features": "Diffusion U-Net features",
    }
    df = df.set_index("prefix").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ys = np.arange(len(df))[::-1]

    for y, (_, r) in zip(ys, df.iterrows()):
        d, lo, hi = r.diff_vs_baseline_mean, r.diff_vs_baseline_ci_lo, r.diff_vs_baseline_ci_hi
        robust = (lo > 0) or (hi < 0)
        # colour: improvement (negative Δ) = teal, worse = terracotta, non-robust = grey
        if not robust:
            c = BASE
        elif d < 0:
            c = GOOD
        else:
            c = BAD
        ax.plot([lo, hi], [y, y], color=c, lw=3, solid_capstyle="round", zorder=2)
        ax.scatter([d], [y], color=c, s=90, zorder=3, edgecolor="white", linewidth=1.2)

    ax.axvline(0, color=MID, lw=1.4, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[p] for p in df.prefix], fontsize=14)
    ax.set_xlabel("Δ MAE vs baseline  (degrees, negative = better)", fontsize=14, color=GREY)
    ax.set_title("Paired difference from aspect-padded baseline\n"
                 "(bootstrap 95% CI, image-level resampling)",
                 fontsize=15.5, color=MID, pad=12)
    ax.tick_params(colors=GREY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)

    # directional shading instead of colliding text labels
    xl = ax.get_xlim()
    ax.axvspan(xl[0], 0, color=GOOD, alpha=0.04, zorder=0)
    ax.axvspan(0, xl[1], color=BAD, alpha=0.04, zorder=0)
    ax.set_xlim(xl)

    fig.tight_layout()
    fig.savefig(OUT / "fig_forest.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote fig_forest.png")


# Figure 2: per-cohort grouped bars
def make_cohort():
    df = pd.read_csv(SRC / "phase_a_per_cohort_results.csv")
    configs = [
        ("matched_4x", "Baseline\n(aspect-padded)", BASE),
        ("matched_tiled", "Tile-and-stitch\n(patch-finetuned)", GOOD),
        ("matched_diffusion_features", "Diffusion\nfeatures", BAD),
    ]
    cohorts = ["mild", "moderate", "severe"]
    n_by_cohort = df[df.prefix == "matched_4x"].set_index("cohort")["n_images"]
    cohort_labels = [f"{c.capitalize()}\n(n={int(n_by_cohort[c])})" for c in cohorts]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(cohorts))
    width = 0.26

    for i, (prefix, label, colour) in enumerate(configs):
        sub = df[df.prefix == prefix].set_index("cohort")
        vals = [sub.loc[c, "mae"] for c in cohorts]
        los  = [sub.loc[c, "mae"] - sub.loc[c, "boot_lo"] for c in cohorts]
        his  = [sub.loc[c, "boot_hi"] - sub.loc[c, "mae"] for c in cohorts]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=label.replace("\n", " "),
                      color=colour, edgecolor="white", linewidth=0.8, zorder=3)
        ax.errorbar(x + offset, vals, yerr=[los, his], fmt="none",
                    ecolor=MID, elinewidth=1.1, capsize=3, alpha=0.6, zorder=4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.18, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=11, color=MID)

    ax.set_xticks(x)
    ax.set_xticklabels(cohort_labels, fontsize=13)
    ax.set_ylabel("Test MAE (degrees)", fontsize=14, color=GREY)
    ax.set_title("MAE by scoliosis severity",
                 fontsize=15.5, color=MID, pad=12)
    ax.tick_params(colors=GREY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=1)
    ax.set_ylim(0, 13.2)

    fig.tight_layout()
    fig.savefig(OUT / "fig_cohort.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote fig_cohort.png")


# Figure 3: ablation decomposition (waterfall-style)
def make_ablation():
    df = pd.read_csv(SRC / "phase_a_bootstrap_results.csv").set_index("prefix")
    base  = df.loc["matched_4x", "mean_mae"]
    pre   = df.loc["matched_tiled_pretrained", "mean_mae"]
    patch = df.loc["matched_tiled", "mean_mae"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    labels = ["Baseline\n(aspect-padded)",
              "+ native-resolution\naccess (tile-and-stitch)",
              "+ patch\nfinetuning"]
    vals = [base, pre, patch]
    colours = [BASE, TEAL, DEEP]

    xs = np.arange(3)
    bars = ax.bar(xs, vals, 0.55, color=colours, edgecolor="white",
                  linewidth=1, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.07, f"{v:.2f}°",
                ha="center", va="bottom", fontsize=14, color=MID, fontweight="bold")

    # connector deltas
    def delta(x0, x1, y0, y1, txt, share):
        ax.annotate("", xy=(x1-0.12, y1), xytext=(x0+0.12, y0),
                    arrowprops=dict(arrowstyle="->", color=MID, lw=1.4))
        ax.text((x0+x1)/2, max(y0, y1)+0.34, txt, ha="center",
                fontsize=12.5, color=MID, fontweight="bold")
        ax.text((x0+x1)/2, max(y0, y1)+0.05, share, ha="center",
                fontsize=10.5, color=GREY)

    total = base - patch   # total MAE change from baseline to patch finetuning
    delta(0, 1, base, pre, f"{pre-base:+.2f}°", f"{100 * (base - pre) / total:.0f}% of total change")
    delta(1, 2, pre, patch, f"{patch-pre:+.2f}°", f"{100 * (pre - patch) / total:.0f}% of total change")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("Test MAE (degrees)", fontsize=14, color=GREY)
    ax.set_title("MAE change from baseline: access pattern vs encoder finetuning",
                 fontsize=15.5, color=MID, pad=14)
    ax.tick_params(colors=GREY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(6.0, 7.8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote fig_ablation.png")


if __name__ == "__main__":
    make_forest()
    make_cohort()
    make_ablation()
    print("All figures written to", OUT)
