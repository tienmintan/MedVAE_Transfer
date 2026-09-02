"""
freq_loss_edge_interior.py — Step 2 of the compression-frontier analysis.

Step 1 showed MedVAE preferentially discards high-frequency content, more so
at 8x. Step 2 asks: does that lost content land on the vertebral edges/corners
that Cobb measurement and landmark localisation depend on?

Method, per test image, per config:
  1. Reconstruct at 384x384 (same path as Step 1 / evaluate_reconstruction.py).
  2. Residual r = original - reconstruction.
  3. Detect content bbox empirically (x > -0.8 mask).
  4. Map the 68 normalised landmarks into that bbox:
         (nx,ny) -> (c0 + nx*(c1-c0), r0 + ny*(r1-r0))
     (exact for aspect-pad-to-square-then-resize, any centering).
  5. EDGE region   = pixels within disk_radius of any landmark.
     INTERIOR region = content pixels farther than excl_radius from every
                       landmark.
  6. Mean squared residual (error power) in each region.

Aggregate across images with image-level bootstrap CIs, and report the
edge/interior ratio per config and whether it grows with compression.

--verify mode: overlays mapped landmarks + the edge/interior masks on a few
images so region placement is confirmed BEFORE trusting any numbers.

Usage (verify first):
  python freq_loss_edge_interior.py --verify \\
      --csv ./medvae/data/boostnet.csv --img-dir ./medvae/data/boostnet \\
      --lookup ./landmark_row_lookup_test_fixed.csv \\
      --landmarks ~/MedVAE/data/boostnet_labeldata/labels/test/landmarks.csv \\
      --out-dir ./outputs/freq_loss

Then full run (drop --verify).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from medvae import MVAE
from medvae.utils.loaders import load_2d_finetune_aspect


CONTENT_THRESHOLD = -0.8
CONFIG_LABELS = {"medvae_4_1_2d": "4x", "medvae_8_1_2d": "8x"}
CONFIG_COLORS = {"medvae_4_1_2d": "#1C7293", "medvae_8_1_2d": "#B85042"}
EDGE_COLOR = "#B85042"
INT_COLOR = "#1C7293"


def transform_image(path):
    return load_2d_finetune_aspect(str(path), merge_channels=True).unsqueeze(0)


@torch.no_grad()
def reconstruct(mvae, x):
    dec, _ = mvae(x, decode=True)
    return dec.unsqueeze(0).unsqueeze(0).clamp(-1, 1)


def content_bbox(x, threshold=CONTENT_THRESHOLD):
    m = (x > threshold).squeeze(0).squeeze(0)
    rows = m.any(dim=1).nonzero(as_tuple=True)[0]
    cols = m.any(dim=0).nonzero(as_tuple=True)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (rows[0].item(), rows[-1].item() + 1, cols[0].item(), cols[-1].item() + 1), m


def map_landmarks(landmarks_norm, bbox):
    """landmarks_norm: (68,2) in [0,1]. bbox: (r0,r1,c0,c1). -> (68,2) px (x,y)."""
    r0, r1, c0, c1 = bbox
    xs = c0 + landmarks_norm[:, 0] * (c1 - c0)
    ys = r0 + landmarks_norm[:, 1] * (r1 - r0)
    return np.stack([xs, ys], axis=1)


def region_masks(landmark_px, shape, content_mask, disk_r, excl_r):
    H, W = shape
    seeds = np.ones((H, W), dtype=bool)   # True everywhere; landmarks become False seeds
    for px, py in landmark_px:
        xi, yi = int(round(px)), int(round(py))
        if 0 <= yi < H and 0 <= xi < W:
            seeds[yi, xi] = False
    dist = distance_transform_edt(seeds)   # distance to nearest landmark
    edge = dist <= disk_r
    interior = content_mask & (dist > excl_r)
    return edge, interior, dist


def parse_landmark_row(row_vals):
    """136 values -> (68,2) normalised (x,y), ordering A: all-x then all-y."""
    xs = np.asarray(row_vals[:68], dtype=float)
    ys = np.asarray(row_vals[68:], dtype=float)
    return np.stack([xs, ys], axis=1)


def bootstrap_ci(per_image, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_image)
    n = len(arr)
    means = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--img-dir", required=True, type=Path)
    ap.add_argument("--lookup", required=True, type=Path,
                    help="landmark_row_lookup_*_fixed.csv from fix_landmark_lookup.py")
    ap.add_argument("--landmarks", required=True, type=Path,
                    help="labels/test/landmarks.csv")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", default=["medvae_4_1_2d", "medvae_8_1_2d"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--modality", default="xray")
    ap.add_argument("--disk-radius", type=float, default=8.0)
    ap.add_argument("--excl-radius", type=float, default=16.0)
    ap.add_argument("--verify", action="store_true",
                    help="Only visualise region placement on a few images.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    lookup = pd.read_csv(args.lookup).set_index("image_uuid")
    landmarks_all = pd.read_csv(args.landmarks, header=None).values.astype(float)

    # keep only images we have a landmark row for
    uuids = [u for u in df["image_uuid"] if u in lookup.index]
    if args.limit:
        uuids = uuids[:args.limit]
    print(f"{len(uuids)}/{len(df)} test images have a landmark row.")

    # ---------- VERIFY MODE ----------
    if args.verify:
        cfg = args.configs[0]
        mvae = MVAE(model_name=cfg, modality=args.modality).to(device).eval()
        mvae.requires_grad_(False)
        n = min(4, len(uuids))
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 11))
        for j, uuid in enumerate(uuids[:n]):
            x = transform_image(args.img_dir / f"{uuid}.jpg").to(device)
            x_hat = reconstruct(mvae, x)
            bbox, cmask = content_bbox(x)
            lm = parse_landmark_row(landmarks_all[int(lookup.loc[uuid, "landmark_row"])])
            lm_px = map_landmarks(lm, bbox)
            edge, interior, _ = region_masks(
                lm_px, x.shape[-2:], cmask.cpu().numpy(),
                args.disk_radius, args.excl_radius)

            img = x.squeeze().cpu().numpy()
            resid = (x - x_hat).squeeze().cpu().numpy()

            # top row: landmarks on the padded frame
            axes[0, j].imshow(img, cmap="gray")
            axes[0, j].scatter(lm_px[:, 0], lm_px[:, 1], s=7, c="red")
            axes[0, j].set_title(f"{uuid[:20]}\nlandmarks on 384 frame", fontsize=8)
            axes[0, j].axis("off")

            # bottom row: region masks over residual magnitude
            axes[1, j].imshow(np.abs(resid), cmap="magma")
            overlay = np.zeros((*edge.shape, 4))
            overlay[edge] = [0.72, 0.31, 0.26, 0.55]      # edge = terracotta
            overlay[interior] = [0.11, 0.45, 0.58, 0.35]  # interior = teal
            axes[1, j].imshow(overlay)
            axes[1, j].set_title("edge (red) vs interior (teal)\non |residual|", fontsize=8)
            axes[1, j].axis("off")
        fig.suptitle("Step 2 verify — confirm dots sit on vertebrae and "
                     "regions are sensible", fontsize=11)
        out = args.out_dir / "edge_interior_verify.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"Saved {out}\nIf placement looks right, re-run WITHOUT --verify.")
        return 0

    # ---------- FULL MEASUREMENT ----------
    records = {c: {"edge": [], "interior": [], "uuid": []} for c in args.configs}
    example = {}

    for cfg in args.configs:
        print(f"\nConfig {cfg}")
        mvae = MVAE(model_name=cfg, modality=args.modality).to(device).eval()
        mvae.requires_grad_(False)
        for i, uuid in enumerate(uuids, 1):
            x = transform_image(args.img_dir / f"{uuid}.jpg").to(device)
            x_hat = reconstruct(mvae, x)
            bbox, cmask = content_bbox(x)
            if bbox is None:
                continue
            lm = parse_landmark_row(landmarks_all[int(lookup.loc[uuid, "landmark_row"])])
            lm_px = map_landmarks(lm, bbox)
            edge, interior, _ = region_masks(
                lm_px, x.shape[-2:], cmask.cpu().numpy(),
                args.disk_radius, args.excl_radius)
            resid_sq = ((x - x_hat).squeeze().cpu().numpy()) ** 2

            records[cfg]["edge"].append(resid_sq[edge].mean())
            records[cfg]["interior"].append(resid_sq[interior].mean())
            records[cfg]["uuid"].append(uuid)

            if cfg == args.configs[-1] and i == 1:
                example = {"resid": np.abs((x - x_hat).squeeze().cpu().numpy()),
                           "lm_px": lm_px, "uuid": uuid, "cfg": cfg}
            if i % 20 == 0 or i == len(uuids):
                print(f"  [{i}/{len(uuids)}]")
        del mvae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- aggregate + bootstrap ----
    summary_rows, per_image_rows = [], []
    print("\n=== Edge vs interior reconstruction error (mean squared residual) ===")
    for cfg in args.configs:
        e = np.array(records[cfg]["edge"])
        it = np.array(records[cfg]["interior"])
        ratio_per_img = e / np.maximum(it, 1e-12)
        em, elo, ehi = bootstrap_ci(e)
        im, ilo, ihi = bootstrap_ci(it)
        rm, rlo, rhi = bootstrap_ci(ratio_per_img)
        print(f"  {CONFIG_LABELS.get(cfg,cfg):>3}: "
              f"edge={em:.4f} [{elo:.4f},{ehi:.4f}]  "
              f"interior={im:.4f} [{ilo:.4f},{ihi:.4f}]  "
              f"ratio={rm:.2f} [{rlo:.2f},{rhi:.2f}]")
        summary_rows.append({"config": cfg, "edge_mean": em, "edge_lo": elo, "edge_hi": ehi,
                             "interior_mean": im, "interior_lo": ilo, "interior_hi": ihi,
                             "ratio_mean": rm, "ratio_lo": rlo, "ratio_hi": rhi})
        for u, ev, iv in zip(records[cfg]["uuid"], e, it):
            per_image_rows.append({"config": cfg, "image_uuid": u,
                                   "edge_err": ev, "interior_err": iv})

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "edge_interior_summary.csv", index=False)
    pd.DataFrame(per_image_rows).to_csv(args.out_dir / "edge_interior_per_image.csv", index=False)

    # ---- ratio-growth check (the key claim) ----
    if len(args.configs) >= 2:
        c4, c8 = args.configs[0], args.configs[1]
        r4 = np.array(records[c4]["edge"]) / np.maximum(np.array(records[c4]["interior"]), 1e-12)
        r8 = np.array(records[c8]["edge"]) / np.maximum(np.array(records[c8]["interior"]), 1e-12)
        # paired per-image growth in the ratio (same images both configs)
        growth = r8 - r4
        gm, glo, ghi = bootstrap_ci(growth)
        print(f"\nEdge/interior ratio growth (8x − 4x): {gm:+.2f} [{glo:+.2f},{ghi:+.2f}]")
        print("  >0 with CI excluding 0 => compression preferentially sacrifices")
        print("  edge fidelity, confirming the frequency loss lands on the anatomy.")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # grouped bars
    x = np.arange(len(args.configs)); w = 0.36
    e_means = [s["edge_mean"] for s in summary_rows]
    e_err = [[s["edge_mean"]-s["edge_lo"] for s in summary_rows],
             [s["edge_hi"]-s["edge_mean"] for s in summary_rows]]
    i_means = [s["interior_mean"] for s in summary_rows]
    i_err = [[s["interior_mean"]-s["interior_lo"] for s in summary_rows],
             [s["interior_hi"]-s["interior_mean"] for s in summary_rows]]
    axes[0].bar(x - w/2, e_means, w, yerr=e_err, capsize=4, color=EDGE_COLOR,
                label="Edge / corner", edgecolor="white")
    axes[0].bar(x + w/2, i_means, w, yerr=i_err, capsize=4, color=INT_COLOR,
                label="Interior", edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels([CONFIG_LABELS.get(c,c) for c in args.configs])
    axes[0].set_ylabel("Mean squared reconstruction error")
    axes[0].set_title("Reconstruction error: vertebral edges vs interior")
    axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=0.3); axes[0].set_axisbelow(True)

    # example residual heatmap with landmarks
    if example:
        axes[1].imshow(example["resid"], cmap="magma")
        axes[1].scatter(example["lm_px"][:, 0], example["lm_px"][:, 1],
                        s=10, facecolors="none", edgecolors="cyan", linewidths=0.8)
        axes[1].set_title(f"|residual| at {CONFIG_LABELS.get(example['cfg'],'')} "
                          f"with landmarks\n{example['uuid'][:24]}", fontsize=9)
        axes[1].axis("off")
    fig.suptitle("Does the lost high-frequency content land on vertebral edges?", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out_dir / "edge_interior.png", dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out_dir/'edge_interior.png'}")
    print(f"Saved {args.out_dir/'edge_interior_summary.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
