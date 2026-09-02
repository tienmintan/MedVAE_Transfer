"""
freq_loss_endplate.py — Step 2 (rebuilt): endplate bands vs body centroids.

The first attempt used disks around all 68 corners; at the needed radius they
merged into whole-vertebra blobs, so "edge" and "interior" measured nearly the
same mixed region and the effect washed out.

This version isolates the structure the mechanism is actually about. Corner
ordering is confirmed: grouped by vertebra, [TL, TR, BL, BR]. So per vertebra:
  superior endplate = segment corner0 -> corner1
  inferior endplate = segment corner2 -> corner3
  body centroid     = mean of the 4 corners

ENDPLATE band  = thin strip (half-width band_hw px) along the 2 endplate lines.
CENTROID disk  = small disk (radius cent_r px) at the body centre.
These don't overlap, and each isolates its target: endplates are the sharp,
oriented, high-frequency edges that define the Cobb angle; centroids sit in the
smooth, low-frequency vertebral-body interior.

Prediction: endplate error > centroid error, and the gap widens with
compression. That would localise the Step-1 high-frequency loss onto the
Cobb-critical structures.

--verify shows the regions over |residual| first. Run it before the full pass.

Usage (verify):
  python freq_loss_endplate.py --verify \\
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
ENDPLATE_COLOR = "#B85042"
CENTROID_COLOR = "#1C7293"


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
        return None, None
    return (rows[0].item(), rows[-1].item() + 1,
            cols[0].item(), cols[-1].item() + 1), m


def parse_landmark_row(row_vals):
    xs = np.asarray(row_vals[:68], dtype=float)
    ys = np.asarray(row_vals[68:], dtype=float)
    return np.stack([xs, ys], axis=1)


def map_landmarks(lm_norm, bbox):
    r0, r1, c0, c1 = bbox
    xs = c0 + lm_norm[:, 0] * (c1 - c0)
    ys = r0 + lm_norm[:, 1] * (r1 - r0)
    return np.stack([xs, ys], axis=1)   # (68,2) px (x,y)


def endplate_centroid_masks(lm_px, shape, content_mask, band_hw, cent_r):
    """Confirmed ordering per vertebra: [TL, TR, BL, BR].
    superior endplate = pts[0]->pts[1], inferior = pts[2]->pts[3]."""
    H, W = shape
    ep_seeds = np.ones((H, W), dtype=bool)
    ct_seeds = np.ones((H, W), dtype=bool)

    for v in range(17):
        quad = lm_px[v * 4:(v + 1) * 4]   # TL, TR, BL, BR
        tl, tr, bl, br = quad
        for pa, pb in [(tl, tr), (bl, br)]:        # superior, inferior endplates
            length = int(max(2, np.hypot(pb[0] - pa[0], pb[1] - pa[1])))
            for t in np.linspace(0, 1, length):
                x = int(round(pa[0] + t * (pb[0] - pa[0])))
                y = int(round(pa[1] + t * (pb[1] - pa[1])))
                if 0 <= y < H and 0 <= x < W:
                    ep_seeds[y, x] = False
        cx, cy = quad[:, 0].mean(), quad[:, 1].mean()
        xi, yi = int(round(cx)), int(round(cy))
        if 0 <= yi < H and 0 <= xi < W:
            ct_seeds[yi, xi] = False

    ep_dist = distance_transform_edt(ep_seeds)
    ct_dist = distance_transform_edt(ct_seeds)
    endplate = (ep_dist <= band_hw) & content_mask
    centroid = (ct_dist <= cent_r) & content_mask & (~endplate)
    return endplate, centroid


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
    ap.add_argument("--lookup", required=True, type=Path)
    ap.add_argument("--landmarks", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", default=["medvae_4_1_2d", "medvae_8_1_2d"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--modality", default="xray")
    ap.add_argument("--band-hw", type=float, default=2.5, help="endplate band half-width px")
    ap.add_argument("--cent-r", type=float, default=4.0, help="centroid disk radius px")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    lookup = pd.read_csv(args.lookup).set_index("image_uuid")
    landmarks_all = pd.read_csv(args.landmarks, header=None).values.astype(float)
    uuids = [u for u in df["image_uuid"] if u in lookup.index]
    if args.limit:
        uuids = uuids[:args.limit]
    print(f"{len(uuids)}/{len(df)} test images have a landmark row.")

    # ---------- VERIFY ----------
    if args.verify:
        cfg = args.configs[0]
        mvae = MVAE(model_name=cfg, modality=args.modality).to(device).eval()
        mvae.requires_grad_(False)
        n = min(4, len(uuids))
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 7))
        if n == 1:
            axes = [axes]
        for ax, uuid in zip(axes, uuids[:n]):
            x = transform_image(args.img_dir / f"{uuid}.jpg").to(device)
            x_hat = reconstruct(mvae, x)
            bbox, cmask = content_bbox(x)
            lm = parse_landmark_row(landmarks_all[int(lookup.loc[uuid, "landmark_row"])])
            lm_px = map_landmarks(lm, bbox)
            ep, ct = endplate_centroid_masks(lm_px, x.shape[-2:], cmask.cpu().numpy(),
                                             args.band_hw, args.cent_r)
            resid = np.abs((x - x_hat).squeeze().cpu().numpy())
            ax.imshow(resid, cmap="magma")
            ov = np.zeros((*ep.shape, 4))
            ov[ep] = [0.72, 0.31, 0.26, 0.9]
            ov[ct] = [0.11, 0.45, 0.58, 0.9]
            ax.imshow(ov)
            ax.set_title(f"{uuid[:18]}\nendplate(red) vs centroid(teal)", fontsize=8)
            ax.axis("off")
        fig.suptitle("Step 2 rebuilt — verify endplate bands & centroids on |residual|",
                     fontsize=11)
        out = args.out_dir / "endplate_verify.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"Saved {out}\nIf bands hug the endplates and teal sits in body centres, "
              f"re-run without --verify.")
        return 0

    # ---------- FULL ----------
    rec = {c: {"ep": [], "ct": [], "uuid": []} for c in args.configs}
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
            ep, ct = endplate_centroid_masks(lm_px, x.shape[-2:], cmask.cpu().numpy(),
                                             args.band_hw, args.cent_r)
            resid_sq = ((x - x_hat).squeeze().cpu().numpy()) ** 2
            if ep.sum() == 0 or ct.sum() == 0:
                continue
            rec[cfg]["ep"].append(resid_sq[ep].mean())
            rec[cfg]["ct"].append(resid_sq[ct].mean())
            rec[cfg]["uuid"].append(uuid)
            if cfg == args.configs[-1] and i == 1:
                example = {"resid": np.abs((x - x_hat).squeeze().cpu().numpy()),
                           "ep": ep, "ct": ct, "uuid": uuid, "cfg": cfg}
            if i % 20 == 0 or i == len(uuids):
                print(f"  [{i}/{len(uuids)}]")
        del mvae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- aggregate ----
    summary, per_image = [], []
    print("\n=== Endplate vs centroid reconstruction error (mean squared residual) ===")
    for cfg in args.configs:
        ep = np.array(rec[cfg]["ep"]); ct = np.array(rec[cfg]["ct"])
        ratio = ep / np.maximum(ct, 1e-12)
        em, elo, ehi = bootstrap_ci(ep)
        cm, clo, chi = bootstrap_ci(ct)
        rm, rlo, rhi = bootstrap_ci(ratio)
        print(f"  {CONFIG_LABELS.get(cfg,cfg):>3}: "
              f"endplate={em:.4f} [{elo:.4f},{ehi:.4f}]  "
              f"centroid={cm:.4f} [{clo:.4f},{chi:.4f}]  "
              f"ratio={rm:.2f} [{rlo:.2f},{rhi:.2f}]")
        summary.append({"config": cfg, "endplate_mean": em, "endplate_lo": elo,
                        "endplate_hi": ehi, "centroid_mean": cm, "centroid_lo": clo,
                        "centroid_hi": chi, "ratio_mean": rm, "ratio_lo": rlo,
                        "ratio_hi": rhi})
        for u, e, c in zip(rec[cfg]["uuid"], ep, ct):
            per_image.append({"config": cfg, "image_uuid": u,
                              "endplate_err": e, "centroid_err": c})

    pd.DataFrame(summary).to_csv(args.out_dir / "endplate_summary.csv", index=False)
    pd.DataFrame(per_image).to_csv(args.out_dir / "endplate_per_image.csv", index=False)

    # within-config: is endplate > centroid? (paired across images)
    print("\n--- Endplate minus centroid, per config (paired) ---")
    for cfg in args.configs:
        ep = np.array(rec[cfg]["ep"]); ct = np.array(rec[cfg]["ct"])
        d = ep - ct
        dm, dlo, dhi = bootstrap_ci(d)
        verdict = "endplate WORSE (as predicted)" if dlo > 0 else (
                  "no clear difference" if dlo <= 0 <= dhi else "endplate BETTER")
        print(f"  {CONFIG_LABELS.get(cfg,cfg):>3}: Δ(ep−ct)={dm:+.4f} "
              f"[{dlo:+.4f},{dhi:+.4f}]  -> {verdict}")

    # across-config: does the endplate/centroid ratio grow with compression?
    if len(args.configs) >= 2:
        c4, c8 = args.configs[0], args.configs[1]
        r4 = np.array(rec[c4]["ep"]) / np.maximum(np.array(rec[c4]["ct"]), 1e-12)
        r8 = np.array(rec[c8]["ep"]) / np.maximum(np.array(rec[c8]["ct"]), 1e-12)
        gm, glo, ghi = bootstrap_ci(r8 - r4)
        print(f"\nRatio growth (8x − 4x): {gm:+.2f} [{glo:+.2f},{ghi:+.2f}]")
        if glo > 0:
            print("  -> compression preferentially sacrifices ENDPLATE fidelity.")
        elif glo <= 0 <= ghi:
            print("  -> no clear compression-dependent change in the ratio.")
        else:
            print("  -> ratio DECREASES with compression (endplates relatively spared).")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(args.configs)); w = 0.36
    ep_m = [s["endplate_mean"] for s in summary]
    ep_e = [[s["endplate_mean"]-s["endplate_lo"] for s in summary],
            [s["endplate_hi"]-s["endplate_mean"] for s in summary]]
    ct_m = [s["centroid_mean"] for s in summary]
    ct_e = [[s["centroid_mean"]-s["centroid_lo"] for s in summary],
            [s["centroid_hi"]-s["centroid_mean"] for s in summary]]
    axes[0].bar(x - w/2, ep_m, w, yerr=ep_e, capsize=4, color=ENDPLATE_COLOR,
                label="Endplate band", edgecolor="white")
    axes[0].bar(x + w/2, ct_m, w, yerr=ct_e, capsize=4, color=CENTROID_COLOR,
                label="Body centroid", edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels([CONFIG_LABELS.get(c, c) for c in args.configs])
    axes[0].set_ylabel("Mean squared reconstruction error")
    axes[0].set_title("Endplate vs body-centroid reconstruction error")
    axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=0.3); axes[0].set_axisbelow(True)

    if example:
        axes[1].imshow(example["resid"], cmap="magma")
        ov = np.zeros((*example["ep"].shape, 4))
        ov[example["ep"]] = [0.72, 0.31, 0.26, 0.9]
        ov[example["ct"]] = [0.11, 0.45, 0.58, 0.9]
        axes[1].imshow(ov)
        axes[1].set_title(f"|residual| at {CONFIG_LABELS.get(example['cfg'],'')}, "
                          f"regions overlaid\n{example['uuid'][:24]}", fontsize=9)
        axes[1].axis("off")
    fig.suptitle("Does compression preferentially blur the vertebral endplates?", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out_dir / "endplate_vs_centroid.png", dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out_dir/'endplate_vs_centroid.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
