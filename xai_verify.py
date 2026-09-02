"""
xai_verify.py — Direction 3: XAI as mechanism verification.

The compression arc concluded that the DIRECT Cobb regressor reads fine endplate
detail (high-frequency, endplate-localised) that compression strips. That gives
a falsifiable saliency prediction:

  (a) the Cobb regressor's saliency should CONCENTRATE on the endplate regions
      (the Step-2 bands), above chance; and
  (b) that concentration should be LOWER / more diffuse at 8x than at 4x, because
      the fine endplate cues are degraded.

This loads the saved Cobb models (train_frontier.py --save-models), computes
ensemble gradient saliency over the latent for each test image, and measures how
much saliency lands on the endplate band vs the body centroids — the same
region decomposition as the Step-2 reconstruction analysis. Both configs are
evaluated on a common 96x96 grid so 4x and 8x are directly comparable.

Metrics per image:
  ep_frac   fraction of total saliency inside the endplate band
  ep_over_uniform   ep_frac / (endplate area fraction)   [>1 = above chance]
  ep_ct_ratio       mean saliency on endplates / mean on centroids

Tests: ep_frac vs chance (within config); 4x-vs-8x paired change (across configs).

Inputs:
  --models-dir : dir with model_{config}_cobb_seed*.pt  (from --save-models)
  --latents-dir: dir with latents_{config}_test.npz  (latents + landmarks)

Run:
  python xai_verify.py --models-dir ./outputs/frontier --latents-dir ./latents_aspect \
      --out-dir ./outputs/xai
  python xai_verify.py --smoke      # toy-model self-test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_frontier import TaskModel   # shared backbone/head

CONFIGS = ["medvae_4_1_2d", "medvae_8_1_2d"]
LABEL = {"medvae_4_1_2d": "4x", "medvae_8_1_2d": "8x"}
N_LM = 68
N_VERT = 17
GRID = 96          # common evaluation grid
EPS = 1e-8


# ==========================================================================
# models
# ==========================================================================
def load_models(models_dir, config, device):
    ckpts = sorted(Path(models_dir).glob(f"model_{config}_cobb_seed*.pt"))
    if not ckpts:
        raise SystemExit(f"no Cobb models for {config} in {models_dir} "
                         f"(did you run train_frontier.py --save-models?)")
    models = []
    for cp in ckpts:
        d = torch.load(cp, map_location=device, weights_only=False)
        m = TaskModel(d["out_dim"], in_ch=d["in_ch"]).to(device)
        m.load_state_dict(d["state_dict"]); m.eval()
        models.append(m)
    return models


# ==========================================================================
# saliency  (gradient of the main-thoracic output w.r.t. the latent)
# ==========================================================================
def saliency(models, latent, device, target=1, smooth=0, noise=0.15):
    """latent: (C,H,W) numpy. Returns (H,W) ensemble mean |gradient| saliency.
    target: output index to attribute (1 = MT). smooth>0 -> SmoothGrad passes."""
    x0 = torch.tensor(latent[None], dtype=torch.float32, device=device)  # (1,C,H,W)
    passes = max(1, smooth)
    sigma = noise * float(x0.std()) if smooth else 0.0
    acc = None
    for m in models:
        for _ in range(passes):
            x = x0 + (torch.randn_like(x0) * sigma if smooth else 0.0)
            x = x.clone().requires_grad_(True)
            out = m(x)
            tgt = min(target, out.shape[1] - 1)
            m.zero_grad(set_to_none=True)
            out[0, tgt].backward()
            g = x.grad.detach().abs().mean(1)[0]           # (H,W) over channels
            acc = g if acc is None else acc + g
    return (acc / (len(models) * passes)).cpu().numpy()


def to_grid(sal, size=GRID):
    arr = np.asarray(sal)
    if arr.ndim == 3:          # (C,H,W) -> take first channel
        arr = arr[0]
    t = torch.tensor(arr[None, None], dtype=torch.float32)
    return F.interpolate(t, size=(size, size), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def grid_at(sal, low, size=GRID):
    """Route a native saliency map through a `low`x`low` bottleneck, then up to
    `size`. With low = the coarsest config's latent resolution, this applies the
    SAME low-pass to every config, removing the upsampling confound in the
    4x-vs-8x concentration comparison."""
    arr = np.asarray(sal)
    if arr.ndim == 3:
        arr = arr[0]
    if low < arr.shape[0]:
        arr = to_grid(arr, low)
    return to_grid(arr, size)


# ==========================================================================
# anatomy masks  (endplate bands + centroid disks, on the common grid)
# ==========================================================================
def content_bbox(latent_grid, freq=0.15):
    a = np.abs(latent_grid)
    m = a > (freq * a.max() + EPS)
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, latent_grid.shape[0], 0, latent_grid.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def landmarks_to_grid(lm136, bbox, size=GRID):
    r0, r1, c0, c1 = bbox
    x = c0 + lm136[:N_LM] * (c1 - c0)
    y = r0 + lm136[N_LM:] * (r1 - r0)
    return np.stack([x, y], 1).reshape(N_VERT, 4, 2)    # (17,4,2) grid coords


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip((P - a) @ ab / (ab @ ab + EPS), 0, 1)
    proj = a + t[:, None] * ab
    return np.linalg.norm(P - proj, axis=1)


def anatomy_masks(V, size=GRID, band=1.6, disk=1.6):
    """V: (17,4,2) grid coords. Corners: 0,1 = top edge; 2,3 = bottom edge.
    Returns (endplate_mask, centroid_mask) boolean (size,size)."""
    yy, xx = np.mgrid[0:size, 0:size]
    P = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    ep = np.zeros(P.shape[0], bool)
    ct = np.zeros(P.shape[0], bool)
    for v in V:
        ep |= _seg_dist(P, v[0], v[1]) < band      # top endplate
        ep |= _seg_dist(P, v[2], v[3]) < band      # bottom endplate
        ct |= np.linalg.norm(P - v.mean(0), axis=1) < disk
    ep &= ~ct                                       # keep regions disjoint
    return ep.reshape(size, size), ct.reshape(size, size)


def concentration(sal, ep, ct):
    total = sal.sum() + EPS
    ep_frac = sal[ep].sum() / total
    ep_area = ep.mean() + EPS
    ep_mean = sal[ep].mean() if ep.any() else 0.0
    ct_mean = sal[ct].mean() if ct.any() else EPS
    return {"ep_frac": ep_frac,
            "ep_over_uniform": ep_frac / ep_area,
            "ep_ct_ratio": ep_mean / (ct_mean + EPS)}


# ==========================================================================
# stats
# ==========================================================================
def boot_ci(v, seed=0, nb=10000):
    rng = np.random.default_rng(seed); v = np.asarray(v); n = len(v)
    b = np.array([v[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(a, b):
    d = np.asarray(b) - np.asarray(a)
    t, pt = stats.ttest_rel(b, a)
    m, lo, hi = boot_ci(d)
    return {"delta": m, "lo": lo, "hi": hi, "p": pt,
            "rel": 100 * m / (np.mean(a) + EPS)}


# ==========================================================================
# main analysis
# ==========================================================================
def analyse(models_dir, latents_dir, out_dir, device, smooth=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_cfg = {"raw": {}, "matched": {}}
    examples = {}

    # coarsest native latent resolution across configs -> shared low-pass
    sizes = [np.load(Path(latents_dir) / f"latents_{c}_test.npz")["latents"].shape[-1]
             for c in CONFIGS]
    LOW = min(sizes)
    print(f"native latent sizes {dict(zip([LABEL[c] for c in CONFIGS], sizes))}; "
          f"resolution-matched low-pass at {LOW}x{LOW}\n")

    for config in CONFIGS:
        d = np.load(Path(latents_dir) / f"latents_{config}_test.npz", allow_pickle=True)
        latents, lms, uuids = d["latents"], d["landmarks"], d["uuids"]
        models = load_models(models_dir, config, device)

        mr = {k: [] for k in ("ep_frac", "ep_over_uniform", "ep_ct_ratio")}
        mm = {k: [] for k in ("ep_frac", "ep_over_uniform", "ep_ct_ratio")}
        for i in range(len(latents)):
            nat = saliency(models, latents[i], device, smooth=smooth)
            lg = to_grid(latents[i][0])
            V = landmarks_to_grid(lms[i], content_bbox(lg))
            ep, ct = anatomy_masks(V)
            c_raw = concentration(to_grid(nat), ep, ct)              # native->96
            c_mat = concentration(grid_at(nat, LOW), ep, ct)         # via LOW bottleneck
            for k in mr:
                mr[k].append(c_raw[k]); mm[k].append(c_mat[k])
            rows.append({"config": config, "image_uuid": str(uuids[i]),
                         **{f"{k}_raw": c_raw[k] for k in mr},
                         **{f"{k}_matched": c_mat[k] for k in mm}})
            if i < 3:
                examples[(config, i)] = (to_grid(nat), ep)
        per_cfg["raw"][config] = {k: np.array(v) for k, v in mr.items()}
        per_cfg["matched"][config] = {k: np.array(v) for k, v in mm.items()}
        m, lo, hi = boot_ci(per_cfg["raw"][config]["ep_over_uniform"])
        print(f"[{LABEL[config]}] endplate saliency vs chance (raw): "
              f"{m:.2f}  CI[{lo:.2f},{hi:.2f}]   (>1 = concentrated on the spine)")

    def report(tag):
        print(f"\nsaliency endplate-concentration, 4x -> 8x [{tag}]:")
        hdr = f"{'metric':<18}{'4x':>9}{'8x':>9}{'Δ':>9}{'rel%':>8}{'paired p':>11}"
        print(hdr); print("-" * len(hdr))
        for k in ("ep_frac", "ep_over_uniform", "ep_ct_ratio"):
            a, b = per_cfg[tag][CONFIGS[0]][k], per_cfg[tag][CONFIGS[1]][k]
            m4, _, _ = boot_ci(a); m8, _, _ = boot_ci(b); pr = paired(a, b)
            print(f"{k:<18}{m4:>9.3f}{m8:>9.3f}{pr['delta']:>9.3f}"
                  f"{pr['rel']:>7.1f}%{pr['p']:>11.3g}")

    report("raw")
    report("matched")
    print("\nThe RAW 4x->8x drop is confounded by upsampling (8x saliency is "
          "48->96). The MATCHED table low-passes both configs to "
          f"{LOW}x{LOW} first — trust that one for the compression claim. "
          "The chance test (>1) is unaffected by this and stands either way.")

    pd.DataFrame(rows).to_csv(out_dir / "xai_per_image.csv", index=False)
    make_figure(examples, out_dir / "xai_saliency_examples.png")
    print(f"\nwrote {out_dir}/xai_per_image.csv and xai_saliency_examples.png")


def make_figure(examples, path):
    n = 3
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for r, config in enumerate(CONFIGS):
        for i in range(n):
            ax = axes[r, i]
            if (config, i) not in examples:
                ax.axis("off"); continue
            sal, ep = examples[(config, i)]
            ax.imshow(sal, cmap="magma")
            ax.contour(ep, levels=[0.5], colors="cyan", linewidths=0.8)
            ax.set_title(f"{LABEL[config]} · example {i}", fontsize=9)
            ax.axis("off")
    fig.suptitle("Cobb-regressor saliency (heat) vs endplate regions (cyan)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


# ==========================================================================
# smoke: toy model whose output depends on a known latent stripe
# ==========================================================================
def smoke():
    torch.manual_seed(0); np.random.seed(0)
    device = "cpu"
    m = TaskModel(3, in_ch=1).to(device).eval()
    lat = np.random.randn(1, 48, 48).astype(np.float32)
    sal = saliency([m], lat, device)
    salg = to_grid(sal)
    lm = np.clip(0.5 + 0.1 * np.random.randn(136), 0.05, 0.95).astype(np.float32)
    V = landmarks_to_grid(lm, content_bbox(to_grid(lat)))
    ep, ct = anatomy_masks(V)
    c = concentration(salg, ep, ct)
    ok = (salg.shape == (GRID, GRID) and ep.any() and ct.any()
          and np.isfinite(list(c.values())).all())
    print(f"saliency grid {salg.shape}, endplate px={ep.sum()}, centroid px={ct.sum()}")
    print(f"concentration keys: {c}")
    print("SMOKE:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--latents-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/xai"))
    ap.add_argument("--smooth", type=int, default=0,
                    help="SmoothGrad passes (0=vanilla gradient).")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.models_dir and args.latents_dir:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        analyse(args.models_dir, args.latents_dir, args.out_dir, device, smooth=args.smooth)
    else:
        raise SystemExit("need --models-dir and --latents-dir (or --smoke)")


if __name__ == "__main__":
    main()
