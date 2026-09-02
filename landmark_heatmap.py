"""
landmark_heatmap.py — a stronger (heatmap-regression) landmark detector.

The frontier used DIRECT coordinate regression (136 outputs). The standard,
usually-stronger approach is HEATMAP regression: predict one 2-D heatmap per
landmark and take its soft-argmax as the coordinate. This trains that detector
on the frozen MedVAE latents at 4x and 8x, so we can answer two review-relevant
questions:

  1. quality — is a proper detector meaningfully better than the direct head?
  2. mechanism — does a *good* detector still resist compression (4x vs 8x),
     or does it, being finer, start to degrade? (Either way is informative:
     robust => strengthens "landmark pathway is compression-tolerant";
     degrades => the coarse head was robust *because* it was coarse.)

Predictions are saved in the shared format (uuids, pred_landmarks, gt_landmarks;
(N,136) ordering A normalised [0,1]) so landmark_pixel_audit.py can convert them
to pixels/mm and audit them the same way.

Run:
  python landmark_heatmap.py --latents-dir ./latents_aspect --out-dir ./outputs/heatmap --seeds 0 1 2
  python landmark_heatmap.py --smoke
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CONFIGS = ["medvae_4_1_2d", "medvae_8_1_2d"]
LABEL = {"medvae_4_1_2d": "4x", "medvae_8_1_2d": "8x"}
N_LM = 68
R = 96          # common heatmap resolution
BETA = 30.0     # soft-argmax sharpness (higher = sharper; set from CLI)
DIRECT_REF = {"medvae_4_1_2d": 0.0418, "medvae_8_1_2d": 0.0418}  # direct-regressor norm error


# ==========================================================================
# model
# ==========================================================================
class HeatmapNet(nn.Module):
    """latent (B,C,h,w) -> (B,68,R,R) heatmaps. Adaptive to 96 or 48 input."""
    def __init__(self, in_ch=1, k=N_LM, res=R):
        super().__init__()
        self.res = res
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, k, 1),
        )

    def forward(self, x):
        f = self.body(x)
        f = F.interpolate(f, size=(self.res, self.res), mode="bilinear", align_corners=False)
        return self.head(f)                                   # (B,k,R,R)


def soft_argmax(hm, beta=None):
    """(B,k,R,R) -> (B,k,2) normalised (x,y) in [0,1]."""
    if beta is None:
        beta = BETA
    B, K, Rr, _ = hm.shape
    p = torch.softmax(hm.reshape(B, K, -1) * beta, dim=-1).reshape(B, K, Rr, Rr)
    coord = torch.linspace(0, 1, Rr, device=hm.device)
    x = (p.sum(2) * coord).sum(-1)                            # weight columns
    y = (p.sum(3) * coord).sum(-1)                            # weight rows
    return torch.stack([x, y], dim=-1)


def coords_to_rowA(coords):
    """(B,68,2) -> (B,136) ordering A (68 xs then 68 ys)."""
    return torch.cat([coords[..., 0], coords[..., 1]], dim=-1)


def gaussian_targets(lm136, res=R, sigma=2.0, device="cpu"):
    """(B,136) normalised -> (B,68,res,res) Gaussian heatmaps."""
    B = lm136.shape[0]
    xs = lm136[:, :N_LM] * (res - 1)
    ys = lm136[:, N_LM:] * (res - 1)
    gy, gx = torch.meshgrid(torch.arange(res, device=device).float(),
                            torch.arange(res, device=device).float(), indexing="ij")
    gx = gx[None, None]; gy = gy[None, None]                  # (1,1,res,res)
    cx = xs[:, :, None, None]; cy = ys[:, :, None, None]      # (B,68,1,1)
    hm = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    return hm


# ==========================================================================
# data / train
# ==========================================================================
class LatentLM(Dataset):
    def __init__(self, latents, lm):
        self.x = torch.as_tensor(latents, dtype=torch.float32)
        self.y = torch.as_tensor(lm, dtype=torch.float32)

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


def norm_err(pred136, gt136):
    dx = pred136[:, :N_LM] - gt136[:, :N_LM]
    dy = pred136[:, N_LM:] - gt136[:, N_LM:]
    return np.sqrt(dx ** 2 + dy ** 2).mean(1)                 # (N,)


def load_split(latents_dir, config, split):
    d = np.load(Path(latents_dir) / f"latents_{config}_{split}.npz", allow_pickle=True)
    return d["latents"].astype(np.float32), d["landmarks"].astype(np.float32), d["uuids"]


def train_one(tr, va, in_ch, *, seed, epochs, bs, lr, wd, patience, sigma, device, aug=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(LatentLM(*tr[:2]), batch_size=bs, shuffle=True)
    xva = torch.as_tensor(va[0], dtype=torch.float32, device=device)
    yva = va[1]
    model = HeatmapNet(in_ch=in_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if aug:
                xb = xb + aug * torch.randn_like(xb)      # latent-noise augmentation
            hm = model(xb)
            tgt = gaussian_targets(yb, sigma=sigma, device=device)
            loss = F.mse_loss(hm, tgt) + 0.1 * F.l1_loss(coords_to_rowA(soft_argmax(hm)), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = coords_to_rowA(soft_argmax(model(xva))).cpu().numpy()
        m = norm_err(pv, yva).mean()
        if m < best - 1e-6:
            best, best_state, bad = m, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def predict(model, latents, device):
    x = torch.as_tensor(latents, dtype=torch.float32, device=device)
    with torch.no_grad():
        return coords_to_rowA(soft_argmax(model(x))).cpu().numpy()


def boot_ci(v, seed=0, nb=10000):
    rng = np.random.default_rng(seed); v = np.asarray(v); n = len(v)
    b = np.array([v[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ==========================================================================
def run(latents_dir, out_dir, seeds, device, **kw):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for config in CONFIGS:
        tr = load_split(latents_dir, config, "train")
        va = load_split(latents_dir, config, "val")
        te = load_split(latents_dir, config, "test")
        in_ch = tr[0].shape[1]
        seed_preds, seed_train = [], []
        for s in seeds:
            m = train_one(tr, va, in_ch, seed=s, device=device, **kw)
            seed_preds.append(predict(m, te[0], device))
            seed_train.append(predict(m, tr[0], device))
        ens = np.mean(seed_preds, axis=0)
        errs = norm_err(ens, te[1])
        train_err = norm_err(np.mean(seed_train, axis=0), tr[1]).mean()
        results[config] = errs
        mean, lo, hi = boot_ci(errs)
        gap = "OVERFIT" if mean > 1.5 * train_err else ("UNDERFIT" if train_err > 0.06 else "ok")
        print(f"[{LABEL[config]}] heatmap norm error: test={mean:.4f} "
              f"CI[{lo:.4f},{hi:.4f}]  train={train_err:.4f}  [{gap}]   "
              f"(direct={DIRECT_REF[config]:.4f})")
        np.savez_compressed(out_dir / f"preds_heatmap_{config}.npz",
                            pred_landmarks=ens.astype(np.float32),
                            gt_landmarks=te[1].astype(np.float32), uuids=te[2])

    from scipy import stats
    a, b = results[CONFIGS[0]], results[CONFIGS[1]]
    d = b - a
    _, pt = stats.ttest_rel(b, a)
    m, lo, hi = boot_ci(d)
    print(f"\nheatmap detector 4x->8x: Δ={m:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
          f"rel={100*m/a.mean():+.1f}% paired p={pt:.3g}")
    print("compare to direct regressor (flat, p~0.99): does the stronger detector "
          "also resist compression, or does it degrade?")


def smoke():
    device = "cpu"
    rng = np.random.default_rng(0)
    # tiny synthetic: landmark position weakly encoded in latent mean
    def synth(H, n):
        lat = rng.standard_normal((n, 1, H, H)).astype(np.float32)
        base = lat.reshape(n, -1).mean(1)
        lm = np.clip(0.5 + 0.1 * base[:, None] + rng.normal(0, 0.03, (n, 136)), 0.05, 0.95)
        return lat, lm.astype(np.float32), np.array([f"i{k}" for k in range(n)])
    tr, va, te = synth(48, 40), synth(48, 12), synth(48, 12)
    m = train_one(tr, va, 1, seed=0, epochs=3, bs=16, lr=1e-3, wd=1e-4,
                  patience=5, sigma=2.0, device=device)
    pv = predict(m, te[0], device)
    e = norm_err(pv, te[1]).mean()
    hm = m(torch.tensor(te[0][:2]))
    ca = soft_argmax(hm)
    ok = hm.shape == (2, N_LM, R, R) and ca.shape == (2, N_LM, 2) and np.isfinite(e)
    print(f"heatmap shape {tuple(hm.shape)}, coords {tuple(ca.shape)}, test err {e:.4f}")
    print("SMOKE:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/heatmap"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian target sigma (heatmap px)")
    ap.add_argument("--beta", type=float, default=30.0, help="soft-argmax sharpness")
    ap.add_argument("--aug", type=float, default=0.0, help="latent-noise augmentation std (e.g. 0.1)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global BETA
    BETA = args.beta
    if args.smoke:
        smoke()
    elif args.latents_dir:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run(args.latents_dir, args.out_dir, args.seeds, device,
            epochs=args.epochs, bs=args.batch_size, lr=args.lr,
            wd=args.weight_decay, patience=args.patience, sigma=args.sigma, aug=args.aug)
    else:
        raise SystemExit("--latents-dir required (or --smoke)")


if __name__ == "__main__":
    main()
