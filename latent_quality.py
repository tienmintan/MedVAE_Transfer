"""
latent_quality.py — is the MedVAE latent a suitable substrate for diffusion?

WHY. Landmark-conditioned diffusion in MedVAE latent space produces realistic
samples but follows its conditioning only weakly (r ~ 0.3). One explanation is
that MedVAE is the wrong kind of autoencoder for this: standard latent
diffusion uses a VAE trained WITH a KL penalty precisely so the latent is
smooth and roughly Gaussian. MedVAE is a COMPRESSIVE, task-agnostic encoder
whose purpose is to discard information — nothing forces its latent to have the
properties diffusion assumes.

That is a hypothesis, and it has been asserted rather than measured. This
script measures it, against a PIXEL-SPACE reference so the numbers mean
something.

WHAT IS MEASURED

  1. GAUSSIANITY. Diffusion assumes the data is roughly standard-normal at
     t=T and interpolates towards it. Heavy tails or strong skew make the
     forward process a poor match for the data.

  2. SPATIAL CORRELATION. Diffusion adds i.i.d. noise. If neighbouring latent
     positions are very strongly correlated, i.i.d. noise is a mismatched
     corruption process.

  3. INTERPOLATION SMOOTHNESS — the most important test. Decode the midpoint
     of two real latents. In a smooth latent this yields a plausible
     intermediate spine; in a latent with holes it yields something incoherent.
     Diffusion trajectories pass through exactly this off-data region, so a
     latent that fails here is a bad substrate REGARDLESS of its statistics.

  4. LOCAL CONTINUITY. Perturb a latent slightly and measure how far the decode
     moves. A latent where tiny moves cause large image changes is hard to
     denoise stably.

  5. EFFECTIVE DIMENSIONALITY. How many PCA components carry the variance. A
     latent whose variance concentrates in very few directions is effectively
     lower-dimensional than its shape suggests, which wastes diffusion capacity.

INTERPRETATION. Every measure is reported alongside the same measure computed
on downsampled PIXELS. Pixels are a substrate that pixel-space diffusion is
known to handle, so "worse than pixels" is the meaningful comparison — not any
absolute threshold.

Run:
  python latent_quality.py --latents-dir ./latents_aspect --img-dir ./medvae/data/boostnet \\
      --config medvae_4_1_2d --out-dir ./outputs/latent_quality
  python latent_quality.py --smoke
"""
import argparse
from pathlib import Path

import numpy as np


# ==========================================================================
# statistics
# ==========================================================================
def moments(x):
    """Mean, sd, skew, excess kurtosis of a flat array."""
    x = np.asarray(x, dtype=np.float64).ravel()
    m, s = x.mean(), x.std()
    z = (x - m) / max(s, 1e-12)
    return dict(mean=float(m), sd=float(s),
                skew=float((z ** 3).mean()),
                kurtosis=float((z ** 4).mean() - 3.0))


def gaussianity_report(name, arr, per_channel=True):
    """Per-channel moments plus a normality statistic.

    Excess kurtosis 0 and skew 0 are Gaussian. Large positive kurtosis means
    heavy tails: rare extreme values the forward diffusion process will not
    reach naturally."""
    print(f"\n  {name}")
    C = arr.shape[1] if (per_channel and arr.ndim == 4) else 1
    rows = []
    for c in range(C):
        sub = arr[:, c] if C > 1 else arr
        m = moments(sub)
        rows.append(m)
        print(f"    ch{c}: mean {m['mean']:+8.3f}  sd {m['sd']:7.3f}  "
              f"skew {m['skew']:+6.3f}  excess kurt {m['kurtosis']:+8.3f}")
    return rows


def spatial_autocorr(arr, max_lag=5):
    """Correlation between a position and its neighbour at distance `lag`,
    averaged over the batch. High values at lag 1 mean the field is very
    smooth spatially."""
    x = arr[:, 0] if arr.ndim == 4 else arr
    x = x - x.mean(axis=(1, 2), keepdims=True)
    out = []
    for lag in range(1, max_lag + 1):
        a, b = x[:, :, :-lag], x[:, :, lag:]
        num = (a * b).mean()
        den = np.sqrt((a ** 2).mean() * (b ** 2).mean()) + 1e-12
        out.append(float(num / den))
    return out


def effective_dim(arr, thresh=0.95):
    """Number of PCA components needed to reach `thresh` of the variance,
    and the participation ratio (a smooth measure of effective rank)."""
    X = arr.reshape(len(arr), -1).astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    n = min(len(X), 400)
    X = X[:n]
    # eigenvalues of the (small) Gram matrix
    G = X @ X.T / max(n - 1, 1)
    w = np.linalg.eigvalsh(G)[::-1]
    w = np.clip(w, 0, None)
    tot = w.sum() + 1e-12
    cum = np.cumsum(w) / tot
    k = int(np.searchsorted(cum, thresh) + 1)
    pr = float((w.sum() ** 2) / ((w ** 2).sum() + 1e-12))
    return k, pr, len(w)


# ==========================================================================
def run(args):
    import torch
    from PIL import Image
    from medvae import MVAE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    d = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz",
                allow_pickle=True)
    lat, uu = d["latents"].astype(np.float32), d["uuids"]
    n = min(args.n, len(lat))
    lat = lat[:n]
    print(f"latents {lat.shape}")

    # ---- pixel-space reference at the same spatial size ----
    size = lat.shape[-1]
    pix = []
    for u in uu[:n]:
        base = str(u).split("__aug")[0]
        p = None
        for ext in (".jpg", ".jpeg", ".png"):
            q = Path(args.img_dir) / f"{base}{ext}"
            if q.exists():
                p = q; break
        if p is None:
            continue
        with Image.open(p) as im:
            im = im.convert("L")
            w, h = im.size
            s = size / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
            canvas = Image.new("L", (size, size), 0)
            canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
            a = np.asarray(canvas, dtype=np.float32) / 255.0
        pix.append(a[None])
    pix = np.stack(pix)
    pix = (pix - pix.mean()) / (pix.std() + 1e-8)          # standardise, as for latents
    print(f"pixel reference {pix.shape} (centre-padded, standardised)\n")

    lat_s = (lat - lat.mean()) / (lat.std() + 1e-8)

    print("=" * 70)
    print("1) GAUSSIANITY  (diffusion assumes ~N(0,1); skew 0, excess kurtosis 0)")
    print("=" * 70)
    lrows = gaussianity_report("MedVAE latent (standardised)", lat_s)
    prows = gaussianity_report("PIXEL reference (standardised)", pix)
    lk = np.mean([abs(r["kurtosis"]) for r in lrows])
    pk = np.mean([abs(r["kurtosis"]) for r in prows])
    ls = np.mean([abs(r["skew"]) for r in lrows])
    ps = np.mean([abs(r["skew"]) for r in prows])
    print(f"\n    mean |excess kurtosis|: latent {lk:.3f}  vs  pixels {pk:.3f}")
    print(f"    mean |skew|           : latent {ls:.3f}  vs  pixels {ps:.3f}")
    print(f"    -> latent is {'WORSE' if lk > pk else 'better'} on tails, "
          f"{'WORSE' if ls > ps else 'better'} on skew")

    print("\n" + "=" * 70)
    print("2) SPATIAL CORRELATION  (diffusion adds i.i.d. noise)")
    print("=" * 70)
    la = spatial_autocorr(lat_s); pa = spatial_autocorr(pix)
    print(f"    lag:        " + "".join(f"{i+1:>8}" for i in range(len(la))))
    print(f"    latent:     " + "".join(f"{v:>8.3f}" for v in la))
    print(f"    pixels:     " + "".join(f"{v:>8.3f}" for v in pa))
    print(f"    -> lag-1 correlation {la[0]:.3f} (latent) vs {pa[0]:.3f} (pixels)")

    print("\n" + "=" * 70)
    print("3) INTERPOLATION SMOOTHNESS  (the most important test)")
    print("=" * 70)
    mvae = MVAE(model_name=args.config, modality="xray").to(device).eval()

    def dec(z):
        with torch.no_grad():
            t = torch.tensor(z, dtype=torch.float32, device=device)
            for obj, nm in ((mvae.model, "decode"), (mvae, "decode")):
                fn = getattr(obj, nm, None)
                if fn is None:
                    continue
                try:
                    o = fn(t)
                    if isinstance(o, (tuple, list)):
                        o = o[0]
                    if torch.is_tensor(o):
                        return o.float().cpu().numpy()
                except Exception:
                    continue
        raise RuntimeError("decode failed")

    rng = np.random.default_rng(0)
    pairs = rng.choice(len(lat), size=(args.n_interp, 2), replace=True)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    # A smooth latent: decode(mix of latents) ~ mix of decode(latents).
    # Large deviation means the midpoint falls into a region the decoder does
    # not map coherently — a "hole" the diffusion trajectory must cross.
    devs, sharp = [], []
    for i, j in pairs:
        za, zb = lat[i:i + 1], lat[j:j + 1]
        zmid = 0.5 * (za + zb)
        da, db, dmid = dec(za)[0], dec(zb)[0], dec(zmid)[0]
        lin = 0.5 * (da + db)
        dev = np.abs(dmid - lin).mean() / (np.abs(lin).mean() + 1e-8)
        devs.append(float(dev))
        # a coherent spine has more local structure than a blurry average
        gx = np.abs(np.diff(dmid.mean(0), axis=1)).mean()
        gl = np.abs(np.diff(lin.mean(0), axis=1)).mean()
        sharp.append(float(gx / (gl + 1e-8)))
    print(f"    |decode(mid) - mid(decode)| / |mid(decode)| : "
          f"median {np.median(devs):.4f}")
    print(f"    gradient energy ratio, decoded-mid vs averaged: "
          f"median {np.median(sharp):.3f}")
    print("      (~1.0 = the midpoint decodes no sharper than a blurry average,")
    print("       i.e. the latent midpoint is NOT a coherent new image;")
    print("       >1.2 = the decoder produces genuine structure at the midpoint)")

    print("\n" + "=" * 70)
    print("4) LOCAL CONTINUITY  (how far does the decode move for a small step?)")
    print("=" * 70)
    ratios = []
    for i in rng.choice(len(lat), size=min(16, len(lat)), replace=False):
        z = lat[i:i + 1]
        base = dec(z)[0]
        for eps in (0.05, 0.10, 0.25):
            zp = z + eps * lat.std() * rng.standard_normal(z.shape).astype(np.float32)
            dp = dec(zp)[0]
            dz = np.abs(zp - z).mean() / (np.abs(z).mean() + 1e-8)
            di = np.abs(dp - base).mean() / (np.abs(base).mean() + 1e-8)
            ratios.append((eps, di / (dz + 1e-8)))
    for eps in (0.05, 0.10, 0.25):
        v = [r for e, r in ratios if e == eps]
        print(f"    perturbation {eps:.2f} sd -> image change / latent change = "
              f"{np.median(v):.2f}")
    print("      (a large amplification means small latent errors become large")
    print("       image errors, which makes stable denoising harder)")

    print("\n" + "=" * 70)
    print("5) EFFECTIVE DIMENSIONALITY")
    print("=" * 70)
    kl, prl, nl = effective_dim(lat_s)
    kp, prp, npx = effective_dim(pix)
    print(f"    latent: {kl}/{nl} components for 95% variance, "
          f"participation ratio {prl:.1f}")
    print(f"    pixels: {kp}/{npx} components for 95% variance, "
          f"participation ratio {prp:.1f}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    flags = []
    if lk > 2 * pk and lk > 1.0:
        flags.append("heavy tails vs pixels -> per-channel whitening may help")
    if np.median(sharp) < 1.05:
        flags.append("latent midpoints decode to blurry averages, not coherent "
                     "images -> the latent is NOT smooth in the sense diffusion needs")
    if la[0] > 0.95:
        flags.append("very high lag-1 spatial correlation -> i.i.d. noise is a "
                     "mismatched corruption process")
    if flags:
        for f in flags:
            print(f"  * {f}")
        print("\n  These support the hypothesis that MedVAE's compressive latent is a")
        print("  poor substrate for diffusion, and motivate a pixel-space comparison.")
    else:
        print("  No strong pathology found relative to pixels. The latent is NOT")
        print("  obviously the problem — the conditioning weakness likely lies in the")
        print("  architecture (concatenation vs cross-attention) instead.")
        print("  Do not claim the latent is at fault without evidence.")


# ==========================================================================
def smoke():
    rng = np.random.default_rng(0)
    print("1) moments on known distributions")
    g = rng.standard_normal(50000)
    m = moments(g)
    print(f"   gaussian     : skew {m['skew']:+.3f} kurt {m['kurtosis']:+.3f} "
          f"(expect ~0, ~0)")
    t = rng.standard_t(3, 50000)
    m = moments(t)
    print(f"   heavy-tailed : skew {m['skew']:+.3f} kurt {m['kurtosis']:+.3f} "
          f"(expect large positive kurt)")
    e = rng.exponential(1.0, 50000)
    m = moments(e)
    print(f"   skewed       : skew {m['skew']:+.3f} kurt {m['kurtosis']:+.3f} "
          f"(expect skew ~2)")

    print("\n2) spatial autocorrelation")
    white = rng.standard_normal((32, 1, 48, 48)).astype(np.float32)
    aw = spatial_autocorr(white)
    from numpy.lib.stride_tricks import sliding_window_view
    sm = np.stack([sliding_window_view(w[0], (5, 5)).mean((-1, -2))
                   for w in white])[:, None]
    as_ = spatial_autocorr(sm)
    print(f"   white noise : lag1 {aw[0]:+.3f} (expect ~0)")
    print(f"   smoothed    : lag1 {as_[0]:+.3f} (expect high)")

    print("\n3) effective dimensionality")
    lowrank = (rng.standard_normal((200, 5)) @ rng.standard_normal((5, 400)))
    full = rng.standard_normal((200, 400))
    kl, prl, _ = effective_dim(lowrank.reshape(200, 1, 20, 20))
    kf, prf, _ = effective_dim(full.reshape(200, 1, 20, 20))
    print(f"   rank-5 data : {kl} comps for 95%, participation ratio {prl:.1f} "
          f"(expect ~5)")
    print(f"   full-rank   : {kf} comps for 95%, participation ratio {prf:.1f} "
          f"(expect large)")

    ok = (abs(moments(g)["kurtosis"]) < 0.2 and moments(t)["kurtosis"] > 2
          and moments(e)["skew"] > 1.5 and abs(aw[0]) < 0.1 and as_[0] > 0.5
          and prl < 10 and prf > 50)
    print("\nSMOKE:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path)
    ap.add_argument("--img-dir", type=Path)
    ap.add_argument("--config", default="medvae_4_1_2d")
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/latent_quality"))
    ap.add_argument("--n", type=int, default=256, help="latents to analyse")
    ap.add_argument("--n-interp", type=int, default=24, help="interpolation pairs")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.latents_dir and args.img_dir:
        run(args)
    else:
        raise SystemExit("need --latents-dir and --img-dir (or --smoke)")


if __name__ == "__main__":
    main()
