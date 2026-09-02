"""
recon_fidelity.py — the reconstruction-fidelity axis (PSNR / SSIM / LPIPS).

WHY THIS EXISTS. Every other task in the framework derives from the same 68
landmarks, so they are different readouts of one annotation rather than
independent tasks. Reconstruction fidelity is computed from PIXELS ALONE — it
is the genuinely landmark-independent axis, and it is what turns a task list
into a framework.

THE POINT IT MAKES. In practice people choose a compression level by looking at
image-quality metrics. If PSNR/SSIM degrade smoothly with total compression
while the clinical tasks track the SPATIAL axis only, then image-quality
metrics give the wrong answer, and a task-aware, axis-aware recommendation is
required instead. That is the framework's contribution.

METHOD NOTES
  * Reconstruction is compared against the ENCODER'S OWN INPUT (the aspect-
    padded square), not the raw file, so padding/resizing is not counted as
    reconstruction error.
  * Metrics are computed on the CONTENT REGION ONLY. The aspect-padded canvas
    is mostly flat black; including it inflates PSNR and SSIM substantially and
    makes every config look better than it is.
  * 3-channel models receive replicated grayscale and return 3 channels; the
    reconstruction is averaged back to one channel before comparison.
  * LPIPS is optional (needs the `lpips` package). PSNR/SSIM are implemented
    here in torch, so the script runs without extra dependencies.

Run:
  python recon_fidelity.py --img-dir ./medvae/data/boostnet \
      --csv ./medvae/data/boostnet_with_angles.csv --split test \
      --out-dir ./outputs/fidelity
  python recon_fidelity.py --smoke
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CONFIGS = ["medvae_4_3_2d", "medvae_4_1_2d", "medvae_8_4_2d", "medvae_8_1_2d"]
LABEL = {"medvae_4_3_2d": "4_3", "medvae_4_1_2d": "4_1",
         "medvae_8_4_2d": "8_4", "medvae_8_1_2d": "8_1"}
COMPRESSION = {"medvae_4_3_2d": 16 / 3, "medvae_4_1_2d": 16.0,
               "medvae_8_4_2d": 16.0, "medvae_8_1_2d": 64.0}
SPATIAL = {"medvae_4_3_2d": "4x", "medvae_4_1_2d": "4x",
           "medvae_8_4_2d": "8x", "medvae_8_1_2d": "8x"}
# clinical reference (Cobb deg MAE) for the misleading-metric comparison
COBB_REF = {"medvae_4_3_2d": 7.783, "medvae_4_1_2d": 7.678,
            "medvae_8_4_2d": 8.550, "medvae_8_1_2d": 8.235}


# ==========================================================================
# metrics (torch, dependency-free)
# ==========================================================================
def psnr(a, b, data_range=1.0):
    """a, b in [0,1], same shape. Returns dB."""
    mse = float(((a - b) ** 2).mean())
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _gauss_win(size=11, sigma=1.5, device="cpu"):
    c = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return (g.t() @ g).unsqueeze(0).unsqueeze(0)        # (1,1,size,size)


def ssim(a, b, data_range=1.0, size=11, sigma=1.5):
    """Standard SSIM with an 11x11 Gaussian window. a,b: (H,W) tensors in [0,1]."""
    a = a.unsqueeze(0).unsqueeze(0).float()
    b = b.unsqueeze(0).unsqueeze(0).float()
    if min(a.shape[-2:]) < size:                        # too small to window
        return float("nan")
    w = _gauss_win(size, sigma, a.device)
    pad = size // 2
    mu_a = F.conv2d(a, w, padding=pad)
    mu_b = F.conv2d(b, w, padding=pad)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sa = F.conv2d(a * a, w, padding=pad) - mu_a2
    sb = F.conv2d(b * b, w, padding=pad) - mu_b2
    sab = F.conv2d(a * b, w, padding=pad) - mu_ab
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return float(m.mean())


def boot_ci(v, seed=0, nb=10000):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed); n = len(v)
    b = np.array([v[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ==========================================================================
# content-region masking
# ==========================================================================
def content_bbox(x, thresh=1e-3):
    """Bounding box of non-padding content in a (H,W) tensor scaled to [0,1].
    Aspect-padding fills the unused canvas with a constant; including it in the
    metrics inflates PSNR/SSIM, so we crop to the real image first."""
    m = (x > (x.min() + thresh))
    rows = torch.where(m.any(1))[0]
    cols = torch.where(m.any(0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, x.shape[0], 0, x.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def to01(x):
    """MedVAE inputs are ~[-1,1]; map to [0,1] for metric conventions."""
    return ((x + 1.0) / 2.0).clamp(0, 1)


# ==========================================================================
# model I/O
# ==========================================================================
def match_in_channels(x, n_in):
    if x.shape[1] == n_in:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, n_in, 1, 1)
    raise ValueError(f"cannot match {x.shape[1]}ch -> {n_in}ch")


@torch.no_grad()
def encode(mvae, x):
    out = mvae.model.encode(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out):
        return out
    for attr in ("mode", "mean", "sample"):
        if hasattr(out, attr):
            a = getattr(out, attr)
            return a() if callable(a) else a
    raise RuntimeError(f"unrecognised encode() return: {type(out)}")


@torch.no_grad()
def decode(mvae, z):
    """Decode a latent back to an image. MedVAE's API varies by version, so try
    the plausible entry points rather than assuming one."""
    last = None
    for obj, name in ((mvae.model, "decode"), (mvae, "decode"),
                      (mvae.model, "decoder"), (mvae.model, "decode_first_stage")):
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        try:
            out = fn(z)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if torch.is_tensor(out):
                return out
        except Exception as e:                       # try the next entry point
            last = e
    raise RuntimeError(
        "could not decode the latent — none of model.decode / decode / decoder / "
        f"decode_first_stage worked (last error: {last}). Check the MedVAE API "
        "and adjust decode() accordingly."
    )


@torch.no_grad()
def reconstruct(mvae, x):
    """Encode then decode. Falls back to a single forward pass if the model
    exposes reconstruction directly."""
    try:
        return decode(mvae, encode(mvae, x))
    except RuntimeError:
        out = mvae(x) if callable(mvae) else mvae.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if torch.is_tensor(out):
            return out
        raise


@torch.no_grad()
def probe_in_channels(mvae, x1, device):
    for c in (1, 3, 4):
        try:
            encode(mvae, match_in_channels(x1, c).to(device))
            return c
        except Exception:
            continue
    raise RuntimeError("could not determine encoder input channels")


# ==========================================================================
def run(args):
    from medvae import MVAE
    from medvae.utils.loaders import load_2d_finetune_aspect

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    sub = df[df["split"] == args.split].reset_index(drop=True)
    uuids = [str(u) for u in sub["image_uuid"]]
    if args.limit:
        uuids = uuids[: args.limit]
    print(f"{len(uuids)} images from split '{args.split}'\n")

    lp = None
    if args.lpips:
        try:
            import lpips as lpips_pkg
            lp = lpips_pkg.LPIPS(net="alex").to(device).eval()
            print("LPIPS: enabled (alex)\n")
        except Exception as e:
            print(f"LPIPS: unavailable ({e}); continuing with PSNR/SSIM only\n")

    rows = []
    for config in args.configs:
        mvae = MVAE(model_name=config, modality="xray").to(device).eval()
        probe = load_2d_finetune_aspect(
            str(Path(args.img_dir) / f"{uuids[0]}.jpg"), merge_channels=True
        ).unsqueeze(0).to(device)
        n_in = probe_in_channels(mvae, probe, device)
        print(f"[{LABEL[config]}] encoder input {n_in}ch — reconstructing…")

        for u in uuids:
            p = Path(args.img_dir) / f"{u}.jpg"
            if not p.exists():
                continue
            x1 = load_2d_finetune_aspect(str(p), merge_channels=True).unsqueeze(0).to(device)
            xin = match_in_channels(x1, n_in)
            xr = reconstruct(mvae, xin)
            if xr.shape[-2:] != x1.shape[-2:]:
                xr = F.interpolate(xr, size=x1.shape[-2:], mode="bilinear",
                                   align_corners=False)
            # back to a single grayscale channel for comparison
            orig = to01(x1[0].mean(0).float().cpu())
            recon = to01(xr[0].mean(0).float().cpu())
            r0, r1, c0, c1 = content_bbox(orig)          # exclude padding
            o, r = orig[r0:r1, c0:c1], recon[r0:r1, c0:c1]

            rec = {"config": config, "image_uuid": u,
                   "psnr": psnr(o, r), "ssim": ssim(o, r)}
            if lp is not None:
                with torch.no_grad():
                    a = (o * 2 - 1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
                    b = (r * 2 - 1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
                    rec["lpips"] = float(lp(a, b).item())
            rows.append(rec)

        del mvae
        if device == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "recon_fidelity_per_image.csv", index=False)
    summarise(out, args.out_dir)


def summarise(out, out_dir):
    metrics = [m for m in ("psnr", "ssim", "lpips") if m in out.columns]
    print("\n=== Reconstruction fidelity (content region only) ===")
    hdr = f"{'config':<8}{'total':>8}{'spatial':>9}" + "".join(f"{m.upper():>22}" for m in metrics)
    print(hdr); print("-" * len(hdr))
    summary = []
    for c in [c for c in CONFIGS if c in set(out.config)]:
        sub = out[out.config == c]
        line = f"{LABEL[c]:<8}{COMPRESSION[c]:>7.1f}x{SPATIAL[c]:>9}"
        rec = {"config": LABEL[c], "compression": COMPRESSION[c], "spatial": SPATIAL[c]}
        for m in metrics:
            mean, lo, hi = boot_ci(sub[m].values)
            line += f"{mean:>10.3f} [{lo:.3f},{hi:.3f}]"
            rec[m] = mean; rec[f"{m}_lo"] = lo; rec[f"{m}_hi"] = hi
        print(line)
        rec["cobb_deg"] = COBB_REF.get(c, np.nan)
        summary.append(rec)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "recon_fidelity_summary.csv", index=False)

    # ---- the framework's point: does fidelity predict clinical utility? ----
    print("\n=== Does image quality predict clinical utility? ===")
    print(f"{'config':<8}{'total':>8}{'spatial':>9}{'PSNR':>9}{'SSIM':>9}{'Cobb (deg)':>12}")
    print("-" * 55)
    for _, r in sdf.iterrows():
        print(f"{r['config']:<8}{r['compression']:>7.1f}x{r['spatial']:>9}"
              f"{r.get('psnr', np.nan):>9.2f}{r.get('ssim', np.nan):>9.3f}"
              f"{r['cobb_deg']:>12.2f}")
    if len(sdf) >= 3 and "psnr" in sdf.columns and sdf["cobb_deg"].notna().all():
        r_psnr = np.corrcoef(sdf["psnr"], sdf["cobb_deg"])[0, 1]
        r_ssim = np.corrcoef(sdf["ssim"], sdf["cobb_deg"])[0, 1]
        print(f"\nacross configs: corr(PSNR, Cobb error) = {r_psnr:+.2f}, "
              f"corr(SSIM, Cobb error) = {r_ssim:+.2f}")
        print("  (strong NEGATIVE corr => better-looking images do give better Cobb;\n"
              "   weak/positive => image-quality metrics MISLEAD about clinical utility)")
        print("\nThe decisive comparison is the matched pair 4_1 vs 8_4 — same total\n"
              "compression (16x). If their PSNR/SSIM are similar but their Cobb errors\n"
              "differ by ~0.87 deg, fidelity cannot distinguish them and a task-aware,\n"
              "axis-aware recommendation is required.")
    print(f"\nwrote {out_dir}/recon_fidelity_per_image.csv and _summary.csv")


# ==========================================================================
def smoke():
    torch.manual_seed(0)
    x = torch.rand(64, 64)

    # identical -> PSNR inf, SSIM 1
    p_id, s_id = psnr(x, x), ssim(x, x)
    # blurred -> finite PSNR, SSIM < 1
    xb = F.avg_pool2d(x.unsqueeze(0).unsqueeze(0), 3, 1, 1)[0, 0]
    p_bl, s_bl = psnr(x, xb), ssim(x, xb)
    # heavier blur -> worse on both
    xb2 = F.avg_pool2d(x.unsqueeze(0).unsqueeze(0), 7, 1, 3)[0, 0]
    p_b2, s_b2 = psnr(x, xb2), ssim(x, xb2)
    # known-MSE case: constant offset 0.1 -> PSNR = 20*log10(1/0.1) = 20 dB
    p_off = psnr(torch.full((32, 32), 0.5), torch.full((32, 32), 0.6))

    print(f"identical      : PSNR {p_id}      SSIM {s_id:.4f}   (expect inf, 1.0000)")
    print(f"blur 3x3       : PSNR {p_bl:6.2f} dB  SSIM {s_bl:.4f}")
    print(f"blur 7x7       : PSNR {p_b2:6.2f} dB  SSIM {s_b2:.4f}   (expect worse than 3x3)")
    print(f"offset 0.1     : PSNR {p_off:6.2f} dB              (expect 20.00)")

    # content-bbox: pad a 20x30 block into a 64x64 canvas of the pad value
    canvas = torch.full((64, 64), -1.0)
    canvas[5:25, 7:37] = torch.rand(20, 30)
    r0, r1, c0, c1 = content_bbox(to01(canvas))

    ok = (np.isinf(p_id) and abs(s_id - 1) < 1e-4
          and p_bl > p_b2 and s_bl > s_b2
          and abs(p_off - 20.0) < 0.01
          and (r0, r1, c0, c1) == (5, 25, 7, 37))
    print(f"content bbox   : {(r0, r1, c0, c1)}   (expect (5, 25, 7, 37))")
    print("\nSMOKE:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/fidelity"))
    ap.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs `lpips`)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.img_dir and args.csv:
        run(args)
    else:
        raise SystemExit("need --img-dir and --csv (or --smoke)")


if __name__ == "__main__":
    main()
