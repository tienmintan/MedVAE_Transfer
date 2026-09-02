"""
extract_latents_multichannel.py — give the 3-channel models something to encode.

THE CLAIM BEING TESTED. We found that adding latent channels never helps
(`4_1`->`4_3` +0.11 deg, `8_1`->`8_4` +0.16 deg) and explained it as: the input
was grayscale REPLICATED across 3 identical channels, so the extra latent
channels had no additional information to encode -- they were spending capacity
on redundancy.

That explanation makes a testable prediction: give the channels genuinely
different content and the channel axis should start to matter. This builds
non-redundant 3-channel inputs and re-extracts:

    channel 0 : the original grayscale radiograph
    channel 1 : CLAHE, moderate  (tile 8,  clip 2.0)  -- local contrast
    channel 2 : CLAHE, aggressive (tile 16, clip 4.0) -- finer structure

CLAHE (Contrast Limited Adaptive Histogram Equalisation) equalises contrast in
small tiles rather than globally, so faint structure in dark regions becomes
visible without saturating bright regions; the clip limit caps the enhancement
so noise is not amplified. It is the standard contrast method for radiographs,
and it is exactly the kind of fine endplate detail our mechanism says
compression destroys -- so enhancing it before encoding is a plausible
mitigation as well as a test.

EITHER OUTCOME IS INFORMATIVE
  * channels now help  -> confirms the redundancy explanation, AND gives a free
    accuracy improvement at no extra storage cost;
  * channels still do not help -> the channel axis is genuinely inert for this
    task, and the "spend the budget on spatial resolution" advice strengthens.

Run:
  python extract_latents_multichannel.py \
      --csv ./medvae/data/boostnet_with_angles.csv \
      --img-dir ./medvae/data/boostnet \
      --landmarks-dir ~/MedVAE/data/boostnet_labeldata/labels \
      --train-lookup ./landmark_row_lookup_train_fixed.csv \
      --test-lookup ./landmark_row_lookup_test_fixed.csv \
      --configs medvae_4_3_2d medvae_8_4_2d \
      --out-dir ./latents_clahe
  python extract_latents_multichannel.py --smoke
"""
import argparse
from pathlib import Path

import numpy as np

N_LM = 68
SPLIT_TO_LMK_SUBDIR = {"train": "training", "val": "training", "test": "test"}
CONFIG_LABELS = {"medvae_4_3_2d": "4_3", "medvae_8_4_2d": "8_4"}
# (tile grid, clip limit) for the two CLAHE channels
CLAHE_PARAMS = [(8, 2.0), (16, 4.0)]


# ==========================================================================
# CLAHE (numpy implementation, so no OpenCV dependency)
# ==========================================================================
def clahe(img, tiles=8, clip=2.0, nbins=256):
    """Contrast Limited Adaptive Histogram Equalisation.

    img: (H,W) float in [0,1]. Returns (H,W) float in [0,1].

    Equalises each tile's histogram separately, clipping the histogram at
    `clip` x the uniform height (redistributing the excess) so noise in flat
    regions is not amplified, then bilinearly interpolates between tile
    mappings to avoid visible tile seams.
    """
    H, W = img.shape
    q = np.clip((img * (nbins - 1)).round().astype(np.int32), 0, nbins - 1)
    th, tw = int(np.ceil(H / tiles)), int(np.ceil(W / tiles))

    # per-tile cumulative mapping
    maps = np.zeros((tiles, tiles, nbins), dtype=np.float32)
    for i in range(tiles):
        for j in range(tiles):
            blk = q[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if blk.size == 0:
                maps[i, j] = np.linspace(0, 1, nbins)
                continue
            hist = np.bincount(blk.ravel(), minlength=nbins).astype(np.float32)
            if clip > 0:                                    # clip and redistribute
                limit = clip * blk.size / nbins
                excess = np.maximum(hist - limit, 0).sum()
                hist = np.minimum(hist, limit) + excess / nbins
            cdf = np.cumsum(hist)
            cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-8)
            maps[i, j] = cdf

    # bilinear interpolation between tile centres
    yy = np.arange(H); xx = np.arange(W)
    fy = np.clip((yy + 0.5) / th - 0.5, 0, tiles - 1)
    fx = np.clip((xx + 0.5) / tw - 0.5, 0, tiles - 1)
    y0 = np.floor(fy).astype(int); y1 = np.minimum(y0 + 1, tiles - 1)
    x0 = np.floor(fx).astype(int); x1 = np.minimum(x0 + 1, tiles - 1)
    wy = (fy - y0)[:, None]; wx = (fx - x0)[None, :]

    m00 = maps[y0[:, None], x0[None, :], q]
    m01 = maps[y0[:, None], x1[None, :], q]
    m10 = maps[y1[:, None], x0[None, :], q]
    m11 = maps[y1[:, None], x1[None, :], q]
    out = ((1 - wy) * ((1 - wx) * m00 + wx * m01)
           + wy * ((1 - wx) * m10 + wx * m11))
    return np.clip(out, 0, 1).astype(np.float32)


def build_channels(gray01, mode="nonredundant"):
    """(H,W) in [0,1] -> (3,H,W).

    mode='nonredundant'  : [original, CLAHE moderate, CLAHE aggressive]
        The experiment — channels carry genuinely different content.

    mode='clahe-replicated' : [CLAHE moderate] x 3
        The CONTROL. CLAHE improves contrast generally, so if the
        non-redundant run improves, the gain could come from better
        preprocessing rather than from channel diversity. This arm applies the
        same CLAHE but replicates it, so the ONLY difference from the
        non-redundant arm is whether the channels differ:
            control ~ non-redundant  => the gain is CLAHE, channels still inert
            control ~ replicated raw => the gain is channel diversity
    """
    if mode == "clahe-replicated":
        tiles, clip = CLAHE_PARAMS[0]
        enh = clahe(gray01, tiles=tiles, clip=clip)
        return np.stack([enh, enh, enh], 0)
    if mode == "raw-replicated":
        return np.stack([gray01, gray01, gray01], 0)
    chans = [gray01]
    for tiles, clip in CLAHE_PARAMS:
        chans.append(clahe(gray01, tiles=tiles, clip=clip))
    return np.stack(chans, 0)


def channel_redundancy(x3):
    """Mean pairwise correlation between channels. Replicated grayscale gives
    1.00; genuinely different content gives noticeably less. This is the
    diagnostic that the experiment is doing what it claims."""
    flat = x3.reshape(x3.shape[0], -1)
    cors = []
    for a in range(len(flat)):
        for b in range(a + 1, len(flat)):
            cors.append(np.corrcoef(flat[a], flat[b])[0, 1])
    return float(np.mean(cors)), [float(c) for c in cors]


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--img-dir", type=Path)
    ap.add_argument("--landmarks-dir", type=Path)
    ap.add_argument("--train-lookup", type=Path)
    ap.add_argument("--test-lookup", type=Path)
    ap.add_argument("--configs", nargs="+", default=["medvae_4_3_2d", "medvae_8_4_2d"])
    ap.add_argument("--out-dir", type=Path, default=Path("./latents_clahe"))
    ap.add_argument("--mode", default="nonredundant",
                    choices=["nonredundant", "clahe-replicated", "raw-replicated"],
                    help="nonredundant = the experiment; clahe-replicated = the "
                         "control isolating CLAHE from channel diversity")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preview", type=Path, default=None,
                    help="save a PNG comparing the three channels for one image")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    run(args)


def run(args):
    import pandas as pd
    import torch
    from PIL import Image
    from medvae import MVAE
    from medvae.utils.loaders import load_2d_finetune_aspect

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    cobb_cols = [c for c in df.columns if c.startswith("angle_")]
    print(f"Cobb columns: {cobb_cols}")
    lookups = {}
    if args.train_lookup:
        lookups["train"] = pd.read_csv(args.train_lookup)
    if args.test_lookup:
        lookups["test"] = pd.read_csv(args.test_lookup)

    def lmk_rows(split):
        sub = SPLIT_TO_LMK_SUBDIR[split]
        arr = pd.read_csv(Path(args.landmarks_dir) / sub / "landmarks.csv",
                          header=None).values
        return np.stack([np.concatenate([np.asarray(r[:N_LM], dtype=float),
                                         np.asarray(r[N_LM:], dtype=float)]) for r in arr])

    def uuid_map(split):
        lk = lookups.get("train" if SPLIT_TO_LMK_SUBDIR[split] == "training" else "test")
        ucol = "image_uuid" if "image_uuid" in lk.columns else lk.columns[0]
        if "landmark_row" in lk.columns:
            return {str(u): int(r) for u, r in zip(lk[ucol], lk["landmark_row"])}
        return {str(u): i for i, u in enumerate(lk[ucol].astype(str))}

    @torch.no_grad()
    def encode(mvae, x):
        o = mvae.model.encode(x)
        if isinstance(o, (tuple, list)):
            o = o[0]
        if torch.is_tensor(o):
            return o
        for a in ("mode", "mean", "sample"):
            if hasattr(o, a):
                f = getattr(o, a)
                return f() if callable(f) else f
        raise RuntimeError("unrecognised encode() return")

    red_stats = []
    for config in args.configs:
        mvae = MVAE(model_name=config, modality="xray").to(device).eval()
        for split in ("train", "val", "test"):
            dst = out / f"latents_{config}_{split}.npz"
            if dst.exists() and not args.force:
                print(f"skip (exists): {dst.name}")
                continue
            sub = df[df["split"] == split].reset_index(drop=True)
            if not len(sub):
                continue
            u2r = uuid_map(split)
            all_lmk = lmk_rows(split)

            lat, uu, cobb, lmk = [], [], [], []
            for _, r in sub.iterrows():
                u = str(r["image_uuid"])
                if u not in u2r:
                    print(f"  WARN no landmark row for {u}"); continue
                p = Path(args.img_dir) / f"{u}.jpg"
                # aspect-pad exactly as the standard pipeline does, then take
                # the grayscale plane and build the three channels from it
                base = load_2d_finetune_aspect(str(p), merge_channels=True)   # (1,H,W) in ~[-1,1]
                g01 = ((base[0].numpy() + 1.0) / 2.0).clip(0, 1)
                x3 = build_channels(g01, mode=args.mode)
                if len(red_stats) < 25:
                    red_stats.append(channel_redundancy(x3)[0])
                t = torch.tensor(x3 * 2 - 1, dtype=torch.float32,
                                 device=device).unsqueeze(0)
                lat.append(encode(mvae, t).squeeze(0).float().cpu().numpy())
                uu.append(u)
                cobb.append(r[cobb_cols].to_numpy(dtype=np.float32))
                lmk.append(all_lmk[u2r[u]].astype(np.float32))

            np.savez_compressed(dst, latents=np.stack(lat).astype(np.float32),
                                uuids=np.array(uu), cobb=np.stack(cobb),
                                landmarks=np.stack(lmk))
            print(f"  wrote {dst.name}  latents={np.stack(lat).shape}  n={len(uu)}")
        del mvae
        if device == "cuda":
            torch.cuda.empty_cache()

    if red_stats:
        m = float(np.mean(red_stats))
        print(f"\nchannel redundancy check: mean pairwise correlation = {m:.3f}")
        print("   (replicated grayscale would be 1.000; lower means the channels")
        print("    genuinely carry different content, which is the point)")
        if m > 0.98:
            print("   !! WARNING: channels are still nearly identical — CLAHE is not")
            print("      adding information. Check the parameters before trusting a null.")

    print(f"\nNow run the frontier on these latents and compare against the "
          f"replicated-grayscale run:\n"
          f"  python train_frontier.py --latents-dir {out} --tasks cobb "
          f"--configs {' '.join(args.configs)} --seeds 0 1 2 3 4 5 6 7 8 9")


# ==========================================================================
def smoke():
    rng = np.random.default_rng(0)
    # a synthetic radiograph-like image: dark background, bright structure,
    # plus a faint low-contrast detail CLAHE should reveal
    H = W = 128
    img = np.zeros((H, W), dtype=np.float32) + 0.08
    img[:, 55:75] = 0.55                                    # bright "spine"
    img[40:44, 55:75] = 0.60                                # faint "endplate"
    img[90:96, 20:40] = 0.11                                # faint dark-region detail
    img += 0.01 * rng.standard_normal((H, W))
    img = np.clip(img, 0, 1)

    out = clahe(img, tiles=8, clip=2.0)
    print(f"clahe: in range [{img.min():.3f},{img.max():.3f}] -> "
          f"out [{out.min():.3f},{out.max():.3f}]  shape {out.shape}")

    # local contrast in the faint dark region should increase
    def local_sd(a):
        return float(a[85:100, 15:45].std())
    print(f"local sd in the faint region: {local_sd(img):.4f} -> {local_sd(out):.4f} "
          f"({'increased' if local_sd(out) > local_sd(img) else 'NOT increased'})")

    x3 = build_channels(img)
    m, cors = channel_redundancy(x3)
    print(f"channels shape {x3.shape}; pairwise correlations "
          f"{[round(c,3) for c in cors]}, mean {m:.3f}")

    rep = np.repeat(img[None], 3, 0)
    mrep, _ = channel_redundancy(rep)
    print(f"replicated-grayscale control: mean correlation {mrep:.3f} (expect 1.000)")

    ok = (out.shape == img.shape and 0 <= out.min() and out.max() <= 1
          and local_sd(out) > local_sd(img)
          and m < 0.999 and abs(mrep - 1.0) < 1e-6)
    print("\nSMOKE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
