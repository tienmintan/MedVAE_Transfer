"""
extract_latents_dino.py — frozen DINOv2 / DINOv3 features as a SOTA baseline.

WHY. Every examiner will ask how a compressive medical encoder compares against
the current general-purpose vision foundation models. This extracts frozen DINO
dense patch features in EXACTLY the same npz format as the MedVAE pipeline, so
`train_frontier.py` runs the identical heads, splits, seeds and metrics on them.

WHAT THE COMPARISON MEANS — read this before interpreting results.
DINO features are NOT a compression baseline. ViT-B/14 at 448px gives a
32x32 grid of 768-dim tokens = 3.1 MB/image in fp32, versus MedVAE's 36 KB.
So this measures something different and more useful:

    "What does COMPRESSION cost, relative to an uncompressed general-purpose
     representation of the same image?"

It bounds how much of the ~7.7 deg Cobb error is the compression versus the
task simply being hard. Either outcome is reportable:
  * MedVAE close to DINO  -> compression is nearly free for these tasks;
  * DINO much better      -> compression costs real, quantified performance.

For a storage-MATCHED comparison, use `--pca-dim`, which projects the token
features down until the latent occupies a comparable number of bytes to the
MedVAE latent. That arm answers a second question: at equal storage, is a
general-purpose encoder better than a medical compressive one?

MODEL CHOICE. DINOv3 is newer, but a recent correspondence benchmark found
DINOv2 outperforms it on correspondence-style tasks, and landmark localisation
is correspondence-like. Run BOTH; the comparison is itself reportable.

Run:
  python extract_latents_dino.py \
      --csv ./medvae/data/boostnet_with_angles.csv \
      --img-dir ./medvae/data/boostnet \
      --landmarks-dir ~/MedVAE/data/boostnet_labeldata/labels \
      --train-lookup ./landmark_row_lookup_train_fixed.csv \
      --test-lookup ./landmark_row_lookup_test_fixed.csv \
      --model facebook/dinov2-base --size 448 --out-dir ./latents_dino

  # storage-matched arm (~36 KB/image, like medvae_4_1_2d)
  python extract_latents_dino.py ... --pca-dim 9 --out-dir ./latents_dino_pca

  python extract_latents_dino.py --smoke

Then:
  python train_frontier.py --latents-dir ./latents_dino \
      --configs dinov2_base --tasks cobb landmark --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
from pathlib import Path

import numpy as np

N_LM = 68
SPLIT_TO_LMK_SUBDIR = {"train": "training", "val": "training", "test": "test"}

# Reuse the augmentation machinery so the DINO arm can be compared FAIRLY
# against the augmented MedVAE arm. Without this the comparison is biased in
# MedVAE's favour, since only MedVAE saw the extra data.
try:
    from augment_and_extract import (PRESETS, augment_once, landmarks_to_points,
                                     points_to_landmarks, max_cobb)
    _AUG_OK = True
except Exception as _e:                      # augmentation optional
    PRESETS, _AUG_OK = {}, False

# Candidate HuggingFace ids. Verify availability before relying on any of them —
# DINOv3 repo names in particular have changed since release.
KNOWN_MODELS = {
    "dinov2-small": "facebook/dinov2-small",     # 384-dim, patch 14
    "dinov2-base": "facebook/dinov2-base",       # 768-dim, patch 14
    "dinov2-large": "facebook/dinov2-large",     # 1024-dim, patch 14
    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


# ==========================================================================
# geometry — matched to the MedVAE pipeline so the arms are comparable
# ==========================================================================
def aspect_pad_square(img, size):
    """PIL image -> (size,size) float array in [0,1], aspect-preserving, padded
    top-left. Mirrors the MedVAE preprocessing so landmark coordinates keep the
    same meaning across arms."""
    import PIL.Image as Image
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(img, (0, 0))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def tokens_to_grid(tokens, n_patch_side, n_extra):
    """(1, n_tokens, D) -> (D, g, g).

    ViTs prepend a CLS token, and DINOv3 also prepends register tokens. Rather
    than assume how many, we take the LAST g*g tokens, which are always the
    patch tokens in raster order."""
    t = tokens[0]                                   # (n_tokens, D)
    need = n_patch_side * n_patch_side
    if t.shape[0] < need:
        raise RuntimeError(f"got {t.shape[0]} tokens, need at least {need}")
    patches = t[-need:]                             # drop CLS/registers
    return patches.reshape(n_patch_side, n_patch_side, -1).transpose(2, 0, 1)


def fit_pca(feats, dim, seed=0):
    """Fit PCA on a sample of TRAIN token vectors only (never val/test), so the
    storage-matched arm carries no test-set information."""
    x = feats.reshape(-1, feats.shape[-1])
    rng = np.random.default_rng(seed)
    if len(x) > 200_000:
        x = x[rng.choice(len(x), 200_000, replace=False)]
    mu = x.mean(0, keepdims=True)
    xc = x - mu
    # economical SVD on the covariance
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1][:dim]
    comps = v[:, idx]                               # (D, dim)
    evr = float(w[idx].sum() / max(w.sum(), 1e-12))
    return mu, comps, evr


def apply_pca(feats, mu, comps):
    """(N, D, g, g) -> (N, dim, g, g)."""
    n, d, g, _ = feats.shape
    x = feats.transpose(0, 2, 3, 1).reshape(-1, d)
    y = (x - mu) @ comps
    return y.reshape(n, g, g, -1).transpose(0, 3, 1, 2)


def bytes_per_image(shape, fp16=False):
    return int(np.prod(shape)) * (2 if fp16 else 4)


# ==========================================================================
def run(args):
    import pandas as pd
    import torch
    from PIL import Image
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = KNOWN_MODELS.get(args.model, args.model)
    print(f"loading {model_id} on {device}")
    try:
        model = AutoModel.from_pretrained(model_id).to(device).eval()
    except Exception as e:
        raise SystemExit(
            f"could not load '{model_id}': {e}\n"
            f"Known shortcuts: {', '.join(KNOWN_MODELS)}\n"
            "DINOv3 repo names have changed since release — check the exact id "
            "on HuggingFace and pass it directly with --model."
        )

    patch = getattr(model.config, "patch_size", 14)
    if args.size % patch:
        newsize = (args.size // patch) * patch
        print(f"  adjusting size {args.size} -> {newsize} (multiple of patch {patch})")
        args.size = newsize
    g = args.size // patch
    dim = getattr(model.config, "hidden_size", None)
    print(f"  patch {patch}, input {args.size}px -> {g}x{g} grid, {dim}-dim tokens")
    raw_bytes = bytes_per_image((dim, g, g))
    print(f"  raw feature size: {raw_bytes/1024:.0f} KB/image fp32 "
          f"(MedVAE 4_1 is 36 KB) — NOT a compression baseline\n")

    # ImageNet normalisation, applied to replicated grayscale
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    df = pd.read_csv(args.csv)
    cobb_cols = [c for c in df.columns if c.startswith("angle_")]
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
                                         np.asarray(r[N_LM:], dtype=float)])
                         for r in arr])

    def uuid_map(split):
        lk = lookups.get("train" if SPLIT_TO_LMK_SUBDIR[split] == "training" else "test")
        ucol = "image_uuid" if "image_uuid" in lk.columns else lk.columns[0]
        if "landmark_row" in lk.columns:
            return {str(u): int(r) for u, r in zip(lk[ucol], lk["landmark_row"])}
        return {str(u): i for i, u in enumerate(lk[ucol].astype(str))}

    @torch.no_grad()
    def encode(x):
        out = model(pixel_values=x)
        return out.last_hidden_state.float().cpu().numpy()

    aug_rng = np.random.default_rng(args.seed)
    pset = PRESETS.get(args.preset, {}) if args.n_aug else {}
    if args.n_aug:
        if not _AUG_OK:
            raise SystemExit("augmentation requested but augment_and_extract.py "
                             "could not be imported — keep it alongside this script")
        print(f"  augmenting TRAIN only: {args.n_aug} copies/image, "
              f"preset '{args.preset}' {pset}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = args.config_name or args.model.replace("/", "_")

    raw = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)
        if not len(sub):
            continue
        u2r = uuid_map(split)
        all_lmk = lmk_rows(split)
        n_aug = args.n_aug if split == "train" else 0     # ONLY augment train
        feats, uu, cobb, lmk = [], [], [], []
        for _, r in sub.iterrows():
            u = str(r["image_uuid"])
            if u not in u2r:
                print(f"  WARN no landmark row for {u}"); continue
            p = Path(args.img_dir) / f"{u}.jpg"
            if not p.exists():
                print(f"  WARN missing image {u}"); continue
            with Image.open(p) as im:
                gray = aspect_pad_square(im.convert("L"), args.size)
            x3 = np.repeat(gray[None], 3, 0)             # grayscale -> 3ch
            x3 = (x3 - mean) / std
            t = torch.tensor(x3[None], dtype=torch.float32, device=device)
            feats.append(tokens_to_grid(encode(t), g, 0))
            uu.append(u)
            cobb_row = r[cobb_cols].to_numpy(dtype=np.float32)
            base_lm = all_lmk[u2r[u]].astype(np.float32)
            cobb.append(cobb_row)
            lmk.append(base_lm)

            # ---- augmented copies (TRAIN only) ----
            if n_aug:
                with Image.open(p) as im0:
                    im0 = im0.convert("L")
                    W0, H0 = im0.size
                    pts0 = landmarks_to_points(base_lm, W0, H0)
                    c0 = max_cobb(pts0)
                    made = 0
                    for _ in range(n_aug * 4):
                        if made >= n_aug:
                            break
                        aug, pts = augment_once(
                            im0, pts0, aug_rng, pset["max_deg"], pset["max_scale"],
                            pset["max_shift"], pset["jitter"], gamma=pset["gamma"],
                            noise=pset["noise"], blur=pset["blur"])
                        if aug is None or abs(max_cobb(pts) - c0) > 0.05:
                            continue
                        ga = aspect_pad_square(aug, args.size)
                        xa = (np.repeat(ga[None], 3, 0) - mean) / std
                        ta = torch.tensor(xa[None], dtype=torch.float32, device=device)
                        feats.append(tokens_to_grid(encode(ta), g, 0))
                        uu.append(f"{u}__aug{made}")
                        cobb.append(cobb_row)
                        lmk.append(points_to_landmarks(pts, W0, H0))
                        made += 1
        raw[split] = {"latents": np.stack(feats), "uuids": np.array(uu),
                      "cobb": np.stack(cobb), "landmarks": np.stack(lmk)}
        print(f"  [{split}] {raw[split]['latents'].shape}")

    # ---- optional PCA for a storage-matched arm (fitted on TRAIN only) ----
    if args.pca_dim:
        print(f"\nfitting PCA to {args.pca_dim} dims on TRAIN tokens only…")
        mu, comps, evr = fit_pca(raw["train"]["latents"].transpose(0, 2, 3, 1),
                                 args.pca_dim)
        print(f"  explained variance retained: {100*evr:.1f}%")
        for split in raw:
            raw[split]["latents"] = apply_pca(raw[split]["latents"], mu, comps)
        newshape = raw["train"]["latents"].shape[1:]
        print(f"  new latent {tuple(newshape)} = "
              f"{bytes_per_image(newshape)/1024:.1f} KB/image fp32")
        cfg_name += f"_pca{args.pca_dim}"

    dtype = np.float16 if args.fp16 else np.float32
    for split, d in raw.items():
        dst = out_dir / f"latents_{cfg_name}_{split}.npz"
        np.savez_compressed(dst, latents=d["latents"].astype(dtype),
                            uuids=d["uuids"], cobb=d["cobb"],
                            landmarks=d["landmarks"])
        print(f"wrote {dst.name}")

    print(f"\nNow run the same heads on these features:\n"
          f"  python train_frontier.py --latents-dir {out_dir} "
          f"--configs {cfg_name} --tasks cobb landmark --seeds 0 1 2 3 4 5 6 7 8 9")
    print("\nNOTE: train_frontier.py needs `{}` added to its CONFIG_LABELS / "
          "COMPRESSION dicts before it will run.".format(cfg_name))
    print("REMINDER: this is a REPRESENTATION-QUALITY baseline, not a compression\n"
          "baseline — the raw features are far larger than the MedVAE latent.")


# ==========================================================================
def smoke():
    import PIL.Image as Image

    # --- aspect padding preserves the 1:2 geometry ---
    im = Image.new("L", (200, 400), 128)
    a = aspect_pad_square(im, 448)
    nz_rows = np.where(a.max(1) > 0)[0]
    nz_cols = np.where(a.max(0) > 0)[0]
    print(f"aspect pad: {a.shape}, content rows 0-{nz_rows[-1]}, cols 0-{nz_cols[-1]} "
          f"(expect ~447 and ~223 for a 1:2 image)")

    # --- token -> grid, with CLS + registers prepended ---
    g, D = 32, 768
    rng = np.random.default_rng(0)
    patches = rng.standard_normal((g * g, D)).astype(np.float32)
    for n_extra in (1, 5):                       # CLS only; CLS + 4 registers
        toks = np.concatenate([rng.standard_normal((n_extra, D)).astype(np.float32),
                               patches], 0)[None]
        grid = tokens_to_grid(toks, g, n_extra)
        ok = np.allclose(grid[:, 0, 0], patches[0]) and np.allclose(grid[:, -1, -1], patches[-1])
        print(f"tokens_to_grid with {n_extra} extra token(s): {grid.shape}, "
              f"patch order preserved = {ok}")

    # --- PCA reduces dimension and retains variance sensibly ---
    feats = rng.standard_normal((20, D, 8, 8)).astype(np.float32)
    feats[:, :4] *= 8.0                          # plant dominant directions
    mu, comps, evr = fit_pca(feats.transpose(0, 2, 3, 1), 9)
    red = apply_pca(feats, mu, comps)
    print(f"PCA: {feats.shape[1]} -> {red.shape[1]} dims, "
          f"{100*evr:.1f}% variance retained, out shape {red.shape}")

    # --- storage accounting ---
    print(f"storage: (768,32,32) = {bytes_per_image((768,32,32))/1024:.0f} KB fp32; "
          f"(9,32,32) = {bytes_per_image((9,32,32))/1024:.1f} KB "
          f"(MedVAE 4_1 = 36.0 KB)")

    # PCA should retain far more variance than the same number of random
    # directions would (9/768 = 1.2%), which is the meaningful check.
    chance = 9 / D
    ok_all = (a.shape == (448, 448) and grid.shape == (D, g, g)
              and red.shape == (20, 9, 8, 8) and evr > 5 * chance)
    print(f"\n(PCA retains {100*evr:.1f}% in 9 of {D} dims; chance would be "
          f"{100*chance:.1f}%)")
    print("SMOKE:", "PASS" if ok_all else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--img-dir", type=Path)
    ap.add_argument("--landmarks-dir", type=Path)
    ap.add_argument("--train-lookup", type=Path)
    ap.add_argument("--test-lookup", type=Path)
    ap.add_argument("--model", default="dinov2-base",
                    help=f"HF id or shortcut: {', '.join(KNOWN_MODELS)}")
    ap.add_argument("--config-name", default=None,
                    help="name used in the output filenames (defaults to the model)")
    ap.add_argument("--size", type=int, default=448,
                    help="input square size; larger = finer patch grid")
    ap.add_argument("--pca-dim", type=int, default=0,
                    help="project tokens to this many dims for a storage-matched arm")
    ap.add_argument("--fp16", action="store_true", help="halve stored feature size")
    ap.add_argument("--n-aug", type=int, default=0,
                    help="augmented copies per TRAIN image (0 = none). Use the same "
                         "value as the MedVAE run for a fair comparison.")
    ap.add_argument("--preset", default="aggressive",
                    help="augmentation preset (see augment_and_extract.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("./latents_dino"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.csv and args.img_dir and args.landmarks_dir:
        run(args)
    else:
        raise SystemExit("need --csv --img-dir --landmarks-dir (or --smoke)")


if __name__ == "__main__":
    main()
