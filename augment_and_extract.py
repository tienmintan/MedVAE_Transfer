"""
augment_and_extract.py: augment training images in image space and re-encode them through the frozen MedVAE encoder.

Tests whether the landmark task is limited by training data rather than by
the representation (a large train/test error gap, and little gain from a
stronger encoder at matched storage, would indicate this). Augmentation
targets that limitation. It is also the control for any later
synthetic-generation experiment: without it, "generation helped" cannot be
distinguished from "any extra data would have helped".

Presets (--preset): conservative (default; intensity only), moderate
(+ uniform scale), aggressive (+ small rotation and translation). Individual
parameters can be overridden on the command line.

Augmentation is applied in image space because latents cannot be augmented
directly: a rotation in image space is not a rotation in latent space. The
image is transformed, the landmarks are transformed with the same affine, and
the result is re-encoded through the frozen encoder.

Label validity: rotation, uniform scale and translation are similarity
transforms, so they preserve angles between lines. A Cobb angle is the angle
between two endplate lines, so it is unchanged. (Shear and non-uniform scale
would not preserve it and are excluded.) The script checks this numerically
per augmented image and rejects copies whose major-curve angle moves by more
than --angle-tol.

No horizontal flip: flipping mirrors curve laterality (a right-convex thoracic
curve becomes left-convex) and puts the heart and stomach bubble on the wrong
side. The result is anatomically implausible and out of distribution.

Only the train split is augmented. Val and test are left untouched, so the
evaluation is unchanged and comparable with the unaugmented runs.

Inputs:  --csv (image_uuid, split, angle_* columns); JPEG images in
         --img-dir; --landmarks-dir; --train-lookup / --test-lookup.
Outputs: <out-dir>/latents_<config>_{train,val,test}.npz (train split
         includes the augmented copies).

Usage:
  python augment_and_extract.py \
      --csv ./medvae/data/boostnet_with_angles.csv \
      --img-dir ./medvae/data/boostnet \
      --landmarks-dir ./data/boostnet_labeldata/labels \
      --train-lookup ./landmark_row_lookup_train_fixed.csv \
      --test-lookup ./landmark_row_lookup_test_fixed.csv \
      --configs medvae_4_1_2d medvae_8_1_2d \
      --n-aug 5 --out-dir ./latents_aug

  python augment_and_extract.py --smoke      # synthetic-data self-test
"""
import argparse
from pathlib import Path

import numpy as np

N_LM = 68
N_VERT = 17
SPLIT_TO_LMK_SUBDIR = {"train": "training", "val": "training", "test": "test"}
CONFIG_LABELS = {"medvae_4_3_2d": "4_3", "medvae_4_1_2d": "4_1",
                 "medvae_8_4_2d": "8_4", "medvae_8_1_2d": "8_1"}


# --- Affine geometry ---
def affine_matrix(deg, scale, tx, ty, cx, cy):
    """Similarity transform about (cx, cy): rotate by `deg`, scale uniformly,
    then translate. Returns a 2x3 matrix mapping (x, y, 1) -> (x', y')."""
    th = np.radians(deg)
    c, s = np.cos(th) * scale, np.sin(th) * scale
    # x' = c(x-cx) - s(y-cy) + cx + tx ; y' = s(x-cx) + c(y-cy) + cy + ty
    return np.array([[c, -s, cx + tx - c * cx + s * cy],
                     [s,  c, cy + ty - s * cx - c * cy]], dtype=np.float64)


def apply_affine_points(pts, M):
    """pts (N,2) -> (N,2)."""
    h = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    return h @ M.T


def landmarks_to_points(lm136, W, H):
    """Ordering-A normalised landmarks -> (68,2) pixel coordinates."""
    return np.stack([lm136[:N_LM] * W, lm136[N_LM:] * H], 1)


def points_to_landmarks(pts, W, H):
    return np.concatenate([pts[:, 0] / W, pts[:, 1] / H]).astype(np.float32)


def endplate_tilts_px(pts):
    """(68,2) pixel points -> (17,) mean endplate tilt per vertebra, degrees.
    Computed in pixel space (project convention: angles are meaningless in the
    anisotropically-normalised space)."""
    V = pts.reshape(N_VERT, 4, 2)
    top = V[:, 1, :] - V[:, 0, :]
    bot = V[:, 3, :] - V[:, 2, :]
    edge = 0.5 * (top + bot)
    return np.degrees(np.arctan2(edge[:, 1], edge[:, 0]))


def max_cobb(pts):
    """Largest pairwise endplate-tilt difference, i.e. the major curve."""
    t = endplate_tilts_px(pts)
    return float(np.abs(t[:, None] - t[None, :]).max())


# --- Augmentation presets ---
PRESETS = {
    # Intensity-only. Everything here varies between real
    # acquisitions (kVp, mAs, patient habitus, detector calibration, dose).
    # No geometric change at all, so no fill artefacts and no deviation from
    # the acquisition protocol. This is the clinically conservative setting.
    "conservative": dict(max_deg=0.0, max_scale=0.0, max_shift=0.0,
                         jitter=0.10, gamma=0.15, noise=0.01, blur=0.0),
    # Adds scale only. Patient size and source-to-detector distance
    # vary, so scale variation is present in real data; rotation and
    # translation are not, because radiographers align and centre the patient.
    "moderate": dict(max_deg=0.0, max_scale=0.08, max_shift=0.0,
                     jitter=0.12, gamma=0.18, noise=0.015, blur=0.5),
    # Adds small rotation and translation. Not clinically motivated: these
    # deviate from the acquisition protocol and introduce fill corners that
    # never appear in real radiographs. Intended as a contrast condition for
    # the conservative preset.
    "aggressive": dict(max_deg=10.0, max_scale=0.10, max_shift=0.05,
                       jitter=0.10, gamma=0.15, noise=0.01, blur=0.5),
}


def intensity_augment(img, rng, jitter, gamma_rng, noise, blur):
    """Photometric augmentation modelling real acquisition variation.

    gain/bias : exposure and detector-offset differences
    gamma     : non-linear detector response
    noise     : quantum noise, which scales with dose
    blur      : focal-spot / patient-motion blur
    """
    import PIL.Image as Image
    from PIL import ImageFilter
    a = np.asarray(img, dtype=np.float32) / 255.0
    if gamma_rng:
        g = float(np.exp(rng.uniform(-gamma_rng, gamma_rng)))   # log-uniform
        a = np.power(np.clip(a, 1e-6, 1.0), g)
    if jitter:
        a = a * (1.0 + rng.uniform(-jitter, jitter)) + rng.uniform(-jitter, jitter)
    if noise:
        # dose-dependent: quantum noise is larger where signal is larger
        a = a + rng.normal(0, noise, a.shape).astype(np.float32) * np.sqrt(np.clip(a, 0, 1))
    a = np.clip(a, 0, 1)
    out = Image.fromarray((a * 255).astype(np.uint8))
    if blur and rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0, blur)))
    return out


# --- Single augmentation ---
def augment_once(img, pts, rng, max_deg, max_scale, max_shift, jitter,
                 gamma=0.0, noise=0.0, blur=0.0):
    """Return (augmented PIL image, transformed points) or (None, None) if the
    transform pushes any landmark outside the image.

    Geometry is applied first (if enabled), then photometry. With the
    conservative preset the geometry is the identity, so the landmarks are
    returned unchanged and no fill artefact is introduced."""
    import PIL.Image as Image
    W, H = img.size
    geometric = (max_deg > 0 or max_scale > 0 or max_shift > 0)

    if not geometric:
        return intensity_augment(img, rng, jitter, gamma, noise, blur), pts.copy()

    deg = rng.uniform(-max_deg, max_deg) if max_deg else 0.0
    scale = rng.uniform(1 - max_scale, 1 + max_scale) if max_scale else 1.0
    tx = rng.uniform(-max_shift, max_shift) * W if max_shift else 0.0
    ty = rng.uniform(-max_shift, max_shift) * H if max_shift else 0.0
    M = affine_matrix(deg, scale, tx, ty, W / 2.0, H / 2.0)

    new_pts = apply_affine_points(pts, M)
    margin = 2.0
    if (new_pts[:, 0].min() < margin or new_pts[:, 0].max() > W - margin
            or new_pts[:, 1].min() < margin or new_pts[:, 1].max() > H - margin):
        return None, None                    # spine would leave the frame

    # PIL's affine takes the inverse map (output -> input)
    Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
    out = img.transform((W, H), Image.AFFINE, data=Minv.flatten(),
                        resample=Image.BILINEAR, fillcolor=0)
    return intensity_augment(out, rng, jitter, gamma, noise, blur), new_pts


# --- Extraction run ---
def run(args):
    import pandas as pd
    import torch
    from PIL import Image
    from medvae import MVAE
    from medvae.utils.loaders import load_2d_finetune_aspect

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv)
    cobb_cols = [c for c in df.columns if c.startswith("angle_")]
    lookups = {}
    if args.train_lookup: lookups["train"] = pd.read_csv(args.train_lookup)
    if args.test_lookup: lookups["test"] = pd.read_csv(args.test_lookup)

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

    def match_ch(x, n):
        return x if x.shape[1] == n else x.repeat(1, n, 1, 1)

    def probe_ch(mvae, x):
        for c in (1, 3, 4):
            try:
                encode(mvae, match_ch(x, c).to(device)); return c
            except Exception:
                continue
        raise RuntimeError("could not determine encoder input channels")

    angle_resid = []
    for config in args.configs:
        mvae = MVAE(model_name=config, modality="xray").to(device).eval()
        lbl = CONFIG_LABELS.get(config, config)

        for split in ("train", "val", "test"):
            dst = out_dir / f"latents_{config}_{split}.npz"
            if dst.exists() and not args.force:
                print(f"skip (exists): {dst.name}"); continue
            sub = df[df["split"] == split].reset_index(drop=True)
            if not len(sub):
                continue
            u2r = uuid_map(split)
            all_lmk = lmk_rows(split)
            n_aug = args.n_aug if split == "train" else 0    # only the train split is augmented

            lat, uu, cobb, lmk = [], [], [], []
            probe = load_2d_finetune_aspect(
                str(Path(args.img_dir) / f"{sub.iloc[0]['image_uuid']}.jpg"),
                merge_channels=True).unsqueeze(0).to(device)
            n_in = probe_ch(mvae, probe)

            for _, r in sub.iterrows():
                u = str(r["image_uuid"])
                if u not in u2r:
                    print(f"  WARNING: no landmark row for {u}; this image is skipped."); continue
                p = Path(args.img_dir) / f"{u}.jpg"
                if not p.exists():
                    continue
                base_lm = all_lmk[u2r[u]].astype(np.float32)
                cobb_row = r[cobb_cols].to_numpy(dtype=np.float32)

                # --- the original ---
                x = match_ch(load_2d_finetune_aspect(str(p), merge_channels=True)
                             .unsqueeze(0), n_in).to(device)
                lat.append(encode(mvae, x).squeeze(0).float().cpu().numpy())
                uu.append(u); cobb.append(cobb_row); lmk.append(base_lm)

                # --- augmented copies ---
                if n_aug:
                    with Image.open(p) as im:
                        im = im.convert("L")
                        W, H = im.size
                        pts0 = landmarks_to_points(base_lm, W, H)
                        c0 = max_cobb(pts0)
                        made = 0
                        for _ in range(n_aug * 4):        # retries for rejections
                            if made >= n_aug:
                                break
                            aug, pts = augment_once(
                                im, pts0, rng, args.max_deg, args.max_scale,
                                args.max_shift, args.jitter,
                                gamma=args.gamma, noise=args.noise, blur=args.blur)
                            if aug is None:
                                continue
                            # Check that the similarity transform preserved the angle
                            resid = abs(max_cobb(pts) - c0)
                            angle_resid.append(resid)
                            if resid > args.angle_tol:
                                continue                   # should not happen
                            tmp = out_dir / "_tmp_aug.jpg"
                            aug.save(tmp, quality=95)
                            xa = match_ch(load_2d_finetune_aspect(
                                str(tmp), merge_channels=True).unsqueeze(0), n_in).to(device)
                            lat.append(encode(mvae, xa).squeeze(0).float().cpu().numpy())
                            uu.append(f"{u}__aug{made}")
                            cobb.append(cobb_row)          # unchanged: angles preserved
                            lmk.append(points_to_landmarks(pts, W, H))
                            made += 1
                        tmp = out_dir / "_tmp_aug.jpg"
                        if tmp.exists():
                            tmp.unlink()

            np.savez_compressed(dst, latents=np.stack(lat).astype(np.float32),
                                uuids=np.array(uu), cobb=np.stack(cobb),
                                landmarks=np.stack(lmk))
            print(f"  Wrote {dst.name}  latents={np.stack(lat).shape}  n={len(uu)}"
                  + (f"  ({n_aug} aug/image)" if n_aug else "  (unaugmented)"))
        del mvae
        if device == "cuda":
            torch.cuda.empty_cache()

    if angle_resid:
        ar = np.array(angle_resid)
        print(f"\nangle-preservation check: median residual {np.median(ar):.4f} deg, "
              f"max {ar.max():.4f} deg")
        print("  (similarity transforms preserve Cobb angles, so the stored angle")
        print("   labels remain valid for the augmented copies)")

    print(f"\nNext: train on the augmented set and compare with the original:\n"
          f"  python train_frontier.py --latents-dir {out_dir} --out-dir ./outputs/aug \\\n"
          f"      --configs {' '.join(args.configs)} --tasks cobb landmark "
          f"--seeds 0 1 2 3 4 5 6 7 8 9 --progress 25\n"
          f"  python compare_runs.py --run-a ./outputs/frontier --run-b ./outputs/aug \\\n"
          f"      --label-a original --label-b augmented")


# --- Synthetic-data self-test ---
def smoke():
    rng = np.random.default_rng(0)
    W, H = 400, 800

    # a synthetic spine: 17 vertebrae with varying tilts
    pts = []
    for i in range(N_VERT):
        deg = 25 * np.sin(i / N_VERT * np.pi)
        th = np.radians(deg); hw, hh = 40.0, 20.0
        corners = np.array([[-hw, -hh], [hw, -hh], [-hw, hh], [hw, hh]])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pts.append(corners @ R.T + [W / 2, 60 + i * 40])
    pts = np.concatenate(pts, 0)
    c0 = max_cobb(pts)
    print(f"original major curve: {c0:.4f} deg")

    # --- 1. similarity transforms must preserve the angle ---
    print("\nangle preservation under similarity transforms:")
    worst = 0.0
    for deg, sc, tx, ty in [(10, 1.0, 0, 0), (-7, 1.15, 20, -30),
                            (15, 0.85, -10, 25), (0, 1.3, 0, 0)]:
        M = affine_matrix(deg, sc, tx, ty, W / 2, H / 2)
        c1 = max_cobb(apply_affine_points(pts, M))
        worst = max(worst, abs(c1 - c0))
        print(f"  rot {deg:>4}deg scale {sc:.2f} shift ({tx:>4},{ty:>4}) "
              f"-> {c1:.4f} deg  (residual {abs(c1-c0):.2e})")

    # --- 2. a shear must change it (shows that the check can fail) ---
    Sh = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0]])
    c_sh = max_cobb(apply_affine_points(pts, Sh))
    print(f"\n  shear (excluded by design) -> {c_sh:.4f} deg "
          f"(residual {abs(c_sh-c0):.3f}; not preserved, as expected)")

    # --- 3. landmark round-trip through normalised coords ---
    lm = points_to_landmarks(pts, W, H)
    back = landmarks_to_points(lm, W, H)
    rt = float(np.abs(back - pts).max())
    print(f"\nnormalise round-trip max error: {rt:.2e} px")

    # --- 4. out-of-bounds rejection ---
    import PIL.Image as Image
    im = Image.new("L", (W, H), 128)
    rejected = sum(augment_once(im, pts, rng, 45, 0.5, 0.4, 0.0)[0] is None
                   for _ in range(40))
    print(f"aggressive transforms rejected (spine leaves frame): {rejected}/40")

    # --- 5. conservative preset must leave landmarks unchanged ---
    print("\nconservative preset (intensity only):")
    cons = PRESETS["conservative"]
    img2 = Image.fromarray((np.linspace(0, 255, W * H).reshape(H, W) % 256).astype(np.uint8))
    unchanged, changed_pixels = True, 0
    for _ in range(10):
        aug, p2 = augment_once(img2, pts, rng, cons["max_deg"], cons["max_scale"],
                               cons["max_shift"], cons["jitter"],
                               gamma=cons["gamma"], noise=cons["noise"], blur=cons["blur"])
        if aug is None or not np.array_equal(p2, pts):
            unchanged = False
        if not np.array_equal(np.asarray(aug), np.asarray(img2)):
            changed_pixels += 1
    print(f"  landmarks identical across 10 draws: {unchanged}")
    print(f"  pixels actually modified in {changed_pixels}/10 draws (should be 10)")
    print(f"  Cobb angle after conservative aug: {max_cobb(pts):.4f} deg (unchanged by construction)")

    # landmarks are stored as float32, so the round-trip is limited by that
    # precision (~1e-4 px on a 800px image), not by the transform maths
    ok = (worst < 1e-6 and abs(c_sh - c0) > 1.0 and rt < 1e-3 and rejected > 0
          and unchanged and changed_pixels == 10)
    print("\nSelf-test:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--img-dir", type=Path)
    ap.add_argument("--landmarks-dir", type=Path)
    ap.add_argument("--train-lookup", type=Path)
    ap.add_argument("--test-lookup", type=Path)
    ap.add_argument("--configs", nargs="+", default=["medvae_4_1_2d", "medvae_8_1_2d"])
    ap.add_argument("--out-dir", type=Path, default=Path("./latents_aug"))
    ap.add_argument("--n-aug", type=int, default=5, help="augmented copies per train image")
    ap.add_argument("--preset", choices=list(PRESETS), default="conservative",
                    help="conservative = intensity only (clinically conservative); "
                         "moderate = + scale; aggressive = + rotation/translation "
                         "(not clinically motivated)")
    ap.add_argument("--max-deg", type=float, default=None)
    ap.add_argument("--max-scale", type=float, default=None)
    ap.add_argument("--max-shift", type=float, default=None, help="fraction of W/H")
    ap.add_argument("--jitter", type=float, default=None, help="exposure gain/bias")
    ap.add_argument("--gamma", type=float, default=None, help="detector response, log-uniform")
    ap.add_argument("--noise", type=float, default=None, help="quantum noise sd")
    ap.add_argument("--blur", type=float, default=None, help="max Gaussian blur radius (px)")
    ap.add_argument("--angle-tol", type=float, default=0.05,
                    help="reject an augmentation if the major curve moves more than this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    preset = PRESETS[args.preset]
    for k, v in preset.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.smoke:
        smoke()
    elif args.csv and args.img_dir and args.landmarks_dir:
        print(f"preset '{args.preset}': rot +/-{args.max_deg}deg, scale +/-{args.max_scale}, "
              f"shift +/-{args.max_shift}, jitter {args.jitter}, gamma {args.gamma}, "
              f"noise {args.noise}, blur {args.blur}")
        if args.max_deg == 0 and args.max_shift == 0:
            print("  (intensity-only geometry: landmarks unchanged, no fill artefacts)")
        run(args)
    else:
        raise SystemExit("--csv, --img-dir and --landmarks-dir are required (or --smoke for the synthetic-data self-test)")


if __name__ == "__main__":
    main()