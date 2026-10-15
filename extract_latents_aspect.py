"""
extract_latents_aspect.py: aspect-padded MedVAE latents, Cobb angles and landmarks for train_frontier.py.

Produces, for each (config, split), one self-contained .npz:
    latents_{config}_{split}.npz  with
        latents   (N, Cz, H, W) float32   # H,W = 96 (4x) or 48 (8x)
        uuids     (N,)                     # image_uuid per row
        cobb      (N, 3) float32           # 3 Cobb angles
        landmarks (N, 136) float32         # normalised, ordering A (68 xs|68 ys)

Same aspect-padded preprocessing as the 96x96 Cobb baseline
(load_2d_finetune_aspect), so both tasks (Cobb and landmarks) read the same
representation at each compression level, which keeps train_frontier.py's
task comparison controlled. Idempotent: an existing .npz is skipped unless
--force.

Landmark-to-image alignment:
landmarks.csv has no ID column, so row order must be mapped to image_uuids.
For the test split the mapping is given by landmark_row_lookup_test_fixed.csv
(from fix_landmark_lookup.py). For the training labels (481 rows covering both
train and val) the mapping has to be established and checked. This script
resolves the mapping in this order:
    1. an explicit --train-lookup / --test-lookup CSV  (col: image_uuid, in row order)
    2. a filenames.csv sitting next to landmarks.csv   (BoostNet canonical order)
    3. fall back to the boostnet.csv order for that split  (source marked as unverified)
--verify overlays the parsed landmarks on sample images to check the
filename-to-label mapping before extraction, and prints which mapping source
was used. If the dots do not sit on the vertebra corners (training
especially), the lookup needs fixing before the full run.

Inputs:  --csv (image_uuid, split, 3 Cobb-angle columns); JPEG images in
         --img-dir; --landmarks-dir with training/ and test/ landmarks.csv;
         optionally --train-lookup / --test-lookup.

Usage (alignment check; no encoding, fast):
    python extract_latents_aspect.py --verify \\
        --csv ./medvae/data/boostnet_with_angles.csv \\
        --img-dir ./medvae/data/boostnet \\
        --landmarks-dir ./data/boostnet_labeldata/labels \\
        --test-lookup ./landmark_row_lookup_test_fixed.csv \\
        --out-dir ./latents_aspect

Full extraction (same command without --verify):
    python extract_latents_aspect.py \\
        --csv ./medvae/data/boostnet_with_angles.csv \\
        --img-dir ./medvae/data/boostnet \\
        --landmarks-dir ./data/boostnet_labeldata/labels \\
        --test-lookup ./landmark_row_lookup_test_fixed.csv \\
        --out-dir ./latents_aspect
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medvae import MVAE
from medvae.utils.loaders import load_2d_finetune_aspect

CONFIGS = ["medvae_4_1_2d", "medvae_8_1_2d"]
CONFIG_LABELS = {"medvae_4_3_2d": "4_3", "medvae_4_1_2d": "4_1",
                 "medvae_8_4_2d": "8_4", "medvae_8_1_2d": "8_1"}
N_LANDMARKS = 68
# training/landmarks.csv covers both the train and val splits.
SPLIT_TO_LMK_SUBDIR = {"train": "training", "val": "training", "test": "test"}
LANDMARK_COLOR = "#B85042"


# --- MedVAE ---
def transform_image(path):
    # identical preprocessing to the Cobb 96x96 baseline / the endplate script
    return load_2d_finetune_aspect(str(path), merge_channels=True).unsqueeze(0)


@torch.no_grad()
def encode_latent_projected(mvae, x):
    """MedVAE's Stage-2 projected latent, the representation the MedVAE paper
    uses for its downstream evaluations.

    z_bar = channel_proj(channel_ds(z) + z)

    channel_ds is Conv1x1 -> ReLU -> Conv3x3 -> ReLU -> Conv1x1 and channel_proj
    is a 1x1 convolution, so the projection mixes channels at each position with
    a receptive field of three positions. It cannot alter spatial resolution or
    recover discarded spatial detail; it is, however, the component trained to
    make additional latent channels useful, which is why the channel comparison
    needs it.

    Falls back to compute_latent_proj's own signalling when a checkpoint has no
    projection layers (apply_channel_ds=False), rather than silently returning
    the unprojected latent.
    """
    m = mvae.model
    if not hasattr(m, "compute_latent_proj"):
        raise RuntimeError(
            "this MedVAE build has no compute_latent_proj; cannot extract the "
            "projected latent. Re-run without --projected.")
    out = m.compute_latent_proj(x, sample_posterior=False)
    if not (isinstance(out, (tuple, list)) and len(out) == 3):
        raise RuntimeError(f"unexpected compute_latent_proj return: {type(out)}")
    z, _posterior, z_bar = out
    if z_bar is None:
        raise RuntimeError(
            "compute_latent_proj returned None for the projected latent, which "
            "means this checkpoint was built with apply_channel_ds=False and has "
            "no Stage-2 projection. Re-run without --projected.")
    return z_bar.squeeze(0).float().cpu()


@torch.no_grad()
def encode_latent(mvae, x):
    """Deterministic latent (posterior mode/mean). Robust to MedVAE's encode
    returning a tensor, a tuple, or a distribution object.

    This is the raw encoder output z, not the Stage-2 projected latent that
    MedVAE's own downstream evaluations use; --projected selects the latter."""
    out = mvae.model.encode(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out):
        z = out
    elif hasattr(out, "mode"):
        z = out.mode()
    elif hasattr(out, "mean"):
        z = out.mean
    elif hasattr(out, "sample"):
        z = out.sample()
    else:
        raise RuntimeError(f"Unrecognised encode() return: {type(out)}")
    return z.squeeze(0).float().cpu()   # (Cz, H, W)


# set in main() from --projected; keeps the two call sites identical
_ENCODE = encode_latent


def match_in_channels(x, n_in):
    """Replicate a 1-channel image to the n_in channels a checkpoint expects.
    The 3c/4c MedVAE models take a 3-channel input; the 1c models take 1."""
    if x.shape[1] == n_in:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, n_in, 1, 1)
    raise ValueError(f"cannot match input {x.shape[1]}ch -> {n_in}ch")


@torch.no_grad()
def probe_in_channels(mvae, x1, device):
    """Find how many input channels this checkpoint's encoder wants (1, 3, or 4)."""
    for c in (1, 3, 4):
        try:
            _ENCODE(mvae, match_in_channels(x1, c).to(device))
            return c
        except Exception:
            continue
    raise RuntimeError("could not determine encoder input channels")


# --- Landmarks ---
def parse_landmark_row(vals):
    xs = np.asarray(vals[:N_LANDMARKS], dtype=float)
    ys = np.asarray(vals[N_LANDMARKS:], dtype=float)
    return np.concatenate([xs, ys])   # (136,) ordering A preserved


def build_uuid_to_row(split, df_split, landmarks_dir, lookups):
    """Return {image_uuid: landmark_row_index} for this split, plus a source
    string. Honours an explicit lookup's `landmark_row` column when present
    (robust to file ordering); otherwise falls back to positional order."""
    subdir = SPLIT_TO_LMK_SUBDIR[split]
    lk = lookups.get("train" if subdir == "training" else "test")
    if lk is not None:
        ucol = "image_uuid" if "image_uuid" in lk.columns else lk.columns[0]
        if "landmark_row" in lk.columns:
            m = {str(u): int(r) for u, r in zip(lk[ucol], lk["landmark_row"])}
        else:
            m = {str(u): i for i, u in enumerate(lk[ucol].astype(str))}
        return m, f"explicit lookup ({subdir}, {len(m)} rows)"
    fnames = Path(landmarks_dir) / subdir / "filenames.csv"
    if fnames.exists():
        f = pd.read_csv(fnames, header=None)[0].astype(str)
        m = {Path(s).stem: i for i, s in enumerate(f)}
        return m, f"filenames.csv ({subdir}); raw stems, not canonicalised"
    return ({str(u): i for i, u in enumerate(df_split["image_uuid"].astype(str))},
            f"fallback: boostnet.csv row order ({split}); mapping not verified")


def load_landmarks_for_split(split, landmarks_dir):
    subdir = SPLIT_TO_LMK_SUBDIR[split]
    arr = pd.read_csv(Path(landmarks_dir) / subdir / "landmarks.csv",
                      header=None).values
    return np.stack([parse_landmark_row(r) for r in arr])   # (R,136)


# --- Verify: overlay landmarks on a few images per split ---
def content_bbox(t, thresh=-0.8):
    m = (t > thresh).squeeze()
    rows = m.any(dim=1).nonzero(as_tuple=True)[0]
    cols = m.any(dim=0).nonzero(as_tuple=True)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, t.shape[-2], 0, t.shape[-1]
    return rows[0].item(), rows[-1].item() + 1, cols[0].item(), cols[-1].item() + 1


def verify(args, df, lookups):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax, split in zip(axes, ("train", "val", "test")):
        sub = df[df["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            ax.set_title(f"{split}: empty"); ax.axis("off"); continue
        uuid_to_row, src = build_uuid_to_row(split, sub, args.landmarks_dir, lookups)
        lmk = load_landmarks_for_split(split, args.landmarks_dir)
        target = sub.iloc[len(sub) // 2]["image_uuid"]
        if str(target) not in uuid_to_row:
            ax.set_title(f"{split}: '{target}' not in mapping\n{src}")
            ax.axis("off"); print(f"[{split}] {src}: target uuid not found in the mapping; overlay skipped for this split."); continue
        row = lmk[uuid_to_row[str(target)]]
        x = transform_image(Path(args.img_dir) / f"{target}.jpg")
        r0, r1, c0, c1 = content_bbox(x)
        img = x.squeeze().numpy()
        xs = c0 + row[:N_LANDMARKS] * (c1 - c0)
        ys = r0 + row[N_LANDMARKS:] * (r1 - r0)
        ax.imshow(img, cmap="gray")
        ax.scatter(xs, ys, s=14, c=LANDMARK_COLOR, edgecolors="white", linewidths=0.4)
        ax.set_title(f"{split}: {target}\n{src}", fontsize=8)
        ax.axis("off")
        print(f"[{split}] source = {src}")
    fig.suptitle("Landmark alignment check: dots should sit on vertebra corners")
    fig.tight_layout()
    p = out / "landmark_alignment_check.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"\nSaved {p}. The overlay is the alignment check for the full run.")


# --- Full extraction ---
def extract(args, df, lookups):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cobb_cols = [c for c in ("pt", "mt", "tl_l") if c in df.columns] or \
                [c for c in df.columns if c.lower().startswith(("angle", "cobb"))][:3]
    if len(cobb_cols) != 3:
        raise SystemExit(f"Could not identify 3 Cobb-angle columns; got {cobb_cols}. "
                         f"Pass the right --csv or edit cobb_cols.")
    print(f"Cobb columns: {cobb_cols}")

    for config in args.configs:
        mvae = MVAE(model_name=config, modality="xray").to(device).eval()
        # 1c vs 3c/4c checkpoints expect different input channel counts
        probe_x = transform_image(Path(args.img_dir) / f"{df.iloc[0]['image_uuid']}.jpg")
        n_in = probe_in_channels(mvae, probe_x, device)
        print(f"[{CONFIG_LABELS[config]}] encoder expects {n_in}-channel input")
        for split in ("train", "val", "test"):
            suffix = "_proj" if args.projected else ""
            dst = out / f"latents_{config}{suffix}_{split}.npz"
            if dst.exists() and not args.force:
                print(f"skip (exists): {dst.name}"); continue
            sub = df[df["split"] == split].reset_index(drop=True)
            if len(sub) == 0:
                print(f"no {split} rows, skip"); continue

            uuid_to_row, src = build_uuid_to_row(split, sub, args.landmarks_dir, lookups)
            lmk_all = load_landmarks_for_split(split, args.landmarks_dir)
            print(f"[{CONFIG_LABELS[config]} {split}] landmark source: {src}")

            lat_list, uuids, cobb, lmk = [], [], [], []
            for _, r in sub.iterrows():
                u = str(r["image_uuid"])
                if u not in uuid_to_row:
                    print(f"  WARNING: no landmark row for {u}; this image is skipped."); continue
                x = match_in_channels(transform_image(Path(args.img_dir) / f"{u}.jpg"), n_in).to(device)
                lat_list.append(_ENCODE(mvae, x).numpy())
                uuids.append(u)
                cobb.append(r[cobb_cols].to_numpy(dtype=np.float32))
                lmk.append(lmk_all[uuid_to_row[u]].astype(np.float32))

            latents = np.stack(lat_list).astype(np.float32)
            np.savez_compressed(dst, latents=latents, uuids=np.array(uuids),
                                cobb=np.stack(cobb), landmarks=np.stack(lmk),
                                representation=np.array(
                                    "projected" if args.projected else "encoder_output"))
            print(f"  Wrote {dst.name}  latents={latents.shape}  n={len(uuids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="boostnet csv with image_uuid, split, and 3 Cobb-angle cols")
    ap.add_argument("--projected", action="store_true",
                    help="extract MedVAE's Stage-2 projected latent (z_bar) "
                         "instead of the raw encoder output (z). This is the "
                         "representation the MedVAE paper evaluates on.")
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--landmarks-dir", type=Path, required=True,
                    help="dir containing training/landmarks.csv and test/landmarks.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-lookup", type=Path)
    ap.add_argument("--test-lookup", type=Path)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # Choose the representation once, here, so both call sites agree.
    global _ENCODE
    if args.projected:
        _ENCODE = encode_latent_projected
        print("Extracting the projected latent (z_bar = channel_proj(channel_ds(z) + z))")
        print("  This is the representation MedVAE's own downstream evaluations use.")
        print("  Output files are suffixed '_proj' so they cannot be confused with z.\n")
    else:
        _ENCODE = encode_latent

    df = pd.read_csv(args.csv)
    lookups = {}
    if args.train_lookup:
        lookups["train"] = pd.read_csv(args.train_lookup)
    if args.test_lookup:
        lookups["test"] = pd.read_csv(args.test_lookup)

    if args.verify:
        verify(args, df, lookups)
    else:
        extract(args, df, lookups)


if __name__ == "__main__":
    raise SystemExit(main())
