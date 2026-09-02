"""
merge_split_latents.py

Merge per-split latent files (latents_train.npz / latents_val.npz /
latents_test.npz) into a single unified .npz file in the schema expected
by train_cobb_regressor_matched.py (and train_cobb_regressor.py):

    latents : (N, C, H, W)
    angles  : (N, 3) — PT, MT, TL
    splits  : (N,)   — "train" / "val" / "test"
    uuids   : (N,)   — image_uuid strings

Cobb angles are joined from the BoostNet labels CSV by image_uuid.
Auto-detects angle column names from common naming conventions.

Usage:
    python merge_split_latents.py \\
        --latent-dir ./latents_whole \\
        --csv        ./medvae/data/boostnet_with_angles.csv \\
        --out        ./latents_whole/latents_whole_merged.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dir", required=True, type=Path,
                    help="Directory containing latents_train.npz, "
                         "latents_val.npz, latents_test.npz.")
    p.add_argument("--csv", required=True, type=Path,
                    help="BoostNet CSV with image_uuid + angle columns.")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    # ---- Load CSV and detect angle columns ----
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"  Columns: {df.columns.tolist()}")

    angle_column_candidates = [
        ("PT", "MT", "TL"),
        ("angle_proximal_thoracic", "angle_main_thoracic", "angle_thoracolumbar"),
        ("proximal_thoracic", "main_thoracic", "thoracolumbar"),
        ("cobb_pt", "cobb_mt", "cobb_tl"),
        ("cobb_angle_1", "cobb_angle_2", "cobb_angle_3"),
    ]
    pt_col = mt_col = tl_col = None
    for pt, mt, tl in angle_column_candidates:
        if pt in df.columns and mt in df.columns and tl in df.columns:
            pt_col, mt_col, tl_col = pt, mt, tl
            break
    if pt_col is None:
        raise ValueError(
            f"Could not find Cobb angle columns. Tried: "
            f"{angle_column_candidates}. Available: {df.columns.tolist()}"
        )
    print(f"  Using angle columns: ({pt_col}, {mt_col}, {tl_col})")

    angle_lookup = {row["image_uuid"]: (row[pt_col], row[mt_col], row[tl_col])
                     for _, row in df.iterrows()}

    # ---- Concatenate per-split files ----
    all_latents = []
    all_uuids   = []
    all_splits  = []
    all_angles  = []

    for split_name in ["train", "val", "test"]:
        path = args.latent_dir / f"latents_{split_name}.npz"
        if not path.exists():
            print(f"  ⚠ {path} missing, skipping")
            continue
        data = np.load(path, allow_pickle=True)
        lats = data["latents"]
        uids = data["uuids"].astype(str)
        print(f"  Loaded {split_name}: {len(lats)} samples, shape={lats.shape}")

        # Look up angles by uuid
        angs = []
        missing = []
        for u in uids:
            if u in angle_lookup:
                angs.append(angle_lookup[u])
            else:
                angs.append((np.nan, np.nan, np.nan))
                missing.append(u)
        if missing:
            print(f"    ⚠ {len(missing)} samples have no angle row in CSV "
                  f"(first 3: {missing[:3]})")

        all_latents.append(lats)
        all_uuids.append(uids)
        all_splits.append(np.array([split_name] * len(lats)))
        all_angles.append(np.array(angs, dtype=np.float32))

    if not all_latents:
        raise RuntimeError(f"No latent files found under {args.latent_dir}")

    merged_latents = np.concatenate(all_latents, axis=0)
    merged_uuids   = np.concatenate(all_uuids)
    merged_splits  = np.concatenate(all_splits)
    merged_angles  = np.concatenate(all_angles, axis=0)

    # Drop any rows with missing angles
    valid = ~np.isnan(merged_angles).any(axis=1)
    if not valid.all():
        n_drop = int((~valid).sum())
        print(f"\n  Dropping {n_drop} rows with missing angles")
        merged_latents = merged_latents[valid]
        merged_uuids   = merged_uuids[valid]
        merged_splits  = merged_splits[valid]
        merged_angles  = merged_angles[valid]

    print(f"\nMerged result:")
    print(f"  latents shape: {merged_latents.shape}")
    print(f"  Splits: {dict(zip(*np.unique(merged_splits, return_counts=True)))}")

    # ---- Save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        latents=merged_latents,
        angles=merged_angles,
        splits=merged_splits,
        uuids=merged_uuids,
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
