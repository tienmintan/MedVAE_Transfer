"""Read-only check of projected latent caches against the split CSV and the raw caches.

For each configuration in --configs (default medvae_4_3_2d and medvae_8_4_2d)
and each split, checks that the projected cache is marked "projected", that its
identifiers match the split CSV and the raw cache without duplicates, that the
latents have the expected shape, are finite float32 and not constant, and that
the cobb and landmark labels match the raw cache after aligning by identifier.
Any failed check raises ValueError.

Requires numpy and pandas. Does not verify checkpoint identity or the
anatomical correctness of labels. It has not been run against the project NPZ
files.

Inputs:  --projected-dir and --raw-dir (latents_{config}_{split}.npz in each),
         --split-csv (image_uuid and split columns).
Outputs: a JSON summary of the passed checks, printed to stdout.

Usage:
  python check_projected_caches.py --projected-dir <dir> --raw-dir <dir> \
      --split-csv <csv> [--configs medvae_4_3_2d medvae_8_4_2d]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHAPES = {"medvae_4_3_2d": (3, 96, 96),
                   "medvae_8_4_2d": (4, 48, 48),
                   "medvae_4_1_2d": (1, 96, 96),
                   "medvae_8_1_2d": (1, 48, 48)}
EXPECTED_COUNTS = {"train": 409, "val": 72, "test": 128}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projected-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", default=list(EXPECTED_SHAPES)[:2],
                    choices=list(EXPECTED_SHAPES))
    args = ap.parse_args()
    frame = pd.read_csv(args.split_csv, dtype={"image_uuid": str})
    require({"image_uuid", "split"}.issubset(frame.columns), "Missing split columns")
    require(not frame.image_uuid.isna().any(), "Missing image identifier")
    require(not frame.image_uuid.duplicated().any(), "Duplicate/overlapping split identifiers")
    require(set(frame.split) == set(EXPECTED_COUNTS), "Unexpected split values")
    expected = {s: set(frame.loc[frame.split == s, "image_uuid"])
                for s in EXPECTED_COUNTS}
    for split, count in EXPECTED_COUNTS.items():
        require(len(expected[split]) == count, f"Wrong split count: {split}")
    records = []
    for config in args.configs:
        for split, count in EXPECTED_COUNTS.items():
            name = f"latents_{config}_{split}.npz"
            with np.load(args.projected_dir / name, allow_pickle=False) as proj, \
                 np.load(args.raw_dir / name, allow_pickle=False) as raw:
                require("representation" in proj.files, f"No representation marker: {name}")
                require(str(proj["representation"].item()) == "projected",
                        f"Cache not marked projected: {name}")
                ids = proj["uuids"].astype(str)
                raw_ids = raw["uuids"].astype(str)
                require(ids.shape == (count,) and len(set(ids)) == count,
                        f"Incorrect/duplicate projected identifiers: {name}")
                require(set(ids) == expected[split], f"Projected IDs differ from split CSV: {name}")
                require(raw_ids.shape == (count,) and len(set(raw_ids)) == count,
                        f"Incorrect/duplicate raw identifiers: {name}")
                require(set(raw_ids) == set(ids), f"Raw/projected IDs differ: {name}")
                p_order, r_order = np.argsort(ids), np.argsort(raw_ids)
                z = proj["latents"]
                require(z.shape == (count, *EXPECTED_SHAPES[config]), f"Wrong shape: {name}")
                require(z.dtype == np.float32 and np.isfinite(z).all(),
                        f"Non-finite or non-float32 latents: {name}")
                require(float(np.std(z)) > 0, f"Constant latent array: {name}")
                for key, size in [("cobb", 3), ("landmarks", 136)]:
                    a, b = proj[key], raw[key]
                    require(a.shape == b.shape == (count, size), f"Wrong {key} shape: {name}")
                    require(np.isfinite(a).all() and np.isfinite(b).all(),
                            f"Non-finite {key}: {name}")
                    require(np.allclose(a[p_order], b[r_order], atol=1e-6, rtol=0),
                            f"Raw/projected {key} labels differ: {name}")
                records.append(dict(config=config, split=split, n=count,
                                    shape=list(z.shape), representation="projected",
                                    latent_sd=float(np.std(z)), raw_labels_match=True))
    print(json.dumps({"cache_checks_passed": records,
                      "scope": "Stored-array checks only; checkpoint weights and run logs are not checked"},
                     indent=2))


if __name__ == "__main__":
    main()
