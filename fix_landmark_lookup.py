"""
fix_landmark_lookup.py — repair the uuid<->landmark-row mapping.

The visual alignment confirmed landmark ROWS map to the right IMAGES. But the
landmark filenames ('sunhl-...-310 C AP.jpg', spaces + mixed case + extension)
don't string-match the pipeline uuids ('sunhl-...-310_C_AP', underscores, no
ext), so the naive cross-check gave 0/128 overlap.

This canonicalises both sides to a common key and rebuilds the lookup:
    canonical(s) = lowercase, spaces/dashes-in-id -> underscores handled,
                   extension stripped.

We DON'T blindly trust it: we report the overlap after canonicalisation and
print any residual mismatches so they can be eyeballed before Step 2.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

HOME = Path.home()
LABELDATA = HOME / "MedVAE" / "data" / "boostnet_labeldata"
PIPELINE_CSV = HOME / "MedVAE" / "MedVAE" / "medvae" / "data" / "boostnet.csv"
SPLIT_DIRNAME = {"train": "training", "val": "training", "test": "test"}
SPLIT_KEY = {"train": "train", "val": "val", "test": "test"}


def canonical(s: str) -> str:
    """Map any filename/uuid variant to a common key."""
    s = str(s).strip()
    s = re.sub(r"\.(jpg|jpeg|png|bmp)$", "", s, flags=re.IGNORECASE)  # drop ext
    s = s.lower()
    s = s.replace(" ", "_").replace("-_", "_")
    s = re.sub(r"_+", "_", s)            # collapse repeats
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    sub = SPLIT_DIRNAME[args.split]
    label_dir = LABELDATA / "labels" / sub
    fn_path = label_dir / "filenames.csv"
    lm_path = label_dir / "landmarks.csv"

    filenames = pd.read_csv(fn_path, header=None).iloc[:, 0].astype(str).tolist()
    n_lm = len(pd.read_csv(lm_path, header=None))
    assert len(filenames) == n_lm, f"{len(filenames)} filenames vs {n_lm} landmark rows"

    pdf = pd.read_csv(PIPELINE_CSV)
    pipe = pdf[pdf["split"] == SPLIT_KEY[args.split]]["image_uuid"].astype(str).tolist()

    lm_canon = {canonical(f): i for i, f in enumerate(filenames)}
    pipe_canon = {canonical(u): u for u in pipe}

    overlap = set(lm_canon) & set(pipe_canon)
    print(f"Split '{args.split}':  {len(filenames)} landmark rows, "
          f"{len(pipe)} pipeline uuids")
    print(f"Canonical overlap: {len(overlap)}/{len(pipe)}")

    if len(overlap) != len(pipe):
        missing = set(pipe_canon) - set(lm_canon)
        print(f"  {len(missing)} pipeline uuids still unmatched. Examples:")
        for k in list(missing)[:5]:
            print(f"    pipeline '{pipe_canon[k]}' -> canonical '{k}'")
        print("  Closest landmark canonicals:")
        for k in list(lm_canon)[:5]:
            print(f"    landmark canonical '{k}'")
    else:
        print("  -> FULL match. Every pipeline image has a landmark row.")

    # Build lookup keyed by the ORIGINAL pipeline uuid (what Step 2 will query with)
    rows = []
    for canon, orig_uuid in pipe_canon.items():
        if canon in lm_canon:
            rows.append({"image_uuid": orig_uuid,
                         "landmark_row": lm_canon[canon],
                         "landmark_filename": filenames[lm_canon[canon]]})
    lookup = pd.DataFrame(rows).sort_values("landmark_row").reset_index(drop=True)
    out = Path.cwd() / f"landmark_row_lookup_{args.split}_fixed.csv"
    lookup.to_csv(out, index=False)
    print(f"\nSaved {out}  ({len(lookup)} matched rows)")
    print("Columns: image_uuid (pipeline key), landmark_row, landmark_filename")
    print("\nSpot-check (first 3):")
    print(lookup.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
