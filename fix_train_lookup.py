"""
fix_train_lookup.py — build the TRAIN uuid<->landmark-row lookup.

Same idea as fix_landmark_lookup.py (which fixed test): training/filenames.csv
is row-paired with landmarks.csv, but its stems use spaces + extensions
('sunhl-...-162 A AP.jpg') while pipeline uuids use underscores
('sunhl-...-162_A_AP'). Canonicalising both sides (lowercase, strip extension,
drop non-alphanumerics) makes them join 1:1 (481/481 confirmed by the
diagnostic). Output schema matches the test lookup:
    image_uuid, landmark_row, landmark_filename

VERIFICATION (non-visual, strong): the labels dir also has training/angles.csv,
positionally row-paired with landmarks.csv. For the join to be correct, row i's
angles must match the pipeline angles of the uuid we mapped to row i. We
cross-correlate the two angle matrices over all matched rows: a correct mapping
shows a clean permutation of ~1.0 correlations; a scrambled mapping shows ~0.
This catches a bad row pairing that a spine overlay would miss (every image
here is a vertical spine, so wrong points still look plausible).

Run:
  python fix_train_lookup.py \
    --csv ./medvae/data/boostnet_with_angles.csv \
    --landmarks-dir ~/MedVAE/data/boostnet_labeldata/labels \
    --out ./landmark_row_lookup_train_fixed.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PIPE_ANGLE_COLS = ["angle_proximal_thoracic", "angle_main_thoracic", "angle_thoracolumbar"]


def strip_ext(s):
    s = str(s)
    while True:
        st = Path(s).stem
        if st == s:
            return s
        s = st


def canon(s):
    return re.sub(r"[^a-z0-9]", "", strip_ext(s).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--landmarks-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./landmark_row_lookup_train_fixed.csv"))
    args = ap.parse_args()

    tdir = args.landmarks_dir / "training"
    fn = pd.read_csv(tdir / "filenames.csv", header=None).iloc[:, 0].astype(str).tolist()
    lm = pd.read_csv(tdir / "landmarks.csv", header=None)
    assert len(fn) == len(lm), f"filenames {len(fn)} != landmark rows {len(lm)}"
    R = len(fn)
    print(f"filenames/landmarks rows: {R} (paired by position)")

    df = pd.read_csv(args.csv)
    trv = df[df["split"].isin(["train", "val"])].copy()
    trv["key"] = trv["image_uuid"].map(canon)
    key_to_uuid = dict(zip(trv["key"], trv["image_uuid"]))
    if len(key_to_uuid) != len(trv):
        print("  WARN: pipeline uuids collide under canonicalisation — inspect!")

    rows, unmatched = [], []
    for i, f in enumerate(fn):
        k = canon(f)
        u = key_to_uuid.get(k)
        if u is None:
            unmatched.append((i, f))
        else:
            rows.append({"image_uuid": u, "landmark_row": i, "landmark_filename": f})
    out = pd.DataFrame(rows).sort_values("landmark_row").reset_index(drop=True)

    print(f"\nmatched {len(out)} / {R} landmark rows to pipeline uuids")
    print(f"pipeline train+val uuids covered: {out['image_uuid'].nunique()} / {len(trv)}")
    if unmatched:
        print(f"  {len(unmatched)} UNMATCHED filenames (first 5): {unmatched[:5]}")
    missing_uuids = set(trv["image_uuid"]) - set(out["image_uuid"])
    if missing_uuids:
        print(f"  {len(missing_uuids)} pipeline uuids with no landmark row (first 5): "
              f"{list(missing_uuids)[:5]}")

    # ---- strong verification: angles cross-check ----
    ang_path = tdir / "angles.csv"
    if ang_path.exists():
        A_all = pd.read_csv(ang_path, header=None).apply(pd.to_numeric, errors="coerce").values
        if A_all.shape[0] == R and A_all.shape[1] >= 3:
            pipe_ang = df.set_index("image_uuid")[PIPE_ANGLE_COLS]
            keep = [r for r in out.to_dict("records")]
            A = np.array([A_all[r["landmark_row"], :3] for r in keep], dtype=float)
            P = np.array([pipe_ang.loc[r["image_uuid"]].values for r in keep], dtype=float)
            ok = np.isfinite(A).all(1) & np.isfinite(P).all(1)
            A, P = A[ok], P[ok]
            print(f"\nangle cross-check on {ok.sum()} rows (angles.csv vs pipeline):")
            corr = np.zeros((3, 3))
            for a in range(3):
                for p in range(3):
                    corr[a, p] = np.corrcoef(A[:, a], P[:, p])[0, 1]
            np.set_printoptions(precision=3, suppress=True)
            print("  corr matrix (rows=angles.csv cols, cols=pipeline cols):")
            print("  " + str(corr).replace("\n", "\n  "))
            best = corr.max(axis=1)
            print(f"  best per-row-column correlation: {np.round(best,3)}")
            if (best > 0.9).all():
                print("  -> CLEAN permutation of ~1.0 correlations: row pairing CONFIRMED.")
            else:
                print("  -> WEAK: row pairing NOT confirmed. Do not trust before inspecting.")
        else:
            print(f"\nangles.csv shape {A_all.shape} unexpected; skipping angle check.")
    else:
        print("\nno training/angles.csv found; skipping angle cross-check "
              "(rely on the extraction --verify overlay instead).")

    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows)")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
