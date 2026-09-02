"""
diagnose_train_alignment.py — find the train landmark<->uuid crosswalk.

Read-only. The test split maps fine via landmark_row_lookup_test_fixed.csv;
training does not, because training/filenames.csv stems and the pipeline's
image_uuids use different naming. This script inspects the real files and
reports the relationship so we can build a correct train lookup (rather than
silently dropping 96% of the training set, as the extraction just did).

Run:
  python diagnose_train_alignment.py \
    --csv ./medvae/data/boostnet_with_angles.csv \
    --landmarks-dir ~/MedVAE/data/boostnet_labeldata/labels \
    --workdir .
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def stem(s):
    return Path(str(s)).stem


def show(label, items, k=8):
    items = list(items)
    print(f"{label} (n={len(items)})")
    print(f"  first {k}: {items[:k]}")
    if len(items) > k:
        print(f"  last  {k}: {items[-k:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--landmarks-dir", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=Path("."))
    args = ap.parse_args()

    print("=" * 70)
    print("1) TRAINING landmark files")
    print("=" * 70)
    tdir = args.landmarks_dir / "training"
    lm = pd.read_csv(tdir / "landmarks.csv", header=None)
    print(f"training/landmarks.csv shape: {lm.shape}  (expect (481, 136))")
    fn_path = tdir / "filenames.csv"
    fn_stems = None
    if fn_path.exists():
        fn = pd.read_csv(fn_path, header=None)
        print(f"training/filenames.csv shape: {fn.shape}")
        col0 = fn.iloc[:, 0].astype(str)
        fn_stems = [stem(x) for x in col0]
        has_ext = any("." in str(x) for x in col0.head(20))
        print(f"  entries have extensions: {has_ext}")
        show("  filenames col0 stems", fn_stems)
        if lm.shape[0] != len(fn_stems):
            print(f"  !! row-count mismatch: {len(fn_stems)} filenames vs "
                  f"{lm.shape[0]} landmark rows — order pairing is suspect.")
    else:
        print("  NO training/filenames.csv found.")
    print("  files in training dir:", [p.name for p in tdir.glob('*')])

    print("\n" + "=" * 70)
    print("2) PIPELINE csv")
    print("=" * 70)
    df = pd.read_csv(args.csv)
    print(f"{args.csv.name} shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(df.head(3).to_string())
    if "split" in df.columns:
        print("\nsplit counts:", df["split"].value_counts().to_dict())
    trv = df[df["split"].isin(["train", "val"])] if "split" in df else df
    uuids = trv["image_uuid"].astype(str).tolist()
    show("\ntrain+val image_uuid", uuids)

    # any column that looks like it could hold an original filename?
    print("\nscanning pipeline columns for a filename-like crosswalk:")
    for c in df.columns:
        sample = df[c].astype(str).head(50)
        if sample.str.contains("sunhl", case=False).any() and c != "image_uuid":
            print(f"  candidate column '{c}': {sample.head(3).tolist()}")

    print("\n" + "=" * 70)
    print("3) OVERLAP under several normalisations")
    print("=" * 70)
    if fn_stems is not None:
        norms = {
            "raw stem": lambda s: s,
            "lower": lambda s: s.lower(),
            "strip _AP": lambda s: s.replace("_AP", "").replace("_ap", ""),
            "strip trailing _X_AP": lambda s: "_".join(s.split("_")[:-2]) if s.count("_") >= 2 else s,
            "alnum only lower": lambda s: "".join(ch for ch in s.lower() if ch.isalnum()),
        }
        uset = set(uuids)
        for name, f in norms.items():
            lm_set = {f(s) for s in fn_stems}
            u_set = {f(u) for u in uuids}
            ov = lm_set & u_set
            print(f"  {name:<22}: overlap {len(ov)} / {len(u_set)} train+val uuids")
        # show the ones that match raw, to see their shape
        raw_match = sorted(set(fn_stems) & uset)
        show("\n  RAW-matching uuids (the ~handful that survived)", raw_match)
        # and a few that DON'T match, for contrast
        nomatch = sorted(uset - set(fn_stems))
        show("  NON-matching pipeline uuids", nomatch)

    print("\n" + "=" * 70)
    print("4) existing lookup/crosswalk files in workdir")
    print("=" * 70)
    pats = ["*lookup*", "*crosswalk*", "*mapping*", "*filename*", "*row*"]
    seen = set()
    for pat in pats:
        for p in args.workdir.glob(pat):
            if p.name in seen:
                continue
            seen.add(p.name)
            try:
                h = pd.read_csv(p, nrows=2)
                print(f"  {p.name}: cols={list(h.columns)}")
            except Exception as e:
                print(f"  {p.name}: (unreadable: {e})")
    # also peek at the working test lookup to mirror its schema for train
    tl = args.workdir / "landmark_row_lookup_test_fixed.csv"
    if tl.exists():
        t = pd.read_csv(tl)
        print(f"\n  test lookup schema: cols={list(t.columns)}  shape={t.shape}")
        print(t.head(3).to_string())


if __name__ == "__main__":
    main()
