"""
verify_angle_semantics.py: determine what the three Cobb angle columns represent.

The dataset provides three unlabelled angle columns; the PT/MT/TL names in
`boostnet_with_angles.csv` were assigned positionally and are unverified, and
the generating code is not available. The semantics of the columns matter more
than their names, and there are two possibilities:

  (A) Anatomical slots: column k always refers to the same spinal region
      (e.g. col0 = proximal thoracic every time). Only the names are then
      affected; after relabelling, per-angle results keep their meaning.

  (B) Rank slots: column 0 is "the largest angle" and cols 1-2 the secondary
      curves, as most `cobb_angle_calc` implementations return. Column 0 is
      then whichever curve is largest for that patient, so its anatomical
      position varies from image to image. No column is "the main thoracic",
      and per-angle results describe major vs secondary curves.

The script distinguishes between them using the ground-truth landmarks only.

Check 1: rank test
    Is column 0 the maximum of the three in ~every image? Are the columns
    sorted descending? If so, they are rank slots by construction.

Check 2: vertebra-pair matching (anatomical identity)
    For each image, compute all 17 vertebral endplate tilts, then for each
    column value find the vertebra pair whose tilt difference reproduces it
    most closely. That identifies which vertebrae the angle spans. The apex
    level per column across images then indicates consistency:
      * tight apex spread  -> that column is an anatomical slot
      * scattered apex     -> that column is a rank slot

Reliability measures:
    * `residual`: how closely the best pair reproduces the value. If large,
      the matching is unreliable and the results that depend on it may be
      unreliable too.
    * `ambiguity`: how many different pairs reproduce the value within 1 deg.
      Angle differences are degenerate, so a high count means the apex
      assignment for that image is weak.

Inputs:  --preds, any preds_*.npz containing gt_landmarks and gt_cobb (and
         uuids, used with --img-dir to rescale landmarks to pixel space).
Outputs: the check results and a verdict printed to stdout; with --out-dir,
         angle_semantics_per_match.csv.

Usage:
    python verify_angle_semantics.py --preds ./outputs/frontier/preds_medvae_4_1_2d.npz
    python verify_angle_semantics.py --smoke      # synthetic-data self-test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_LM = 68
N_VERT = 17
COLS = ["angle_proximal_thoracic", "angle_main_thoracic", "angle_thoracolumbar"]
VERT_NAMES = [f"T{i+1}" for i in range(12)] + [f"L{i+1}" for i in range(5)]
TOL_AMBIG = 1.0        # deg: another pair this close counts as ambiguous
RESID_OK = 2.0         # deg: median residual above this -> matching unreliable
APEX_TIGHT = 2.0       # vertebrae: apex IQR below this -> anatomically consistent


# --- Geometry ---
# Validated grouping: vmajor, identity corner order; corners 0,1 = top edge,
# 2,3 = bottom edge.
def to_vertebrae(rows):
    x, y = rows[:, :N_LM], rows[:, N_LM:]
    return np.stack([x, y], -1).reshape(-1, N_VERT, 4, 2)


def find_image(img_dir, uuid):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = Path(img_dir) / f"{uuid}{ext}"
        if p.exists():
            return p
    return None


def rescale_to_pixels(gt_lm, uuids, img_dir):
    """Landmarks are normalised anisotropically: x by image width, y by image
    height. Spine radiographs are ~1:2 (tall), so in that space every vertical
    displacement is shrunk ~2x relative to horizontal, which flattens every
    endplate tilt by roughly half. Angles are therefore computed in pixel
    space, not in normalised space. Returns (landmarks_px, n_found)."""
    from PIL import Image
    out = gt_lm.astype(float).copy()
    found = 0
    for i, u in enumerate(uuids):
        p = find_image(img_dir, str(u))
        if p is None:
            continue
        try:
            with Image.open(p) as im:
                W, H = im.width, im.height
        except Exception:
            continue
        out[i, :N_LM] = gt_lm[i, :N_LM] * W
        out[i, N_LM:] = gt_lm[i, N_LM:] * H
        found += 1
    return out, found


def endplate_tilts(V):
    """Mean tilt per vertebra (averaged-edge construction, labelled 'legacy' in
    the output; used for comparison)."""
    top = V[:, :, 1, :] - V[:, :, 0, :]
    bot = V[:, :, 3, :] - V[:, :, 2, :]
    edge = 0.5 * (top + bot)
    return np.degrees(np.arctan2(edge[..., 1], edge[..., 0]))       # (N,17)


def endplate_tilts_tb(V):
    """Separate superior and inferior endplate tilts per vertebra.

    A clinical Cobb angle is measured between the superior endplate of the
    upper end vertebra and the inferior endplate of the lower end vertebra
    (two different edges). Averaging a vertebra's two edges (as endplate_tilts
    does) averages out vertebral wedging, so the two constructions give
    different angles. construction_comparison measures which of them
    reproduces the stored angles more closely and uses that one."""
    top = V[:, :, 1, :] - V[:, :, 0, :]      # superior endplate
    bot = V[:, :, 3, :] - V[:, :, 2, :]      # inferior endplate
    t_sup = np.degrees(np.arctan2(top[..., 1], top[..., 0]))
    t_inf = np.degrees(np.arctan2(bot[..., 1], bot[..., 0]))
    return t_sup, t_inf                                             # (N,17) each


# --- Check 1: rank test ---
def rank_test(gt_cobb):
    n = len(gt_cobb)
    is_max0 = (np.argmax(gt_cobb, axis=1) == 0).mean()
    is_min2 = (np.argmin(gt_cobb, axis=1) == 2).mean()
    desc = ((gt_cobb[:, 0] >= gt_cobb[:, 1]) & (gt_cobb[:, 1] >= gt_cobb[:, 2])).mean()
    print("Check 1: rank test")
    print(f"   column 0 is the largest of the three : {100*is_max0:5.1f}% of images")
    print(f"   column 2 is the smallest             : {100*is_min2:5.1f}% of images")
    print(f"   columns sorted descending (0>=1>=2)  : {100*desc:5.1f}% of images")
    if is_max0 > 0.95:
        print("   -> column 0 is the maximum by construction (rank slot).")
    elif is_max0 > 0.6:
        print("   -> column 0 is usually but not always the largest "
              "(consistent with an anatomical slot that is often the major curve).")
    else:
        print("   -> column 0 is not systematically the largest.")
    return {"col0_is_max": is_max0, "col2_is_min": is_min2, "descending": desc}


# --- Check 2: vertebra-pair matching ---
def match_pairs(t_sup_row, t_inf_row, value):
    """Find the vertebra pair (i<j) whose clinical Cobb angle
    (|superior endplate of i  -  inferior endplate of j|) best reproduces
    `value`. Returns (i, j, residual, n_ambiguous)."""
    d = np.abs(t_sup_row[:, None] - t_inf_row[None, :])       # (17,17)
    iu = np.triu_indices(N_VERT, k=1)
    diffs = d[iu]
    resid = np.abs(diffs - value)
    k = int(np.argmin(resid))
    n_amb = int((resid < TOL_AMBIG).sum())
    return int(iu[0][k]), int(iu[1][k]), float(resid[k]), n_amb


def construction_comparison(V, gt_cobb):
    """Which construction reproduces the stored angles better? Reports the
    median best-match residual for each, so the geometry is validated before
    any anatomical conclusion is drawn."""
    theta = endplate_tilts(V)
    t_sup, t_inf = endplate_tilts_tb(V)
    res_legacy, res_clin = [], []
    for n in range(len(gt_cobb)):
        dl = np.abs(theta[n][:, None] - theta[n][None, :])
        dc = np.abs(t_sup[n][:, None] - t_inf[n][None, :])
        iu = np.triu_indices(N_VERT, k=1)
        for c in range(3):
            res_legacy.append(np.abs(dl[iu] - gt_cobb[n, c]).min())
            res_clin.append(np.abs(dc[iu] - gt_cobb[n, c]).min())
    ml, mc = float(np.median(res_legacy)), float(np.median(res_clin))
    print("Construction check: which geometry reproduces the stored angles?")
    print(f"   mean-tilt per vertebra (legacy)          : median residual {ml:6.2f} deg")
    print(f"   superior-vs-inferior endplate (clinical) : median residual {mc:6.2f} deg")
    better = "clinical" if mc < ml else "legacy"
    print(f"   -> using the {better} construction below\n")
    return better


def pair_analysis(V, gt_cobb, construction):
    if construction == "clinical":
        t_sup, t_inf = endplate_tilts_tb(V)
    else:
        theta = endplate_tilts(V)
        t_sup = t_inf = theta
    rows = []
    for n in range(len(gt_cobb)):
        for c in range(3):
            i, j, r, amb = match_pairs(t_sup[n], t_inf[n], gt_cobb[n, c])
            rows.append({"image": n, "col": c, "upper": i, "lower": j,
                         "apex": (i + j) / 2.0, "span": j - i,
                         "residual": r, "ambiguity": amb,
                         "value": gt_cobb[n, c]})
    return pd.DataFrame(rows)


def report_pairs(df):
    print("\nCheck 2: vertebra-pair matching")
    med_resid = df.residual.median()
    print(f"   median residual across all matches: {med_resid:.2f} deg "
          f"({'OK' if med_resid < RESID_OK else 'too large; see below'})")
    print(f"   median ambiguity (pairs within {TOL_AMBIG:.0f} deg): "
          f"{df.ambiguity.median():.0f}")
    print(f"\n   {'column':<26}{'median apex':>13}{'apex IQR':>10}"
          f"{'median span':>13}{'resid':>8}")
    stats = {}
    for c in range(3):
        sub = df[df.col == c]
        apex_med = sub.apex.median()
        iqr = sub.apex.quantile(0.75) - sub.apex.quantile(0.25)
        stats[c] = {"apex_med": apex_med, "apex_iqr": iqr,
                    "span": sub.span.median(), "resid": sub.residual.median()}
        nm = VERT_NAMES[int(round(apex_med))]
        print(f"   {COLS[c]:<26}{nm+f' ({apex_med:.1f})':>13}{iqr:>10.1f}"
              f"{sub.span.median():>13.0f}{sub.residual.median():>8.2f}")
    return stats


# --- Verdict ---
def verdict(rank, stats, df):
    print("\n" + "=" * 70)
    med_resid = df.residual.median()
    if med_resid >= RESID_OK:
        print("Verdict: inconclusive; the pair matching does not reproduce the stored angles\n"
              f"(median residual {med_resid:.2f} deg vs threshold {RESID_OK:.1f} deg). The dataset's angles were likely\n"
              "computed by a different construction than max-tilt-between-endplates.")
        print("=" * 70)
        return

    tight = {c: stats[c]["apex_iqr"] <= APEX_TIGHT for c in range(3)}
    if rank["col0_is_max"] > 0.95:
        print("Verdict: rank slots.")
        print("  Column 0 is the maximum angle in essentially every image, so it holds\n"
              "  whichever curve is largest for that patient, not a fixed region.")
        print("\n  The columns correspond to 'the major curve' and 'the two secondary curves',\n"
              "  not to fixed anatomical regions (PT/MT/TL).")
    elif all(tight.values()):
        print("Verdict: anatomical slots.")
        print("  Each column's apex sits at a consistent spinal level across images,\n"
              "  so each refers to a fixed region. Only the column names are affected.")
        print("\n  Measured identity of each column:")
        for c in range(3):
            nm = VERT_NAMES[int(round(stats[c]["apex_med"]))]
            region = ("proximal thoracic" if stats[c]["apex_med"] <= 5 else
                      "main thoracic" if stats[c]["apex_med"] <= 10 else
                      "thoracolumbar / lumbar")
            print(f"    {COLS[c]:<26} apex {nm:<4} -> {region}")
        print("\n  The columns can be relabelled accordingly; per-angle results keep their meaning.")
    else:
        print("Verdict: mixed (partially consistent).")
        for c in range(3):
            nm = VERT_NAMES[int(round(stats[c]["apex_med"]))]
            tag = "consistent" if tight[c] else "SCATTERED"
            print(f"    {COLS[c]:<26} apex {nm:<4} IQR {stats[c]['apex_iqr']:.1f}  [{tag}]")
        print("\n  Columns with a scattered apex are rank-like and correspond to a\n"
              "  measured identity, not an anatomical name.")
    amb = df.ambiguity.median()
    if amb > 3:
        print(f"\nWARNING: median ambiguity is {amb:.0f} pairs (threshold 3); several vertebra pairs reproduce\n"
              "each value within tolerance, so per-image apex assignment may be unreliable.\n"
              "The aggregate pattern is still informative; individual rows are not.")
    print("=" * 70)


def relative_position(df):
    """If col0 is the major curve, are cols 1 and 2 the curves above and below
    it? Compares each column's apex to col0's apex, per image."""
    piv = df.pivot(index="image", columns="col", values="apex")
    if not set([0, 1, 2]).issubset(piv.columns):
        return
    print("\nRelative position: where do cols 1 and 2 sit vs col 0's apex?")
    for c in (1, 2):
        d = piv[c] - piv[0]
        above = (d < -0.5).mean() * 100
        below = (d > 0.5).mean() * 100
        print(f"   col {c}: above col0 in {above:5.1f}% of images, "
              f"below in {below:5.1f}%  (median offset {d.median():+.1f} vertebrae)")
    print("   (a clear above/below split supports the standard "
          "[major, upper, lower] construction)")


def run(gt_lm, gt_cobb, out_dir=None, uuids=None, img_dir=None):
    print(f"N = {len(gt_lm)} images\n")
    if img_dir is not None and uuids is not None:
        gt_lm, found = rescale_to_pixels(gt_lm, uuids, img_dir)
        print(f"Geometry: landmarks rescaled to pixel space for {found}/{len(gt_lm)} "
              f"images (angles measured in true image geometry).\n")
        if found < len(gt_lm):
            print("   WARNING: image lookup found fewer images than landmark rows. Angles for the missing images remain distorted.\n")
    else:
        print("WARNING: no --img-dir given, so angles are computed in\n"
              "  anisotropically normalised space (x/width, y/height). For ~1:2 spine\n"
              "  radiographs this flattens every tilt by roughly half, so the angles are\n"
              "  likely distorted. Passing --img-dir gives pixel-space angles.\n")
    V = to_vertebrae(gt_lm)
    rank = rank_test(gt_cobb)
    print()
    construction = construction_comparison(V, gt_cobb)
    df = pair_analysis(V, gt_cobb, construction)
    stats = report_pairs(df)
    relative_position(df)
    verdict(rank, stats, df)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "angle_semantics_per_match.csv", index=False)
        print(f"\nWrote {out_dir}/angle_semantics_per_match.csv")
    return df, stats


# --- Synthetic-data self-test: one anatomical and one rank dataset ---
def _synth(mode, n=100, seed=0):
    rng = np.random.default_rng(seed)
    lm, cobb = [], []
    for _ in range(n):
        tilts = np.zeros(N_VERT)
        # three curves with region-specific, randomly-sized magnitudes
        pt_mag = rng.uniform(5, 45)
        mt_mag = rng.uniform(5, 45)
        tl_mag = rng.uniform(5, 45)
        tilts[0:6] += np.linspace(0, pt_mag, 6)
        tilts[6:12] += np.linspace(0, mt_mag, 6)
        tilts[11:17] += np.linspace(0, tl_mag, 6)
        V = np.zeros((N_VERT, 4, 2))
        for i, deg in enumerate(tilts):
            th = np.radians(deg); hw, hh = 6.0, 4.0
            corners = np.array([[-hw, -hh], [hw, -hh], [-hw, hh], [hw, hh]])
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            V[i] = corners @ R.T + [50, i * 10.0]
        pts = V.reshape(68, 2)
        lm.append(np.concatenate([pts[:, 0], pts[:, 1]]))

        def rc(lo, hi):
            t = tilts[lo:hi + 1]
            return np.abs(t[:, None] - t[None, :]).max()
        angles = [rc(0, 5), rc(6, 11), rc(11, 16)]      # PT, MT, TL by region
        if mode == "rank":
            angles = sorted(angles, reverse=True)        # rank-ordered columns
        cobb.append(angles)
    return np.array(lm), np.array(cobb)


def smoke():
    for mode, expect in [("anatomical", "ANATOMICAL"), ("rank", "RANK")]:
        print("\n" + "#" * 70)
        print(f"# Self-test: planted dataset with {mode.upper()} columns")
        print("#" * 70)
        lm, cobb = _synth(mode)
        run(lm, cobb)
    print("\nSelf-test: the two verdicts above should read 'anatomical slots' and\n"
          "'rank slots' respectively.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path,
                    help="any preds_*.npz containing gt_landmarks and gt_cobb")
    ap.add_argument("--img-dir", type=Path, default=None,
                    help="image dir; needed for correct angles (pixel-space geometry)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.preds:
        d = np.load(args.preds, allow_pickle=True)
        run(d["gt_landmarks"].astype(float), d["gt_cobb"].astype(float),
            args.out_dir, uuids=d["uuids"] if "uuids" in d.files else None,
            img_dir=args.img_dir)
    else:
        raise SystemExit("--preds is required (or --smoke for the synthetic-data self-test)")


if __name__ == "__main__":
    main()
