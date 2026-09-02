"""
verify_angle_columns.py — which named column holds which Cobb curve?

Three independent signals suggested the dataset's angle columns may be
mislabelled:
  1. `angle_proximal_thoracic` has the LARGEST values (mean ~39 deg) while
     `angle_main_thoracic` is ~20 deg. Anatomically backwards: in adolescent
     idiopathic scoliosis the MAIN THORACIC is usually the major curve and the
     proximal thoracic is minor/compensatory.
  2. The landmark-derived MAX-pairwise angle (which IS, by definition, the major
     curve) correlated r=0.97 with the column named PT.
  3. A supervisor predicted MT should carry the larger values.

This settles it from the data alone, ANATOMICALLY: compute PT/MT/TL from the
ground-truth landmarks using FIXED vertebra ranges (not an ad-hoc apex split),
then see which named column each one matches.

Vertebra indexing (17 = T1..T12, L1..L5):
    idx 0..11 = T1..T12,  idx 12..16 = L1..L5
Regions (Lenke-style apex conventions, widened to end-vertebra ranges):
    PT  apex T3-T5   -> vertebrae T1..T6   (idx 0..5)
    MT  apex T6-T11  -> vertebrae T5..T12  (idx 4..11)
    TL  apex T12-L4  -> vertebrae T11..L4  (idx 10..15)

Why this is decisive: the region definitions are fixed by anatomy and do not
depend on the column names, so the correlation structure can only be explained
by the actual contents of the columns.

Run:
  python verify_angle_columns.py --preds ./outputs/frontier/preds_medvae_4_1_2d.npz
  python verify_angle_columns.py --smoke
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

N_LM = 68
N_VERT = 17
COLS = ["angle_proximal_thoracic", "angle_main_thoracic", "angle_thoracolumbar"]
COL_SHORT = ["PT-col", "MT-col", "TL-col"]
REGIONS = {"PT (T1-T6)": (0, 5), "MT (T5-T12)": (4, 11), "TL (T11-L4)": (10, 15)}
REG_SHORT = list(REGIONS.keys())
PERMS = list(itertools.permutations(range(3)))
VERT_NAMES = [f"T{i+1}" for i in range(12)] + [f"L{i+1}" for i in range(5)]


# --------------------------------------------------------------------------
def to_vertebrae(rows):
    """(N,136) ordering A, vmajor grouping -> (N,17,4,2)."""
    x, y = rows[:, :N_LM], rows[:, N_LM:]
    return np.stack([x, y], -1).reshape(-1, N_VERT, 4, 2)


def endplate_tilts(V):
    """(N,17,4,2) -> (N,17) endplate tilt in degrees.
    Corners 0,1 = top edge; 2,3 = bottom edge (validated ordering)."""
    top = V[:, :, 1, :] - V[:, :, 0, :]
    bot = V[:, :, 3, :] - V[:, :, 2, :]
    edge = 0.5 * (top + bot)
    return np.degrees(np.arctan2(edge[..., 1], edge[..., 0]))


def region_cobb(theta, lo, hi):
    """Max pairwise tilt difference within a fixed vertebra range -> (N,)."""
    t = theta[:, lo:hi + 1]
    d = np.abs(t[:, :, None] - t[:, None, :])
    return d.reshape(t.shape[0], -1).max(1)


def apex_of_major(theta):
    """(N,17) -> (N,) index of the apex (most-deviating) vertebra of the global
    major curve, i.e. the vertebra whose tilt is farthest from the spine mean."""
    return np.argmax(np.abs(theta - theta.mean(1, keepdims=True)), axis=1)


def best_assignment(cm):
    """cm[i,j] = corr(region_i, column_j). Return perm a where a[j] = region
    index matched to column j, maximising total |corr|."""
    best, score = None, -np.inf
    for a in PERMS:
        s = sum(abs(cm[a[j], j]) for j in range(3))
        if s > score:
            score, best = s, a
    return best


# --------------------------------------------------------------------------
def check(gt_lm, gt_cobb):
    theta = endplate_tilts(to_vertebrae(gt_lm))
    derived = np.stack([region_cobb(theta, lo, hi) for (lo, hi) in REGIONS.values()], 1)

    print(f"N = {len(gt_lm)} images\n")

    # ---- 1. value levels -------------------------------------------------
    print("1) VALUE LEVELS — which column holds the largest curve?")
    print(f"   {'column':<24}{'mean':>8}{'sd':>8}{'max':>8}")
    for j, c in enumerate(COLS):
        v = gt_cobb[:, j]
        print(f"   {c:<24}{v.mean():>8.1f}{v.std():>8.1f}{v.max():>8.1f}")
    biggest = int(np.argmax(gt_cobb.mean(0)))
    print(f"   -> largest-valued column: {COLS[biggest]}")
    print("      (anatomically the MAIN THORACIC is usually the major curve in AIS)\n")

    # ---- 2. anatomical correlation --------------------------------------
    cm = np.array([[np.corrcoef(derived[:, i], gt_cobb[:, j])[0, 1]
                    for j in range(3)] for i in range(3)])
    np.set_printoptions(precision=2, suppress=True)
    print("2) ANATOMICAL CORRELATION — landmark-derived regions vs named columns")
    print(f"   {'':<14}" + "".join(f"{c:>12}" for c in COL_SHORT))
    for i, r in enumerate(REG_SHORT):
        print(f"   {r:<14}" + "".join(f"{cm[i,j]:>12.2f}" for j in range(3)))
    a = best_assignment(cm)
    print("\n   best assignment (by |corr|):")
    for j in range(3):
        print(f"     {COLS[j]:<24} <- {REG_SHORT[a[j]]:<14} (r={cm[a[j], j]:.2f})")

    # ---- 3. apex location of the major curve ----------------------------
    apex = apex_of_major(theta)
    print("\n3) APEX LOCATION of the global major curve (from landmarks):")
    vals, counts = np.unique(apex, return_counts=True)
    top = sorted(zip(counts, vals), reverse=True)[:5]
    for c, v in top:
        print(f"     {VERT_NAMES[v]:<4} (idx {v:>2}): {c:>4} images  "
              f"({100*c/len(apex):.0f}%)")
    med = int(np.median(apex))
    print(f"   -> median apex: {VERT_NAMES[med]} (idx {med})")

    # ---- verdict ---------------------------------------------------------
    assigned = [abs(cm[a[j], j]) for j in range(3)]
    strong = [j for j in range(3) if assigned[j] >= 0.8]
    weak = [j for j in range(3) if assigned[j] < 0.8]
    print("\n" + "=" * 66)
    if not strong:
        print("VERDICT: INCONCLUSIVE — no column matches a region at r>=0.8.")
    else:
        print("VERDICT (only the strongly-supported part):")
        for j in strong:
            same = (a[j] == j)
            print(f"  '{COLS[j]}' holds the {REG_SHORT[a[j]]} curve (r={assigned[j]:.2f})"
                  + ("  <- matches its name" if same else "  <- does NOT match its name"))
    if weak:
        print("\n  NOT ESTABLISHED (weak, r<0.8 — do not rely on these):")
        for j in weak:
            print(f"    '{COLS[j]}' best match {REG_SHORT[a[j]]} at only r={assigned[j]:.2f}")
        print("\n  Note: the regions overlap (MT=T5-T12, TL=T11-L4 share T11-T12), so a\n"
              "  column holding the MAJOR curve will correlate with whichever region\n"
              "  happens to contain each patient's apex. That degeneracy can produce a\n"
              "  spurious 'best assignment' for the remaining columns.")
    print("\n  => Any per-angle claim keyed on column NAMES must be re-checked before\n"
          "     it goes in the report. Safest: refer to columns by MEASURED identity\n"
          "     (e.g. 'the major curve') rather than by their assumed names.")
    print("=" * 66)
    return cm, a


# --------------------------------------------------------------------------
def smoke():
    """Plant a KNOWN swap (major curve written into column 0) and check we
    detect it."""
    rng = np.random.default_rng(0)
    N = 120
    gt_lm, gt_cobb = [], []
    for _ in range(N):
        # MT region (idx 4..11) gets the big curve; PT small; TL medium
        tilts = np.zeros(N_VERT)
        tilts[0:6] += np.linspace(0, 6, 6) + rng.normal(0, 0.5, 6)      # small PT
        tilts[4:12] += np.linspace(0, 35, 8) + rng.normal(0, 1.0, 8)    # BIG MT
        tilts[10:16] += np.linspace(0, 12, 6) + rng.normal(0, 0.8, 6)   # medium TL
        V = np.zeros((N_VERT, 4, 2))
        for i, deg in enumerate(tilts):
            th = np.radians(deg); hw, hh = 6.0, 4.0
            corners = np.array([[-hw, -hh], [hw, -hh], [-hw, hh], [hw, hh]])
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            V[i] = corners @ R.T + [50, i * 10.0]
        pts = V.reshape(68, 2)
        gt_lm.append(np.concatenate([pts[:, 0], pts[:, 1]]))
        # true region angles
        def rc(lo, hi):
            t = tilts[lo:hi + 1]
            return np.abs(t[:, None] - t[None, :]).max()
        pt, mt, tl = rc(0, 5), rc(4, 11), rc(10, 15)
        # PLANT THE SWAP: column0 <- MT, column1 <- PT, column2 <- TL
        gt_cobb.append([mt, pt, tl])
    gt_lm = np.array(gt_lm); gt_cobb = np.array(gt_cobb)
    print(">>> SMOKE: planted swap = column0 holds MT, column1 holds PT\n")
    cm, a = check(gt_lm, gt_cobb)
    ok = (a[0] == 1 and a[1] == 0 and a[2] == 2)   # col0<-MT(idx1), col1<-PT(idx0)
    print("\nSMOKE:", "PASS (planted swap detected)" if ok else f"FAIL (got {a})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path,
                    help="any preds_*.npz with gt_landmarks and gt_cobb")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.preds:
        d = np.load(args.preds, allow_pickle=True)
        check(d["gt_landmarks"].astype(float), d["gt_cobb"].astype(float))
    else:
        raise SystemExit("--preds required (or --smoke)")


if __name__ == "__main__":
    main()
