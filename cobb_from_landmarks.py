"""
cobb_from_landmarks.py: test the position -> orientation -> Cobb mechanism on predicted landmarks.

Tests how landmark position error can be insensitive to compression while
Cobb error (built from the same landmarks) is not. Errors are measured at
three levels on the *predicted* landmarks:

  1. position    mean landmark Euclidean error
  2. orientation per-vertebra endplate slope error
  3. Cobb        error of the landmark-derived angle matched to the provided
                 MT column (col 1)

Interpretation:
  - sensitivity increasing from 1 to 3 -> compression preserves where structures
                                          are but erodes fine angular detail
  - landmark-derived Cobb degrades  -> apex-orientation loss the position metric hid
  - landmark-derived Cobb stays flat -> the direct Cobb regressor used a texture
                                        shortcut that compression erodes

The Cobb-from-landmarks geometry is validated before predictions are
processed: it must reproduce the provided angles from ground-truth landmarks.
The vertebra corner ordering is unknown, so all 24 orderings (under both point
groupings) are tried, and the one that best reproduces the provided MT angle
is kept.

Inputs: preds_{config}.npz from `train_frontier.py --save-preds`, each with
    uuids, gt_landmarks (N,136), gt_cobb (N,3)=[PT,MT,TL],
    pred_landmarks (N,136), pred_cobb (N,3).

Outputs: the geometry validation, the ladder table and the per-angle
    direct-Cobb table, printed to stdout.

Usage:
    python cobb_from_landmarks.py --preds-dir ./outputs/frontier
    python cobb_from_landmarks.py --smoke        # synthetic-data self-test (corner-ordering recovery)
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy import stats

CONFIGS = ["medvae_4_1_2d", "medvae_8_1_2d"]
LABEL = {"medvae_4_1_2d": "4x", "medvae_8_1_2d": "8x"}
N_LM = 68
N_VERT = 17
MT = 1  # column index of the angle named main thoracic (col 1) in [PT, MT, TL]
PERMS = list(itertools.permutations(range(4)))


# --- Geometry ---
def to_vertebrae(rows, grouping="vmajor"):
    """(N,136) ordering A -> (N,17,4,2), under one of two point layouts:
       'vmajor' : 4 consecutive points per vertebra (v0c0,v0c1,v0c2,v0c3,v1c0,...)
       'cmajor' : all 17 of corner-0, then all 17 of corner-1, ...  """
    x, y = rows[:, :N_LM], rows[:, N_LM:]
    pts = np.stack([x, y], axis=-1)            # (N,68,2)
    if grouping == "vmajor":
        return pts.reshape(-1, N_VERT, 4, 2)
    elif grouping == "cmajor":
        return pts.reshape(-1, 4, N_VERT, 2).transpose(0, 2, 1, 3)
    raise ValueError(grouping)


def endplate_angles(V, perm):
    """(N,17,4,2) -> (N,17) endplate tilt in degrees, under a corner ordering
    perm that maps corners to [TL, TR, BL, BR]. Uses the mean of the top and
    bottom edge vectors (the vertebra's endplate orientation)."""
    c = V[:, :, perm, :]
    top = c[:, :, 1, :] - c[:, :, 0, :]        # TR - TL
    bot = c[:, :, 3, :] - c[:, :, 2, :]        # BR - BL
    edge = 0.5 * (top + bot)
    return np.degrees(np.arctan2(edge[..., 1], edge[..., 0]))   # (N,17)


def mt_cobb(theta):
    """(N,17) endplate angles -> (N,) maximum pairwise tilt difference."""
    d = np.abs(theta[:, :, None] - theta[:, None, :])          # (N,17,17)
    return d.reshape(theta.shape[0], -1).max(axis=1)


def three_cobb(theta):
    """(N,17) -> (N,3) [PT, MT, TL] with a simple region split about the apex."""
    out = np.zeros((theta.shape[0], 3))
    for k in range(theta.shape[0]):
        t = theta[k]
        D = np.abs(t[:, None] - t[None, :])
        mt = D.max()
        u, l = np.unravel_index(D.argmax(), D.shape)
        u, l = sorted((int(u), int(l)))
        pt = np.abs(t[:u + 1, None] - t[None, :u + 1]).max() if u >= 1 else 0.0
        tl = np.abs(t[l:, None] - t[None, l:]).max() if l <= N_VERT - 2 else 0.0
        out[k] = [pt, mt, tl]
    return out


def apex_pair(theta_gt):
    """(N,17) GT angles -> (N,2) the two vertebra indices forming the MT pair."""
    pairs = np.zeros((theta_gt.shape[0], 2), dtype=int)
    for k in range(theta_gt.shape[0]):
        D = np.abs(theta_gt[k][:, None] - theta_gt[k][None, :])
        u, l = np.unravel_index(D.argmax(), D.shape)
        pairs[k] = sorted((int(u), int(l)))
    return pairs


# --- Statistics ---
def boot_ci(v, seed=0, nb=10000):
    rng = np.random.default_rng(seed); v = np.asarray(v); n = len(v)
    b = np.array([v[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(a, b):
    """b - a paired stats."""
    d = np.asarray(b) - np.asarray(a)
    t, pt = stats.ttest_rel(b, a)
    try:
        _, pw = stats.wilcoxon(b, a)
    except ValueError:
        pw = np.nan
    m, lo, hi = boot_ci(d)
    return {"delta": m, "lo": lo, "hi": hi, "p_t": pt, "p_w": pw,
            "rel": 100 * m / (np.mean(a) + 1e-12)}


# --- Corner-ordering discovery (run before predictions are processed) ---
def best_assignment(cm):
    """cm[i,j] = corr(derived_i, provided_j). Return the derived->provided
    column assignment maximising total |corr|, as a tuple where a[j] is the
    derived index matched to provided column j."""
    best, best_score = None, -np.inf
    for a in itertools.permutations(range(3)):
        score = sum(abs(cm[a[j], j]) for j in range(3))
        if score > best_score:
            best_score, best = score, a
    return best


def discover_geometry(gt_rows, provided):
    """Search {grouping} x {corner ordering} for the layout whose derived curve
    angles best match the provided PT/MT/TL (under the best column assignment).
    Returns (grouping, perm, assignment, corr_matrix). assignment[j] = derived
    index matched to provided column j; assignment[MT] is the MT tracker."""
    best, best_score = None, -np.inf
    for grouping in ("vmajor", "cmajor"):
        V = to_vertebrae(gt_rows, grouping)
        for perm in PERMS:
            der = three_cobb(endplate_angles(V, perm))          # (N,3)
            cm = np.array([[np.corrcoef(der[:, i], provided[:, j])[0, 1]
                            for j in range(3)] for i in range(3)])
            a = best_assignment(cm)
            score = abs(cm[a[MT], MT])          # prioritise matching main-thoracic
            if score > best_score:
                best_score, best = score, (grouping, perm, a, cm)
    return best


# --- Evaluation ---
def load(preds_dir, config):
    d = np.load(Path(preds_dir) / f"preds_{config}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def position_error(pred, gt):
    px, py = pred[:, :N_LM], pred[:, N_LM:]
    gx, gy = gt[:, :N_LM], gt[:, N_LM:]
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2).mean(1)


def _find_image(img_dir, uuid):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = Path(img_dir) / f"{uuid}{ext}"
        if p.exists():
            return p
    return None


def to_pixels(lm, uuids, img_dir):
    """Landmarks are normalised anisotropically (x/width, y/height). Spine
    radiographs are ~1:2, so angles computed in that space are flattened by
    roughly half. Landmarks are rescaled to pixel space before any angle
    computation. (Position error stays in normalised units, the units of the
    reported metric.)"""
    from PIL import Image
    out = lm.astype(float).copy()
    found = 0
    for i, u in enumerate(uuids):
        p = _find_image(img_dir, str(u))
        if p is None:
            continue
        try:
            with Image.open(p) as im:
                W, H = im.width, im.height
        except Exception:
            continue
        out[i, :N_LM] = lm[i, :N_LM] * W
        out[i, N_LM:] = lm[i, N_LM:] * H
        found += 1
    return out, found


def run(preds_dir, img_dir=None):
    data = {c: load(preds_dir, c) for c in CONFIGS}

    # Rescale to pixel space for angle geometry
    if img_dir is not None:
        for c in CONFIGS:
            d = data[c]
            d["gt_landmarks_px"], nf = to_pixels(d["gt_landmarks"], d["uuids"], img_dir)
            d["pred_landmarks_px"], _ = to_pixels(d["pred_landmarks"], d["uuids"], img_dir)
        print(f"Geometry: landmarks rescaled to pixel space ({nf}/{len(data[CONFIGS[0]]['uuids'])} "
              f"images); angles in true image geometry.\n")
    else:
        for c in CONFIGS:
            data[c]["gt_landmarks_px"] = data[c]["gt_landmarks"].astype(float)
            data[c]["pred_landmarks_px"] = data[c]["pred_landmarks"].astype(float)
        print("WARNING: no --img-dir given, so angles are computed in anisotropically\n"
              "  normalised space and are flattened (~2x for 1:2 radiographs).\n"
              "  Absolute degree values below are likely invalid. Passing --img-dir avoids this.\n")

    # Verify the geometry on GT landmarks (config-independent labels)
    ref = data[CONFIGS[0]]
    provided = ref["gt_cobb"]
    grouping, perm, assign, cm = discover_geometry(ref["gt_landmarks_px"], provided)
    mt_idx = assign[MT]           # derived index that tracks provided MT
    names = ["PT", "MT", "TL"]
    print(f"geometry discovered: grouping={grouping}  corner-order={perm}")
    print(f"derived->provided assignment: " +
          ", ".join(f"{names[assign[j]]}der->{names[j]}prov" for j in range(3)))
    Vgt = to_vertebrae(ref["gt_landmarks_px"], grouping)
    derived_gt = three_cobb(endplate_angles(Vgt, perm))
    print("validation under assignment (derived matched to provided):")
    for j, nm in enumerate(names):
        dcol = derived_gt[:, assign[j]]
        mae = np.abs(dcol - provided[:, j]).mean()
        # offset-corrected MAE (systematic convention offset cancels in the
        # paired 4x-vs-8x test anyway):
        mae_dt = np.abs((dcol - dcol.mean()) - (provided[:, j] - provided[:, j].mean())).mean()
        r = np.corrcoef(dcol, provided[:, j])[0, 1]
        flag = "  <-- primary (MT)" if j == MT else ""
        print(f"   {nm}: r={r:.3f}  MAE={mae:5.2f}deg  (offset-corrected {mae_dt:4.2f}deg){flag}")
    np.set_printoptions(precision=2, suppress=True)
    print("   raw 3x3 corr (derived rows x provided cols):")
    print("   " + str(cm).replace("\n", "\n   "))
    mt_r = cm[mt_idx, MT]
    if abs(mt_r) < 0.85:
        print(f"   WARNING: MT tracker correlation is {mt_r:.2f} (|r| threshold 0.85). "
              "The derived-Cobb results may be unreliable; the position and orientation results are unaffected.")
    else:
        print(f"   Check passed: MT tracker r={mt_r:.2f} (|r| threshold 0.85); the derived-Cobb geometry is validated.")

    theta_gt = endplate_angles(Vgt, perm)
    ap = apex_pair(theta_gt)   # (N,2) apex vertebra indices per image

    # The ladder, per configuration
    rungs = {}
    for c in CONFIGS:
        d = data[c]
        pos = position_error(d["pred_landmarks"], d["gt_landmarks"])   # normalised
        Vp = to_vertebrae(d["pred_landmarks_px"], grouping)            # pixels for angles
        th_p = endplate_angles(Vp, perm)
        th_g = endplate_angles(to_vertebrae(d["gt_landmarks_px"], grouping), perm)
        orient = np.abs(th_p - th_g).mean(1)                        # mean over 17
        apex = np.array([np.abs(th_p[k, ap[k]] - th_g[k, ap[k]]).mean()
                         for k in range(len(th_p))])                # apex pair
        mt_pred = three_cobb(th_p)[:, mt_idx]                      # MT-aligned tracker
        mt_geom = three_cobb(th_g)[:, mt_idx]
        mt_err_true = np.abs(mt_pred - d["gt_cobb"][:, MT])         # vs provided MT
        mt_err_geom = np.abs(mt_pred - mt_geom)                     # vs GT-landmark geom
        direct = np.abs(d["pred_cobb"] - d["gt_cobb"]).mean(1)      # direct-reg Cobb MAE
        rungs[c] = {"position": pos, "orientation": orient, "apex_orientation": apex,
                    "mt_cobb_vs_true": mt_err_true, "mt_cobb_vs_geom": mt_err_geom,
                    "direct_cobb": direct}

    order = ["position", "orientation", "apex_orientation",
             "mt_cobb_vs_geom", "mt_cobb_vs_true", "direct_cobb"]
    units = {"position": "norm", "orientation": "deg", "apex_orientation": "deg",
             "mt_cobb_vs_geom": "deg", "mt_cobb_vs_true": "deg", "direct_cobb": "deg"}
    print("\nLadder: error at each level, and the 4x->8x paired change:\n")
    hdr = f"{'rung':<20}{'4x':>10}{'8x':>10}{'Δ(8x-4x)':>12}{'rel%':>8}{'paired p':>11}"
    print(hdr); print("-" * len(hdr))
    for r in order:
        a, b = rungs[CONFIGS[0]][r], rungs[CONFIGS[1]][r]
        m4, _, _ = boot_ci(a); m8, _, _ = boot_ci(b)
        pr = paired(a, b)
        print(f"{r:<20}{m4:>10.4f}{m8:>10.4f}{pr['delta']:>12.4f}"
              f"{pr['rel']:>7.1f}%{pr['p_t']:>11.3g}  ({units[r]})")

    print("\nInterpretation: position should be ~flat; if orientation/apex/Cobb grow, "
          "compression erodes angular detail. If landmark-derived MT (mt_cobb_vs_geom) "
          "stays flat but direct_cobb grows, the direct regressor used a texture shortcut.")

    # Per-angle decomposition of the direct-Cobb degradation
    print("\nDirect-Cobb per angle (does the degradation concentrate in one curve?):")
    hdr = f"{'angle':<8}{'4x':>9}{'8x':>9}{'Δ':>9}{'rel%':>8}{'paired p':>11}"
    print(hdr); print("-" * len(hdr))
    for k, nm in enumerate(["PT", "MT", "TL"]):
        a = np.abs(data[CONFIGS[0]]["pred_cobb"][:, k] - data[CONFIGS[0]]["gt_cobb"][:, k])
        b = np.abs(data[CONFIGS[1]]["pred_cobb"][:, k] - data[CONFIGS[1]]["gt_cobb"][:, k])
        m4, _, _ = boot_ci(a); m8, _, _ = boot_ci(b)
        pr = paired(a, b)
        print(f"{nm:<8}{m4:>9.3f}{m8:>9.3f}{pr['delta']:>9.3f}"
              f"{pr['rel']:>7.1f}%{pr['p_t']:>11.3g}")

    return grouping, perm, mt_idx, rungs


# --- Synthetic-data self-test: known tilts and a scrambled corner order ---
def smoke():
    rng = np.random.default_rng(0)
    N = 30
    # a curve: tilt increases then decreases down the spine
    base = np.concatenate([np.linspace(-5, 25, 9), np.linspace(25, -3, 8)])  # 17 deg
    true_perm = (2, 0, 3, 1)   # the "unknown" ordering to be recovered
    inv = np.argsort(true_perm)
    gt_lm, gt_cobb = [], []
    for _ in range(N):
        tilts = base + rng.normal(0, 1.5, N_VERT)
        V = np.zeros((N_VERT, 4, 2))
        for i, deg in enumerate(tilts):
            cy = i * 10.0
            th = np.radians(deg)
            hw, hh = 6.0, 4.0
            TL = [-hw, -hh]; TR = [hw, -hh]; BL = [-hw, hh]; BR = [hw, hh]
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            corners = np.array([TL, TR, BL, BR]) @ R.T + [50, cy]  # [TL,TR,BL,BR]
            V[i] = corners[list(true_perm)]   # scramble into stored order
        # ordering A row
        pts = V.reshape(68, 2)
        gt_lm.append(np.concatenate([pts[:, 0], pts[:, 1]]))
        D = np.abs(tilts[:, None] - tilts[None, :]); mt = D.max()
        u, l = np.unravel_index(D.argmax(), D.shape); u, l = sorted((u, l))
        pt = np.abs(tilts[:u + 1, None] - tilts[None, :u + 1]).max()
        tl = np.abs(tilts[l:, None] - tilts[None, l:]).max()
        gt_cobb.append([pt, mt, tl])
    gt_lm = np.array(gt_lm); gt_cobb = np.array(gt_cobb)

    Vgt = to_vertebrae(gt_lm, "vmajor")
    grouping, perm, assign, cm = discover_geometry(gt_lm, gt_cobb)
    print(f"true grouping=vmajor perm={true_perm}  discovered={grouping},{perm}  "
          f"assignment(MT<-der{assign[MT]}) MT-corr={cm[assign[MT],MT]:.3f}")
    derived = three_cobb(endplate_angles(to_vertebrae(gt_lm, grouping), perm))
    for j, nm in enumerate(["PT", "MT", "TL"]):
        print(f"  {nm}: derived vs true MAE = {np.abs(derived[:,assign[j]]-gt_cobb[:,j]).mean():.3f} deg")
    ok = np.abs(derived[:, assign[MT]] - gt_cobb[:, MT]).mean() < 0.5
    print("Self-test:", "PASS" if ok else "FAIL", "(MT recovered from scrambled corners)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", type=Path)
    ap.add_argument("--img-dir", type=Path, default=None,
                    help="image dir; needed for correct angle geometry")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.preds_dir:
        run(args.preds_dir, args.img_dir)
    else:
        raise SystemExit("--preds-dir is required (or --smoke for the synthetic-data self-test)")


if __name__ == "__main__":
    main()
