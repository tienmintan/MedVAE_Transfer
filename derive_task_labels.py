"""
derive_task_labels.py: derive additional downstream task labels from the existing ground truth.

The tasks sit at different points on a granularity axis, from coarse position
to fine angular detail, so that results across them form a dose-response curve
rather than a single contrast. This tests whether compression removes fine
angular detail while preserving coarse position:

    coarse ......................................................... fine
    surgical threshold -> severity -> curve pattern -> landmarks -> Cobb -> wedging

Expected: compression tolerance should decrease monotonically along it.

The threshold tasks are of clinical interest. A 0.5 deg rise in Cobb MAE
rarely moves a patient across the 45 deg surgical boundary, so severity
classification may be robust to compression even while the underlying
regression degrades; in that case the clinical decision would tolerate more
compression than the MAE implies.

Caveat: all of these derive from the same 68 landmarks, so they are not
statistically independent tasks; they are different readouts of one
annotation. They do demand different information and are scored with
different metric families (accuracy/AUC vs degrees/pixels).
Reconstruction-fidelity metrics (PSNR/SSIM) are an independent axis that
complements them.

Tasks produced:
    severity        3-class  : <25 / 25-45 / >=45 deg on the major curve
    surgical        binary   : major curve >= SURGICAL_DEG (45 deg)
    curve_pattern   binary   : region holding the major-curve apex
                               (0 thoracic, 1 thoracolumbar / lumbar)
    wedging         17 x reg : per-vertebra |superior - inferior| endplate angle

Conventions:
    * Column 0 of the stored Cobb array is the major curve (rank slot), not
      the proximal thoracic curve, despite its name.
    * Angles are computed in pixel space (x*W, y*H). Computing them in the
      anisotropically normalised space flattens every tilt by ~2x.

Inputs:  --latents-dir (latents_{config}_{split}.npz, providing landmarks,
         cobb and uuids) and --img-dir (images, for pixel-space rescaling).
Outputs: <out-dir>/task_labels_{train,val,test}.npz with the task labels and
         an `implausible` mask; a per-split summary printed to stdout.

Usage:
    python derive_task_labels.py --latents-dir ./latents_aspect \
        --img-dir ./medvae/data/boostnet --out-dir ./labels_tasks
    python derive_task_labels.py --smoke      # synthetic-data self-test
"""
import argparse
from pathlib import Path

import numpy as np

N_LM = 68
N_VERT = 17
MAJOR = 0                 # stored column holding the major curve (established)
SURGICAL_DEG = 45.0       # standard surgical-consideration threshold
SEVERITY_EDGES = [25.0, 45.0]            # -> 3 classes (a separate <10 bin is near-empty)
SEVERITY_NAMES = ["<25 (mild)", "25-45 (moderate)", ">=45 (severe)"]
# plausibility limits: beyond these the annotation is almost certainly corrupt
MAX_PLAUSIBLE_WEDGE = 30.0     # a single vertebra's sup-vs-inf endplate angle
MAX_PLAUSIBLE_COBB = 100.0     # major curve
# vertebra index ranges by region (0..11 = T1..T12, 12..16 = L1..L5)
# Binary curve pattern. A 3-class version (proximal / main / thoracolumbar)
# leaves too few proximal-thoracic cases to learn or measure, so the classes
# are thoracic vs thoracolumbar-lumbar, which is a clinical distinction
# (thoracic and lumbar curves differ in bracing and surgical approach).
PATTERN_REGIONS = [("thoracic (proximal + main)", 0, 10),
                   ("thoracolumbar / lumbar", 11, 16)]
VERT_NAMES = [f"T{i+1}" for i in range(12)] + [f"L{i+1}" for i in range(5)]


# --- Geometry (pixel space) ---
def find_image(img_dir, uuid):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = Path(img_dir) / f"{uuid}{ext}"
        if p.exists():
            return p
    return None


def to_pixels(lm, uuids, img_dir):
    """Rescale anisotropically-normalised landmarks to pixel space so angles
    are measured in true image geometry. Returns (lm_px, n_found)."""
    from PIL import Image
    out = lm.astype(float).copy()
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
        out[i, :N_LM] = lm[i, :N_LM] * W
        out[i, N_LM:] = lm[i, N_LM:] * H
        found += 1
    return out, found


def to_vertebrae(rows):
    x, y = rows[:, :N_LM], rows[:, N_LM:]
    return np.stack([x, y], -1).reshape(-1, N_VERT, 4, 2)


def endplate_tilts(V):
    """Mean tilt per vertebra: the construction the dataset itself used."""
    top = V[:, :, 1, :] - V[:, :, 0, :]
    bot = V[:, :, 3, :] - V[:, :, 2, :]
    edge = 0.5 * (top + bot)
    return np.degrees(np.arctan2(edge[..., 1], edge[..., 0]))          # (N,17)


def wedging(V):
    """Per-vertebra wedge angle = |superior endplate - inferior endplate|.
    The finest-grained angular target available: one number per vertebra."""
    top = V[:, :, 1, :] - V[:, :, 0, :]
    bot = V[:, :, 3, :] - V[:, :, 2, :]
    t_sup = np.degrees(np.arctan2(top[..., 1], top[..., 0]))
    t_inf = np.degrees(np.arctan2(bot[..., 1], bot[..., 0]))
    return np.abs(t_sup - t_inf)                                        # (N,17)


def major_apex(theta):
    """Apex of the major curve = midpoint of the vertebra pair whose endplate
    tilt difference is largest (i.e. the pair that defines the Cobb angle).

    This matches the definition used in verify_angle_semantics.py. The
    alternative definition ('vertebra whose tilt deviates most from the spine
    mean') is biased toward the ends of the spine, where tilts are extreme."""
    n = theta.shape[0]
    apex = np.zeros(n, dtype=np.int64)
    for k in range(n):
        d = np.abs(theta[k][:, None] - theta[k][None, :])
        i, j = np.unravel_index(int(d.argmax()), d.shape)
        apex[k] = int(round((int(i) + int(j)) / 2.0))
    return apex


def flag_outliers(labels):
    """Anatomically implausible geometry -> almost certainly corrupt landmarks.
    A single vertebra cannot be wedged by 150 deg; a Cobb angle cannot exceed
    ~100 deg in this cohort. Flagged rows are saved in the `implausible` mask
    so that they can be excluded."""
    wedge_max = labels["wedging"].max(1)
    bad_wedge = wedge_max > MAX_PLAUSIBLE_WEDGE
    bad_cobb = labels["major_angle"] > MAX_PLAUSIBLE_COBB
    return bad_wedge | bad_cobb, wedge_max


# --- Label derivation ---
def derive(cobb, lm_px):
    major = cobb[:, MAJOR]
    severity = np.digitize(major, SEVERITY_EDGES)                       # 0..3
    surgical = (major >= SURGICAL_DEG).astype(np.int64)

    V = to_vertebrae(lm_px)
    theta = endplate_tilts(V)
    apex = major_apex(theta)
    pattern = np.zeros(len(apex), dtype=np.int64)
    for k, (_, lo, hi) in enumerate(PATTERN_REGIONS):
        pattern[(apex >= lo) & (apex <= hi)] = k

    return {"severity": severity.astype(np.int64),
            "surgical": surgical,
            "curve_pattern": pattern,
            "wedging": wedging(V).astype(np.float32),
            "major_angle": major.astype(np.float32),
            "apex_index": apex.astype(np.int64)}


# --- Reporting ---
def report(name, labels, n, uuids=None):
    print(f"\n--- {name} (n={n}) ---")
    maj = labels["major_angle"]
    print(f"  major curve: mean {maj.mean():.1f} deg, median {np.median(maj):.1f}, "
          f"range {maj.min():.1f}-{maj.max():.1f}")

    # Plausibility check first: implausible rows distort the statistics below
    bad, wedge_max = flag_outliers(labels)
    nbad = int(bad.sum())
    if nbad:
        print(f"  WARNING: plausibility check flagged {nbad} row(s) outside the limits below. The landmark annotation is likely corrupt:")
        for i in np.where(bad)[0]:
            uid = str(uuids[i]) if uuids is not None else f"index {i}"
            print(f"       {uid:<40} major={maj[i]:7.1f} deg  max wedge={wedge_max[i]:7.1f} deg")
        print(f"     (limits: wedge <= {MAX_PLAUSIBLE_WEDGE:.0f} deg, "
              f"Cobb <= {MAX_PLAUSIBLE_COBB:.0f} deg)")
        print("     These rows are saved with implausible=True so they can be excluded from training.")
    else:
        print(f"  Check passed: plausibility (max wedge {wedge_max.max():.1f} deg, "
              f"max Cobb {maj.max():.1f} deg)")

    print("  severity (3-class):")
    for k, nm in enumerate(SEVERITY_NAMES):
        c = int((labels["severity"] == k).sum())
        print(f"     {k} {nm:<20} {c:>4}  ({100*c/n:5.1f}%)")

    pos = int(labels["surgical"].sum())
    print(f"  surgical (>= {SURGICAL_DEG:.0f} deg): {pos} positive / {n-pos} negative "
          f"({100*pos/n:.1f}% positive)")
    if pos < 15 or (n - pos) < 15:
        print("     WARNING: class balance check found fewer than 15 cases in one class. "
              "This task may not be viable; a lower threshold or dropping the task may help.")

    print("  curve pattern (region of major-curve apex):")
    for k, (nm, _, _) in enumerate(PATTERN_REGIONS):
        c = int((labels["curve_pattern"] == k).sum())
        print(f"     {k} {nm:<24} {c:>4}  ({100*c/n:5.1f}%)")
    ap = labels["apex_index"]
    print(f"     median apex vertebra: {VERT_NAMES[int(np.median(ap))]} "
          f"(compare with the median apex reported by verify_angle_semantics.py)")

    w = labels["wedging"]
    keep = ~bad
    if keep.any():
        wk = w[keep]
        print(f"  wedging (excluding implausible): mean {wk.mean():.2f} deg/vertebra, "
              f"max {wk.max():.2f}")


def run(latents_dir, img_dir, out_dir, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Deriving task labels from {config} splits "
          f"(labels are config-independent; using this one for uuids/GT)\n")
    for split in ("train", "val", "test"):
        f = Path(latents_dir) / f"latents_{config}_{split}.npz"
        if not f.exists():
            print(f"Skipping {split}: {f.name} not found")
            continue
        d = np.load(f, allow_pickle=True)
        lm, cobb, uuids = d["landmarks"], d["cobb"], d["uuids"]
        lm_px, found = to_pixels(lm, uuids, img_dir)
        if found < len(lm):
            print(f"  WARNING [{split}]: image lookup found {found}/{len(lm)} images. "
                  "Angles for the rest are in distorted units and are likely invalid.")
        labels = derive(cobb, lm_px)
        report(split, labels, len(lm), uuids=uuids)
        bad, _ = flag_outliers(labels)
        np.savez_compressed(out_dir / f"task_labels_{split}.npz",
                            uuids=uuids, implausible=bad, **labels)
    print(f"\nWrote task_labels_*.npz to {out_dir}")
    print("  (each file carries an `implausible` boolean mask marking rows to exclude)")
    print("\nNote: these all derive from the same 68 landmarks; they are different readouts,\n"
          "not independent tasks. Reconstruction-fidelity metrics (PSNR/SSIM) provide\n"
          "a landmark-independent axis that complements them.")


# --- Synthetic-data self-test ---
def smoke():
    rng = np.random.default_rng(0)
    n = 60
    # synthetic spines with a controllable major curve at a controllable apex
    lm, cobb = [], []
    for k in range(n):
        apex = rng.integers(2, 15)
        mag = rng.uniform(5, 60)
        tilts = np.zeros(N_VERT)
        for i in range(N_VERT):
            tilts[i] = mag * np.exp(-((i - apex) ** 2) / 8.0) * (1 if i < apex else -1)
        V = np.zeros((N_VERT, 4, 2))
        for i, deg in enumerate(tilts):
            th = np.radians(deg); hw, hh = 60.0, 40.0
            corners = np.array([[-hw, -hh], [hw, -hh], [-hw, hh], [hw, hh]])
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            V[i] = corners @ R.T + [500.0, i * 100.0]
        pts = V.reshape(68, 2)
        lm.append(np.concatenate([pts[:, 0], pts[:, 1]]))
        d = np.abs(tilts[:, None] - tilts[None, :])
        cobb.append([d.max(), d.max() * 0.5, d.max() * 0.4])
    lm = np.array(lm); cobb = np.array(cobb)

    labels = derive(cobb, lm)           # already in "pixel" units here
    report("smoke", labels, n, uuids=np.array([f"img{k}" for k in range(n)]))
    ok = (labels["severity"].shape == (n,) and labels["surgical"].shape == (n,)
          and labels["curve_pattern"].shape == (n,)
          and labels["wedging"].shape == (n, N_VERT)
          and set(np.unique(labels["severity"])).issubset({0, 1, 2})
          and set(np.unique(labels["curve_pattern"])).issubset({0, 1}))
    print("\nSelf-test:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path)
    ap.add_argument("--img-dir", type=Path, help="needed for pixel-space angles")
    ap.add_argument("--out-dir", type=Path, default=Path("./labels_tasks"))
    ap.add_argument("--config", default="medvae_4_1_2d",
                    help="which split files to read uuids/GT from (labels are the same for all)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.latents_dir and args.img_dir:
        run(args.latents_dir, args.img_dir, args.out_dir, args.config)
    else:
        raise SystemExit("--latents-dir and --img-dir are required (or --smoke for the synthetic-data self-test)")


if __name__ == "__main__":
    main()
