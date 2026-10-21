"""Shared checks for the bounded MedVAE follow-up. No Torch import at startup."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from projection_common import atomic_json, atomic_npz, fingerprint, import_source, require, sha256

HERE = Path(__file__).resolve().parent
CONFIG = "medvae_4_3_2d"
VARIANTS = ("raw_keep", "raw_mean", "projected_keep", "projected_mean")
TASKS = ("cobb", "landmark")
SEEDS = [0, 1, 2]
TRAINING = dict(epochs=200, batch_size=32, lr=0.001, weight_decay=0.0001, patience=25)
EXCLUDE = ["sunhl-1th-20-Jul-2016-coronal6", "sunhl-1th-30-Dec-2016-161_A_AP"]
TRAINER_SHA = "221c04e67ab526fdc2bfc75d2eeb7d6ff31c01ff2e89d32c930782d84dadf108"


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {str(p.relative_to(HERE)): sha256(p) for p in sorted(HERE.rglob("*.py"))}


def trainer_path():
    p = HERE / "references/train_frontier.py"
    require(sha256(p) == TRAINER_SHA, "Bundled trainer changed. Restore the supplied file; do not disable this check.")
    return p


def set_plan(dst, plan):
    """Once written, the scientific plan cannot change in this output directory."""
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    plan = dict(plan, source_sha256=source_hashes())
    plan["fingerprint"] = fingerprint(plan)
    p = dst / "plan.json"
    if p.exists():
        require(read(p) == plan, f"Plan differs from {p}. Use a new output directory for a changed design.")
    else:
        atomic_json(p, plan)
    return plan


def get_plan(dst, check_source=True):
    p = read(Path(dst) / "plan.json")
    require(fingerprint({k: v for k, v in p.items() if k != "fingerprint"}) == p["fingerprint"], "Plan fingerprint differs")
    if check_source:
        require(p["source_sha256"] == source_hashes(), "Script files changed after planning. Restore the package or use a new study directory.")
    return p


def portable_path(study, stored, preferred):
    """Prefer the relocated study tree; never select another cache just by basename."""
    p = Path(study) / preferred
    if p.is_file():
        return p.resolve()
    p = Path(stored)
    require(p.is_file(), f"Required file missing: {Path(study) / preferred}")
    return p.resolve()


def old_manifest(study):
    m = read(Path(study) / "manifest.json")
    require(m.get("study_kind") == "fresh_paired" and m.get("status") == "ready", "Use the completed fresh paired projection study")
    ids = m["expected_ids"]
    for s, n in (("train", 409), ("val", 72), ("test", 128)):
        require(len(ids[s]) == len(set(ids[s])) == n, f"Unexpected {s} identifiers/count")
    require(not (set(ids["train"]) & set(ids["val"]) or set(ids["train"]) & set(ids["test"]) or set(ids["val"]) & set(ids["test"])), "Splits overlap")
    require(m["precision_mode"] == "ieee", "Expected fresh IEEE extraction")
    return m


def old_cache(study, m, variant, split, config=CONFIG):
    cell = m["cells"][variant][config]
    p = portable_path(study, cell["paths"][split], f"caches/{variant}/latents_{config}_{split}.npz")
    require(sha256(p) == cell["sha256"][split], f"Cache hash differs: {p}")
    return p


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def cache(path, expected_ids=None, channels=None):
    a = arrays(path)
    require(all(k in a for k in ("latents", "uuids", "cobb", "landmarks")), f"Incomplete cache {path}")
    ids = a["uuids"].astype(str)
    n = len(ids)
    require(n == len(set(ids)), "Duplicate cache identifiers")
    if expected_ids is not None:
        require(list(ids) == list(expected_ids), f"Cache ordering differs: {path}")
    require(a["latents"].ndim == 4 and len(a["latents"]) == n, "Invalid latent shape")
    if channels is not None:
        require(a["latents"].shape[1] == channels, "Unexpected channels")
    require(a["cobb"].shape == (n, 3) and a["landmarks"].shape == (n, 136), "Invalid target dimensions")
    for k in ("latents", "cobb", "landmarks"):
        require(a[k].dtype == np.float32 and np.isfinite(a[k]).all(), f"Invalid dtype/values: {k}")
    return a


def same_labels(a, b):
    for k in ("uuids", "cobb", "landmarks"):
        require(np.array_equal(a[k], b[k]), f"Different identifiers or labels: {k}")


def subset(a, keep):
    n = len(a["uuids"])
    return {k: v[keep] if v.ndim and v.shape[0] == n else v.copy() for k, v in a.items()}


def environment(torch, device):
    from projection_validation import precision_snapshot
    return dict(torch=torch.__version__, numpy=np.__version__, device="cuda" if device.startswith("cuda") else device,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
                precision=precision_snapshot(torch), required_precision_mode="ieee")


def check_environment(actual, expected, reason):
    diff = {k: {"expected": expected.get(k), "actual": actual.get(k)} for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)}
    require(not diff, reason + ": " + json.dumps(diff, sort_keys=True) + ". Do not bypass this gate. For historical-control mismatch, prepare a new annotation directory with --control-mode refit.")


def image_path(img_dir, uid):
    hits = [Path(img_dir) / (uid + ext) for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG")]
    hits = [p for p in hits if p.is_file()]
    require(len(hits) == 1, f"Expected exactly one image for {uid}; found {hits}")
    return hits[0]


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def load_prediction(path, ids, true):
    a = arrays(path)
    require(list(a["uuids"].astype(str)) == list(ids) and np.array_equal(a["true"], true), f"Prediction pairing/targets differ: {path}")
    require(a["pred"].shape == true.shape and np.isfinite(a["pred"]).all(), f"Invalid prediction: {path}")
    return a["pred"]


def errors(task, pred, true, dimensions=None):
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    d = pred - true
    if task == "cobb":
        return np.abs(d).mean(1)
    dx, dy = d[:, :68], d[:, 68:]
    if dimensions is None:
        return np.hypot(dx, dy).mean(1)
    w, h = dimensions[:, 0], dimensions[:, 1]
    span = np.ptp(true[:, 68:] * h[:, None], axis=1)
    require(np.all(span > 0), "Zero ground-truth vertical spine extent")
    return np.hypot(dx * w[:, None], dy * h[:, None]).mean(1) / span * 100


def paired_summary(values, seed=20260918, boot=10000):
    from scipy.stats import ttest_1samp
    v = np.asarray(values, dtype=np.float64)
    require(v.ndim == 1 and len(v) >= 2 and np.isfinite(v).all(), "Invalid paired differences")
    rng = np.random.default_rng(seed)
    b = np.concatenate([v[rng.integers(0, len(v), (min(500, boot-i), len(v)))].mean(1) for i in range(0, boot, 500)])
    p = float(ttest_1samp(v, 0).pvalue) if np.std(v) else (1.0 if v[0] == 0 else 0.0)
    return dict(n=len(v), mean_difference=float(v.mean()), ci_low=float(np.quantile(b, .025)), ci_high=float(np.quantile(b, .975)), p_raw=p)


def dimensions_from_images(img_dir, ids):
    from PIL import Image
    out = []
    for uid in ids:
        with Image.open(image_path(img_dir, uid)) as im:
            out.append({"image_uuid": uid, "width": im.width, "height": im.height})
    return out


def install_repo_on_path():
    # Commands run from the MedVAE repository root, not from inside this package.
    # Keep this kit's helpers ahead of similarly named repository-level files.
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(Path.cwd()))


class StudyLock:
    def __init__(self, root):
        self.path = Path(root) / "run.lock"
    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ValueError(f"Another run or stale lock exists: {self.path}. Check active processes before removing a stale lock.")
        os.write(self.fd, str(os.getpid()).encode()); return self
    def __exit__(self, *args):
        os.close(self.fd); self.path.unlink(missing_ok=True)
