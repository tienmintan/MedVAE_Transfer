"""
train_frontier.py: compression-frontier task pipeline (Cobb + landmarks).

Trains several tasks on the same aspect-padded MedVAE latents at different
compression levels, with a backbone held identical across the whole grid, so
that compression tolerance can be compared between tasks rather than read from
a single degradation curve.

Design:
    Grid = {task} x {config}: by default {cobb, landmark} (--tasks) x the four
    MedVAE configurations 4_3, 4_1, 8_4, 8_1 (--configs).
    The backbone (LatentBackbone) is identical in every cell.
    An adaptive pool before the head means the same backbone ingests both the
    96x96 (4x) and 48x48 (8x) latent without any architecture change. The only
    things that differ across cells are:
        (a) the compression level baked into the latent, and
        (b) the task head's output dim (e.g. 3 angles, 136 landmark coords,
            or class logits), and the input channel count of the first conv.
    Any difference in degradation between the task curves is therefore
    attributable to the task, not to a regressor mismatch.

Scope: with only two spatial factors (4x and 8x), the spatial axis has two
points per task, not a curve. The result is *relative compression sensitivity
per task* (how much each task's error grows from 4x->8x), not a "breakpoint".
A breakpoint would need intermediate compression levels.

Metrics:
    cobb     -> mean absolute error in degrees (mean over the 3 angles), per image
    landmark -> mean Euclidean landmark error in image-normalised units
                (mean over the 68 points), per image
    wedging  -> mean absolute error in degrees (mean over the 17 vertebrae)
    surgical / severity / curve_pattern -> 0/1 misclassification per image
                (mean = error rate), plus accuracy, macro-F1, AUC (binary)
                and quadratic-weighted kappa (severity)
All get image-level bootstrap CIs (consistent with the other analysis
scripts), plus an across-seed sd as a secondary robustness number.

Multi-seed: train --seeds models per cell, ensemble their test predictions
(average angles / coords) for the per-image numbers, and also record each
seed's standalone test metric for the +/- sd.

Inputs: one .npz per (config, split) from extract_latents_aspect.py, each with
    latents (N,C,H,W) float32, uuids (N,), cobb (N,3) float32, landmarks (N,136)
    float32 (normalised [0,1], ordering A: 68 xs then 68 ys). Derived-task
    labels come from task_labels_{split}.npz in --labels-dir.
Outputs: <out-dir>/frontier_per_image.csv, frontier_summary.csv,
    frontier_curve.csv, frontier_curve.png and compression_tolerance_ladder.csv;
    matched_pair_4_1_vs_8_4.csv and clean_spatial_4_1_vs_8_1.csv when those
    configurations are present; preds_{config}.npz with --save-preds and
    model_{config}_{task}_seed{seed}.pt with --save-models.

Usage:
    # full run (after extraction):
    python train_frontier.py \\
        --latents-dir ./latents_aspect \\
        --out-dir     ./outputs/frontier \\
        --seeds 0 1 2

    # synthetic-data self-test (no MedVAE / no latents needed):
    python train_frontier.py --smoke --out-dir /tmp/frontier_smoke
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Shared conventions ---
# config -> short label and total compression ratio (f/C, f=area factor, C=channels)
CONFIG_LABELS = {"medvae_4_3_2d": "4_3", "medvae_4_1_2d": "4_1",
                 "medvae_8_4_2d": "8_4", "medvae_8_1_2d": "8_1"}
COMPRESSION = {"medvae_4_3_2d": 16 / 3, "medvae_4_1_2d": 16 / 1,
               "medvae_8_4_2d": 64 / 4, "medvae_8_1_2d": 64 / 1}   # ~5.3, 16, 16, 64
CONFIG_ORDER = ["medvae_4_3_2d", "medvae_4_1_2d", "medvae_8_4_2d", "medvae_8_1_2d"]
MATCHED_PAIR = ("medvae_4_1_2d", "medvae_8_4_2d")   # both 16x: spatial-vs-channel test

# --- runtime knobs (set from the CLI) ---
PROGRESS = 0            # print a line every N epochs; 0 = silent
PRELOAD_LIMIT_GB = 8.0  # preload a split onto the GPU if it fits under this


def args_preload_limit_bytes():
    return PRELOAD_LIMIT_GB * (1024 ** 3)


def register_config(name):
    """Allow non-MedVAE configurations (e.g. frozen DINOv2/DINOv3 features)
    through the same grid. They have no meaningful f/C ratio, so they get a
    nominal 1.0 and should be read as a representation-quality reference point,
    not as a point on the compression curve."""
    if name not in CONFIG_LABELS:
        CONFIG_LABELS[name] = name.replace("facebook_", "").replace("-", "_")[:14]
        COMPRESSION[name] = 1.0
    return name

# Task specification. `kind` drives the loss, the metric, and whether targets
# are standardised. The tasks form a granularity ladder (coarse -> fine):
#   surgical -> severity -> curve_pattern -> landmark -> cobb -> wedging
# Prediction: compression tolerance decreases down the ladder.
TASK_SPEC = {
    "surgical":      {"out": 2,   "kind": "clf", "src": "surgical",      "unit": "err rate"},
    "severity":      {"out": 3,   "kind": "clf", "src": "severity",      "unit": "err rate"},
    "curve_pattern": {"out": 2,   "kind": "clf", "src": "curve_pattern", "unit": "err rate"},
    "landmark":      {"out": 136, "kind": "reg", "src": "landmarks",     "unit": "norm"},
    "cobb":          {"out": 3,   "kind": "reg", "src": "cobb",          "unit": "deg"},
    "wedging":       {"out": 17,  "kind": "reg", "src": "wedging",       "unit": "deg"},
}
TASK_OUT_DIM = {k: v["out"] for k, v in TASK_SPEC.items()}
LADDER = ["surgical", "severity", "curve_pattern", "landmark", "cobb", "wedging"]
COBB_COLOR = "#1C7293"      # teal
LANDMARK_COLOR = "#B85042"  # accent
N_LANDMARKS = 68


# --- Model: one backbone for every cell of the grid; the task only changes the head ---
class LatentBackbone(nn.Module):
    """(B, Cz, H, W) -> (B, feat_dim). Identical for 96x96 and 48x48 inputs.

    Stride-2 conv blocks then AdaptiveAvgPool2d(POOL) so the flattened feature
    dim is fixed regardless of input spatial size. Kept deliberately small
    (~few hundred k params) because there are only ~400 training images.
    """
    POOL = 4

    def __init__(self, in_ch=1, width=(32, 64, 128)):
        super().__init__()
        c0, c1, c2 = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c0, 3, padding=1), nn.BatchNorm2d(c0), nn.GELU(),
        )
        self.block1 = self._block(c0, c0)   # /2
        self.block2 = self._block(c0, c1)   # /2
        self.block3 = self._block(c1, c2)   # /2
        self.pool = nn.AdaptiveAvgPool2d(self.POOL)
        self.feat_dim = c2 * self.POOL * self.POOL

    @staticmethod
    def _block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1),
            nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.GELU(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class TaskModel(nn.Module):
    def __init__(self, out_dim, in_ch=1, dropout=0.2):
        super().__init__()
        self.backbone = LatentBackbone(in_ch=in_ch)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# --- Data ---
class LatentDataset(Dataset):
    def __init__(self, latents, targets, is_class=False):
        self.x = torch.as_tensor(latents, dtype=torch.float32)
        self.y = torch.as_tensor(targets,
                                 dtype=torch.long if is_class else torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_split(latents_dir, config, split, labels_dir=None, exclude_implausible=False):
    p = Path(latents_dir) / f"latents_{config}_{split}.npz"
    d = np.load(p, allow_pickle=True)
    out = {
        "latents": d["latents"].astype(np.float32),
        "uuids": d["uuids"],
        "cobb": d["cobb"].astype(np.float32),
        "landmarks": d["landmarks"].astype(np.float32),
    }
    if labels_dir is not None:
        lf = Path(labels_dir) / f"task_labels_{split}.npz"
        if lf.exists():
            L = np.load(lf, allow_pickle=True)
            # align by uuid rather than assuming the row order matches
            idx = {str(u): i for i, u in enumerate(L["uuids"])}
            order = np.array([idx[str(u)] for u in out["uuids"] if str(u) in idx])
            keep = np.array([i for i, u in enumerate(out["uuids"]) if str(u) in idx])
            if len(keep) != len(out["uuids"]):
                print(f"    note [{split}]: {len(out['uuids']) - len(keep)} images have "
                      f"no task label and are dropped for the extra tasks")
            for k in ("surgical", "severity", "curve_pattern"):
                if k in L.files:
                    out[k] = L[k][order].astype(np.int64)
            if "wedging" in L.files:
                out["wedging"] = L["wedging"][order].astype(np.float32)
            out["_label_rows"] = keep
            if exclude_implausible and "implausible" in L.files:
                bad = L["implausible"][order]
                if bad.any():
                    good = ~bad
                    print(f"    [{split}] excluding {int(bad.sum())} implausible "
                          f"annotation(s)")
                    sel = keep[good]
                    for k in ("latents", "uuids", "cobb", "landmarks"):
                        out[k] = out[k][sel]
                    for k in ("surgical", "severity", "curve_pattern", "wedging"):
                        if k in out:
                            out[k] = out[k][good]
    return out


def targets_for(task, data):
    src = TASK_SPEC[task]["src"]
    if src not in data:
        raise SystemExit(
            f"task '{task}' needs '{src}' — pass --labels-dir pointing at the "
            f"task_labels_*.npz produced by derive_task_labels.py"
        )
    return data[src]


# --- Metrics ---
def cobb_per_image_error(pred, true):
    # pred/true: (N, 3) in degrees -> (N,) MAE over the 3 angles
    return np.abs(pred - true).mean(axis=1)


def landmark_per_image_error(pred, true):
    # pred/true: (N, 136) normalised, ordering A (68 xs then 68 ys)
    n = pred.shape[0]
    px, py = pred[:, :N_LANDMARKS], pred[:, N_LANDMARKS:]
    tx, ty = true[:, :N_LANDMARKS], true[:, N_LANDMARKS:]
    dist = np.sqrt((px - tx) ** 2 + (py - ty) ** 2)   # (N, 68)
    return dist.mean(axis=1)                           # (N,)


def per_image_error(task, pred, true):
    """One number per image, so bootstrap CIs and paired tests work uniformly
    across regression and classification. For classification this is the 0/1
    misclassification indicator, whose mean is the error rate."""
    kind = TASK_SPEC[task]["kind"]
    if kind == "clf":
        return (pred.argmax(1) != true).astype(np.float64)
    if task == "landmark":
        return landmark_per_image_error(pred, true)
    return np.abs(pred - true).mean(axis=1)      # cobb (deg), wedging (deg)


def extra_metrics(task, logits, true):
    """Secondary metrics that a single error rate hides: AUC for binary tasks
    (threshold-free, robust to class imbalance) and quadratic-weighted kappa
    for the ordinal severity task (penalises distant confusions more)."""
    kind = TASK_SPEC[task]["kind"]
    if kind != "clf":
        return {}
    out = {}
    pred = logits.argmax(1)
    out["accuracy"] = float((pred == true).mean())
    # majority-class baseline: an error rate is interpreted against this
    vals, counts = np.unique(true, return_counts=True)
    out["majority_baseline"] = float(counts.max() / len(true))
    try:
        from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score
        out["macro_f1"] = float(f1_score(true, pred, average="macro"))
        if TASK_SPEC[task]["out"] == 2:
            e = np.exp(logits - logits.max(1, keepdims=True))
            prob1 = (e / e.sum(1, keepdims=True))[:, 1]
            if len(np.unique(true)) > 1:
                out["auc"] = float(roc_auc_score(true, prob1))
        if task == "severity":
            out["qwk"] = float(cohen_kappa_score(true, pred, weights="quadratic"))
    except ImportError:
        pass
    return out


def bootstrap_ci(values, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    n = len(v)
    boot = np.array([v[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# --- Train one model (one task, one config, one seed) ---
def train_one(task, train, val, in_ch, *, seed, epochs, batch_size, lr,
              weight_decay, patience, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ytr = targets_for(task, train)
    kind = TASK_SPEC[task]["kind"]

    if kind == "clf":
        # integer class labels: no standardisation, cross-entropy loss
        mu = sd = None

        def std(y):
            return y

        def unstd(y):
            return y
    elif task in ("cobb", "wedging"):
        # degrees: standardise, since scales differ across outputs
        mu = ytr.mean(axis=0, keepdims=True)
        sd = ytr.std(axis=0, keepdims=True) + 1e-8

        def std(y):
            return (y - mu) / sd

        def unstd(y):
            return y * sd + mu
    else:
        mu = np.zeros((1, ytr.shape[1]), dtype=np.float32)
        sd = np.ones((1, ytr.shape[1]), dtype=np.float32)

        def std(y):
            return (y - mu) / sd

        def unstd(y):
            return y * sd + mu

    # Batching
    # Large feature tensors (e.g. frozen DINO at 768x32x32 = 3 MB/image, ~1.3 GB
    # for the train split) are dominated by DataLoader collation: every batch
    # stacks 32 multi-MB tensors in Python, then copies to GPU. Preloading the
    # whole split onto the GPU once and indexing it directly removes that
    # overhead; this is typically an order of magnitude faster for these
    # features, and numerically identical.
    ytr_std = std(ytr)
    tr_bytes = int(np.prod(train["latents"].shape)) * 4
    use_gpu_resident = (str(device).startswith("cuda")
                        and tr_bytes < args_preload_limit_bytes())
    if use_gpu_resident:
        Xtr = torch.as_tensor(train["latents"], dtype=torch.float32, device=device)
        Ytr = torch.as_tensor(ytr_std,
                              dtype=torch.long if kind == "clf" else torch.float32,
                              device=device)
        dl_tr = None
    else:
        Xtr = Ytr = None
        dl_tr = DataLoader(LatentDataset(train["latents"], ytr_std, is_class=(kind == "clf")),
                           batch_size=batch_size, shuffle=True, drop_last=False)
    xval = torch.as_tensor(val["latents"], dtype=torch.float32, device=device)
    yval_true = targets_for(task, val)

    model = TaskModel(TASK_OUT_DIM[task], in_ch=in_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if kind == "clf":
        # inverse-frequency weights so a skewed task isn't solved by predicting
        # the majority class
        cnt = np.bincount(ytr, minlength=TASK_OUT_DIM[task]).astype(np.float32)
        w = torch.tensor(len(ytr) / (len(cnt) * np.maximum(cnt, 1)),
                         dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.SmoothL1Loss(beta=0.1 if task == "landmark" else 1.0)

    best_metric, best_state, bad = np.inf, None, 0
    n_tr = len(train["latents"])
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if use_gpu_resident:
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xtr[idx]), Ytr[idx])
                loss.backward()
                opt.step()
        else:
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            # chunked so a large validation tensor cannot exhaust memory
            pv = np.concatenate([model(xval[i:i + 64]).cpu().numpy()
                                 for i in range(0, len(xval), 64)], 0)
        pv = unstd(pv)
        metric = per_image_error(task, pv, yval_true).mean()
        if metric < best_metric - 1e-9:
            best_metric, best_state, bad = metric, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if PROGRESS and (ep + 1) % PROGRESS == 0:
            print(f"      ep {ep+1:>4}/{epochs}  val {metric:.5f}  "
                  f"best {best_metric:.5f}  patience {bad}/{patience}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    if use_gpu_resident:
        del Xtr, Ytr
        torch.cuda.empty_cache()
    model.load_state_dict(best_state)
    model.eval()
    return model, (mu, sd)


def predict(model, mu, sd, latents, device):
    """Returns raw logits for classification tasks (mu/sd are None) and
    de-standardised values for regression."""
    x = torch.as_tensor(latents, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x).cpu().numpy()
    if mu is None or sd is None:
        return out
    return out * sd + mu


# --- Run the full grid ---
def run_grid(splits_by_config, tasks, configs, *, seeds, device, models_dir=None, **train_kw):
    """splits_by_config[config] = {'train':..,'val':..,'test':..}.
    Returns per-image rows and a summary dict keyed by (config, task)."""
    per_image_rows = []
    summary = {}
    preds = {}   # (config, task) -> ensemble test predictions (N, out_dim)

    for config in configs:
        sp = splits_by_config[config]
        in_ch = sp["train"]["latents"].shape[1]
        test_uuids = sp["test"]["uuids"]

        for task in tasks:
            ytest_true = targets_for(task, sp["test"])
            seed_preds, seed_metrics = [], []
            task_t0 = time.time()
            if PROGRESS:
                mb = int(np.prod(sp["train"]["latents"].shape)) * 4 / 1024 ** 3
                print(f"  [{CONFIG_LABELS[config]} | {task}] {len(seeds)} seeds, "
                      f"train tensor {mb:.2f} GB", flush=True)
            for seed in seeds:
                model, (mu, sd) = train_one(
                    task, sp["train"], sp["val"], in_ch,
                    seed=seed, device=device, **train_kw)
                pt = predict(model, mu, sd, sp["test"]["latents"], device)
                seed_preds.append(pt)
                seed_metrics.append(per_image_error(task, pt, ytest_true).mean())
                if PROGRESS:
                    done = len(seed_preds)
                    el = time.time() - task_t0
                    eta = el / done * (len(seeds) - done)
                    print(f"    seed {seed} done ({done}/{len(seeds)})  "
                          f"test {seed_metrics[-1]:.5f}  "
                          f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
                if models_dir is not None:
                    torch.save({"state_dict": model.state_dict(), "in_ch": in_ch,
                                "out_dim": TASK_OUT_DIM[task], "task": task,
                                "config": config, "seed": seed,
                                "mu": mu, "sd": sd},
                               Path(models_dir) / f"model_{config}_{task}_seed{seed}.pt")

            ens_pred = np.mean(seed_preds, axis=0)                  # ensemble
            preds[(config, task)] = ens_pred
            errs = per_image_error(task, ens_pred, ytest_true)      # (N,)
            mean, lo, hi = bootstrap_ci(errs)
            extra = extra_metrics(task, ens_pred, ytest_true)
            summary[(config, task)] = {
                "config": config, "task": task,
                "error_mean": mean, "error_lo": lo, "error_hi": hi,
                "seed_sd": float(np.std(seed_metrics)),
                "n": int(len(errs)), **extra,
            }
            for u, e in zip(test_uuids, errs):
                per_image_rows.append({"config": config, "task": task,
                                       "image_uuid": str(u), "error": float(e)})
            unit = TASK_SPEC[task]["unit"]
            line = (f"[{CONFIG_LABELS[config]:>3} | {task:<13}] "
                    f"error={mean:.5f} ({unit})  CI=[{lo:.5f},{hi:.5f}]  "
                    f"seed_sd={summary[(config, task)]['seed_sd']:.5f}")
            if extra:
                bits = [f"acc={extra['accuracy']:.3f}",
                        f"base={extra['majority_baseline']:.3f}"]
                if "auc" in extra:
                    bits.append(f"AUC={extra['auc']:.3f}")
                if "qwk" in extra:
                    bits.append(f"QWK={extra['qwk']:.3f}")
                if "macro_f1" in extra:
                    bits.append(f"F1={extra['macro_f1']:.3f}")
                line += "  " + " ".join(bits)
            print(line)

    return pd.DataFrame(per_image_rows), summary, preds


def ladder_table(summary, tasks, configs, anchor=None):
    """The compression-tolerance ladder: for each task, how much does error
    grow along the spatial axis?

    Anchored by default on the 1-channel pair (4_1 -> 8_1), not on the lowest
    total compression. `4_3` has the least total compression but is a
    3-channel model fed replicated greyscale, and it is not the best
    configuration for several tasks; anchoring there conflates the spatial and
    channel axes and makes tasks appear to *improve* under compression."""
    order = [t for t in LADDER if t in tasks]
    if len(order) < 2:
        return None
    if anchor is not None:
        light, heavy = anchor
    elif "medvae_4_1_2d" in configs and "medvae_8_1_2d" in configs:
        light, heavy = "medvae_4_1_2d", "medvae_8_1_2d"       # 1-channel spatial pair
    elif len(configs) < 2:
        return None                                           # nothing to compare
    else:
        light = min(configs, key=lambda c: COMPRESSION[c])
        heavy = max(configs, key=lambda c: COMPRESSION[c])
    if light == heavy:
        return None
    rows = []
    print(f"\n=== Compression-tolerance ladder ({CONFIG_LABELS[light]} -> "
          f"{CONFIG_LABELS[heavy]}, 1-channel spatial axis) ===")
    print("  coarse tasks first; prediction is that % degradation grows downward")
    hdr = (f"{'task':<15}{'unit':>10}{CONFIG_LABELS[light]:>11}"
           f"{CONFIG_LABELS[heavy]:>11}{'rel %':>9}{'CI overlap?':>13}")
    print(hdr); print("-" * len(hdr))
    for t in order:
        if (light, t) not in summary or (heavy, t) not in summary:
            continue
        sa, sb = summary[(light, t)], summary[(heavy, t)]
        a, b = sa["error_mean"], sb["error_mean"]
        rel = 100.0 * (b - a) / a if a > 0 else float("nan")
        # do the two CIs overlap? if so the change is not distinguishable
        overlap = not (sa["error_hi"] < sb["error_lo"] or sb["error_hi"] < sa["error_lo"])
        print(f"{t:<15}{TASK_SPEC[t]['unit']:>10}{a:>11.4f}{b:>11.4f}{rel:>8.1f}%"
              f"{('yes (n.s.)' if overlap else 'no'):>13}")
        rows.append({"task": t, "unit": TASK_SPEC[t]["unit"], "anchor_light": CONFIG_LABELS[light],
                     "anchor_heavy": CONFIG_LABELS[heavy], "err_light": a, "err_heavy": b,
                     "rel_pct": rel, "ci_overlap": overlap})
    print("  Note: where the CIs overlap, the change is not distinguishable from noise.")
    return pd.DataFrame(rows)


def frontier_table(summary, tasks, configs):
    """Per (task, config): raw error + error relative to the lightest config,
    plus the total compression ratio. The compression-curve read-out."""
    order = sorted(configs, key=lambda c: COMPRESSION[c])
    light = order[0]
    rows = []
    for task in tasks:
        base = summary[(light, task)]["error_mean"]
        for c in order:
            e = summary[(c, task)]["error_mean"]
            rows.append({"task": task, "config": CONFIG_LABELS[c],
                         "compression": round(COMPRESSION[c], 1),
                         "error": e, "rel_to_lightest": e / base})
    return pd.DataFrame(rows)


def paired_config_test(df_per_image, tasks, a_cfg, b_cfg, title, note):
    """Paired test between two configs on the same images, per task."""
    out = []
    for task in tasks:
        sub = df_per_image[df_per_image.task == task]
        piv = sub.pivot(index="image_uuid", columns="config", values="error").dropna()
        if a_cfg not in piv.columns or b_cfg not in piv.columns:
            continue
        a, b = piv[a_cfg].values, piv[b_cfg].values
        d = b - a
        t, p = stats.ttest_rel(b, a)
        rng = np.random.default_rng(0); n = len(d)
        boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
        out.append({"task": task,
                    f"err_{CONFIG_LABELS[a_cfg]}": a.mean(),
                    f"err_{CONFIG_LABELS[b_cfg]}": b.mean(),
                    "delta": d.mean(),
                    "rel_pct": 100.0 * d.mean() / a.mean() if a.mean() else np.nan,
                    "ci_lo": np.percentile(boot, 2.5),
                    "ci_hi": np.percentile(boot, 97.5), "p": p})
    tab = pd.DataFrame(out)
    if len(tab):
        print(f"\n=== {title} ===")
        print(f"  {note}")
        print(tab.to_string(index=False))
        if len(tab) > 1:
            alpha = 0.05 / len(tab)
            print(f"  Note: {len(tab)} tasks tested; Bonferroni threshold is "
                  f"p < {alpha:.4f}. Single-task p-values above that are "
                  f"suggestive only.")
    return tab


def matched_pair_test(df_per_image, tasks):
    """4_1 vs 8_4 (both 16x total): does the more-spatially-compressed 8_4
    hurt each task more than 4_1 at equal total compression?"""
    return paired_config_test(
        df_per_image, tasks, *MATCHED_PAIR,
        "Matched-compression test: 4_1 vs 8_4 (both 16x)",
        "positive delta = 8_4 worse; tests SPATIAL vs channel at equal total compression")


def clean_spatial_test(df_per_image, tasks):
    """4_1 vs 8_1: both are native 1-channel models, so this varies only the
    spatial factor, with no channel-count or input-replication confound. This
    comparison supports a 'spatial compression' interpretation."""
    return paired_config_test(
        df_per_image, tasks, "medvae_4_1_2d", "medvae_8_1_2d",
        "Clean spatial test: 4_1 vs 8_1 (both 1-channel)",
        "positive delta = 8_1 worse; ONLY the spatial factor differs (no confound)")


# --- Figure: compression curve ---
# Error vs total compression ratio, per task, normalised to the lightest
# config so both tasks share one axis. The two 16x configs (4_1, 8_4) appear
# at the same x, so their vertical gap is the spatial-vs-channel effect.
def make_figure(summary, tasks, configs, out_path):
    order = sorted(configs, key=lambda c: COMPRESSION[c])
    xs = [COMPRESSION[c] for c in order]
    palette = [COBB_COLOR, LANDMARK_COLOR, "#065A82", "#64748B", "#21295C", "#9DC3D4"]
    colors = {t: palette[i % len(palette)] for i, t in enumerate(tasks)}
    names = {t: f"{t} ({TASK_SPEC[t]['unit']})" for t in tasks}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    light = order[0]
    for task in tasks:
        if any((c, task) not in summary for c in order):
            continue
        base = summary[(light, task)]["error_mean"]
        if base == 0:
            continue
        ys = [summary[(c, task)]["error_mean"] / base for c in order]
        ax.plot(xs, ys, "-o", color=colors[task], lw=2.2, ms=8, label=names[task])
        for c, x, y in zip(order, xs, ys):
            ax.annotate(CONFIG_LABELS[c], (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color=colors[task])
    ax.axhline(1.0, color="0.7", lw=1, ls="--", zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(xs)))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("total compression ratio (log scale)")
    ax.set_ylabel(f"error relative to lightest ({CONFIG_LABELS[light]})")
    ax.set_title("Compression curve: Cobb vs landmark localisation")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --- Synthetic data for the self-test (no MedVAE, no latents; exercises the whole pipeline) ---
def synthetic_splits(seed=0):
    rng = np.random.default_rng(seed)
    CH = {"medvae_4_3_2d": 3, "medvae_4_1_2d": 1, "medvae_8_4_2d": 4, "medvae_8_1_2d": 1}
    out = {}
    for config in CONFIG_ORDER:
        H = 96 if config.split("_")[1] == "4" else 48
        C = CH.get(config, 1)
        # more total compression -> noisier Cobb signal (crude, for the self-test only)
        noise = 0.2 + 0.01 * COMPRESSION[config]
        sp = {}
        for split, n in [("train", 120), ("val", 30), ("test", 40)]:
            lat = rng.standard_normal((n, C, H, H)).astype(np.float32)
            base = lat.reshape(n, -1).mean(1)
            cobb = np.stack([20 + 30 * base + rng.normal(0, 8 * noise, n),
                             15 + 20 * base + rng.normal(0, 6 * noise, n),
                             10 + 15 * base + rng.normal(0, 5 * noise, n)], 1)
            lmk = np.clip(0.5 + 0.2 * base[:, None]
                          + rng.normal(0, 0.05 * (1 + noise), (n, 136)), 0, 1)
            sp[split] = {"latents": lat,
                         "uuids": np.array([f"{split}_{i}" for i in range(n)]),
                         "cobb": cobb.astype(np.float32),
                         "landmarks": lmk.astype(np.float32)}
        out[config] = sp
    return out


# --- Entry point ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["cobb", "landmark"],
                    help=f"any of: {', '.join(LADDER)}")
    ap.add_argument("--labels-dir", type=Path, default=None,
                    help="dir with task_labels_*.npz (needed for surgical/"
                         "severity/curve_pattern/wedging)")
    ap.add_argument("--progress", type=int, default=0, metavar="N",
                    help="print a line every N epochs and after each seed "
                         "(e.g. 25). Useful for large feature tensors.")
    ap.add_argument("--preload-limit-gb", type=float, default=8.0,
                    help="preload a split onto the GPU if it fits under this "
                         "(avoids DataLoader collation; big speedup for DINO)")
    ap.add_argument("--exclude-implausible", action="store_true",
                    help="drop images flagged as having corrupt landmark annotation")
    ap.add_argument("--configs", nargs="+", default=CONFIG_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--smoke", action="store_true",
                    help="Synthetic-data self-test (no MedVAE/latents); checks the wiring.")
    ap.add_argument("--save-preds", action="store_true",
                    help="Also dump ensemble test predictions per config for the "
                         "Cobb-from-landmarks mechanism analysis.")
    ap.add_argument("--save-models", action="store_true",
                    help="Save trained model weights per config/task/seed for XAI.")
    args = ap.parse_args()
    global PROGRESS, PRELOAD_LIMIT_GB
    PROGRESS = args.progress
    PRELOAD_LIMIT_GB = args.preload_limit_gb
    for c in args.configs:
        register_config(c)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        print("Self-test: synthetic data, small training budget.")
        splits = synthetic_splits()
        args.epochs = min(args.epochs, 12)
        args.seeds = args.seeds[:2]
    else:
        if args.latents_dir is None:
            raise SystemExit("--latents-dir is required (or --smoke for the synthetic-data self-test)")
        splits = {c: {s: load_split(args.latents_dir, c, s,
                                    labels_dir=args.labels_dir,
                                    exclude_implausible=args.exclude_implausible)
                      for s in ("train", "val", "test")} for c in args.configs}

    train_kw = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, patience=args.patience)
    df_pi, summary, preds = run_grid(splits, args.tasks, args.configs,
                                     seeds=args.seeds, device=device,
                                     models_dir=(args.out_dir if args.save_models else None),
                                     **train_kw)

    if args.save_preds:
        for config in args.configs:
            tst = splits[config]["test"]
            out = {"uuids": tst["uuids"],
                   "gt_landmarks": tst["landmarks"].astype(np.float32),
                   "gt_cobb": tst["cobb"].astype(np.float32)}
            if (config, "landmark") in preds:
                out["pred_landmarks"] = preds[(config, "landmark")].astype(np.float32)
            if (config, "cobb") in preds:
                out["pred_cobb"] = preds[(config, "cobb")].astype(np.float32)
            p = args.out_dir / f"preds_{config}.npz"
            np.savez_compressed(p, **out)
            print(f"Saved predictions -> {p.name}")

    df_pi.to_csv(args.out_dir / "frontier_per_image.csv", index=False)
    df_sum = pd.DataFrame(list(summary.values()))
    df_sum.to_csv(args.out_dir / "frontier_summary.csv", index=False)

    ltab = ladder_table(summary, args.tasks, args.configs)
    if ltab is not None and len(ltab):
        ltab.to_csv(args.out_dir / "compression_tolerance_ladder.csv", index=False)

    ftab = frontier_table(summary, args.tasks, args.configs)
    ftab.to_csv(args.out_dir / "frontier_curve.csv", index=False)
    print("\n=== Compression curve (error vs total compression) ===")
    print(ftab.to_string(index=False))

    if all(c in args.configs for c in MATCHED_PAIR):
        mtab = matched_pair_test(df_pi, args.tasks)
        if len(mtab):
            mtab.to_csv(args.out_dir / "matched_pair_4_1_vs_8_4.csv", index=False)

    if all(c in args.configs for c in ("medvae_4_1_2d", "medvae_8_1_2d")):
        stab = clean_spatial_test(df_pi, args.tasks)
        if len(stab):
            stab.to_csv(args.out_dir / "clean_spatial_4_1_vs_8_1.csv", index=False)

    make_figure(summary, args.tasks, args.configs, args.out_dir / "frontier_curve.png")
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
