"""
task_baselines.py: score each task with a trivial predictor that ignores the image.

An error is interpreted relative to the trivial predictor. This fits the
simplest possible model to the train split and evaluates it on test, using the
same metrics as the trained-model results (frontier_summary.csv), so "% better
than baseline" is comparable across all six tasks.

    surgical / severity / curve_pattern : predict the majority class
    landmark                            : predict the mean shape
    cobb                                : predict the mean angle vector
    wedging                             : predict the per-vertebra mean wedge

Comparing each task's error with the trivial predictor distinguishes a
compression-insensitive task from a head that predicts a constant: an error
that is flat across configurations is consistent with either, and only the
baseline separates them. The same comparison shows whether a classification
task performs better than its majority-class baseline.

Inputs:  --latents-dir (latents_{config}_{split}.npz, providing uuids, cobb and
         landmarks), --labels-dir (task_labels_{split}.npz, joined by uuid) and
         optionally --results (frontier_summary.csv) for the model comparison.
Outputs: the baseline table and verdicts printed to stdout;
         <out-dir>/task_baselines.csv.

Usage:
  python task_baselines.py --labels-dir ./labels_tasks --latents-dir ./latents_aspect \
      --results ./outputs/ladder/frontier_summary.csv
  python task_baselines.py --smoke      # synthetic-data self-test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_LM = 68
N_VERT = 17
CLF_TASKS = ["surgical", "severity", "curve_pattern"]
REG_TASKS = ["landmark", "cobb", "wedging"]
UNITS = {"surgical": "err rate", "severity": "err rate", "curve_pattern": "err rate",
         "landmark": "norm", "cobb": "deg", "wedging": "deg"}


def load(labels_dir, latents_dir, config, split):
    """Task labels joined to the latent split by uuid (never by row order)."""
    out = {}
    lp = Path(latents_dir) / f"latents_{config}_{split}.npz"
    d = np.load(lp, allow_pickle=True)
    out["uuids"] = d["uuids"]
    out["cobb"] = d["cobb"].astype(np.float64)
    out["landmark"] = d["landmarks"].astype(np.float64)

    lf = Path(labels_dir) / f"task_labels_{split}.npz"
    if lf.exists():
        L = np.load(lf, allow_pickle=True)
        idx = {str(u): i for i, u in enumerate(L["uuids"])}
        order = np.array([idx[str(u)] for u in out["uuids"] if str(u) in idx])
        keep = np.array([i for i, u in enumerate(out["uuids"]) if str(u) in idx])
        out["cobb"] = out["cobb"][keep]
        out["landmark"] = out["landmark"][keep]
        out["uuids"] = out["uuids"][keep]
        for k in CLF_TASKS:
            if k in L.files:
                out[k] = L[k][order].astype(np.int64)
        if "wedging" in L.files:
            out["wedging"] = L["wedging"][order].astype(np.float64)
        if "implausible" in L.files:
            out["implausible"] = L["implausible"][order]
    return out


def drop_implausible(d):
    if "implausible" not in d:
        return d
    good = ~d["implausible"]
    out = {}
    for k, v in d.items():
        if k == "implausible":
            continue
        out[k] = v[good] if isinstance(v, np.ndarray) and len(v) == len(good) else v
    return out


def landmark_error(pred, true):
    dx = pred[:, :N_LM] - true[:, :N_LM]
    dy = pred[:, N_LM:] - true[:, N_LM:]
    return np.sqrt(dx ** 2 + dy ** 2).mean(1)


def boot_ci(v, seed=0, nb=10000):
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed); n = len(v)
    b = np.array([v[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def baseline_for(task, train, test):
    """Fit the trivial predictor on train, score it on test. Returns
    (per-image error array, description of the predictor)."""
    if task in CLF_TASKS:
        if task not in train:
            return None, None
        counts = np.bincount(train[task])
        maj = int(counts.argmax())
        err = (test[task] != maj).astype(float)
        return err, f"always predict class {maj} ({100*counts[maj]/counts.sum():.0f}% of train)"

    if task == "landmark":
        mean_shape = train["landmark"].mean(0, keepdims=True)
        err = landmark_error(np.repeat(mean_shape, len(test["landmark"]), 0),
                             test["landmark"])
        return err, "mean shape over train"

    if task == "cobb":
        mu = train["cobb"].mean(0, keepdims=True)
        err = np.abs(np.repeat(mu, len(test["cobb"]), 0) - test["cobb"]).mean(1)
        return err, "mean angle vector over train"

    if task == "wedging":
        if "wedging" not in train:
            return None, None
        mu = train["wedging"].mean(0, keepdims=True)     # per-vertebra mean
        err = np.abs(np.repeat(mu, len(test["wedging"]), 0) - test["wedging"]).mean(1)
        return err, "per-vertebra mean wedge over train"

    return None, None


def report(train, test, results_csv=None, config_label="4_1"):
    model = {}
    if results_csv and Path(results_csv).exists():
        r = pd.read_csv(results_csv)
        col = "config" if "config" in r.columns else None
        for _, row in r.iterrows():
            cfg = str(row[col]) if col else ""
            if config_label in cfg:
                model[str(row["task"])] = float(row["error_mean"])
        if model:
            print(f"comparing against trained model results for config "
                  f"containing '{config_label}'\n")

    tasks = CLF_TASKS + REG_TASKS
    hdr = (f"{'task':<15}{'unit':>9}{'baseline':>11}{'CI':>20}"
           f"{'model':>10}{'better by':>11}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for t in tasks:
        err, desc = baseline_for(t, train, test)
        if err is None:
            continue
        m, lo, hi = boot_ci(err)
        mv = model.get(t, np.nan)
        better = 100.0 * (m - mv) / m if (m > 0 and np.isfinite(mv)) else np.nan
        bstr = f"{better:>10.1f}%" if np.isfinite(better) else f"{'—':>11}"
        mstr = f"{mv:>10.4f}" if np.isfinite(mv) else f"{'—':>10}"
        print(f"{t:<15}{UNITS[t]:>9}{m:>11.4f}   [{lo:.4f},{hi:.4f}]{mstr}{bstr}")
        rows.append({"task": t, "unit": UNITS[t], "baseline": m,
                     "baseline_lo": lo, "baseline_hi": hi,
                     "model": mv, "better_pct": better, "predictor": desc})

    print("\npredictors used:")
    for r in rows:
        print(f"   {r['task']:<15} {r['predictor']}")

    # Verdicts
    print("\n" + "=" * 62)
    print("Verdict: does the model improve on the trivial predictor?")
    print("=" * 62)
    for r in rows:
        b = r["better_pct"]
        if not np.isfinite(b):
            print(f"  {r['task']:<15} (no model result supplied)")
            continue
        if b < 2:
            v = "NOT LEARNING — indistinguishable from the trivial predictor"
        elif b < 10:
            v = "weak — barely beats guessing; treat results with caution"
        elif b < 30:
            v = "moderate"
        else:
            v = "learns well"
        print(f"  {r['task']:<15} {b:>6.1f}% better  ->  {v}")
    print("\nA task that barely beats its baseline gives little evidence of compression")
    print("tolerance: flat error across configurations would then only mean that the")
    print("model predicts a constant regardless of the input.")
    return pd.DataFrame(rows)


# Synthetic-data self-test
def smoke():
    rng = np.random.default_rng(0)
    ntr, nte = 300, 120

    def make(n, informative):
        d = {}
        d["cobb"] = rng.normal([39, 20, 18], [16, 11, 12], (n, 3))
        d["landmark"] = np.clip(0.5 + 0.08 * rng.standard_normal((n, 136)), 0, 1)
        d["wedging"] = np.abs(rng.normal(3.6, 1.8, (n, N_VERT)))
        d["surgical"] = (d["cobb"][:, 0] >= 45).astype(np.int64)
        d["severity"] = np.digitize(d["cobb"][:, 0], [25, 45]).astype(np.int64)
        d["curve_pattern"] = (rng.random(n) < 0.3).astype(np.int64)
        return d

    train, test = make(ntr, True), make(nte, True)
    # no model results are supplied; only the baselines and the machinery checks are tested
    tab = report(train, test)
    print("\n(the self-test supplies no model results, so the 'better by' column is blank)")

    # explicit check of the baseline machinery itself
    err, desc = baseline_for("wedging", train, test)
    mu = train["wedging"].mean(0)
    manual = np.abs(test["wedging"] - mu[None, :]).mean(1)
    ok_wedge = np.allclose(err, manual)
    err_s, _ = baseline_for("surgical", train, test)
    maj = int(np.bincount(train["surgical"]).argmax())
    ok_clf = np.allclose(err_s, (test["surgical"] != maj).astype(float))
    print(f"\nwedging baseline matches manual computation: {ok_wedge}")
    print(f"majority-class baseline matches manual computation: {ok_clf}")
    print("\nSelf-test:", "PASS" if (ok_wedge and ok_clf) else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", type=Path)
    ap.add_argument("--latents-dir", type=Path)
    ap.add_argument("--config", default="medvae_4_1_2d")
    ap.add_argument("--results", type=Path, default=None,
                    help="frontier_summary.csv, to auto-compare against the model")
    ap.add_argument("--config-label", default="4_1")
    ap.add_argument("--exclude-implausible", action="store_true", default=True)
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/baselines"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    if not (args.labels_dir and args.latents_dir):
        raise SystemExit("--labels-dir and --latents-dir are required (or --smoke for the synthetic-data self-test)")
    train = load(args.labels_dir, args.latents_dir, args.config, "train")
    test = load(args.labels_dir, args.latents_dir, args.config, "test")
    if args.exclude_implausible:
        train, test = drop_implausible(train), drop_implausible(test)
    print(f"train n={len(train['uuids'])}, test n={len(test['uuids'])}\n")
    tab = report(train, test, args.results, args.config_label)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out_dir / "task_baselines.csv", index=False)
    print(f"\nWrote {args.out_dir}/task_baselines.csv")


if __name__ == "__main__":
    main()
