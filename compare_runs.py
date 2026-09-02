"""
compare_runs.py — paired per-image comparison between two frontier runs.

WHY THIS IS NEEDED. Comparing two runs by their reported means is not enough
when the difference is smaller than the seed-to-seed spread. The CLAHE run
improved Cobb by ~0.2 deg while the across-seed sd was 0.25-0.42 deg, so a
difference of means proves nothing. But both runs score the SAME test images,
so the comparison can be paired image-by-image, which removes patient-to-
patient variation and is far more sensitive.

Reads each run's `frontier_per_image.csv` (written by train_frontier.py) and
joins on image_uuid — never on row order.

Run:
  python compare_runs.py --run-a ./outputs/frontier --run-b ./outputs/clahe \
      --label-a replicated --label-b clahe
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_run(path):
    p = Path(path)
    f = p / "frontier_per_image.csv" if p.is_dir() else p
    if not f.exists():
        raise SystemExit(f"no per-image CSV at {f}")
    return pd.read_csv(f)


def paired(a, b, seed=0, nb=10000):
    d = b - a
    t, p = stats.ttest_rel(b, a)
    try:
        w = stats.wilcoxon(b, a).pvalue
    except Exception:
        w = float("nan")
    rng = np.random.default_rng(seed); n = len(d)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return {"delta": float(d.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "p_t": float(p), "p_wilcoxon": float(w),
            "n": int(n), "frac_better": float((d < 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="baseline run dir or CSV")
    ap.add_argument("--run-b", required=True, help="comparison run dir or CSV")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--config-a", default=None,
                    help="config in run A to compare (needed when the two runs "
                         "use different config names, e.g. MedVAE vs DINO)")
    ap.add_argument("--config-b", default=None, help="config in run B to compare")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    A, B = load_run(args.run_a), load_run(args.run_b)

    if args.config_a or args.config_b:
        ca = args.config_a or args.config_b
        cb = args.config_b or args.config_a
        missing = [c for c, d in ((ca, A), (cb, B)) if c not in set(d.config)]
        if missing:
            raise SystemExit(f"config(s) not found: {missing}\n"
                             f"  run A has: {sorted(set(A.config))}\n"
                             f"  run B has: {sorted(set(B.config))}")
        tasks = sorted(set(A[A.config == ca].task) & set(B[B.config == cb].task))
        keys = [(ca, cb, t) for t in tasks]
    else:
        common = (set(map(tuple, A[["config", "task"]].drop_duplicates().values))
                  & set(map(tuple, B[["config", "task"]].drop_duplicates().values)))
        keys = [(c, c, t) for c, t in sorted(common)]
    if not keys:
        raise SystemExit("no comparable (config, task) pairs — use --config-a/--config-b")

    print(f"paired comparison: {args.label_b} vs {args.label_a}")
    print(f"  negative delta = {args.label_b} is BETTER\n")
    hdr = (f"{'compared':<22}{'task':<10}{f'{args.label_a}':>10}{f'{args.label_b}':>10}"
           f"{'delta':>9}{'rel %':>8}{'95% CI':>20}{'p':>9}{'% imgs better':>14}")
    print(hdr); print("-" * len(hdr))

    rows = []
    for ca, cb, task in keys:
        sa = A[(A.config == ca) & (A.task == task)][["image_uuid", "error"]]
        sb = B[(B.config == cb) & (B.task == task)][["image_uuid", "error"]]
        m = sa.merge(sb, on="image_uuid", suffixes=("_a", "_b"))
        if len(m) < 5:
            print(f"  (skipping {ca} vs {cb} / {task}: only {len(m)} shared images)")
            continue
        a, b = m["error_a"].values, m["error_b"].values
        r = paired(a, b)
        rel = 100.0 * r["delta"] / a.mean() if a.mean() else np.nan
        short = (ca if ca == cb else f"{ca[:9]}\u2192{cb[:9]}").replace("medvae_", "").replace("_2d", "")
        print(f"{short:<22}{task:<10}"
              f"{a.mean():>10.4f}{b.mean():>10.4f}{r['delta']:>9.4f}{rel:>7.1f}%"
              f"   [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]{r['p_t']:>9.4g}"
              f"{100*r['frac_better']:>13.0f}%")
        rows.append({"config_a": ca, "config_b": cb, "task": task,
                     "mean_a": a.mean(), "mean_b": b.mean(), "rel_pct": rel, **r})

    tab = pd.DataFrame(rows)
    if len(tab) > 1:
        alpha = 0.05 / len(tab)
        print(f"\n  {len(tab)} comparisons — Bonferroni threshold p < {alpha:.4f}")
    print("\n  '% imgs better' is the fraction of individual images improved;")
    print("  a real effect should move well over half of them, not just the mean.")
    print("  NOTE: both runs must share the same test images for the pairing to be valid.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
