"""
run_multiseed_matched.py

Multi-seed reproducibility runner for train_cobb_regressor_matched.py.

For each (latent-source, seed) pair, runs the matched regressor and
collects the resulting test MAE. At the end, reports:
  1. Per-(condition, seed) test MAE table
  2. Per-condition mean ± std across seeds
  3. Pairwise comparisons of each condition against the chosen baseline
     condition, with a sign-test count of how many seeds favour which side

Usage:
    python run_multiseed_matched.py \\
        --out-dir ./outputs/multiseed \\
        --seeds 0 1 42 100 7

Conditions are configured in the CONDITIONS list below. Add/remove entries
to expand or restrict the sweep. The baseline condition (named in
--baseline-key) is used for pairwise comparisons.

If a seed run fails for any condition, the runner continues with the
remaining seeds and reports the failure in the summary. Partial runs are
resumable: if a per-seed output file exists with valid test MAE, the
runner skips re-running that combination.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Conditions to evaluate
# ---------------------------------------------------------------------------

CONDITIONS = [
    {
        "key":     "matched_4x",
        "label":   "MedVAE 4× aspect-padded (baseline)",
        "latents": "./latents_pretrained_4x.npz",
    },
    {
        "key":     "matched_8x",
        "label":   "MedVAE 8× aspect-padded",
        "latents": "./latents_pretrained_8x.npz",
    },
    {
        "key":     "matched_whole_768",
        "label":   "MedVAE 4× whole-image 384×768",
        "latents": "./latents_whole/latents_whole_merged.npz",
    },
    {
        "key":     "matched_whole_1024",
        "label":   "MedVAE 4× whole-image 384×1024",
        "latents": "./latents_whole_1024/latents_whole_1024_merged.npz",
    },
    {
        "key":     "matched_tiled_pretrained",
        "label":   "MedVAE 4× tile-and-stitch (pretrained encoder)",
        "latents": "./latents_pretrained_tiled/latents_pretrained_tiled_merged.npz",
    },
    {
        "key":     "matched_tiled",
        "label":   "MedVAE 4× tile-and-stitch (patch encoder)",
        "latents": "./latents_tiled/latents_tiled_merged.npz",
    },
    {
        "key":     "matched_diffusion_features",
        "label":   "Diffusion U-Net features (level 1, t=250)",
        "latents": "./latents_diffusion_features/latents_diffusion_features.npz",
    },
]


# ---------------------------------------------------------------------------
# Per-(condition, seed) execution
# ---------------------------------------------------------------------------

def run_single(condition: dict, seed: int, out_dir: Path,
                training_script: str = "train_cobb_regressor_matched.py",
                ) -> Optional[float]:
    """Run one (condition, seed) job. Returns the test MAE, or None on failure.

    If the output NPZ already exists with valid per_image_mae, skip and
    return the cached value. This makes the runner safely resumable.
    """
    output_path = out_dir / f"{condition['key']}_seed{seed}.npz"

    # Resume: if output already exists with valid MAE, just read it
    if output_path.exists():
        try:
            d = np.load(output_path, allow_pickle=True)
            test_mae = float(d["per_image_mae"].mean())
            print(f"    [cached] {condition['key']} seed={seed} -> {test_mae:.3f}°")
            return test_mae
        except Exception as e:
            print(f"    [cached file unreadable, will re-run: {e}]")

    cmd = [
        sys.executable, training_script,
        "--latents", condition["latents"],
        "--out", str(output_path),
        "--seed", str(seed),
    ]
    print(f"    Running {condition['key']} seed={seed}...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1200,
        )
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT after 20 minutes]")
        return None

    if result.returncode != 0:
        print(f"    [FAIL with code {result.returncode}]")
        # Print last 20 lines of stderr for diagnosis
        err_lines = result.stderr.strip().splitlines()[-20:]
        for line in err_lines:
            print(f"      {line}")
        return None

    # Read the output file to extract test MAE
    try:
        d = np.load(output_path, allow_pickle=True)
        test_mae = float(d["per_image_mae"].mean())
        print(f"    {condition['key']} seed={seed} -> {test_mae:.3f}°")
        return test_mae
    except Exception as e:
        print(f"    [could not parse output: {e}]")
        return None


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def per_condition_summary(results: dict) -> dict:
    """results: {condition_key: {seed: mae_or_None}}.

    Returns: {condition_key: {"mean": float, "std": float, "n": int,
                              "values": [...]}}
    """
    summary = {}
    for cond_key, seed_results in results.items():
        valid = [v for v in seed_results.values() if v is not None]
        if not valid:
            summary[cond_key] = {"mean": float("nan"),
                                   "std":  float("nan"),
                                   "n": 0,
                                   "values": []}
        else:
            arr = np.array(valid)
            summary[cond_key] = {
                "mean":   float(arr.mean()),
                "std":    float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n":      len(arr),
                "values": list(arr),
            }
    return summary


def sign_test_count(condition_results: dict, baseline_results: dict,
                     seeds: list) -> tuple:
    """Per-seed, did condition beat baseline? Returns (wins, n_paired)
    where wins = number of seeds where condition.test_mae < baseline.test_mae.
    """
    wins = 0
    n_paired = 0
    for seed in seeds:
        c_val = condition_results.get(seed)
        b_val = baseline_results.get(seed)
        if c_val is None or b_val is None:
            continue
        n_paired += 1
        if c_val < b_val:
            wins += 1
    return wins, n_paired


def print_summary(results: dict, conditions: list, seeds: list,
                   baseline_key: str):
    """Pretty-print the full summary."""

    # Build a lookup so we can find condition labels
    cond_lookup = {c["key"]: c for c in conditions}

    print("\n" + "=" * 90)
    print("MULTI-SEED MATCHED-REGRESSOR REPRODUCIBILITY SUMMARY")
    print("=" * 90)

    # ---- Per-(condition, seed) raw table ----
    print(f"\nPer-seed test MAE (degrees):")
    header_seeds = "  ".join(f"seed={s:<4}" for s in seeds)
    print(f"  {'condition':<40}  {header_seeds}")
    print(f"  {'-' * 40}  {'-' * len(header_seeds)}")
    for cond in conditions:
        key = cond["key"]
        row = "  ".join(
            f"{results[key][s]:>7.3f}" if results[key].get(s) is not None
            else "    FAIL"
            for s in seeds
        )
        print(f"  {cond['label']:<40}  {row}")

    # ---- Per-condition aggregated ----
    print(f"\nAggregated (mean ± std, n=seeds with valid result):")
    print(f"  {'condition':<40}  {'mean ± std':<18}  {'n':<3}  {'min':<7}  {'max':<7}")
    print(f"  {'-' * 40}  {'-' * 18}  {'-' * 3}  {'-' * 7}  {'-' * 7}")
    summary = per_condition_summary(results)
    for cond in conditions:
        key = cond["key"]
        s = summary[key]
        if s["n"] == 0:
            print(f"  {cond['label']:<40}  {'(all failed)':<18}  0")
            continue
        mae_str = f"{s['mean']:.3f} ± {s['std']:.3f}"
        min_v = min(s["values"])
        max_v = max(s["values"])
        print(f"  {cond['label']:<40}  {mae_str:<18}  {s['n']:<3}  "
              f"{min_v:<7.3f}  {max_v:<7.3f}")

    # ---- Pairwise comparisons against baseline ----
    if baseline_key not in cond_lookup:
        print(f"\n⚠ baseline_key={baseline_key} not in conditions; "
              f"skipping pairwise comparisons.")
        return summary

    print(f"\nPairwise vs baseline ({cond_lookup[baseline_key]['label']}):")
    print(f"  {'condition':<40}  {'mean Δ':<12}  {'sign test':<14}  {'verdict':<20}")
    print(f"  {'-' * 40}  {'-' * 12}  {'-' * 14}  {'-' * 20}")

    b_summary = summary[baseline_key]
    b_results = results[baseline_key]

    for cond in conditions:
        key = cond["key"]
        if key == baseline_key:
            continue
        c_summary = summary[key]
        c_results = results[key]

        if c_summary["n"] == 0:
            print(f"  {cond['label']:<40}  {'(no data)':<12}")
            continue

        # Mean delta (positive = condition better than baseline,
        # since lower MAE is better, so we use baseline - condition)
        mean_delta = b_summary["mean"] - c_summary["mean"]
        wins, n_paired = sign_test_count(c_results, b_results, seeds)

        # Verdict
        if n_paired == 0:
            verdict = "(no paired seeds)"
        elif wins == n_paired:
            verdict = "consistently better"
        elif wins == 0:
            verdict = "consistently worse"
        elif wins / n_paired >= 0.8:
            verdict = "mostly better"
        elif wins / n_paired <= 0.2:
            verdict = "mostly worse"
        else:
            verdict = "mixed / unclear"

        sign_str = f"{wins}/{n_paired} better"
        print(f"  {cond['label']:<40}  {mean_delta:+.3f}°       "
              f"{sign_str:<14}  {verdict}")

    print("=" * 90)
    return summary


def write_csv(results: dict, conditions: list, seeds: list, csv_path: Path):
    """Write a per-(condition, seed) CSV for downstream analysis."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition_key", "condition_label", "seed", "test_mae"])
        for cond in conditions:
            for seed in seeds:
                mae = results[cond["key"]].get(seed)
                writer.writerow([
                    cond["key"], cond["label"], seed,
                    f"{mae:.4f}" if mae is not None else "FAIL",
                ])
    print(f"\nWrote per-(condition, seed) results to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, type=Path,
                    help="Directory for per-seed output NPZs and the summary CSV.")
    p.add_argument("--seeds", type=int, nargs="+",
                    default=[0, 1, 42, 100, 7])
    p.add_argument("--baseline-key", default="matched_4x",
                    help="Which condition (key) to use as baseline for "
                         "pairwise comparisons. Default: matched_4x.")
    p.add_argument("--training-script", default="train_cobb_regressor_matched.py",
                    help="Path to the training script. Default: in cwd.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # results[condition_key][seed] = test_mae or None
    results = {c["key"]: {s: None for s in args.seeds} for c in CONDITIONS}

    print(f"Running {len(CONDITIONS)} conditions × {len(args.seeds)} seeds = "
          f"{len(CONDITIONS) * len(args.seeds)} jobs")
    print(f"Seeds: {args.seeds}")
    print(f"Conditions: {[c['key'] for c in CONDITIONS]}")
    print(f"Outputs will go to: {args.out_dir}")
    print()

    for cond in CONDITIONS:
        print(f"\n--- {cond['label']} ({cond['key']}) ---")
        # Verify latents file exists before running any seeds for this condition
        if not Path(cond["latents"]).exists():
            print(f"  ⚠ Latents file {cond['latents']} not found; "
                  f"skipping condition.")
            continue
        for seed in args.seeds:
            mae = run_single(cond, seed, args.out_dir,
                              training_script=args.training_script)
            results[cond["key"]][seed] = mae

    # Final summary
    print_summary(results, CONDITIONS, args.seeds, args.baseline_key)
    write_csv(results, CONDITIONS, args.seeds,
                args.out_dir / "multiseed_summary.csv")


if __name__ == "__main__":
    main()
