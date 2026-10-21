"""Analyse complete, paired three-seed ensembles. Never treat seeds as test images."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import *
from run_heads import done, jobs, paths


def contrast_definitions():
    out = []
    for task in TASKS:
        for variant in VARIANTS:
            out.append((f"exclusion_{variant}_{task}", task, {("clean", variant): 1, ("unclean", variant): -1}))
        for cohort in ("unclean", "clean"):
            out.append((f"projection_after_averaging_{cohort}_{task}", task, {(cohort, "projected_mean"): 1, (cohort, "raw_mean"): -1}))
            out.append((f"interaction_{cohort}_{task}", task, {(cohort, "projected_mean"): 1, (cohort, "raw_mean"): -1,
                (cohort, "projected_keep"): -1, (cohort, "raw_keep"): 1}))
        out.append((f"change_in_interaction_{task}", task,
            {(c, v): sign * weight for c, sign in (("clean", 1), ("unclean", -1)) for v, weight in
             (("projected_mean", 1), ("raw_mean", -1), ("projected_keep", -1), ("raw_keep", 1))}))
    require(len(out) == 18, "Unexpected comparison count")
    return out


def contrast_vector(name, metric, weights, effects):
    def interaction(cohort):
        return ((effects[(cohort,"projected_mean",metric)]-effects[(cohort,"raw_mean",metric)])
                -(effects[(cohort,"projected_keep",metric)]-effects[(cohort,"raw_keep",metric)]))
    if name.startswith("change_in_interaction_"):
        return interaction("clean")-interaction("unclean")
    if name.startswith("interaction_"):
        return interaction(next(iter(weights))[0])
    return sum(w*effects[(c,v,metric)] for (c,v),w in weights.items())


def analyse(dst):
    dst = Path(dst).resolve(); plan = get_plan(dst); m = read(dst / "manifest.json")
    require(m["plan_fingerprint"] == plan["fingerprint"], "Plan/manifest differ")
    planned = jobs(plan)
    require(all(done(dst, plan, j) for j in planned), "The declared fits are incomplete. Return the pilot/diagnostics instead of analysing a selected subset.")
    require(sha256(dst / "evaluation.npz") == m["evaluation_sha256"], "Evaluation labels changed")
    require(sha256(dst / "image_dimensions.csv") == m["dimensions_sha256"], "Image dimensions changed")
    ev = arrays(dst / "evaluation.npz"); ids = list(ev["uuids"].astype(str))
    dimensions = pd.read_csv(dst / "image_dimensions.csv").set_index("image_uuid").loc[ids, ["width", "height"]].to_numpy(float)
    summary, per_image, seed_rows, effects = [], [], [], {}
    cohorts = ("unclean", "clean") if plan["kind"] == "annotation" else ("clean", "augmented")
    variants = VARIANTS if plan["kind"] == "annotation" else ("projected_mean",)
    export = dst / "analysis"; export.mkdir(exist_ok=True)
    for cohort in cohorts:
        for variant in variants:
            for task in TASKS:
                true = ev["cobb" if task == "cobb" else "landmarks"]
                predictions = []
                for seed in SEEDS:
                    key = f"{cohort}/{variant}/{task}/{seed}"
                    if key in m.get("imported", {}):
                        record = m["imported"][key]; p = dst / record["path"]
                        require(sha256(p) == record["sha256"], f"Changed imported prediction: {p}")
                    else:
                        p, meta = paths(dst, (cohort, variant, task, seed))
                        require(done(dst, plan, (cohort, variant, task, seed)), f"Missing fit {p}")
                    pred = load_prediction(p, ids, true); predictions.append(pred)
                    seed_rows.append(dict(cohort=cohort, variant=variant, task=task, seed=seed,
                        mean_error=float(errors(task, pred, true).mean()),
                        isotropic_percent=float(errors(task, pred, true, dimensions).mean()) if task == "landmark" else ""))
                # Match the original pipeline: average predictions first, then calculate error.
                ensemble = np.stack(predictions).mean(0, dtype=np.float32)
                atomic_npz(export / f"ensemble_{cohort}_{variant}_{task}.npz", pred=ensemble, true=true, uuids=ev["uuids"])
                for metric, dims in ((task, None), *(([("landmark_isotropic", dimensions)]) if task == "landmark" else [])):
                    e = errors(task, ensemble, true, dims)
                    effects[(cohort, variant, metric)] = e
                    individual = [float(errors(task, p, true, dims).mean()) for p in predictions]
                    interval = paired_summary(e)
                    summary.append(dict(cohort=cohort, variant=variant, task=task, metric=metric,
                        ensemble_mean=float(e.mean()), ci_low=interval["ci_low"], ci_high=interval["ci_high"],
                        seed_metric_mean=float(np.mean(individual)), seed_metric_sd=float(np.std(individual, ddof=1)), n_images=len(ids), n_seeds=3))
                    per_image.extend(dict(image_uuid=u, cohort=cohort, variant=variant, metric=metric, error=float(x)) for u, x in zip(ids, e))
    definitions = contrast_definitions() if plan["kind"] == "annotation" else [
        (f"augmentation_{t}", t, {("augmented", "projected_mean"): 1, ("clean", "projected_mean"): -1}) for t in TASKS]
    primary_n, secondary_n = (18, 9) if plan["kind"] == "annotation" else (2, 1)
    comparisons, differences = [], []
    for name, task, weights in definitions:
        for metric in ([task, "landmark_isotropic"] if task == "landmark" else [task]):
            d = contrast_vector(name, metric, weights, effects)
            row = paired_summary(d)
            family = "secondary" if metric == "landmark_isotropic" else "primary"
            nfamily = secondary_n if family == "secondary" else primary_n
            row.update(comparison=name, metric=metric, family=family, family_size=nfamily,
                p_bonferroni=min(1., row["p_raw"] * nfamily), p_all_tests_sensitivity=min(1., row["p_raw"] * (primary_n+secondary_n)),
                unit="degrees" if task == "cobb" else "percent_vertical_spine_extent" if metric == "landmark_isotropic" else "image_normalised_distance",
                coefficients=json.dumps({f"{c}/{v}": w for (c,v), w in weights.items()}, sort_keys=True))
            comparisons.append(row)
            differences.extend(dict(image_uuid=u, comparison=name, metric=metric, difference=float(x)) for u,x in zip(ids,d))
    require(sum(r["family"] == "primary" for r in comparisons) == primary_n and sum(r["family"] == "secondary" for r in comparisons) == secondary_n, "Wrong family sizes")
    for filename, rows in (("summary.csv", summary), ("per_image.csv", per_image), ("per_seed.csv", seed_rows),
                            ("comparisons.csv", comparisons), ("paired_differences.csv", differences)):
        write_csv(export / filename, rows)
    atomic_json(export / "analysis_complete.json", dict(plan_fingerprint=plan["fingerprint"], n_images=len(ids), seeds=SEEDS,
        primary_tests=primary_n, secondary_tests=secondary_n, input_prediction_sha256={str(p.relative_to(dst)): sha256(p) for p in sorted(dst.rglob("*_seed*.npz"))},
        output_sha256={p.name: sha256(p) for p in sorted(export.glob("*.csv"))},
        interpretation=["Negative cell differences favour the first named pipeline.",
            "Interaction = (projected_mean - raw_mean) - (projected_keep - raw_keep). Negative means projection helps more when channels are averaged.",
            "Change in interaction = clean interaction - unclean interaction.",
            "Intervals are unadjusted and conditional on these three fitted seeds; the resampling unit is the image.",
            "A non-significant difference does not establish equivalence or robustness. These tests reuse the thesis test set."]))
    print(f"Analysis complete: {export}. {primary_n} primary and {secondary_n} secondary comparisons; no pooled seed/image pseudo-replication.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out-dir", type=Path, required=True)
    analyse(p.parse_args().out_dir)
