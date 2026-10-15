"""Combine verified reference results with completed follow-up cells.

All contrasts are paired by image identifier. Positive changes mean higher
error in B. Families are fixed in the code, including not-yet-completed tests;
missing cells never reduce the Bonferroni denominator.

Before analysis, prepared-cache hashes, reference-file hashes and completed-run
outputs are checked against their recorded SHA-256 values.

Inputs:  --study-dir (manifest.json, prepared caches and completed runs) and
         the reference CSVs in references/.
Outputs: in <study-dir>/analysis: all_per_image.csv, all_summary.csv,
         paired_comparisons.csv, relative_task_changes.csv,
         pending_comparisons.csv, analysis_plan.json and RESULTS.md (also
         printed to stdout).

Usage:
  python analyse_projection_study.py --study-dir <study> [--resamples 10000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projection_common import (CONFIGS, MULTI, SHAPES, SINGLE, TASKS, atomic_json,
    checked_manifest, mean_interval, paired, read_json, require, sha256, projected_reference_configs, reference_configs)


def comparison_plan():
    specs = []
    def add(family, name, a, b):
        for task in TASKS:
            specs.append(dict(family=family, comparison=name, task=task, a=a, b=b))
    def cell(variant, config): return [(1, variant, config)]
    add("matched_storage", "projected retained 4_1 to 8_4 (36 KiB each)",
        cell("projected_keep", CONFIGS[0]), cell("projected_keep", CONFIGS[3]))
    add("single_channel_spatial", "projected 4_1 to 8_1", cell("projected_keep", SINGLE[0]),
        cell("projected_keep", SINGLE[1]))
    for handling, configs in (("keep", CONFIGS), ("mean", MULTI)):
        for c in configs:
            add("projection_effect", f"raw to projected, {handling}, {c}",
                cell("raw_"+handling, c), cell("projected_"+handling, c))
    for stage in ("raw", "projected"):
        for c in MULTI:
            add("pooling_effect", f"retained to mean channels, {stage}, {c}",
                cell(stage+"_keep", c), cell(stage+"_mean", c))
    for handling in ("keep", "mean"):
        for a, b in ((CONFIGS[0], CONFIGS[1]), (CONFIGS[2], CONFIGS[3])):
            # A mean over the one-channel checkpoint is identical to the keep variant.
            add("same_grid_checkpoints", f"projected {handling}, {a} to {b}",
                cell("projected_keep", a), cell("projected_"+handling, b))
    for c in MULTI:
        add("projection_pooling_interaction", f"change in pooling penalty, {c}",
            [(1, "raw_mean", c), (-1, "raw_keep", c)],
            [(1, "projected_mean", c), (-1, "projected_keep", c)])
    add("matched_storage_interaction", "change in 8_4-minus-4_1 penalty after projection",
        [(1, "raw_keep", CONFIGS[3]), (-1, "raw_keep", CONFIGS[0])],
        [(1, "projected_keep", CONFIGS[3]), (-1, "projected_keep", CONFIGS[0])])
    return specs


def read_errors(path, expected_ids, variant=None):
    df = pd.read_csv(path, dtype={"image_uuid": str})
    require({"config", "task", "image_uuid", "error"}.issubset(df.columns), f"Missing columns: {path}")
    if variant is not None: df["variant"] = variant
    require("variant" in df, "Variant missing")
    require(not df.duplicated(["variant", "config", "task", "image_uuid"]).any(), "Repeated error rows")
    require(np.isfinite(df.error).all() and (df.error >= 0).all(), "Invalid error values")
    for key, sub in df.groupby(["variant", "config", "task"]):
        require(set(sub.image_uuid) == set(expected_ids), f"Test cohort mismatch: {key}")
    return df


def checked_run(path):
    complete = read_json(path / "complete.json")
    for name, digest in complete["sha256"].items():
        require(sha256(path / name) == digest, f"Changed/incomplete run output: {path / name}")
    require(read_json(path / "protocol.json")["fingerprint"] == complete["fingerprint"],
            "Completed run protocol mismatch")
    return complete


def load_results(study, manifest):
    ref = Path(__file__).parent / "references"
    ids = sorted(manifest["expected_ids"]["test"])
    frames, seed_sd = [], {}
    reference_manifest = read_json(ref / "protocol.json")
    for stage in ("raw", "projected"):
        eligible = reference_configs(manifest, stage)
        if not eligible: continue
        name = stage+"_frontier_per_image.csv"
        require(sha256(ref/name) == reference_manifest["files_sha256"][name], "Reference CSV changed")
        frame = read_errors(ref/name, ids, stage+"_keep")
        configs = CONFIGS if stage == "raw" else MULTI
        require(set(zip(frame.config, frame.task)) == {(c, t) for c in configs for t in TASKS},
                "Unexpected reference grid")
        name = stage+"_frontier_summary.csv"
        require(sha256(ref/name) == reference_manifest["files_sha256"][name], "Reference summary changed")
        summary = pd.read_csv(ref/name)
        for row in summary.itertuples():
            values = frame.loc[(frame.config == row.config) & (frame.task == row.task), "error"]
            require(len(values) == 128 and abs(values.mean()-row.error_mean) < 1e-5,
                    "Reference summary does not reproduce per-image errors")
            if row.config in eligible:
                seed_sd[stage+"_keep", row.config, row.task] = float(row.seed_sd)
        frame = frame.loc[frame.config.isin(eligible)]
        frames.append(frame)
    for complete in sorted((study/"runs").glob("*/*/complete.json")):
        path = complete.parent; checked_run(path)
        protocol = read_json(path/"protocol.json")
        v, c = protocol["variant"], protocol["config"]
        if v in ("raw_keep", "projected_keep"):
            require(c not in reference_configs(manifest, v.removesuffix("_keep")),
                    "A new run conflicts with a reused historical reference")
        require(protocol["manifest_sha256"] == sha256(study/"manifest.json"), "Manifest changed since fitting")
        require(protocol["input_sha256"] == manifest["cells"][v][c]["sha256"], "Different training inputs")
        frame = read_errors(path/"per_image.csv", ids)
        require(set(zip(frame.variant, frame.config, frame.task)) == {(v, c, t) for t in TASKS},
                "Unexpected completed run grid")
        require(protocol["seeds"] == list(range(10)), "Different ensemble size/seeds")
        frames.append(frame)
        for row in pd.read_csv(path/"summary.csv").itertuples():
            seed_sd[v, c, row.task] = float(row.seed_sd)
    require(frames and any(len(f) for f in frames), "No completed results: no primary or core runs have completed")
    all_rows = pd.concat(frames, ignore_index=True)
    require(not all_rows.duplicated(["variant", "config", "task", "image_uuid"]).any(), "Duplicate result cells")
    values = {(v,c,t): sub.set_index("image_uuid").loc[ids, "error"].to_numpy()
              for (v,c,t), sub in all_rows.groupby(["variant", "config", "task"])}
    return all_rows, values, seed_sd


def evaluate_plan(values, specs, resamples=10000):
    family_n = {f: sum(x["family"] == f for x in specs) for f in {x["family"] for x in specs}}
    done, missing = [], []
    for spec in specs:
        def combine(terms):
            keys = [(v,c,spec["task"]) for _,v,c in terms]
            if any(k not in values for k in keys): return None
            return sum(weight * values[k] for (weight,_,_), k in zip(terms, keys))
        a, b = combine(spec["a"]), combine(spec["b"])
        public = {k: spec[k] for k in ("family", "comparison", "task")}
        if a is None or b is None:
            missing.append(dict(**public, family_size=family_n[spec["family"]])); continue
        result = paired(a, b, resamples=resamples)
        if "interaction" in spec["family"]: result["relative_percent"] = None
        n = family_n[spec["family"]]
        done.append(dict(**public, n=len(a), **result, family_size=n, family_alpha=.05/n,
            p_bonf_family=min(1., result["p"]*n), p_bonf_all=min(1., result["p"]*len(specs)),
            passes_family=bool(result["p"] < .05/n)))
    return pd.DataFrame(done), pd.DataFrame(missing), family_n


def relative_task_changes(values, resamples=10000):
    """Descriptive cross-task intervals using the same image draws for all cells."""
    rows = []
    pairs = (("single_channel", SINGLE[0], SINGLE[1]),
             ("matched_storage", CONFIGS[0], CONFIGS[3]),
             ("multichannel_reference", CONFIGS[1], CONFIGS[3]))
    for stage in ("raw_keep", "projected_keep"):
        for name, a, b in pairs:
            if not all((stage,c,t) in values for c in (a,b) for t in TASKS): continue
            n = len(values[stage,a,"cobb"])
            draws = np.random.default_rng(0).integers(0,n,(resamples,n))
            observed, boot = {}, {}
            for task in TASKS:
                av, bv = values[stage,a,task], values[stage,b,task]
                denom = av[draws].mean(1)
                require(av.mean() > 0 and np.all(denom > 0), "Undefined relative error change")
                observed[task] = 100*(bv.mean()/av.mean()-1)
                boot[task] = 100*(bv[draws].mean(1)/denom-1)
            lo, hi = np.percentile(boot["cobb"]-boot["landmark"],[2.5,97.5])
            rows.append(dict(variant=stage, comparison=name, config_a=a, config_b=b,
                cobb_relative_percent=float(observed["cobb"]), landmark_relative_percent=float(observed["landmark"]),
                cobb_minus_landmark_pp=float(observed["cobb"]-observed["landmark"]),
                ci95_lo_pp=float(lo), ci95_hi_pp=float(hi),
                scope="descriptive unadjusted cross-task interval; not an equivalence test"))
    return pd.DataFrame(rows)


def analyse(args):
    study = args.study_dir.resolve(); m = checked_manifest(study)
    # Check cache hashes without repeatedly loading full arrays.
    for cells in m["cells"].values():
        for entry in cells.values():
            if "alias" in entry: continue
            for split, path in entry["paths"].items():
                require(sha256(path) == entry["sha256"][split], f"Prepared cache changed: {path}")
    rows, values, seed_sd = load_results(study, m)
    specs = comparison_plan()
    paired_table, missing, families = evaluate_plan(values, specs, args.resamples)
    summary = []
    for (v,c,t), errors in sorted(values.items()):
        channels,h,w = SHAPES[c]; channels = 1 if v.endswith("mean") else channels
        mean, lo, hi = mean_interval(errors, args.resamples)
        summary.append(dict(variant=v, config=c, task=t, error_mean=mean, error_lo=lo, error_hi=hi,
            n=len(errors), n_seeds=10, seed_sd=seed_sd[v,c,t], actual_channels=channels,
            fp32_kib=channels*h*w*4/1024, compression_ratio=384**2/(channels*h*w)))
    out = study/"analysis"; out.mkdir(exist_ok=True)
    rows.to_csv(out/"all_per_image.csv", index=False)
    pd.DataFrame(summary).to_csv(out/"all_summary.csv", index=False)
    paired_table.to_csv(out/"paired_comparisons.csv", index=False)
    relative_task_changes(values, args.resamples).to_csv(out/"relative_task_changes.csv", index=False)
    missing.to_csv(out/"pending_comparisons.csv", index=False)
    atomic_json(out/"analysis_plan.json", dict(specs=specs, family_sizes=families,
        total_tests=len(specs), image_bootstrap_resamples=args.resamples, bootstrap_seed=0,
        reference_reuse=m.get("reference_reuse", dict(raw_keep=list(CONFIGS), projected_keep=list(MULTI))),
        study_kind=m.get("study_kind", "historical_cache"),
        raw_cache_policy=m.get("raw_cache_policy", "not_applicable_fresh_extraction" if m.get("study_kind") == "fresh_paired" else "strict"),
        historical_reencoding_summary=m.get("historical_reencoding_summary"),
        note="Exploratory, fixed fitted ensembles; image resampling does not include refitting uncertainty.",
        script_sha256=sha256(__file__), common_sha256=sha256(Path(__file__).parent/"projection_common.py")))
    text = ["# Projected-latent follow-up results", "",
        f"Completed {len(paired_table)} of {len(specs)} specified paired contrasts. "
        f"The {len(missing)} pending contrasts are listed separately.", "",
        "Positive differences mean higher error for B. Interaction differences compare paired error penalties. "
        "Bootstrap intervals are unadjusted 95% image intervals; family and all-contrast Bonferroni p-values "
        "are separate columns. Family denominators include pending tests.", "",
        "No result here establishes equivalence or an intrinsic causal effect of channel count. "
        "The same-grid comparisons use separately trained MedVAE checkpoints. "
        "Channel-mean and retained-channel results have different storage and input-layer sizes.", "",
        ("All results in this fresh paired study come from its own completed fits; historical raw and "
         "projected reference scores are excluded. Raw and projected features were extracted together."
         if m.get("study_kind") == "fresh_paired" else
         "The preparation manifest records which historical projected results were reused. "
         "Incompatible historical projected scores are excluded until replacement fits complete. "
         "Fixed-cache validation is not a downstream equivalence test."), "",
        "relative_task_changes.csv supplies descriptive intervals for differences between the "
        "percentage changes in Cobb and landmark errors, using shared image resamples. These "
        "intervals are unadjusted and separate from the 38 listed paired tests.", "",
        "| Family | Planned tests | Completed |", "|---|---:|---:|"]
    for family, n in sorted(families.items()):
        text.append(f"| {family} | {n} | {sum(paired_table.family == family) if len(paired_table) else 0} |")
    (out/"RESULTS.md").write_text("\n".join(text)+"\n")
    print("\n".join(text)); print(f"Outputs: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, required=True)
    p.add_argument("--resamples", type=int, default=10000)
    args = p.parse_args(); require(args.resamples >= 10000, "--resamples must be at least 10,000")
    analyse(args)


if __name__ == "__main__": main()
