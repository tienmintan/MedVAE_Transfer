"""Prepare the 4_3 two-case exclusion sensitivity without re-extracting features."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from common import *


def prepare(args):
    study, dst = args.study_dir.resolve(), args.out_dir.resolve()
    require(study != dst and study not in dst.parents, "Put the follow-up outside the original study directory")
    m = old_manifest(study)
    require(set(EXCLUDE).issubset(m["expected_ids"]["train"]), "The two confirmed cases are not both in training")
    require(not set(EXCLUDE) & set(m["expected_ids"]["val"] + m["expected_ids"]["test"]), "An exclusion is outside training")
    trainer_path()
    plan = set_plan(dst, dict(kind="annotation", config=CONFIG, variants=list(VARIANTS), seeds=SEEDS,
        tasks=list(TASKS), training=TRAINING, excluded_training_ids=EXCLUDE, control_mode=args.control_mode,
        original_study=str(study), original_manifest_sha256=sha256(study / "manifest.json"),
        expected_new_fits=24 if args.control_mode == "existing" else 48,
        primary_family=dict(n=18, tests="8 clean-minus-unclean cell differences; 4 projection-after-averaging effects (two tasks x two cohorts); 4 projection-by-averaging interactions; 2 changes in the interaction after exclusion"),
        secondary_family=dict(n=9, tests="The nine landmark comparisons repeated with isotropic pixel error / ground-truth vertical spine extent"),
        multiplicity="Bonferroni within 18 primary and 9 secondary tests; also export 27-test sensitivity",
        inference="Paired images, two-sided t-tests; 10000 paired image bootstrap samples; unadjusted 95% intervals. Fixed three-seed ensembles. Exploratory reused test set.",
        stopping="Runtime/integrity only; no choice of seeds or exclusions based on test results"))
    manifest = dict(kind="annotation", plan_fingerprint=plan["fingerprint"], status="preparing", cells={}, imported={},
        expected_ids={s: [u for u in ids if s != "train" or u not in EXCLUDE] for s, ids in m["expected_ids"].items()},
        original_study=str(study), original_manifest_sha256=sha256(study / "manifest.json"))
    labels = {}
    for variant in VARIANTS:
        clean, unclean = dict(paths={}, sha256={}), dict(paths={}, sha256={})
        for split in ("train", "val", "test"):
            p = old_cache(study, m, variant, split)
            a = cache(p, m["expected_ids"][split], channels=3 if variant.endswith("keep") else 1)
            require(a["latents"].shape[2:] == (96, 96), "Expected 4_3 96x96 features")
            if split in labels:
                same_labels(a, labels[split])
            else:
                labels[split] = {k: a[k].copy() for k in ("uuids", "cobb", "landmarks")}
            unclean["paths"][split] = str(p); unclean["sha256"][split] = sha256(p)
            if split == "train":
                keep = ~np.isin(a["uuids"].astype(str), EXCLUDE)
                require(keep.sum() == 407, "Expected exactly 407 retained training images")
                a = subset(a, keep)
                a["annotation_exclusion_ids"] = np.asarray(EXCLUDE)
                q = dst / "caches" / "clean" / variant / f"latents_{CONFIG}_{split}.npz"
                atomic_npz(q, **a)
            else:
                q = p
            clean["paths"][split] = str(q); clean["sha256"][split] = sha256(q)
        manifest["cells"][f"clean/{variant}"] = clean
        manifest["cells"][f"unclean/{variant}"] = unclean
        if args.control_mode == "existing":
            run = study / "runs" / variant / CONFIG
            protocol, complete = read(run / "protocol.json"), read(run / "complete.json")
            require(complete["sha256"]["protocol.json"] == sha256(run / "protocol.json"), "Original protocol file hash differs")
            require(complete["fingerprint"] == protocol["fingerprint"], "Original completion fingerprint differs")
            require(protocol["trainer_sha256"] == TRAINER_SHA and protocol["training"] == TRAINING, "Historical trainer/training differs")
            require(protocol["input_sha256"] == unclean["sha256"], "Historical cache hashes differ")
            require(set(SEEDS).issubset(protocol["seeds"]), "Historical predictions lack required seeds")
            env = protocol["environment"]
            if "control_environment" in manifest:
                require(manifest["control_environment"] == env, "Historical cells used different environments")
            manifest["control_environment"] = env
            target = dst / "imported_controls" / variant
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / "protocol.json", target / "protocol.json")
            for task in TASKS:
                true = labels["test"]["cobb" if task == "cobb" else "landmarks"]
                for seed in SEEDS:
                    name = f"{task}_seed{seed}.npz"; p = run / name
                    require(sha256(p) == complete["sha256"][name], f"Historical prediction hash differs: {p}")
                    load_prediction(p, m["expected_ids"]["test"], true)
                    require(str(arrays(p)["fingerprint"].item()) == protocol["fingerprint"], "Historical prediction protocol differs")
                    q = target / name; shutil.copy2(p, q)
                    manifest["imported"][f"unclean/{variant}/{task}/{seed}"] = {"path": str(q.relative_to(dst)), "sha256": sha256(q)}
    # Verify that averaged caches really are means of the retained-channel arrays.
    for prefix in ("raw", "projected"):
        for split in ("train", "val", "test"):
            full = arrays(old_cache(study, m, prefix + "_keep", split))["latents"]
            avg = arrays(old_cache(study, m, prefix + "_mean", split))["latents"]
            require(np.array_equal(full.mean(1, keepdims=True, dtype=np.float32), avg), f"{prefix}/{split}: mean cache is not the exact channel mean")
    atomic_npz(dst / "evaluation.npz", **labels["test"])
    manifest["evaluation_sha256"] = sha256(dst / "evaluation.npz")
    dims = dimensions_from_images(args.img_dir, m["expected_ids"]["test"])
    write_csv(dst / "image_dimensions.csv", dims)
    manifest["dimensions_sha256"] = sha256(dst / "image_dimensions.csv")
    before = labels["train"]["cobb"]
    after = before[~np.isin(labels["train"]["uuids"].astype(str), EXCLUDE)]
    atomic_json(dst / "annotation_audit.json", dict(excluded_ids=EXCLUDE, train_before=409, train_after=407, val=72, test=128,
        cobb_mean_before=before.mean(0).tolist(), cobb_sd_before=(before.std(0)+1e-8).tolist(),
        cobb_mean_after=after.mean(0).tolist(), cobb_sd_after=(after.std(0)+1e-8).tolist(),
        note="Training targets will be standardised separately by the unchanged trainer. Landmark targets remain unstandardised."))
    manifest["status"] = "ready"
    atomic_json(dst / "manifest.json", manifest)
    print(f"Ready: {dst}. {plan['expected_new_fits']} new fits; seeds {SEEDS}; train 407 / val 72 / test 128.")
    print("Next run run_heads.py --pilot. Historical-control reuse additionally requires a matching runtime.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--control-mode", choices=("existing", "refit"), default="existing")
    prepare(p.parse_args())
