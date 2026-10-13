"""Train the planned cells using the audited frontier model and optimiser.

Fresh study: primary pair 80 fits; full retained-channel core 160 fits total;
channel pooling adds 80 fits. Historical-cache studies retain their old plan.
Completed seeds are reused only when data, code and protocol fingerprints match.
"""
import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from projection_common import (CONFIGS, MULTI, SINGLE, TASKS, TRAINING, atomic_json, atomic_npz,
    checked_manifest, fingerprint, import_source, mean_interval,
    read_json, require, sha256, validate_cell, projected_reference_configs, verify_protocol_source)
from projection_validation import precision_mode, precision_snapshot


def planned_cells(phase, manifest=None):
    if manifest is not None and manifest.get("study_kind") == "fresh_paired":
        projected_reference_configs(manifest)
        primary = [(v,c) for v in ("raw_keep", "projected_keep") for c in (CONFIGS[0], CONFIGS[3])]
        rest = [(v,c) for v in ("raw_keep", "projected_keep") for c in (CONFIGS[1], CONFIGS[2])]
        core = primary + rest
        pooling = [(v,c) for v in ("raw_mean", "projected_mean") for c in MULTI]
        return primary if phase == "primary" else core if phase == "core" else pooling if phase == "pooling" else core+pooling
    require(phase != "primary", "The primary phase is for a fresh paired study")
    reused = projected_reference_configs(manifest) if manifest is not None else list(MULTI)
    core = [("projected_keep", c) for c in (*SINGLE, *MULTI) if c not in reused]
    pooling = [(v, c) for v in ("raw_mean", "projected_mean") for c in MULTI]
    return core if phase == "core" else pooling if phase == "pooling" else core + pooling


def check_protocol(path):
    return verify_protocol_source(path, "training")


def _run(args):
    require(args.seeds == list(range(10)), "This matched follow-up uses seeds 0 through 9")
    study = args.study_dir.resolve(); m = checked_manifest(study)
    check_protocol(args.trainer)
    cells = planned_cells(args.phase, m)
    # Validate before importing Torch, so --dry-run is also an integrity check.
    for variant, config in cells:
        sp = validate_cell(m, variant, config)
        print(f"{variant}/{config}: C={sp['train']['latents'].shape[1]}, "
              f"{m['cells'][variant][config]['fp32_kib']:g} KiB/image", flush=True)
        del sp
    print(f"Requested {len(cells) * len(TASKS) * len(args.seeds)} fits; completed matching seeds will be reused.")
    if args.dry_run: return
    import torch
    trainer = import_source(args.trainer, "medvae_projection_frontier")
    require(trainer.N_LANDMARKS == 68 and trainer.TASK_OUT_DIM["cobb"] == 3
            and trainer.TASK_OUT_DIM["landmark"] == 136, "Task dimensions differ from the reference")
    for task, src in (("cobb", "cobb"), ("landmark", "landmarks")):
        require(trainer.TASK_SPEC[task]["kind"] == "reg" and trainer.TASK_SPEC[task]["src"] == src,
                "Target/loss specification differs from the reference")
    trainer.PROGRESS = args.progress
    trainer.PRELOAD_LIMIT_GB = 8.0
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    environment = dict(torch=torch.__version__, numpy=np.__version__, device=device,
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(device) if device.startswith("cuda") else None)
    if m.get("study_kind") == "fresh_paired":
        environment["precision"] = precision_snapshot(torch)
        environment["required_precision_mode"] = m["precision_mode"]
        require(torch.__version__ == m["torch_version"] and np.__version__ == m["numpy_version"],
                "Fresh-study extraction and training package versions differ")
        anchor = study/"training_environment.json"
        if anchor.exists():
            require(read_json(anchor) == environment, "Training runtime differs from earlier cells in this fresh study")
        else: atomic_json(anchor, environment)
    for variant, config in cells:
        sp = validate_cell(m, variant, config)
        entry = m["cells"][variant][config]; test = sp["test"]
        dst = study / "runs" / variant / config; dst.mkdir(parents=True, exist_ok=True)
        protocol = dict(variant=variant, config=config, seeds=args.seeds, training=TRAINING,
            input_sha256=entry["sha256"], trainer_sha256=sha256(args.trainer),
            runner_sha256=sha256(__file__), common_sha256=sha256(Path(__file__).parent / "projection_common.py"),
            manifest_sha256=sha256(study / "manifest.json"), environment=environment,
            prediction_rule="average predictions across seeds, then compute per-image error",
            save_models=not args.no_save_models)
        token = fingerprint(protocol)
        if (dst / "protocol.json").exists():
            require(read_json(dst / "protocol.json")["fingerprint"] == token,
                    f"Incompatible existing run: {dst}. Use a separate study for a changed protocol.")
        atomic_json(dst / "protocol.json", dict(fingerprint=token, **protocol))
        all_predictions, summary, rows, seed_rows = {}, [], [], []
        start = time.time()
        for task in TASKS:
            true = test["cobb" if task == "cobb" else "landmarks"]
            seed_predictions, seed_metrics = [], []
            for seed in args.seeds:
                pred_file = dst / f"{task}_seed{seed}.npz"
                model_file = dst / f"{task}_seed{seed}.pt"
                if pred_file.exists():
                    with np.load(pred_file, allow_pickle=False) as data:
                        require(str(data["fingerprint"].item()) == token, "Seed protocol mismatch")
                        require(np.array_equal(data["uuids"], test["uuids"]), "Seed ID order mismatch")
                        require(np.array_equal(data["true"], true), "Seed labels changed")
                        pt = data["pred"].copy()
                    require(args.no_save_models or model_file.is_file(), "Seed checkpoint missing")
                    print(f"Reuse {variant}/{config}/{task}/seed{seed}", flush=True)
                else:
                    model, (mu, sd) = trainer.train_one(task, sp["train"], sp["val"],
                        entry["channels"], seed=seed, device=device, **TRAINING)
                    pt = trainer.predict(model, mu, sd, test["latents"], device)
                    require(pt.shape == true.shape and np.isfinite(pt).all(), "Invalid predictions")
                    if not args.no_save_models:
                        temp = model_file.with_suffix(".pt.tmp")
                        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            mu=mu, sd=sd, task=task, config=config, variant=variant, seed=seed,
                            in_ch=entry["channels"], out_dim=true.shape[1], fingerprint=token), temp)
                        os.replace(temp, model_file)
                    atomic_npz(pred_file, pred=pt, true=true, uuids=test["uuids"], fingerprint=np.array(token))
                    del model; gc.collect()
                    if device.startswith("cuda"): torch.cuda.empty_cache()
                require(pt.shape == true.shape and np.isfinite(pt).all(), "Invalid saved predictions")
                err = trainer.per_image_error(task, pt, true)
                seed_predictions.append(pt); seed_metrics.append(float(err.mean()))
                seed_rows.append(dict(variant=variant, config=config, task=task, seed=seed,
                                      error_mean=float(err.mean()), n=len(err)))
                print(f"{variant}/{config}/{task}/seed{seed}: {err.mean():.6f}", flush=True)
            ensemble = np.mean(seed_predictions, axis=0)
            all_predictions[task] = ensemble
            errs = trainer.per_image_error(task, ensemble, true)
            mean, lo, hi = mean_interval(errs)
            summary.append(dict(variant=variant, config=config, task=task, error_mean=mean,
                error_lo=lo, error_hi=hi, seed_sd=float(np.std(seed_metrics, ddof=0)), n=len(errs),
                n_seeds=len(args.seeds), channels=entry["channels"], fp32_kib=entry["fp32_kib"],
                compression_ratio=entry["compression_ratio"]))
            rows.extend(dict(variant=variant, config=config, task=task, image_uuid=str(u), error=float(e))
                        for u, e in zip(test["uuids"], errs))
        pred_name = f"preds_{config}.npz"
        atomic_npz(dst / pred_name, uuids=test["uuids"], gt_cobb=test["cobb"],
            gt_landmarks=test["landmarks"], pred_cobb=all_predictions["cobb"],
            pred_landmarks=all_predictions["landmark"], fingerprint=np.array(token))
        pd.DataFrame(summary).to_csv(dst / "summary.csv", index=False)
        pd.DataFrame(rows).to_csv(dst / "per_image.csv", index=False)
        pd.DataFrame(seed_rows).to_csv(dst / "per_seed.csv", index=False)
        files = [pred_name, "summary.csv", "per_image.csv", "per_seed.csv", "protocol.json"]
        files += [f"{task}_seed{seed}.npz" for task in TASKS for seed in args.seeds]
        if not args.no_save_models:
            files += [f"{task}_seed{seed}.pt" for task in TASKS for seed in args.seeds]
        atomic_json(dst / "complete.json", dict(fingerprint=token, elapsed_seconds=time.time()-start,
                                                sha256={f: sha256(dst / f) for f in files}))
        print(f"Complete: {dst}", flush=True)
        del sp


def run(args):
    m = checked_manifest(args.study_dir)
    if m.get("study_kind") == "fresh_paired" and not args.dry_run:
        import torch
        with precision_mode(torch, m["precision_mode"]):
            return _run(args)
    return _run(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, required=True)
    p.add_argument("--trainer", type=Path,
                   default=Path(__file__).resolve().parent / "references" / "train_frontier.py",
                   help="Trainer source; default: the bundled audited reference")
    p.add_argument("--phase", choices=("primary", "core", "pooling", "all"), default="core")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--device")
    p.add_argument("--progress", type=int, default=25)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-save-models", action="store_true", help="Keep predictions but omit checkpoints")
    run(p.parse_args())


if __name__ == "__main__": main()
