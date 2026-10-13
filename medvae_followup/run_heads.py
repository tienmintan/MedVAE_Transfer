"""Run the exact audited trainer, with a reusable pilot and a wall-time budget."""
import argparse
import subprocess
import time
from pathlib import Path
import sys

import numpy as np

from common import *


def jobs(plan):
    if plan["kind"] == "annotation":
        cohorts = ["clean"] + (["unclean"] if plan["control_mode"] == "refit" else [])
        return [(c, v, t, s) for s in SEEDS for c in cohorts for v in VARIANTS for t in TASKS]
    require(plan["kind"] == "augmentation", "Unknown head study kind")
    return [("augmented", "projected_mean", t, s) for s in SEEDS for t in TASKS]


def paths(dst, job):
    c, v, t, s = job
    base = Path(dst) / "runs" / c / v
    return base / f"{t}_seed{s}.npz", base / f"{t}_seed{s}.json"


def done(dst, plan, job):
    pred, meta = paths(dst, job)
    if not (pred.exists() and meta.exists()):
        return False
    a = read(meta)
    require(a["plan_fingerprint"] == plan["fingerprint"] and a["prediction_sha256"] == sha256(pred), f"Completed fit changed: {pred}")
    return True


def verify_inputs(dst, manifest):
    require(manifest["status"] == "ready", "Study preparation is incomplete")
    for cell in manifest["cells"].values():
        for split, p in cell["paths"].items():
            require(Path(p).is_file() and sha256(p) == cell["sha256"][split], f"Prepared input missing/changed: {p}")
    for rec in manifest.get("imported", {}).values():
        require(sha256(Path(dst) / rec["path"]) == rec["sha256"], "Imported control changed")


def current_environment(dst, plan, manifest, device):
    import torch
    from projection_validation import precision_mode
    require(not device.startswith("cuda") or torch.cuda.is_available(), "CUDA unavailable; this study will not silently fall back to CPU")
    with precision_mode(torch, "ieee"):
        env = environment(torch, device)
    if plan["kind"] == "annotation" and plan["control_mode"] == "existing":
        check_environment(env, manifest["control_environment"], "Historical controls cannot be reused in this runtime")
    if plan["kind"] == "augmentation":
        check_environment(env, manifest["control_environment"], "Augmentation and its unaugmented controls need the same training runtime")
    p = Path(dst) / "training_environment.json"
    if p.exists():
        check_environment(env, read(p), "Runtime changed within this follow-up")
    else:
        atomic_json(p, env)
    return env


def worker(args):
    dst = args.out_dir.resolve(); plan = get_plan(dst); manifest = read(dst / "manifest.json")
    job = (args.cohort, args.variant, args.task, args.seed)
    require(job in jobs(plan), "Fit not declared in the study plan")
    require(manifest["plan_fingerprint"] == plan["fingerprint"], "Manifest/plan differ")
    env = current_environment(dst, plan, manifest, args.device)
    import torch
    from projection_validation import precision_mode
    trainer = import_source(trainer_path(), "medvae_followup_audited_trainer")
    trainer.PROGRESS = 25; trainer.PRELOAD_LIMIT_GB = 8.0
    cell = manifest["cells"][args.cohort + "/" + args.variant]
    data = {}
    for split, path in cell["paths"].items():
        require(sha256(path) == cell["sha256"][split], f"Changed input: {path}")
        data[split] = cache(path)
    require(not (set(data["train"]["uuids"]) & set(data["test"]["uuids"])), "Training/test overlap")
    start = time.monotonic()
    with precision_mode(torch, "ieee"):
        model, (mu, sd) = trainer.train_one(args.task, data["train"], data["val"], data["train"]["latents"].shape[1],
            seed=args.seed, device=args.device, **TRAINING)
        pred = trainer.predict(model, mu, sd, data["test"]["latents"], args.device)
    true = data["test"]["cobb" if args.task == "cobb" else "landmarks"]
    require(pred.shape == true.shape and np.isfinite(pred).all(), "Invalid task predictions")
    target, meta = paths(dst, job)
    atomic_npz(target, pred=pred.astype(np.float32), true=true, uuids=data["test"]["uuids"], fingerprint=np.asarray(plan["fingerprint"]))
    atomic_json(meta, dict(plan_fingerprint=plan["fingerprint"], cohort=args.cohort, variant=args.variant, task=args.task,
        seed=args.seed, prediction_sha256=sha256(target), elapsed_seconds=time.monotonic()-start,
        train_n=len(data["train"]["uuids"]), val_n=len(data["val"]["uuids"]), test_n=len(true),
        parameter_count=sum(p.numel() for p in model.parameters()), target_mean=mu.tolist(), target_sd=sd.tolist(),
        input_sha256=cell["sha256"], trainer_sha256=TRAINER_SHA, environment=env))
    print(f"Saved {target}", flush=True)


def run(args):
    dst = args.out_dir.resolve(); plan = get_plan(dst); m = read(dst / "manifest.json")
    require(m["plan_fingerprint"] == plan["fingerprint"], "Manifest/plan differ")
    trainer_path(); verify_inputs(dst, m)
    all_jobs = jobs(plan)
    selected = all_jobs
    pilot_jobs = [("clean", "raw_keep", t, 0) for t in TASKS] if plan["kind"] == "annotation" else [("augmented", "projected_mean", t, 0) for t in TASKS]
    if args.pilot:
        selected = pilot_jobs
    pending = [j for j in selected if not done(dst, plan, j)]
    print(f"Plan: {len(all_jobs)} new fits. Requested now: {len(selected)}; remaining in this command: {len(pending)}.")
    for j in pending:
        print("  " + "/".join(map(str, j)))
    if args.dry_run:
        print("Input integrity check passed. CUDA/runtime compatibility is checked when --pilot or a full run starts.")
        return
    with StudyLock(dst):
        env = current_environment(dst, plan, m, args.device)
        ledger_path = dst / "attempts.json"
        ledger = read(ledger_path) if ledger_path.exists() else []
        spent = sum(x["seconds"] for x in ledger)
        require(args.max_hours > 0, "--max-hours must be positive")
        budget = args.max_hours * 3600
        if not args.pilot and pending:
            require(all(done(dst, plan, j) for j in pilot_jobs), "Run --pilot first so the time estimate comes from this machine")
            pilot_times = [read(paths(dst, j)[1])["elapsed_seconds"] for j in pilot_jobs]
            estimate = spent + 2 * max(pilot_times) * len(pending)
            atomic_json(dst / "budget_estimate.json", dict(total_new_fits=len(all_jobs), remaining_fits=len(pending),
                pilot_seconds=pilot_times, spent_seconds=spent, conservative_total_seconds=estimate, budget_seconds=budget,
                rule="twice the slower pilot fit per remaining fit; elapsed attempt time counts across resumes"))
            require(estimate <= budget, f"Conservative estimate {estimate/3600:.2f} h exceeds {args.max_hours:g} h. Stop and return the pilot; do not select a smaller favourable subset of test results.")
        for job in pending:
            remaining = budget - spent
            require(remaining > 0, "Study time budget exhausted; return completed diagnostics")
            pred, meta = paths(dst, job); pred.parent.mkdir(parents=True, exist_ok=True)
            c, v, t, s = job
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--out-dir", str(dst), "--device", args.device,
                   "--worker", "--cohort", c, "--variant", v, "--task", t, "--seed", str(s)]
            log = pred.with_suffix(".log"); start = time.monotonic(); status = "interrupted"
            print(f"Fitting {'/'.join(map(str,job))}; progress log: {log}", flush=True)
            try:
                with log.open("w") as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=remaining)
                status = "ok" if result.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
            finally:
                seconds = time.monotonic() - start
                ledger.append(dict(job=list(job), seconds=seconds, status=status)); spent += seconds
                atomic_json(ledger_path, ledger)
            require(status == "ok", f"Fit {status}; see {log}. No partial prediction is accepted. Fix the cause and repeat the same command, or return diagnostics.")
            require(done(dst, plan, job), "Worker exited without a valid completed fit")
        completed = sum(done(dst, plan, j) for j in all_jobs)
        if all(done(dst, plan, j) for j in pilot_jobs):
            times = [read(paths(dst, j)[1])["elapsed_seconds"] for j in pilot_jobs]
            estimate = spent + 2 * max(times) * (len(all_jobs)-completed)
            atomic_json(dst / "budget_estimate.json", dict(total_new_fits=len(all_jobs), completed_fits=completed,
                remaining_fits=len(all_jobs)-completed, pilot_seconds=times, spent_seconds=spent,
                conservative_total_seconds=estimate, budget_seconds=budget, within_budget=estimate <= budget))
            print(f"Conservative total estimate: {estimate/3600:.2f} hours; budget: {args.max_hours:g} hours.")
        print(f"Completed {completed}/{len(all_jobs)} planned fits. Pilot fits are reused.")
        if completed == len(all_jobs):
            atomic_json(dst / "heads_complete.json", dict(plan_fingerprint=plan["fingerprint"], fits=completed,
                fit_metadata_sha256={str(paths(dst, j)[1].relative_to(dst)): sha256(paths(dst, j)[1]) for j in all_jobs}))
            print(f"Next: python medvae_followup/analyse_heads.py --out-dir {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--pilot", action="store_true")
    p.add_argument("--max-hours", type=float, default=36)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for flag in ("cohort", "variant", "task"):
        p.add_argument("--" + flag, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    worker(a) if a.worker else run(a)
