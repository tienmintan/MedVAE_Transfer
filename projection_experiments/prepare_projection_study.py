"""Prepare a matched raw/projected x retained/mean-channel study from existing caches.

Run from the MedVAE repository root. Existing caches are read only. The new
directory must be empty; an interrupted preparation can be restarted in a new
directory. Fixed-cache mode preserves historical encoder features while checking
their consistency with fresh encodings separately from projection correctness.

For each configuration, --spot-checks images per split are re-encoded; the
projection formula on the API's own raw latent and encode().mode() are checked
against the API output, and the historical raw cache is checked under
--raw-cache-policy. For multi-channel configurations, the existing projected
cache is reused (--projected-reuse auto) only if every split passes the
original comparison; otherwise the projected caches are recomputed and both
projected tasks are refitted. A failed check writes numerical_failure.json.

Inputs:  --raw-dir and --existing-projected-dir (latents_{config}_{split}.npz),
         --split-csv, --img-dir ({uuid}.jpg) and
         references/{raw,projected}_frontier_per_image.csv next to this script.
Outputs: <out-dir>/manifest.json (status "ready" when complete) and
         <out-dir>/caches/{projected_keep,raw_mean,projected_mean}/latents_{config}_{split}.npz;
         the raw_keep variant points at the existing raw caches.

Usage:
  python prepare_projection_study.py --raw-dir <dir> --existing-projected-dir <dir> \
      --split-csv <csv> --img-dir <dir> --out-dir <new empty dir> \
      [--raw-cache-policy strict|fixed] [--projected-reuse auto|never]
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np

from projection_common import (CONFIGS, COUNTS, MULTI, SHAPES, align_to, atomic_json,
    atomic_npz, cache_path, load_cache, require, sha256, split_ids)

from projection_validation import (NumericalMismatch, build_verified_model, collect_spot,
    comparison_report, enforce, load_source_image, precision_mode, runtime_info,
    historical_cache_decision, projected_reuse_decision)


def _prepare_impl(args):
    import torch
    sys.path.insert(0, str(Path.cwd()))

    out = args.out_dir.resolve()
    require(not out.exists() or not any(out.iterdir()),
            f"Study directory is not empty, so nothing was overwritten; a new empty directory is required: {out}")
    expected = split_ids(args.split_csv)
    references = Path(__file__).parent / "references"
    import pandas as pd
    for name in ("raw_frontier_per_image.csv", "projected_frontier_per_image.csv"):
        ref = pd.read_csv(references / name, dtype={"image_uuid": str})
        require(set(ref.image_uuid) == set(expected["test"]),
                f"Test IDs differ from the reference file: {name}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw_policy = getattr(args, "raw_cache_policy", "strict")
    reuse_policy = getattr(args, "projected_reuse", "auto")
    manifest = dict(schema=2, status="preparing", expected_ids=expected, cells={},
                    validation=[], checkpoints={}, device=device,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    split_csv_sha256=sha256(args.split_csv),
                    preparation_script_sha256=sha256(__file__),
                    tolerances=dict(atol=args.atol, rtol=args.rtol),
                    precision_mode=args.precision, runtime=runtime_info(torch, device),
                    raw_cache_policy=raw_policy, projected_reuse_policy=reuse_policy,
                    reference_reuse=dict(raw_keep=list(CONFIGS), projected_keep=[]),
                    projected_reuse_decisions={},
                    scope="Projection applied to fixed historical raw caches; fresh-encoder identity is a separate recorded check",
                    validation_script_sha256=sha256(Path(__file__).parent / "projection_validation.py"))
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "manifest.json", manifest)
    try:
        label_reference = {}
        for config in CONFIGS:
            model, checkpoint_info = build_verified_model(config, device)
            ck_hash = checkpoint_info["sha256"]
            manifest["checkpoints"][config] = checkpoint_info
            pending = {}
            for variant in ("raw_keep", "projected_keep", "raw_mean", "projected_mean"):
                manifest["cells"].setdefault(variant, {})[config] = {"paths": {}, "sha256": {}}
            for split in COUNTS:
                raw_file = cache_path(args.raw_dir, config, split).resolve()
                raw = load_cache(raw_file, config, expected[split])
                if "representation" in raw:
                    require(str(np.asarray(raw["representation"]).item()) in
                            ("encoder_output", "raw", "unprojected", "raw_keep"),
                            f"Input cache is not marked as raw: {raw_file}")
                if split in label_reference:
                    align_to(label_reference[split], raw)
                    if split != "test":
                        require(np.array_equal(label_reference[split]["uuids"], raw["uuids"]),
                                f"Cross-config {split} row order differs; existing runs may not be comparable")
                else: label_reference[split] = {k: raw[k].copy() for k in ("uuids", "cobb", "landmarks")}
                raw_hash = sha256(raw_file)
                record = dict(config=config, split=split, n=len(raw["uuids"]), spot_checks=[])
                order = np.argsort(raw["uuids"])
                indices = order[np.linspace(0, len(order)-1, args.spot_checks, dtype=int)]
                with torch.no_grad():
                    for idx in indices:
                        image_path = args.img_dir / f"{raw['uuids'][idx]}.jpg"
                        x, image_info = load_source_image(model, image_path)
                        reports, _ = collect_spot(model, x, raw["latents"][idx:idx+1],
                                                  device, args.atol, args.rtol, check_encode=True)
                        decision = historical_cache_decision(reports["fresh_raw_vs_cached_raw"], raw_policy)
                        reports["historical_cache_consistency"] = decision
                        spot = dict(image_uuid=str(raw["uuids"][idx]), image=image_info, **reports)
                        record["spot_checks"].append(spot)
                        manifest["active_check"] = dict(config=config, split=split, **spot)
                        atomic_json(out / "manifest.json", manifest)
                        # Check the API on its own returned raw latent, separately
                        # from reproducibility against a historical cached latent.
                        enforce(reports["formula_vs_api_same_fresh_z"],
                                f"{config}/{split}/{raw['uuids'][idx]} projection formula on API z")
                        enforce(reports["encode_mode_vs_api_raw_same_input"],
                                f"{config}/{split}/{raw['uuids'][idx]} encode mode versus API raw")
                        enforce(decision, f"{config}/{split}/{raw['uuids'][idx]} historical cache consistency ({raw_policy})")
                        if not reports["fresh_raw_vs_cached_raw"]["passed"]:
                            print(f"WARNING: {config}/{split}/{raw['uuids'][idx]}: original elementwise check failed; "
                                  f"fixed-cache consistency passed (RMS/SD={decision['rms_over_cached_sd']:.3g}, "
                                  f"max/SD={decision['max_over_cached_sd']:.3g}). Historical raw values retained.", flush=True)
                    chunks = []
                    for start in range(0, len(raw["uuids"]), args.batch_size):
                        z = torch.as_tensor(raw["latents"][start:start+args.batch_size], device=device)
                        chunks.append(model.channel_proj(model.channel_ds(z) + z).float().cpu().numpy())
                    projected = np.concatenate(chunks)
                require(np.isfinite(projected).all(), f"Non-finite projected features: {config}/{split}")
                existing_file = cache_path(args.existing_projected_dir, config, split)
                existing = None
                if config in MULTI and existing_file.is_file():
                    existing = load_cache(existing_file, config, expected[split])
                    require("representation" in existing and
                            str(np.asarray(existing["representation"]).item()) == "projected",
                            f"Missing/wrong projected metadata: {existing_file}")
                    record["existing_projected_row_order_matches_raw"] = bool(
                        np.array_equal(existing["uuids"], raw["uuids"]))
                    existing = align_to(raw, existing)
                    manifest["active_check"] = dict(config=config, split=split,
                        operation="full existing projected cache", existing_file=str(existing_file.resolve()))
                    atomic_json(out / "manifest.json", manifest)
                    record["existing_projected_agreement"] = comparison_report(
                        projected, existing["latents"], args.atol, args.rtol)
                    record["existing_projected_sha256"] = sha256(existing_file)
                elif config in MULTI:
                    record["existing_projected_agreement"] = dict(passed=False, reason="File missing")
                pending[split] = dict(raw=raw, raw_file=raw_file, raw_hash=raw_hash,
                                      calculated_projected=projected, existing=existing, record=record)
                manifest["validation"].append(record)
                atomic_json(out / "manifest.json", manifest)
                print(f"Checked {config}/{split}: {len(raw['uuids'])} images", flush=True)
            reuse = False
            if config in MULTI:
                reuse_decision = projected_reuse_decision([d["record"] for d in pending.values()], reuse_policy)
                manifest["projected_reuse_decisions"][config] = reuse_decision
                reuse = reuse_decision["reuse"]
                if reuse: manifest["reference_reuse"]["projected_keep"].append(config)
                print(f"{config}: {'reusing existing projected results' if reuse else 'refitting both projected tasks (20 fits)'}; "
                      f"{reuse_decision['reason']}", flush=True)
            for split, data in pending.items():
                raw, raw_file, raw_hash = data["raw"], data["raw_file"], data["raw_hash"]
                # Select one source for all splits. If refitting, all three use P(cached z).
                # If reusing, preserve the exact aligned features used by the old heads.
                projected = data["existing"]["latents"] if reuse else data["calculated_projected"]
                sources = {"raw_keep": raw["latents"], "projected_keep": projected}
                if config in MULTI:
                    sources["raw_mean"] = raw["latents"].mean(axis=1, keepdims=True, dtype=np.float32)
                    sources["projected_mean"] = projected.mean(axis=1, keepdims=True, dtype=np.float32)
                for variant, values in sources.items():
                    if variant == "raw_keep": dst = raw_file
                    else:
                        dst = cache_path(out / "caches" / variant, config, split)
                        atomic_npz(dst, latents=values.astype(np.float32), uuids=raw["uuids"],
                            cobb=raw["cobb"], landmarks=raw["landmarks"],
                            representation=np.array(variant), source_raw_sha256=np.array(raw_hash),
                            projection_checkpoint_sha256=np.array(ck_hash))
                    entry = manifest["cells"][variant][config]
                    entry["paths"][split] = str(dst.resolve()); entry["sha256"][split] = sha256(dst)
                    entry["channels"] = int(values.shape[1])
                    entry["fp32_kib"] = float(values[0].nbytes / 1024)
                    entry["compression_ratio"] = float(384**2 / values[0].size)
                if config not in MULTI:
                    # A mean over one channel is an identity; it requires no new fit.
                    for prefix in ("raw", "projected"):
                        manifest["cells"][prefix+"_mean"][config] = dict(
                            alias=f"{prefix}_keep/{config}")
                atomic_json(out / "manifest.json", manifest)
            del pending
            del model; gc.collect()
            if device.startswith("cuda"): torch.cuda.empty_cache()
        manifest.pop("active_check", None)
        comparisons = [s["fresh_raw_vs_cached_raw"] for r in manifest["validation"] for s in r["spot_checks"]]
        manifest["historical_reencoding_summary"] = dict(
            checked_images=len(comparisons), elementwise_failures=sum(not r["passed"] for r in comparisons),
            all_elementwise_passed=all(r["passed"] for r in comparisons),
            note="A ready study means the selected validation policy passed; it does not turn failed elementwise comparisons into passes")
        manifest["status"] = "ready"
        atomic_json(out / "manifest.json", manifest)
        print(f"Ready: {out / 'manifest.json'}")
    except Exception as exc:
        manifest["status"] = "failed"
        failure = dict(error_type=type(exc).__name__, message=str(exc),
                       active_check=manifest.get("active_check"),
                       precision_mode=args.precision, runtime=manifest["runtime"])
        if isinstance(exc, NumericalMismatch):
            failure.update(context=exc.context, comparison=exc.report)
        atomic_json(out / "numerical_failure.json", failure)
        atomic_json(out / "manifest.json", manifest)
        raise


def prepare(args):
    import torch
    sys.path.insert(0, str(Path.cwd()))
    with precision_mode(torch, args.precision):
        _prepare_impl(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--existing-projected-dir", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device")
    p.add_argument("--precision", choices=("current", "ieee", "tf32"), default="current",
                   help="Explicit float32 math mode; default preserves current backend settings")
    p.add_argument("--raw-cache-policy", choices=("strict", "fixed"), default="strict",
                   help="strict: original elementwise re-encoding check; fixed: retain historical raw inputs with separate RMS and local consistency limits")
    p.add_argument("--projected-reuse", choices=("auto", "never"), default="auto",
                   help="auto: reuse only if every split passes the original comparison; never: regenerate all projected caches and refit all four configurations")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--spot-checks", type=int, default=2)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    args = p.parse_args()
    require(args.spot_checks > 0 and args.batch_size > 0, "Positive batch and check counts required")
    require(args.spot_checks <= min(COUNTS.values()), "Spot count exceeds the smallest split")
    require(np.isfinite([args.atol,args.rtol]).all() and args.atol >= 0 and args.rtol >= 0,
            "Invalid elementwise tolerances")
    prepare(args)


if __name__ == "__main__": main()
