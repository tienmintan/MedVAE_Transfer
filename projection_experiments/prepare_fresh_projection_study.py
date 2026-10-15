"""Extract a self-contained, paired raw/projected study from source images.

Historical caches supply IDs and labels only. Their latent values are used for
descriptive spot comparisons, never as training features or an acceptance gate.
Every raw/projected pair comes from one deterministic compute_latent_proj call.
All scores must be fitted again in this study; no historical scores are imported.

On --spot-checks images per split, the projection formula is compared with the
API output and encode().mode() with the API raw latent; a failed comparison
stops preparation and writes numerical_failure.json. Source-image bytes are
required to be identical across configurations.

Inputs:  --label-cache-dir (historical latents_{config}_{split}.npz, for IDs and
         labels), --split-csv and --img-dir ({uuid}.jpg).
Outputs: <out-dir>/manifest.json (status "ready" when complete),
         <out-dir>/caches/{raw_keep,projected_keep,raw_mean,projected_mean}/latents_{config}_{split}.npz
         (mean variants for multi-channel configurations only) and
         <out-dir>/source_images/{config}_{split}.json.

Usage:
  python prepare_fresh_projection_study.py --label-cache-dir <dir> --split-csv <csv> \
      --img-dir <dir> --out-dir <new empty dir> [--precision ieee]
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np

from projection_common import (CONFIGS, COUNTS, MULTI, SHAPES, atomic_json,
    atomic_npz, cache_path, load_cache, require, sha256, split_ids)
from projection_validation import (NumericalMismatch, build_verified_model, comparison_report,
    enforce, load_source_image, precision_mode, runtime_info)


def extract_pair(model, x, device, check_api=False):
    import torch
    with torch.no_grad():
        x = x.to(device=device, dtype=torch.float32)
        z, _, projected = model.compute_latent_proj(x, sample_posterior=False)
        require(projected is not None, "Checkpoint has no projected representation")
        raw_array = z.float().cpu().numpy()
        projected_array = projected.float().cpu().numpy()
        checks = {}
        if check_api:
            formula = model.channel_proj(model.channel_ds(z) + z).float().cpu().numpy()
            checks["formula_vs_api_same_fresh_z"] = comparison_report(formula, projected_array)
            encoded = model.encode(x).mode().float().cpu().numpy()
            checks["encode_mode_vs_api_raw_same_input"] = comparison_report(encoded, raw_array)
    return raw_array, projected_array, checks


def _prepare(args, torch):
    out = args.out_dir.resolve()
    require(not out.exists() or not any(out.iterdir()), f"Study directory is not empty; a new empty directory is required: {out}")
    expected = split_ids(args.split_csv)
    require(1 <= args.spot_checks <= min(map(len, expected.values())), "Invalid source-image check count")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = dict(schema=3, study_kind="fresh_paired", status="preparing", expected_ids=expected,
        cells={v:{} for v in ("raw_keep", "projected_keep", "raw_mean", "projected_mean")},
        reference_reuse=dict(raw_keep=[], projected_keep=[]), checkpoints={}, validation=[], label_caches={},
        device=device, precision_mode=args.precision, runtime=runtime_info(torch, device),
        torch_version=torch.__version__, numpy_version=np.__version__,
        split_csv_sha256=sha256(args.split_csv), preparation_script_sha256=sha256(__file__),
        validation_script_sha256=sha256(Path(__file__).parent/"projection_validation.py"),
        input_rule="raw z and projected zbar returned by the same compute_latent_proj(sample_posterior=False) call",
        historical_comparison_scope="diagnostic only; historical features and fitted scores are excluded",
        recommended_training_phases=dict(primary_fits=80, full_core_fits=160, additional_pooling_fits=80))
    out.mkdir(parents=True, exist_ok=True); atomic_json(out/"manifest.json", m)
    labels_by_split, input_hashes = {}, {}
    try:
        for config in CONFIGS:
            model, checkpoint = build_verified_model(config, device)
            m["checkpoints"][config] = checkpoint; m["label_caches"][config] = {}
            for variant in m["cells"]:
                m["cells"][variant][config] = dict(paths={}, sha256={})
            for split in COUNTS:
                file = cache_path(args.label_cache_dir, config, split).resolve()
                original = load_cache(file, config, expected[split])
                lookup = {u:i for i,u in enumerate(original["uuids"])}
                order = np.array([lookup[u] for u in expected[split]])
                old_latents = original["latents"][order]
                labels = {k:original[k][order].copy() for k in ("uuids", "cobb", "landmarks")}
                if split in labels_by_split:
                    require(np.array_equal(labels_by_split[split]["uuids"], labels["uuids"]), "Cross-configuration ID order differs")
                    for key in ("cobb", "landmarks"):
                        require(np.array_equal(labels_by_split[split][key], labels[key]),
                                f"Cross-configuration {key} labels differ: {config}/{split}")
                else: labels_by_split[split] = labels
                source_hash = sha256(file)
                m["label_caches"][config][split] = dict(path=str(file), sha256=source_hash)
                sorted_indices = np.argsort(labels["uuids"])
                selected = set(sorted_indices[np.linspace(0,len(sorted_indices)-1,args.spot_checks,dtype=int)])
                raw_values, projected_values, image_records, checks = [], [], [], []
                for idx, uuid in enumerate(labels["uuids"]):
                    m["active_check"] = dict(config=config, split=split, image_uuid=str(uuid))
                    x, image_info = load_source_image(model, args.img_dir/f"{uuid}.jpg")
                    # Enforce identical source file bytes across separately loaded checkpoints.
                    if str(uuid) in input_hashes:
                        require(input_hashes[str(uuid)] == image_info["file_sha256"], "Source image changed during extraction")
                    input_hashes[str(uuid)] = image_info["file_sha256"]
                    raw, projected, reports = extract_pair(model, x, device, check_api=idx in selected)
                    shape = (1, *SHAPES[config])
                    require(raw.shape == projected.shape == shape, f"Unexpected paired shapes: {config}/{split}/{uuid}")
                    require(raw.dtype == projected.dtype == np.float32 and np.isfinite(raw).all()
                            and np.isfinite(projected).all(), f"Invalid paired features: {config}/{split}/{uuid}")
                    if idx in selected:
                        record = dict(image_uuid=str(uuid), **reports,
                            fresh_raw_vs_historical_raw=comparison_report(raw, old_latents[idx:idx+1]),
                            historical_comparison_is_acceptance_gate=False)
                        m["active_check"].update(record)
                        atomic_json(out/"manifest.json", m)
                        for key, report in reports.items():
                            enforce(report, f"{config}/{split}/{uuid} {key}")
                        require(set(reports) == {"formula_vs_api_same_fresh_z", "encode_mode_vs_api_raw_same_input"},
                                "Required API validation missing")
                        checks.append(record)
                    raw_values.append(raw[0]); projected_values.append(projected[0])
                    image_records.append(dict(image_uuid=str(uuid), **image_info))
                    if (idx+1) % 50 == 0 or idx+1 == len(labels["uuids"]):
                        print(f"{config}/{split}: encoded {idx+1}/{len(labels['uuids'])} paired images", flush=True)
                values = dict(raw_keep=np.stack(raw_values), projected_keep=np.stack(projected_values))
                if config in MULTI:
                    for stage in ("raw", "projected"):
                        values[stage+"_mean"] = values[stage+"_keep"].mean(axis=1,keepdims=True,dtype=np.float32)
                for variant, array in values.items():
                    require(float(array.std()) > 0, f"Constant feature cache: {variant}/{config}/{split}")
                    dst = cache_path(out/"caches"/variant, config, split)
                    atomic_npz(dst, latents=array, **labels, representation=np.array(variant),
                        encoder_checkpoint_sha256=np.array(checkpoint["sha256"]),
                        projection_checkpoint_sha256=np.array(checkpoint["sha256"]),
                        source_label_cache_sha256=np.array(source_hash), precision_mode=np.array(args.precision),
                        extraction_kind=np.array("fresh_paired"))
                    entry = m["cells"][variant][config]
                    entry["paths"][split] = str(dst.resolve()); entry["sha256"][split] = sha256(dst)
                    entry.update(channels=int(array.shape[1]), fp32_kib=float(array[0].nbytes/1024),
                                 compression_ratio=float(384**2/array[0].size))
                if config not in MULTI:
                    for stage in ("raw", "projected"):
                        m["cells"][stage+"_mean"][config] = dict(alias=f"{stage}_keep/{config}")
                source_info = out/"source_images"/f"{config}_{split}.json"
                atomic_json(source_info, image_records)
                m["validation"].append(dict(config=config, split=split, n=len(labels["uuids"]),
                    source_images_file=str(source_info.resolve()), source_images_sha256=sha256(source_info),
                    spot_checks=checks))
                atomic_json(out/"manifest.json", m)
                del original, old_latents, raw_values, projected_values, values
            del model; gc.collect()
            if device.startswith("cuda"): torch.cuda.empty_cache()
        m.pop("active_check", None); m["status"] = "ready"
        atomic_json(out/"manifest.json", m)
        print(f"Ready: {out/'manifest.json'}", flush=True)
        print("Fresh paired study: no historical raw or projected results will be imported.", flush=True)
    except Exception as exc:
        m["status"] = "failed"
        failure = dict(error_type=type(exc).__name__, message=str(exc), active_check=m.get("active_check"))
        if isinstance(exc, NumericalMismatch): failure.update(context=exc.context, comparison=exc.report)
        atomic_json(out/"numerical_failure.json", failure); atomic_json(out/"manifest.json", m)
        raise


def prepare(args):
    import torch
    sys.path.insert(0, str(Path.cwd()))
    with precision_mode(torch, args.precision): return _prepare(args, torch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-cache-dir", type=Path, required=True, help="Historical raw-cache directory; labels/IDs only, not reused training features")
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device")
    p.add_argument("--precision", choices=("ieee", "current", "tf32"), default="ieee")
    p.add_argument("--spot-checks", type=int, default=8)
    prepare(p.parse_args())


if __name__ == "__main__": main()
