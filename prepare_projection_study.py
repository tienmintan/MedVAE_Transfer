"""Prepare a matched raw/projected x retained/mean-channel study from existing caches.

Run from the MedVAE repository root. Existing caches are read only. The new
directory must be empty; an interrupted preparation can be restarted in a new
directory. All projection weights and representative encoder/API outputs are
checked before the manifest is marked ready for training.
"""
import argparse
import gc
import inspect
import sys
from pathlib import Path

import numpy as np

from projection_common import (CONFIGS, COUNTS, MULTI, SHAPES, align_to, atomic_json,
    atomic_npz, cache_path, load_cache, numeric_agreement, require, sha256, split_ids)


def prepare(args):
    import torch
    sys.path.insert(0, str(Path.cwd()))
    from medvae.utils.factory import (FILE_DICT_ASSOCIATIONS, build_model,
                                       download_model_weights)
    from medvae.utils.loaders import load_2d_finetune_aspect

    out = args.out_dir.resolve()
    require(not out.exists() or not any(out.iterdir()),
            f"Use a new empty study directory; refusing to overwrite {out}")
    expected = split_ids(args.split_csv)
    references = Path(__file__).parent / "references"
    import pandas as pd
    for name in ("raw_frontier_per_image.csv", "projected_frontier_per_image.csv"):
        ref = pd.read_csv(references / name, dtype={"image_uuid": str})
        require(set(ref.image_uuid) == set(expected["test"]),
                f"Test IDs differ from the report reference: {name}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = dict(schema=1, status="preparing", expected_ids=expected, cells={},
                    validation=[], checkpoints={}, device=device,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    split_csv_sha256=sha256(args.split_csv),
                    preparation_script_sha256=sha256(__file__),
                    tolerances=dict(atol=args.atol, rtol=args.rtol))
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "manifest.json", manifest)
    label_reference = {}
    for config in CONFIGS:
        paths = FILE_DICT_ASSOCIATIONS[config]
        config_file = download_model_weights(paths["config"])
        checkpoint_file = download_model_weights(paths["ckpt"])
        model = build_model(config, config_file, checkpoint_file, training=False).eval()
        kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False  # same trusted MedVAE checkpoint as its loader
        checkpoint = torch.load(checkpoint_file, **kwargs)["state_dict"]
        current = model.state_dict()
        keys = [k for k in current if k.startswith(("channel_ds.", "channel_proj."))]
        require(len(keys) == 8 and all(k in checkpoint for k in keys),
                f"Incomplete trained projection weights for {config}")
        require(all(torch.equal(current[k].cpu(), checkpoint[k].cpu()) for k in keys),
                f"Loaded projection weights differ from checkpoint: {config}")
        ck_hash = sha256(checkpoint_file)
        manifest["checkpoints"][config] = dict(path=str(checkpoint_file), sha256=ck_hash,
            config_sha256=sha256(config_file), projection_keys=keys,
            implementation_sha256=sha256(inspect.getfile(type(model))))
        del checkpoint, current
        model = model.requires_grad_(False).to(device)
        in_ch = int(model.encoder.conv_in.in_channels)
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
                            f"Cross-config {split} row order differs; verify existing-run comparability before reuse")
            else: label_reference[split] = {k: raw[k].copy() for k in ("uuids", "cobb", "landmarks")}
            raw_hash = sha256(raw_file)
            record = dict(config=config, split=split, n=len(raw["uuids"]), spot_checks=[])
            order = np.argsort(raw["uuids"])
            indices = order[np.linspace(0, len(order)-1, args.spot_checks, dtype=int)]
            with torch.no_grad():
                for idx in indices:
                    image_path = args.img_dir / f"{raw['uuids'][idx]}.jpg"
                    require(image_path.is_file(), f"Missing source image: {image_path}")
                    x = load_2d_finetune_aspect(str(image_path), merge_channels=True).unsqueeze(0)
                    require(bool(torch.isfinite(x).all()) and float(x.std()) > 0,
                            f"Invalid source image or loader fallback: {image_path}")
                    require(x.shape[1] in (1, in_ch), "Unexpected image channel count")
                    if x.shape[1] == 1 and in_ch != 1: x = x.repeat(1, in_ch, 1, 1)
                    z, _, zbar = model.compute_latent_proj(x.to(device), sample_posterior=False)
                    require(zbar is not None, f"No projected output: {config}")
                    raw_check = numeric_agreement(z[0].cpu().numpy(), raw["latents"][idx],
                        args.atol, args.rtol, f"{config}/{split} encoder posterior mode")
                    cached_z = torch.as_tensor(raw["latents"][idx:idx+1], device=device)
                    from_cache = model.channel_proj(model.channel_ds(cached_z) + cached_z)
                    api_check = numeric_agreement(from_cache.cpu().numpy(), zbar.cpu().numpy(),
                        args.atol, args.rtol, f"{config}/{split} cached projection versus API")
                    record["spot_checks"].append(dict(image_uuid=str(raw["uuids"][idx]),
                                                       raw=raw_check, projected=api_check))
                chunks = []
                for start in range(0, len(raw["uuids"]), args.batch_size):
                    z = torch.as_tensor(raw["latents"][start:start+args.batch_size], device=device)
                    chunks.append(model.channel_proj(model.channel_ds(z) + z).float().cpu().numpy())
                projected = np.concatenate(chunks)
            require(np.isfinite(projected).all(), f"Non-finite projected features: {config}/{split}")
            existing_file = cache_path(args.existing_projected_dir, config, split)
            if config in MULTI:
                existing = load_cache(existing_file, config, expected[split])
                require("representation" in existing and
                        str(np.asarray(existing["representation"]).item()) == "projected",
                        f"Missing/wrong projected metadata: {existing_file}")
                record["existing_projected_row_order_matches_raw"] = bool(
                    np.array_equal(existing["uuids"], raw["uuids"]))
                if split != "test":
                    require(record["existing_projected_row_order_matches_raw"],
                            f"Raw/projected {split} row order differs; verify the old head runs before reuse")
                existing = align_to(raw, existing)
                record["existing_projected_agreement"] = numeric_agreement(
                    projected, existing["latents"], args.atol, args.rtol,
                    f"{config}/{split} full existing projected cache")
                record["existing_projected_sha256"] = sha256(existing_file)
                # Preserve the EXACT features used by the existing fitted heads.
                projected = existing["latents"]
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
            manifest["validation"].append(record)
            atomic_json(out / "manifest.json", manifest)
            print(f"Validated {config}/{split}: {len(raw['uuids'])} images", flush=True)
        del model; gc.collect()
        if device.startswith("cuda"): torch.cuda.empty_cache()
    manifest["status"] = "ready"
    atomic_json(out / "manifest.json", manifest)
    print(f"Ready: {out / 'manifest.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--existing-projected-dir", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--spot-checks", type=int, default=2)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    args = p.parse_args()
    require(args.spot_checks > 0 and args.batch_size > 0, "Positive batch and check counts required")
    prepare(args)


if __name__ == "__main__": main()
