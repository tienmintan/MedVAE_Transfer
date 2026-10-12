"""Separate historical cache consistency from same-input projection correctness.

Historical cache reproducibility and correctness of the projection formula are
different checks. The latter must compare computations on the SAME latent.
"""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import os
import platform
from pathlib import Path

import numpy as np

from projection_common import require, sha256


class NumericalMismatch(ValueError):
    def __init__(self, context, report):
        self.context, self.report = context, report
        super().__init__(f"Numerical check failed ({context}): "
            f"max_abs={report.get('max_abs')}, rms={report.get('rms')}, "
            f"failing_elements={report.get('failing_elements')}/{report.get('elements')}. "
            "The detailed record is saved in the study's numerical_failure.json. "
            "Inspect that record and CACHE_MISMATCH_RECOVERY.md before retrying.")


def comparison_report(actual, expected, atol=1e-3, rtol=1e-3):
    a0, b0 = np.asarray(actual), np.asarray(expected)
    result = dict(shape_actual=list(a0.shape), shape_expected=list(b0.shape),
                  dtype_actual=str(a0.dtype), dtype_expected=str(b0.dtype), atol=atol, rtol=rtol)
    if a0.shape != b0.shape or a0.size == 0:
        return dict(result, passed=False, reason="shape mismatch or empty array")
    a, b = a0.astype(np.float64), b0.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(result, passed=False, reason="non-finite values",
                    nonfinite_actual=int((~np.isfinite(a)).sum()), nonfinite_expected=int((~np.isfinite(b)).sum()))
    d = a-b; ad = np.abs(d); tolerance = atol+rtol*np.abs(b)
    fail = ad > tolerance
    rms = float(np.sqrt(np.mean(d*d))); scale = float(np.sqrt(np.mean(b*b)))
    flat = ad.ravel(); count = min(5, len(flat))
    top = np.argpartition(flat, len(flat)-count)[-count:]
    top = top[np.argsort(flat[top])[::-1]]
    worst = []
    for i in top:
        ix = np.unravel_index(i, a.shape)
        worst.append(dict(index=[int(j) for j in ix], actual=float(a[ix]), expected=float(b[ix]),
                          absolute_difference=float(ad[ix]), allowed_difference=float(tolerance[ix])))
    result.update(passed=bool(not fail.any()), elements=int(a.size), failing_elements=int(fail.sum()),
        failing_fraction=float(fail.mean()), max_abs=float(ad.max()), rms=rms,
        mean_signed_difference=float(d.mean()), expected_mean=float(b.mean()), expected_sd=float(b.std()),
        expected_rms=scale, rms_relative_to_expected_rms=rms/scale if scale else None,
        absolute_difference_percentiles=dict(zip(("p50","p95","p99","p99_9"),
                                                   map(float,np.percentile(ad,[50,95,99,99.9])))),
        worst_positions=worst)
    return result


def enforce(report, context):
    if not report["passed"]: raise NumericalMismatch(context, report)
    return report


def historical_cache_decision(report, policy="strict"):
    """Engineering consistency screen, NOT downstream or clinical equivalence.

    Strict retains the original elementwise rule. Fixed deliberately anchors the
    experiment to saved raw features; fresh encoding is a consistency check.
    Its additional limits are RMS <= 0.1% and maximum <= 1% of the cached tensor
    SD. Both must pass; a small average cannot hide a large local discrepancy.
    These are reviewed engineering limits, not measured machine-error bounds or
    a claim that we recovered the historical execution environment.
    """
    require(policy in ("strict", "fixed"), "Unknown raw-cache policy")
    result = dict(policy=policy, elementwise_passed=bool(report["passed"]),
                  max_abs=report.get("max_abs"), rms=report.get("rms"),
                  elements=report.get("elements"), failing_elements=report.get("failing_elements"),
                  scope="cache consistency only; not historical identity or downstream equivalence")
    if policy == "strict":
        return dict(result, passed=bool(report["passed"]))
    result.update(rms_over_cached_sd_limit=0.001, max_over_cached_sd_limit=0.01)
    if "reason" in report:
        return dict(result, passed=False, reason=report["reason"])
    sd = float(report["expected_sd"])
    if not np.isfinite(sd) or sd <= 0:
        return dict(result, passed=False, reason="Nonpositive cached SD; relative screen undefined")
    rms_ratio, max_ratio = report["rms"]/sd, report["max_abs"]/sd
    result.update(rms_over_cached_sd=rms_ratio, max_over_cached_sd=max_ratio)
    return dict(result, passed=bool(np.isfinite([rms_ratio, max_ratio]).all()
                                   and rms_ratio <= 0.001 and max_ratio <= 0.01))


def projected_reuse_decision(records, mode="auto"):
    """Reuse a configuration only if ALL three aligned caches remain compatible.

    Otherwise all its projected splits are rebuilt, and both retained-channel
    tasks are refitted. This avoids a mixture of old/new splits under old scores.
    """
    require(mode in ("auto", "never"), "Unknown projected-reuse policy")
    from projection_common import COUNTS
    require(len(records) == len(COUNTS) and {r["split"] for r in records} == set(COUNTS),
            "All three splits are required for a reuse decision")
    failed = [r["split"] for r in records
              if not r.get("existing_projected_agreement", {}).get("passed", False)
              or (r["split"] != "test" and not r.get("existing_projected_row_order_matches_raw", False))]
    reuse = mode == "auto" and not failed
    return dict(reuse=reuse, mode=mode, incompatible_splits=failed,
        selected_source="existing_projected_exact_values" if reuse else "projection_of_fixed_raw_cache",
        additional_retained_fits=0 if reuse else 20,
        reason=("All three caches pass the original elementwise comparison and training row-order checks"
                if reuse else "Explicit rebuild requested" if mode == "never"
                else "At least one old projected split is missing, numerically incompatible, or differently ordered"))


def compare_spot_arrays(fresh_z, cached_z, formula_on_fresh_z, api_zbar, formula_on_cached_z, atol, rtol):
    return dict(
        fresh_raw_vs_cached_raw=comparison_report(fresh_z, cached_z, atol, rtol),
        formula_vs_api_same_fresh_z=comparison_report(formula_on_fresh_z, api_zbar, atol, rtol),
        projected_drift_from_different_raw_inputs=comparison_report(formula_on_cached_z, api_zbar, atol, rtol))


def collect_spot(model, x, cached_z, device, atol, rtol, check_encode=False):
    import torch
    with torch.no_grad():
        x = x.to(device=device, dtype=torch.float32)
        z, _, zbar = model.compute_latent_proj(x, sample_posterior=False)
        require(zbar is not None, "Checkpoint has no projected output")
        on_fresh = model.channel_proj(model.channel_ds(z) + z)
        cached = torch.as_tensor(cached_z, dtype=torch.float32, device=device)
        on_cached = model.channel_proj(model.channel_ds(cached) + cached)
        arrays = dict(fresh_z=z.cpu().numpy(), cached_z=np.asarray(cached_z),
                      formula_on_fresh_z=on_fresh.cpu().numpy(), api_zbar=zbar.cpu().numpy(),
                      formula_on_cached_z=on_cached.cpu().numpy())
        reports = compare_spot_arrays(**arrays, atol=atol, rtol=rtol)
        if check_encode:
            legacy_z = model.encode(x).mode().cpu().numpy()
            reports["encode_mode_vs_api_raw_same_input"] = comparison_report(legacy_z, arrays["fresh_z"], atol, rtol)
    return reports, arrays


def precision_controls(torch):
    back = torch.backends
    try:
        owners = [("global", back), ("cuda_matmul", back.cuda.matmul),
                  ("cudnn", back.cudnn), ("cudnn_conv", back.cudnn.conv), ("cudnn_rnn", back.cudnn.rnn)]
        values = [getattr(owner,"fp32_precision") for _,owner in owners]
        return "fp32_precision", [(name,owner,"fp32_precision",value)
                                  for (name,owner),value in zip(owners,values)]
    except AttributeError:
        owners = [("cuda_matmul", back.cuda.matmul), ("cudnn", back.cudnn)]
        return "allow_tf32", [(name,owner,"allow_tf32",getattr(owner,"allow_tf32")) for name,owner in owners]


def precision_snapshot(torch):
    api, controls = precision_controls(torch)
    return dict(api=api, values={name:value for name,_,_,value in controls},
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic))


@contextmanager
def precision_mode(torch, mode):
    require(mode in ("current","ieee","tf32"), "Unknown precision mode")
    api, controls = precision_controls(torch)
    try:
        if mode != "current":
            for _,owner,attr,_ in controls:
                setattr(owner,attr,mode if api == "fp32_precision" else mode == "tf32")
        yield precision_snapshot(torch)
    finally:
        if mode != "current":
            for _,owner,attr,value in controls: setattr(owner,attr,value)


def runtime_info(torch, device):
    versions = {}
    for name in ("torch","torchvision","monai","numpy","Pillow","huggingface-hub"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    gpu = None
    if str(device).startswith("cuda"):
        prop = torch.cuda.get_device_properties(device)
        gpu = dict(name=prop.name, compute_capability=[prop.major,prop.minor], memory_bytes=prop.total_memory)
    return dict(python=platform.python_version(), packages=versions, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=gpu, precision=precision_snapshot(torch),
        environment={key:os.environ.get(key) for key in
            ("NVIDIA_TF32_OVERRIDE","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE","CUBLAS_WORKSPACE_CONFIG")})


def build_verified_model(config, device):
    import torch
    from medvae.utils.factory import FILE_DICT_ASSOCIATIONS, build_model, download_model_weights
    paths = FILE_DICT_ASSOCIATIONS[config]
    config_file = download_model_weights(paths["config"])
    checkpoint_file = download_model_weights(paths["ckpt"])
    model = build_model(config, config_file, checkpoint_file, training=False).eval()
    kwargs = {"map_location":"cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters: kwargs["weights_only"] = False
    checkpoint = torch.load(checkpoint_file, **kwargs)["state_dict"]
    current = model.state_dict()
    keys = [k for k in current if k.startswith(("channel_ds.","channel_proj."))]
    require(len(keys)==8 and all(k in checkpoint for k in keys), f"Incomplete projection weights: {config}")
    require(all(torch.equal(current[k].cpu(),checkpoint[k].cpu()) for k in keys),
            f"Projection weights differ from the loaded checkpoint: {config}")
    encoder_keys = [k for k in current if k.startswith(("encoder.","quant_conv."))]
    require(all(k in checkpoint and torch.equal(current[k].cpu(),checkpoint[k].cpu()) for k in encoder_keys),
            f"Encoder weights differ from the loaded checkpoint: {config}")
    record = dict(path=str(checkpoint_file), sha256=sha256(checkpoint_file),
        config_sha256=sha256(config_file), projection_keys=keys, encoder_tensor_count_checked=len(encoder_keys),
        implementation_sha256=sha256(inspect.getfile(type(model))))
    return model.requires_grad_(False).float().to(device), record


def load_source_image(model, image_path):
    import torch
    from medvae.utils.loaders import load_2d_finetune_aspect
    require(Path(image_path).is_file(), f"Missing source image: {image_path}")
    x = load_2d_finetune_aspect(str(image_path),merge_channels=True).unsqueeze(0)
    require(bool(torch.isfinite(x).all()) and float(x.std())>0, f"Invalid image or loader fallback: {image_path}")
    channels = int(model.encoder.conv_in.in_channels)
    require(x.shape[1] in (1,channels), "Unexpected image channel count")
    if x.shape[1]==1 and channels!=1: x=x.repeat(1,channels,1,1)
    a=x.cpu().numpy()
    details=dict(path=str(Path(image_path).resolve()), file_sha256=sha256(image_path),
        input_sha256=hashlib.sha256(a.tobytes()).hexdigest(), input_shape=list(a.shape),
        input_min=float(a.min()), input_max=float(a.max()), input_mean=float(a.mean()), input_sd=float(a.std()),
        loader_source_sha256=sha256(inspect.getfile(load_2d_finetune_aspect)))
    return x, details
