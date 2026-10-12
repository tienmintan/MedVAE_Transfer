"""Shared validation and statistics for the MedVAE projection follow-up."""
import hashlib
import ast
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONFIGS = ("medvae_4_1_2d", "medvae_4_3_2d", "medvae_8_1_2d", "medvae_8_4_2d")
SINGLE = (CONFIGS[0], CONFIGS[2])
MULTI = (CONFIGS[1], CONFIGS[3])
SHAPES = {CONFIGS[0]: (1, 96, 96), CONFIGS[1]: (3, 96, 96),
          CONFIGS[2]: (1, 48, 48), CONFIGS[3]: (4, 48, 48)}
COUNTS = {"train": 409, "val": 72, "test": 128}
TASKS = ("cobb", "landmark")
VARIANTS = ("raw_keep", "projected_keep", "raw_mean", "projected_mean")
TRAINING = dict(epochs=200, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=25)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2); handle.write("\n"); tmp = handle.name
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays); tmp = handle.name
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def cache_path(directory, config, split):
    return Path(directory) / f"latents_{config}_{split}.npz"


def load_cache(path, config, expected_ids, *, mean=False):
    with np.load(path, allow_pickle=False) as source:
        require({"latents", "uuids", "cobb", "landmarks"}.issubset(source.files),
                f"Incomplete cache: {path}")
        d = {k: source[k] for k in source.files}
    ids = d["uuids"].astype(str)
    require(ids.ndim == 1 and len(set(ids)) == len(ids), f"Duplicate/invalid IDs: {path}")
    require(set(ids) == set(expected_ids), f"Split membership mismatch: {path}")
    c, h, w = SHAPES[config]
    require(d["latents"].shape == (len(ids), 1 if mean else c, h, w), f"Shape mismatch: {path}")
    require(d["latents"].dtype == np.float32, f"Expected fp32 latents: {path}")
    for key, shape in (("cobb", (len(ids), 3)), ("landmarks", (len(ids), 136))):
        require(d[key].shape == shape, f"Wrong {key} shape: {path}")
    for key in ("latents", "cobb", "landmarks"):
        require(np.isfinite(d[key]).all(), f"Non-finite {key}: {path}")
    require(float(np.std(d["latents"])) > 0, f"Constant cache: {path}")
    d["uuids"] = ids
    return d


def align_to(reference, candidate):
    ri, ci = reference["uuids"].astype(str), candidate["uuids"].astype(str)
    require(set(ri) == set(ci) and len(ri) == len(ci), "Identifier mismatch")
    index = {u: i for i, u in enumerate(ci)}
    order = np.array([index[u] for u in ri])
    out = dict(candidate)
    for key in ("latents", "uuids", "cobb", "landmarks"):
        out[key] = candidate[key][order]
    for key in ("cobb", "landmarks"):
        require(np.allclose(reference[key], out[key], atol=1e-6, rtol=0),
                f"Raw/projected or cross-config {key} labels differ")
    return out


def numeric_agreement(actual, expected, atol, rtol, context):
    require(actual.shape == expected.shape, f"Shape mismatch: {context}")
    diff = np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)
    record = dict(max_abs=float(np.abs(diff).max()), rms=float(np.sqrt(np.mean(diff**2))),
                  atol=atol, rtol=rtol)
    require(np.allclose(actual, expected, atol=atol, rtol=rtol),
            f"Numerical check failed ({context}): {record}. Check weights, preprocessing and precision; do not automatically loosen the tolerance.")
    return record


def import_source(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    require(spec is not None and spec.loader is not None, f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def definition_hashes(path, names):
    """Hash definitions for comparison within the current Python interpreter.

    AST fields and their dump format change between Python versions. These
    hashes must not be compared with AST hashes saved by another interpreter.
    verify_protocol_source checks the pinned source bytes before comparing two
    source files parsed by the same interpreter.
    """
    definitions = [n for n in ast.parse(Path(path).read_text(encoding="utf-8")).body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    nodes = {n.name: n for n in definitions}
    require(set(names).issubset(nodes), f"Missing protocol definitions in {path}")
    duplicate = [name for name in names if sum(n.name == name for n in definitions) != 1]
    require(not duplicate, f"Duplicate protocol definitions in {path}: {duplicate}")
    return {name: hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
            for name in names}


def verify_protocol_source(path, family, *, references_dir=None):
    """Compare against an unchanged audited source using one Python version.

    The original full-file SHA256 anchors the reference independently of the
    Python version. The old per-definition hashes remain in protocol.json as
    provenance, but are not a portable representation of Python syntax.
    This function reads source only; it does not import or execute either file.
    """
    filenames = {"training": "train_frontier.py", "geometry": "cobb_from_landmarks.py"}
    require(family in filenames, f"Unknown protocol family: {family}")
    root = Path(references_dir) if references_dir is not None else Path(__file__).resolve().parent / "references"
    protocol = read_json(root / "protocol.json")
    reference_path = root / filenames[family]
    require(reference_path.is_file(), f"Missing bundled reference: {reference_path}")
    expected_sha = protocol["files_sha256"].get(reference_path.name)
    require(expected_sha is not None and sha256(reference_path) == expected_sha,
            f"Bundled reference {reference_path} differs from its pinned SHA256. "
            "Restore the original bundled source; do not rewrite protocol.json.")
    names = protocol[family + "_definitions"]
    expected = definition_hashes(reference_path, names)
    actual = definition_hashes(path, names)
    different = [name for name in names if actual[name] != expected[name]]
    require(not different, f"{family.capitalize()} source differs from the audited protocol: {different}. "
            f"Compare {path} with {reference_path} before running.")
    return protocol


def checked_manifest(study_dir):
    m = read_json(Path(study_dir) / "manifest.json")
    require(m.get("status") == "ready", "Cache preparation is incomplete; do not train/analyse it")
    return m


def reference_configs(manifest, stage):
    """Fresh paired studies never import historical raw or projected scores."""
    require(stage in ("raw", "projected"), "Unknown reference stage")
    if manifest.get("study_kind") == "fresh_paired":
        require(manifest.get("reference_reuse") == dict(raw_keep=[], projected_keep=[]),
                "Fresh paired study cannot reuse historical scores")
        return []
    if manifest.get("schema", 1) < 2:
        return list(CONFIGS if stage == "raw" else MULTI)
    reused = manifest.get("reference_reuse", {}).get(stage+"_keep")
    allowed = CONFIGS if stage == "raw" else MULTI
    require(isinstance(reused, list) and len(set(reused)) == len(reused)
            and set(reused).issubset(allowed), "Missing or invalid reference-reuse decision")
    return reused


def projected_reference_configs(manifest):
    return reference_configs(manifest, "projected")


def validate_cell(m, variant, config):
    entry = m["cells"][variant][config]
    require("alias" not in entry, "Identity aliases do not need another fit")
    out = {}
    for split in COUNTS:
        path = entry["paths"][split]
        require(sha256(path) == entry["sha256"][split], f"Cache changed after preparation: {path}")
        out[split] = load_cache(path, config, m["expected_ids"][split], mean=variant.endswith("mean"))
    return out


def mean_interval(values, resamples=10000):
    values = np.asarray(values, dtype=float)
    ix = np.random.default_rng(0).integers(0, len(values), (resamples, len(values)))
    lo, hi = np.percentile(values[ix].mean(1), [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def paired(a, b, *, resamples=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(a.shape == b.shape and a.ndim == 1 and len(a) > 1, "Invalid paired arrays")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Non-finite paired arrays")
    delta = b - a
    draws = np.random.default_rng(seed).integers(0, len(a), (resamples, len(a)))
    lo, hi = np.percentile(delta[draws].mean(1), [2.5, 97.5])
    p = 1.0 if np.all(delta == 0) else float(stats.ttest_rel(b, a).pvalue)
    return dict(mean_a=float(a.mean()), mean_b=float(b.mean()),
                delta=float(delta.mean()), ci95_lo=float(lo), ci95_hi=float(hi), p=p,
                relative_percent=float(100 * delta.mean() / a.mean()) if a.mean() else None)


def error_vectors(pred_cobb, pred_landmarks, gt_cobb, gt_landmarks):
    d = pred_landmarks - gt_landmarks
    return {"cobb": np.abs(pred_cobb - gt_cobb).mean(1),
            "landmark": np.sqrt(d[:, :68]**2 + d[:, 68:]**2).mean(1)}


def split_ids(csv_path):
    df = pd.read_csv(csv_path, dtype={"image_uuid": str})
    require({"image_uuid", "split"}.issubset(df.columns), "Missing split columns")
    require(not df.image_uuid.isna().any() and not df.image_uuid.duplicated().any(),
            "Missing or repeated identifiers in the split manifest")
    require(set(df.split) == set(COUNTS), "Unexpected split names")
    out = {s: df.loc[df.split == s, "image_uuid"].tolist() for s in COUNTS}
    for s, n in COUNTS.items(): require(len(out[s]) == n, f"Expected {n} {s} images")
    return out
