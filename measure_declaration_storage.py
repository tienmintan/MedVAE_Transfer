#!/usr/bin/env python3
"""Read-only inventory of likely caches and checkpoint state under one repository.

Uses file metadata only: no PyTorch, NumPy, GPU work or checkpoint loading.
Prints JSON to stdout; redirect it to a file if required. Totals are logical
regular-file byte counts, not physical allocated disk blocks. Classification
uses directory names/extensions and is explicitly a candidate inventory.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


SKIP_DIRS = {
    ".git", ".Trash-1000", ".venv", "venv", "env", "miniforge3",
    "miniconda3", "node_modules", "site-packages", "__pycache__",
    ".ipynb_checkpoints", ".cache",
}
ARRAY_SUFFIXES = {".npz", ".npy"}
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def classify(relative):
    """Return a category and contributing directory, without loading a file."""
    parts = relative.parts
    directories = parts[:-1]
    cache_index = next((i for i, name in enumerate(directories)
                        if name.lower().startswith("latents")
                        or name.lower() == "caches"), None)
    if relative.suffix.lower() in ARRAY_SUFFIXES:
        if cache_index is not None:
            owner = Path(*parts[:cache_index + 1]).as_posix()
            if "dino" in owner.lower() or "diffusion_features" in owner.lower():
                category = "candidate_baseline_feature_caches"
            else:
                category = "candidate_latent_caches"
            return category, owner
        return "other_saved_arrays_review_for_uncategorised_caches", relative.parent.as_posix()

    checkpoint_index = next((i for i, name in enumerate(directories)
                             if name.lower() in {"checkpoint", "checkpoints"}
                             or name.lower().startswith("diffusion_ckpt")
                             or name.lower().endswith((".pt", ".ckpt"))), None)
    suffix = relative.suffix.lower()
    if suffix in WEIGHT_SUFFIXES or (suffix == ".bin" and checkpoint_index is not None):
        owner = (Path(*parts[:checkpoint_index + 1]).as_posix()
                 if checkpoint_index is not None else relative.parent.as_posix())
        return "candidate_checkpoint_state", owner
    if checkpoint_index is not None:
        return "other_checkpoint_directory_files_review", Path(*parts[:checkpoint_index + 1]).as_posix()
    if suffix == ".bin":
        return "other_bin_files_review_for_uncategorised_weights", relative.parent.as_posix()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")
    totals = defaultdict(lambda: {"files": 0, "bytes": 0})
    grouped = defaultdict(lambda: {"files": 0, "bytes": 0})
    errors, skipped_links = [], []

    def onerror(error):
        errors.append(str(error))

    for directory, child_dirs, filenames in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(directory)
        retained = []
        for name in child_dirs:
            child = parent / name
            if child.is_symlink():
                skipped_links.append(str(child.relative_to(root)))
            elif name not in SKIP_DIRS:
                retained.append(name)
        child_dirs[:] = retained
        for name in filenames:
            path = parent / name
            relative = path.relative_to(root)
            if path.is_symlink():
                skipped_links.append(str(relative))
                continue
            classification = classify(relative)
            if classification is None:
                continue
            try:
                if not path.is_file():
                    continue
                size = path.stat().st_size
            except OSError as error:
                errors.append(f"{relative}: {error}")
                continue
            category, owner = classification
            for accumulator in (totals[category], grouped[(category, owner)]):
                accumulator["files"] += 1
                accumulator["bytes"] += size

    def with_units(row):
        return {**row, "GB_decimal": round(row["bytes"] / 1e9, 6),
                "GiB_binary": round(row["bytes"] / 2**30, 6)}

    result = {
        "root": str(root),
        "measurement": "Logical sizes of regular files; no content deserialisation.",
        "scope": "Current files under this root, excluding the listed directories and symbolic links.",
        "classification": "Candidates based on names and extensions. Review categories and directories before citing totals.",
        "excluded_directory_names": sorted(SKIP_DIRS),
        "totals": {key: with_units(value) for key, value in sorted(totals.items())},
        "directories": [{"category": category, "directory": owner, **with_units(value)}
                        for (category, owner), value in sorted(grouped.items())],
        "errors": errors,
        "skipped_symbolic_links": skipped_links,
        "historical_encoder_calls": None,
        "historical_GPU_hours": None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
