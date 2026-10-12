"""Derive raw/projected geometry and per-angle results from saved core predictions.

No model fitting. Uses the original geometric definitions, restores pixel
aspect ratio, verifies the annotation convention before reading model outputs,
and names stored Cobb columns by their validated anatomical meanings.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analyse_projection_study import checked_run
from projection_common import (SINGLE, atomic_json, checked_manifest,
    error_vectors, import_source, paired, read_json, require, sha256, verify_protocol_source)


ANGLE_NAMES = ("major", "upper_proximal_thoracic", "lower_thoracolumbar_lumbar")


def pixel_scale(uuids, img_dir):
    scales, dimensions = [], []
    for uuid in uuids:
        candidates = [Path(img_dir)/f"{uuid}{ext}" for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")]
        matches = [p for p in candidates if p.is_file()]
        require(len(matches) == 1, f"Missing or ambiguous image dimensions for {uuid}")
        with Image.open(matches[0]) as im: width, height = im.size
        require(width > 0 and height > 0, f"Invalid image dimensions for {uuid}")
        scales.append([width]*68 + [height]*68)
        dimensions.append(dict(image_uuid=str(uuid), width=width, height=height))
    return np.array(scales, dtype=float), dimensions


def geometry_errors(g, predictions, gt, scale, grouping, perm, assignment):
    th_gt = g.endplate_angles(g.to_vertebrae(gt["landmarks"]*scale, grouping), perm)
    th_pred = g.endplate_angles(g.to_vertebrae(predictions["landmarks"]*scale, grouping), perm)
    selected = g.apex_pair(th_gt)
    orient_error = np.abs(th_pred - th_gt)
    derived_gt = g.three_cobb(th_gt)[:, assignment[1]]
    derived_pred = g.three_cobb(th_pred)[:, assignment[1]]
    basic = error_vectors(predictions["cobb"], predictions["landmarks"], gt["cobb"], gt["landmarks"])
    return dict(position=basic["landmark"], orientation=orient_error.mean(1),
        selected_end_vertebra_orientation=np.take_along_axis(orient_error, selected, axis=1).mean(1),
        upper_cobb_from_landmarks=np.abs(derived_pred-derived_gt), direct_cobb=basic["cobb"])


def _analyse_variant(args, variant):
    study = args.study_dir.resolve(); m = checked_manifest(study)
    verify_protocol_source(args.geometry_source, "geometry")
    g = import_source(args.geometry_source, "medvae_projection_geometry")
    ids = np.array(sorted(m["expected_ids"]["test"])); data = {}
    for c in SINGLE:
        path = study/"runs"/variant/c
        checked_run(path)
        protocol = read_json(path/"protocol.json")
        require(protocol["manifest_sha256"] == sha256(study/"manifest.json"), "Manifest changed since fitting")
        require(protocol["seeds"] == list(range(10)), "Geometry requires the complete ten-seed ensemble")
        with np.load(path/f"preds_{c}.npz", allow_pickle=False) as source:
            d = {k: source[k] for k in source.files}
        source_ids = d["uuids"].astype(str)
        require(len(set(source_ids)) == len(ids) and set(source_ids) == set(ids), "Prediction ID mismatch")
        index = dict(zip(source_ids, range(len(ids))))
        order = np.array([index[u] for u in ids])
        data[c] = {k: d[k][order] for k in ("gt_cobb", "gt_landmarks", "pred_cobb", "pred_landmarks")}
    for key in ("gt_cobb", "gt_landmarks"):
        require(np.array_equal(data[SINGLE[0]][key], data[SINGLE[1]][key]), "Cross-config annotation mismatch")
    gt = dict(cobb=data[SINGLE[0]]["gt_cobb"], landmarks=data[SINGLE[0]]["gt_landmarks"])
    require(np.isfinite(gt["cobb"]).all() and np.isfinite(gt["landmarks"]).all(), "Invalid annotations")
    scale, dimensions = pixel_scale(ids, args.img_dir)
    # Only ground-truth coordinates and labels enter this convention check.
    result = g.discover_geometry(gt["landmarks"]*scale, gt["cobb"])
    require(result is not None, "Could not identify the annotation convention")
    grouping, perm, assignment, _ = result
    require(tuple(assignment) == (1,0,2), "Annotation output identities differ from the report")
    theta = g.endplate_angles(g.to_vertebrae(gt["landmarks"]*scale, grouping), perm)
    derived = g.three_cobb(theta)[:, assignment]
    residual = np.abs(derived-gt["cobb"]).mean(0)
    require(residual[0] < .05 and residual[1] < .5,
            f"Ground-truth geometry failed report calibration: MAEs {residual.tolist()}")
    calibration = [dict(output_index=j, angle=name, mae_degrees=float(residual[j]),
        correlation=float(np.corrcoef(derived[:,j], gt["cobb"][:,j])[0,1]))
        for j, name in enumerate(ANGLE_NAMES)]
    errors, per_angle = {}, {}
    for c in SINGLE:
        pred = dict(cobb=data[c]["pred_cobb"], landmarks=data[c]["pred_landmarks"])
        require(pred["cobb"].shape == gt["cobb"].shape and pred["landmarks"].shape == gt["landmarks"].shape,
                "Prediction shape mismatch")
        require(np.isfinite(pred["cobb"]).all() and np.isfinite(pred["landmarks"]).all(), "Invalid predictions")
        errors[c] = geometry_errors(g, pred, gt, scale, grouping, perm, assignment)
        per_angle[c] = {name: np.abs(pred["cobb"][:,j]-gt["cobb"][:,j]) for j,name in enumerate(ANGLE_NAMES)}
    out = study/"geometry"
    if m.get("study_kind") == "fresh_paired" or variant != "projected_keep": out = out/variant
    out.mkdir(parents=True, exist_ok=True)
    long_rows, tables = [], {}
    for family, values, size in (("geometric_readouts", errors, 5), ("direct_per_angle", per_angle, 3)):
        table = []
        for metric in values[SINGLE[0]]:
            a, b = values[SINGLE[0]][metric], values[SINGLE[1]][metric]
            result = paired(a, b, resamples=args.resamples)
            table.append(dict(metric=metric, config_a=SINGLE[0], config_b=SINGLE[1], n=len(ids),
                **result, family_size=size, p_bonf_family=min(1., size*result["p"]),
                p_bonf_all_eight=min(1., 8*result["p"])))
            for c in SINGLE:
                long_rows.extend(dict(family=family, metric=metric, config=c, image_uuid=str(u), error=float(e))
                                 for u,e in zip(ids, values[c][metric]))
        tables[family] = pd.DataFrame(table)
        tables[family].to_csv(out/f"{family}.csv", index=False)
    pd.DataFrame(long_rows).to_csv(out/"geometry_per_image.csv", index=False)
    pd.DataFrame(calibration).to_csv(out/"annotation_geometry_validation.csv", index=False)
    pd.DataFrame(dimensions).to_csv(out/"image_dimensions.csv", index=False)
    atomic_json(out/"geometry_protocol.json", dict(variant=variant, grouping=grouping, corner_permutation=list(perm),
        provided_to_derived_assignment=list(assignment), stored_angle_names=list(ANGLE_NAMES),
        geometry_source_sha256=sha256(args.geometry_source), script_sha256=sha256(__file__),
        common_sha256=sha256(Path(__file__).parent/"projection_common.py"),
        bootstrap_resamples=args.resamples, bootstrap_seed=0, n=128, families=dict(geometric_readouts=5,direct_per_angle=3),
        convention="Original pixel-space endplate-vector geometry; no new clipping or angular wrapping",
        interpretation="Differences between readouts are descriptive; no ordering or causal mechanism test is performed."))
    (out/"README.md").write_text("""# Projected geometry follow-up

These results use saved ten-seed predictions for projected 4_1 and 8_1. The five
readouts reproduce the thesis definitions, including pixel-space angles and
GT-selected end vertebrae. The regional Cobb readout concerns the **upper** curve
and compares derived predictions with geometry from GT landmarks. Direct Cobb
is instead the mean of three supplied angle targets. They are different targets.

The per-angle table names stored outputs 0/1/2 as major/upper/lower. The lower
geometric convention has a known residual against the supplied labels; it is not
used as another derived-Cobb endpoint here. Raw angular differences match the
original script; no new wrapping or clipping is introduced.

Positive deltas indicate higher error for projected 8_1. Family-adjusted and
all-eight adjusted p-values are supplied separately. Intervals resample images
for fixed fitted ensembles. The five readouts are correlated and an ordered set
of percentage changes alone does not establish selective information loss.
""".replace("Projected", "Raw" if variant == "raw_keep" else "Projected").replace(
        "projected", "raw" if variant == "raw_keep" else "projected"))
    print(pd.DataFrame(calibration).to_string(index=False))
    for name, table in tables.items(): print(name); print(table.to_string(index=False))
    print(f"Outputs: {out}")


def analyse(args):
    m = checked_manifest(args.study_dir)
    variant = getattr(args, "variant", None) or ("both" if m.get("study_kind") == "fresh_paired" else "projected_keep")
    for v in (("raw_keep", "projected_keep") if variant == "both" else (variant,)):
        _analyse_variant(args, v)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--geometry-source", type=Path,
                   default=Path(__file__).resolve().parent / "references" / "cobb_from_landmarks.py",
                   help="Geometry source; default: the bundled audited reference")
    p.add_argument("--variant", choices=("raw_keep", "projected_keep", "both"),
                   help="Default: both for fresh paired studies, projected_keep for historical-cache studies")
    p.add_argument("--resamples", type=int, default=10000)
    args = p.parse_args(); require(args.resamples >= 10000, "Use at least 10,000 draws")
    analyse(args)


if __name__ == "__main__": main()
