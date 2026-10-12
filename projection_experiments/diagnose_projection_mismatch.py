"""Read existing caches and compare raw encoding under explicit precision modes.

Writes diagnostic JSON only. Does not overwrite caches, train heads, relax
tolerances or produce a training-ready study. Default: the failing 4_1 training
split, including the same first image checked by prepare_projection_study.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from projection_common import CONFIGS, COUNTS, atomic_json, cache_path, load_cache, require, sha256, split_ids
from projection_validation import (build_verified_model, collect_spot, comparison_report,
                                    load_source_image, precision_mode, runtime_info)


def summarise_modes(report):
    compatible = []
    for mode in report["requested_modes"]:
        rows = [r for r in report["checks"] if r["mode"] == mode]
        if len(rows) != report["expected_checks_per_mode"]: continue
        if all("comparisons" in r and all(r["comparisons"][key]["passed"] for key in
               ("fresh_raw_vs_cached_raw", "formula_vs_api_same_fresh_z", "encode_mode_vs_api_raw_same_input"))
               for r in rows): compatible.append(mode)
    return compatible


def diagnose(args):
    import torch
    sys.path.insert(0,str(Path.cwd()))
    expected = split_ids(args.split_csv)
    require(1 <= args.spot_checks <= COUNTS[args.split], "Invalid number of spot checks")
    out = args.out_dir.resolve()
    require(not out.exists() or not any(out.iterdir()), f"Use a new empty diagnostic directory: {out}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = dict(schema=1, status="diagnostic_only", expected_checks_per_mode=args.spot_checks,
        config=args.config, split=args.split, requested_modes=args.modes, checks=[],
        runtime=runtime_info(torch,device), tolerances=dict(atol=args.atol,rtol=args.rtol),
        script_sha256=sha256(__file__), helper_sha256=sha256(Path(__file__).parent/"projection_validation.py"),
        scope="Numerical diagnosis on selected cases, not a full-cache validation or proof of historical checkpoint identity")
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"diagnosis.json",report)
    try:
        raw_file = cache_path(args.raw_dir,args.config,args.split)
        raw = load_cache(raw_file,args.config,expected[args.split])
        report["raw_cache"] = dict(path=str(raw_file.resolve()),sha256=sha256(raw_file),
            shape=list(raw["latents"].shape), dtype=str(raw["latents"].dtype),
            representation=str(raw["representation"].item()) if "representation" in raw else None)
        model,checkpoint = build_verified_model(args.config,device)
        report["checkpoint"] = checkpoint
        indices = np.argsort(raw["uuids"])[np.linspace(0,len(raw["uuids"])-1,args.spot_checks,dtype=int)]
        report["selected_uuids"] = [str(raw["uuids"][i]) for i in indices]
        for idx in indices:
            uuid = str(raw["uuids"][idx])
            x,image_info = load_source_image(model,args.img_dir/f"{uuid}.jpg")
            by_mode = {}
            for mode in args.modes:
                row = dict(image_uuid=uuid, mode=mode, image=image_info)
                try:
                    with precision_mode(torch,mode) as settings:
                        row["precision"] = settings
                        checks,arrays = collect_spot(model,x,raw["latents"][idx:idx+1],device,
                                                     args.atol,args.rtol,check_encode=True)
                    row["comparisons"] = checks
                    by_mode[mode] = arrays["fresh_z"]
                    r = checks["fresh_raw_vs_cached_raw"]
                    api = checks["formula_vs_api_same_fresh_z"]
                    print(f"{uuid} | {mode}: cached-raw match={r['passed']}, "
                          f"max={r.get('max_abs')}, rms={r.get('rms')}, "
                          f"failed={r.get('failing_elements')}/{r.get('elements')}, "
                          f"same-latent API match={api['passed']}",flush=True)
                except (RuntimeError,ValueError,AttributeError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"{uuid} | {mode}: {row['error']}",flush=True)
                report["checks"].append(row)
                atomic_json(out/"diagnosis.json",report)
            if "current" in by_mode:
                for mode,values in by_mode.items():
                    if mode != "current":
                        report.setdefault("precision_changes_same_input",[]).append(dict(
                            image_uuid=uuid, mode_a="current", mode_b=mode,
                            comparison=comparison_report(values,by_mode["current"],args.atol,args.rtol)))
        report["compatible_modes_for_checked_cases"] = summarise_modes(report)
        report["diagnostic_complete"] = True
        report["next_step"] = (
            "A listed mode satisfies the unchanged checks on these cases. Use that mode explicitly in a fresh "
            "preparation run; only completion of every preparation check permits training."
            if report["compatible_modes_for_checked_cases"] else
            "No tested mode passed every required check. Return diagnosis.json for review; do not loosen "
            "the threshold or start training from the incomplete study.")
        atomic_json(out/"diagnosis.json",report)
        print("\nCompatible modes on checked cases:",report["compatible_modes_for_checked_cases"])
        print(report["next_step"])
        print(f"Diagnostic saved: {out/'diagnosis.json'}")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["diagnostic_complete"] = False
        atomic_json(out/"diagnosis.json",report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir",type=Path,required=True)
    p.add_argument("--split-csv",type=Path,required=True)
    p.add_argument("--img-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--device")
    p.add_argument("--config",choices=CONFIGS,default="medvae_4_1_2d")
    p.add_argument("--split",choices=COUNTS,default="train")
    p.add_argument("--spot-checks",type=int,default=2)
    p.add_argument("--modes",choices=("current","ieee","tf32"),nargs="+",default=["current","ieee","tf32"])
    p.add_argument("--atol",type=float,default=1e-3)
    p.add_argument("--rtol",type=float,default=1e-3)
    args=p.parse_args()
    require(len(set(args.modes)) == len(args.modes), "Repeated precision mode")
    require(np.isfinite([args.atol,args.rtol]).all() and args.atol>=0 and args.rtol>=0,"Invalid tolerances")
    diagnose(args)


if __name__=="__main__":main()
