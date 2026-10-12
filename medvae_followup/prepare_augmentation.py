"""OPTIONAL: five moderate augmented copies per cleaned training image, projected then averaged."""
import argparse
import inspect
import tempfile
import time
from pathlib import Path
import shutil

import numpy as np

from common import *
from run_heads import done, jobs, paths


def run(args):
    parent,dst=args.annotation_dir.resolve(),args.out_dir.resolve()
    require(parent!=dst and parent not in dst.parents,"Use a separate augmentation output directory")
    ap=get_plan(parent);am=read(parent/"manifest.json")
    require(ap["kind"]=="annotation" and all(done(parent,ap,j) for j in jobs(ap)),"Complete the annotation study before starting augmentation")
    require((parent/"analysis/analysis_complete.json").is_file(),"Analyse the annotation check first so its policy is settled")
    study=Path(am["original_study"]);om=old_manifest(study)
    plan=set_plan(dst,dict(kind="augmentation",config=CONFIG,variants=["projected_mean"],seeds=SEEDS,tasks=list(TASKS),training=TRAINING,
        parent_annotation=str(parent),parent_plan_fingerprint=ap["fingerprint"],original_study=str(study),
        n_original=407,n_aug_per_original=5,train_total=2442,val=72,test=128,excluded_training_ids=EXCLUDE,
        preset=dict(max_deg=0.,max_scale=.08,max_shift=0.,jitter=.12,gamma=.18,noise=.015,blur=.5),
        geometric_rule="Uniform scale about the native image centre; reject if any landmark leaves the 2-pixel image margin; no rotation, translation or flip",
        sampling="SHA256-derived RNG seed per image, master seed 0; exactly five accepted variants, at most 100 attempts per image",
        representation="compute_latent_proj(x,sample_posterior=False)[2], then float32 channel mean",
        encoding="Original cached features retained; raw and projected fresh encodings of EVERY original training image checked against the verified caches at atol=rtol=0.001",
        photometric_serialisation="Pillow grayscale, source preset, JPEG quality 95, same aspect loader; original and validation/test caches unchanged",
        expected_new_fits=6,primary_family=dict(n=2,tests="augmented-minus-clean Cobb and landmark errors"),
        secondary_family=dict(n=1,tests="isotropic landmark sensitivity"),
        scope="Practical augmentation pipeline with the same maximum epochs; additional training examples also increase updates per epoch. This does not isolate augmentation from additional optimisation.",
        inference="Three-seed ensembles; paired images; two-sided t-tests, 10000 bootstrap samples; Bonferroni across two primary comparisons, plus three-test sensitivity. Exploratory reused test set."))
    cell=am["cells"]["clean/projected_mean"]
    for split,p in cell["paths"].items():require(sha256(p)==cell["sha256"][split],"Clean input cache changed")
    train=cache(cell["paths"]["train"],am["expected_ids"]["train"],channels=1)
    raw_cell=am["cells"]["clean/raw_keep"]
    require(sha256(raw_cell["paths"]["train"])==raw_cell["sha256"]["train"],"Clean raw cache changed")
    raw=cache(raw_cell["paths"]["train"],am["expected_ids"]["train"],channels=3);same_labels(raw,train)
    manifest=dict(kind="augmentation",status="preparing",plan_fingerprint=plan["fingerprint"],cells={},imported={},
        control_environment=read(parent/"training_environment.json"),expected_ids=am["expected_ids"])
    for name,key in (("evaluation.npz","evaluation_sha256"),("image_dimensions.csv","dimensions_sha256")):
        require(sha256(parent/name)==am[key],"Parent evaluation input changed")
        shutil.copy2(parent/name,dst/name);manifest[key]=sha256(dst/name)
    for task in TASKS:
        for seed in SEEDS:
            p,meta=paths(parent,("clean","projected_mean",task,seed))
            require(done(parent,ap,("clean","projected_mean",task,seed)),"Unaugmented control incomplete")
            q=dst/"imported_controls/projected_mean"/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            shutil.copy2(meta,q.with_suffix(".json"))
            manifest["imported"][f"clean/projected_mean/{task}/{seed}"]={"path":str(q.relative_to(dst)),"sha256":sha256(q)}
    atomic_json(dst/"manifest.json",manifest)
    install_repo_on_path()
    import torch
    from PIL import Image
    from projection_common import numeric_agreement
    from projection_validation import build_verified_model,load_source_image,precision_mode,runtime_info
    from medvae.utils.loaders import load_2d_finetune_aspect
    helper=import_source(HERE/"references/augment_and_extract.py","medvae_original_augmentation")
    require(helper.PRESETS["moderate"]==plan["preset"],"Augmentation preset changed")
    with precision_mode(torch,"ieee"):
        check_environment(environment(torch,args.device),manifest["control_environment"],"Optional augmentation needs the same runtime as its controls")
    model,record=build_verified_model(CONFIG,args.device)
    for key in ("sha256","config_sha256","implementation_sha256"):
        require(record[key]==om["checkpoints"][CONFIG][key],f"Different model/checkpoint: {key}")
    with precision_mode(torch,"ieee"):runtime=runtime_info(torch,args.device)
    runtime["loader_sha256"]=sha256(inspect.getfile(load_2d_finetune_aspect))
    anchor=dst/"extraction_environment.json"
    if anchor.exists():require(read(anchor)==runtime,"Augmentation extraction runtime changed")
    else:atomic_json(anchor,runtime)
    atomic_json(dst/"model_record.json",record)
    parts=dst/"augmentation_parts";parts.mkdir(exist_ok=True)
    limit=4 if args.phase=="pilot" else 407
    ledger_path=dst/"extraction_attempts.json";ledger=read(ledger_path) if ledger_path.exists() else []
    spent=sum(r["seconds"] for r in ledger);budget=args.max_hours*3600
    def valid(uid):
        p=parts/(uid+".npz");j=parts/(uid+".json")
        if not(p.exists() and j.exists()):return False
        a=read(j);require(a["plan_fingerprint"]==plan["fingerprint"] and a["sha256"]==sha256(p),"Augmentation part changed")
        return True
    ids=list(train["uuids"].astype(str))
    if args.phase=="run":
        require(all(valid(u) for u in ids[:4]),"Run the four-image augmentation extraction pilot first")
        estimate=spent+2*max(read(parts/(u+".json"))["elapsed_seconds"] for u in ids[:4])*sum(not valid(u) for u in ids)
        require(estimate<=budget,f"Conservative augmented extraction estimate {estimate/3600:.2f}h exceeds {args.max_hours:g}h; defer this optional experiment")
    with StudyLock(dst):
        for i,uid in enumerate(ids[:limit]):
            if valid(uid):continue
            require(spent<budget,"Augmentation extraction budget exhausted; no partial cache will be used for training")
            start=time.monotonic();status="failed"
            try:
                path=image_path(args.img_dir,uid)
                x,source=load_source_image(model,path)
                with torch.inference_mode(),precision_mode(torch,"ieee"):
                    z,_,zp=model.compute_latent_proj(x.to(args.device),sample_posterior=False)
                raw_check=numeric_agreement(z[0].cpu().numpy(),raw["latents"][i],atol=.001,rtol=.001,context=f"{uid} raw")
                projected=zp[0].cpu().numpy().mean(0,keepdims=True,dtype=np.float32)
                proj_check=numeric_agreement(projected,train["latents"][i],atol=.001,rtol=.001,context=f"{uid} projected mean")
                with Image.open(path) as im:base=im.convert("L")
                w,h=base.size;points=helper.landmarks_to_points(train["landmarks"][i],w,h).astype(np.float64)
                original_tilts=helper.endplate_tilts_px(points)
                seed=int(hashlib.sha256((uid+":augmentation0").encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed)
                latent_list=[train["latents"][i]];lm_list=[train["landmarks"][i]];names=[uid]
                attempts=0;max_tilt_difference=0.
                with tempfile.TemporaryDirectory(prefix="medvae_aug_") as temp:
                    while len(latent_list)<6:
                        require(attempts<100,f"Cannot produce five valid augmentations for {uid}; do not silently drop this case")
                        require(spent+(time.monotonic()-start)<budget,"Augmentation extraction budget reached")
                        attempts+=1
                        aug,pts=helper.augment_once(base,points,rng,**plan["preset"])
                        if aug is None:continue
                        tilt_delta=(helper.endplate_tilts_px(pts)-original_tilts+180)%360-180
                        difference=float(np.max(abs(tilt_delta)))
                        require(difference<1e-5,"Uniform-scale augmentation changed vertebral tilts")
                        max_tilt_difference=max(max_tilt_difference,difference)
                        q=Path(temp)/"image.jpg";aug.save(q,quality=95)
                        ax,_=load_source_image(model,q)
                        with torch.inference_mode(),precision_mode(torch,"ieee"):
                            _,_,aproj=model.compute_latent_proj(ax.to(args.device),sample_posterior=False)
                        feature=aproj[0].cpu().numpy().mean(0,keepdims=True,dtype=np.float32)
                        require(feature.shape==(1,96,96) and np.isfinite(feature).all(),"Invalid augmented projected feature")
                        names.append(uid+f"__aug{len(latent_list):02d}");latent_list.append(feature)
                        lm_list.append(helper.points_to_landmarks(pts,w,h))
                target=parts/(uid+".npz")
                atomic_npz(target,latents=np.stack(latent_list).astype(np.float32),uuids=np.asarray(names),
                    cobb=np.repeat(train["cobb"][i:i+1],6,axis=0),landmarks=np.stack(lm_list).astype(np.float32),parent_uuid=np.asarray([uid]*6))
                atomic_json(parts/(uid+".json"),dict(plan_fingerprint=plan["fingerprint"],sha256=sha256(target),
                    source=source,rng_seed=seed,attempts=attempts,accepted=5,raw_check=raw_check,projected_check=proj_check,
                    max_tilt_difference=max_tilt_difference,elapsed_seconds=time.monotonic()-start))
                status="ok"
                print(f"{uid}: five projected-mean augmentations; {i+1}/{limit}",flush=True)
            except Exception as exc:
                atomic_json(dst/"extraction_failure.json",dict(image_uuid=uid,error=str(exc),plan_fingerprint=plan["fingerprint"]))
                raise
            finally:
                seconds=time.monotonic()-start;spent+=seconds
                ledger.append(dict(image_uuid=uid,seconds=seconds,status=status));atomic_json(ledger_path,ledger)
        completed=sum(valid(u) for u in ids)
        estimate=spent+2*max(read(parts/(u+".json"))["elapsed_seconds"] for u in ids[:4])*(407-completed)
        atomic_json(dst/"extraction_budget.json",dict(completed_images=completed,total_images=407,spent_seconds=spent,
            conservative_total_seconds=estimate,budget_seconds=budget))
        print(f"Augmented extraction {completed}/407. Conservative total {estimate/3600:.2f}h.")
        if completed==407:
            all_parts=[arrays(parts/(u+".npz")) for u in ids]
            out={k:np.concatenate([a[k] for a in all_parts]) for k in ("latents","uuids","cobb","landmarks","parent_uuid")}
            out.update(representation=np.asarray("projected_mean"),checkpoint_sha256=np.asarray(record["sha256"]),precision_mode=np.asarray("ieee"))
            p=dst/"caches/augmented/projected_mean"/f"latents_{CONFIG}_train.npz";atomic_npz(p,**out)
            cache(p,channels=1);require(len(out["uuids"])==2442,"Wrong augmented training count")
            aug_cell=dict(paths=dict(cell["paths"],train=str(p)),sha256=dict(cell["sha256"],train=sha256(p)))
            manifest["cells"]={"augmented/projected_mean":aug_cell};manifest["status"]="ready"
            atomic_json(dst/"manifest.json",manifest)
            print("Ready for run_heads.py --pilot, then six augmented fits.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation-dir",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--img-dir",type=Path,required=True);p.add_argument("--phase",choices=("pilot","run"),required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--max-hours",type=float,default=2)
    run(p.parse_args())
