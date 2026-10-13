"""Paired-resolution reconstruction audit. Decode RAW posterior latents only."""
import argparse
import inspect
import subprocess
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from common import *
from reconstruction_metrics import bbox, crop, fidelity, regional, spectrum, to01

RECON_CONFIGS = ["medvae_4_1_2d", "medvae_8_1_2d", "medvae_8_4_2d"]


def prepare(args):
    study, dst = args.study_dir.resolve(), args.out_dir.resolve()
    require(study != dst and study not in dst.parents, "Keep reconstruction outputs outside the original study")
    m = old_manifest(study)
    require(len(args.resolutions) == 2 and args.resolutions[0] == 384 and args.resolutions[1] > 384
        and all(s % 32 == 0 for s in args.resolutions), "Choose 384 and one larger multiple of 32 before running")
    plan = set_plan(dst, dict(kind="reconstruction", original_study=str(study), original_manifest_sha256=sha256(study / "manifest.json"),
        configs=RECON_CONFIGS, resolutions=args.resolutions, expected_ids=m["expected_ids"]["test"],
        input_rule="same centered square padding, bicubic antialiased resize, per-image intensity scaling; only target size changes",
        reconstruction="decode(encode(x).mode()); never decode projected features",
        sample_diagnostic="First 8 test identifiers, 4_1 and 8_1, at 384 only: one fixed seeded posterior sample per image; descriptive, not a variance estimate",
        channel_rule="score each decoded image channel and their grayscale mean; average BEFORE clipping, as in recon_fidelity.py",
        fidelity="[0,1], data_range=1; legacy zero-padded SSIM full map vs scikit-image Gaussian SSIM with population covariance and interior averaging; 11 pixels/sigma1.5 plus resolution-scaled window",
        regional="[-1,1] squared error; endplate half-width 2.5 and centroid radius 4 at 384, scaled with resolution; centroid excludes endplate; foreground input > -0.8",
        mapping="geometric centered-padding map is primary; threshold-box map retained as a separate legacy diagnostic",
        frequency="64 original corner-normalised bins plus shared cycles-per-canvas bins 0..192; record counts, signal and residual powers, floors, and pooled power ratios",
        inference="Descriptive paired effect sizes and unadjusted image-bootstrap intervals; no claim of a new confirmatory significance family",
        stopping="Pilot memory/runtime only; no replacement of cases after inspecting reconstruction scores"))
    labels = arrays(old_cache(study,m,"raw_keep","test"))
    atomic_npz(dst / "evaluation.npz", **{k: labels[k] for k in ("uuids","cobb","landmarks")})
    images=[]
    for uid in plan["expected_ids"]:
        p = image_path(args.img_dir,uid)
        from PIL import Image
        with Image.open(p) as im:
            w,h=im.size
        images.append(dict(image_uuid=uid,path=str(p.resolve()),sha256=sha256(p),width=w,height=h))
    atomic_json(dst / "input_images.json",images)
    atomic_json(dst / "manifest.json",dict(status="ready",plan_fingerprint=plan["fingerprint"],
        evaluation_sha256=sha256(dst / "evaluation.npz"), images_sha256=sha256(dst / "input_images.json"),
        checkpoints={c:m["checkpoints"][c] for c in RECON_CONFIGS}))
    print(f"Ready: {dst}. {len(RECON_CONFIGS)*2*128} deterministic reconstructions plus 16 sampled diagnostics. Run --phase pilot first.")


def record_path(dst, config, resolution, uid):
    return Path(dst) / "records" / config / str(resolution) / (uid + ".json")


def valid_record(path, plan):
    if not path.is_file():
        return False
    r = read(path)
    require(r["plan_fingerprint"] == plan["fingerprint"], f"Stale reconstruction record: {path}")
    require(r["record_fingerprint"] == fingerprint({k:v for k,v in r.items() if k != "record_fingerprint"}), f"Altered reconstruction record: {path}")
    return True


def preview(dst, config, resolution, uid, original, decoded, lm, width, height):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result,e,c,points=regional(original,decoded.mean(0),lm,width,height,resolution,"canvas_exact")
    fig, ax=plt.subplots(1,4,figsize=(11,5),constrained_layout=True)
    for a in ax: a.axis("off")
    ax[0].imshow(original,cmap="gray",vmin=-1,vmax=1);ax[0].set_title("Resized encoder input")
    ax[1].imshow(decoded.mean(0),cmap="gray",vmin=-1,vmax=1);ax[1].set_title("Decoded channel mean")
    ax[2].imshow(abs(original-np.clip(decoded.mean(0),-1,1)),cmap="magma",vmin=0,vmax=.5);ax[2].set_title("Absolute residual")
    ax[3].imshow(original,cmap="gray",vmin=-1,vmax=1)
    overlay=np.zeros((*original.shape,4));overlay[e]=[.85,.25,.15,.7];overlay[c]=[.05,.5,.85,.7]
    ax[3].imshow(overlay);ax[3].scatter(points[:,0],points[:,1],s=2,c="yellow")
    ax[3].set_title("Endplate (red), centre (blue)")
    fig.suptitle(f"{config} / {resolution}px / {uid}",fontsize=9)
    p=Path(dst)/"previews"/f"{config}_{resolution}_{uid}.png";p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=125);plt.close(fig)


def worker(args):
    dst=args.out_dir.resolve();plan=get_plan(dst);m=read(dst/"manifest.json")
    require(args.config in plan["configs"] and args.resolution in plan["resolutions"],"Undeclared reconstruction cell")
    require(sha256(dst/"evaluation.npz")==m["evaluation_sha256"] and sha256(dst/"input_images.json")==m["images_sha256"],"Changed audit inputs")
    install_repo_on_path()
    import torch
    import skimage
    from medvae.utils.loaders import load_2d_finetune_aspect
    from projection_validation import build_verified_model, precision_mode, runtime_info
    require(not args.device.startswith("cuda") or torch.cuda.is_available(),"CUDA unavailable; refusing silent CPU fallback")
    model,model_record=build_verified_model(args.config,args.device)
    old=m["checkpoints"][args.config]
    for key in ("sha256","config_sha256","implementation_sha256"):
        require(model_record[key]==old[key],f"Checkpoint/implementation changed: {key}")
    ckpt=torch.load(model_record["path"],map_location="cpu",weights_only=False)["state_dict"]
    decoder_keys=[k for k in model.state_dict() if k.startswith(("decoder.","post_quant_conv."))]
    require(bool(decoder_keys) and all(k in ckpt and torch.equal(v.cpu(),ckpt[k].cpu()) for k,v in model.state_dict().items() if k in decoder_keys),"Decoder weights do not match checkpoint")
    del ckpt
    with precision_mode(torch,"ieee"):
        runtime=runtime_info(torch,args.device)
    runtime.update(skimage=skimage.__version__,loader_sha256=sha256(inspect.getfile(load_2d_finetune_aspect)))
    anchor=dst/"reconstruction_environment.json"
    if anchor.exists():
        require(read(anchor)==runtime,"Reconstruction runtime/loader changed; use a new output directory")
    else: atomic_json(anchor,runtime)
    atomic_json(dst/"model_records"/(args.config+".json"),dict(**model_record,decoder_tensors_checked=len(decoder_keys)))
    ev=arrays(dst/"evaluation.npz");images=read(dst/"input_images.json")
    selected=images[:2] if args.pilot else images
    lm_lookup={str(u):v for u,v in zip(ev["uuids"],ev["landmarks"])}
    for entry in selected:
        uid=entry["image_uuid"];target=record_path(dst,args.config,args.resolution,uid)
        if valid_record(target,plan):continue
        start=time.monotonic();path=Path(entry["path"])
        require(sha256(path)==entry["sha256"],f"Image changed: {path}")
        x=load_2d_finetune_aspect(str(path),merge_channels=True,target_size=args.resolution).unsqueeze(0)
        require(tuple(x.shape)==(1,1,args.resolution,args.resolution) and torch.isfinite(x).all().item() and x.std().item()>0,"Image loader failed or returned a constant image")
        original=x[0,0].numpy().astype(np.float64)
        ch=int(model.encoder.conv_in.in_channels)
        x=x.repeat(1,ch,1,1).to(args.device)
        if args.device.startswith("cuda"):torch.cuda.reset_peak_memory_stats(args.device)
        diagnostic=args.resolution==384 and args.config in RECON_CONFIGS[:2] and uid in plan["expected_ids"][:8]
        decodes={};sample_seed=None
        with torch.inference_mode(),precision_mode(torch,"ieee"):
            posterior=model.encode(x)
            decodes["posterior_mode"]=model.decode(posterior.mode())[0].cpu().numpy()
            if diagnostic:
                sample_seed=int(hashlib.sha256((args.config+":"+uid+":sample0").encode()).hexdigest()[:8],16)
                torch.manual_seed(sample_seed)
                decodes["sample_seed0"]=model.decode(posterior.sample())[0].cpu().numpy()
        peak=int(torch.cuda.max_memory_allocated(args.device)) if args.device.startswith("cuda") else None
        fidelity_rows,region_rows,spectral_rows,band_rows=[],[],[],[];example=None
        original01=to01(original);box_fidelity=bbox(original01>original01.min()+1e-3)
        for sampling,decoded in decodes.items():
            require(decoded.ndim==3 and decoded.shape[1:]==original.shape and np.isfinite(decoded).all(),"Decoder output shape/values invalid; no silent resize is allowed")
            outputs={f"channel_{i}":decoded[i] for i in range(len(decoded))}
            outputs["grayscale_mean"]=decoded.mean(0,dtype=np.float64)
            for channel,recon in outputs.items():
                # The per-channel results are descriptive diagnostics; do not pick the best channel using test scores.
                fr=fidelity(crop(original01,box_fidelity),crop(to01(recon),box_fidelity),args.resolution)
                fr.update(sampling=sampling,channel=channel,clipped_fraction=float(np.mean((recon< -1)|(recon>1))),crop_box=list(box_fidelity))
                fidelity_rows.append(fr)
                for mapping in ("canvas_exact","legacy_content_box"):
                    rr,_,_,_=regional(original,recon,lm_lookup[uid],entry["width"],entry["height"],args.resolution,mapping)
                    region_rows.append(dict(rr,sampling=sampling,channel=channel))
                if channel=="grayscale_mean":
                    sr,br=spectrum(original,recon,args.resolution)
                    spectral_rows.extend(dict(r,sampling=sampling,channel=channel) for r in sr)
                    band_rows.extend(dict(r,sampling=sampling,channel=channel) for r in br)
            if sampling=="posterior_mode" and uid in plan["expected_ids"][:2]:
                preview(dst,args.config,args.resolution,uid,original,decoded,lm_lookup[uid],entry["width"],entry["height"])
                ep=dst/"examples"/f"{args.config}_{args.resolution}_{uid}.npz"
                atomic_npz(ep,input=original.astype(np.float32),decoded_channels=decoded.astype(np.float32),
                    landmarks=lm_lookup[uid],width=np.asarray(entry["width"]),height=np.asarray(entry["height"]),
                    resolution=np.asarray(args.resolution),image_uuid=np.asarray(uid))
                example=dict(path=str(ep.relative_to(dst)),sha256=sha256(ep))
        record=dict(plan_fingerprint=plan["fingerprint"],image_uuid=uid,config=args.config,resolution=args.resolution,
            input_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),source_image_sha256=entry["sha256"],
            decoded_channels=next(iter(decodes.values())).shape[0],elapsed_seconds=time.monotonic()-start,peak_cuda_bytes=peak,posterior_sample_seed=sample_seed,
            fidelity=fidelity_rows,regional=region_rows,spectra=spectral_rows,bands=band_rows,example=example)
        record["record_fingerprint"]=fingerprint(record);atomic_json(target,record)
        print(f"{args.config}/{args.resolution}: {uid}, {record['elapsed_seconds']:.1f}s, peak {peak}",flush=True)


def run(args):
    dst=args.out_dir.resolve();plan=get_plan(dst)
    cells=[(c,s) for c in plan["configs"] for s in plan["resolutions"]]
    count=2 if args.phase=="pilot" else 128
    with StudyLock(dst):
        ledger_path=dst/"attempts.json";ledger=read(ledger_path) if ledger_path.exists() else []
        spent=sum(x["seconds"] for x in ledger);budget=args.max_hours*3600
        pilot_records=[record_path(dst,c,s,u) for c,s in cells for u in plan["expected_ids"][:2]]
        if args.phase=="run":
            require(all(valid_record(p,plan) for p in pilot_records),"Run --phase pilot first")
            estimate=spent+sum(2*max(read(record_path(dst,c,s,u))["elapsed_seconds"] for u in plan["expected_ids"][:2])*
                sum(not valid_record(record_path(dst,c,s,u),plan) for u in plan["expected_ids"]) for c,s in cells)
            require(estimate<=budget,f"Conservative reconstruction estimate {estimate/3600:.2f}h exceeds {args.max_hours:g}h. Return the pilot; do not change cases based on scores.")
        for c,s in cells:
            if all(valid_record(record_path(dst,c,s,u),plan) for u in plan["expected_ids"][:count]):continue
            remaining=budget-spent;require(remaining>0,"Reconstruction time budget exhausted")
            log=dst/"logs"/f"{c}_{s}_{args.phase}.log";log.parent.mkdir(exist_ok=True)
            cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--phase","worker","--out-dir",str(dst),"--config",c,
                 "--resolution",str(s),"--device",args.device]
            if args.phase=="pilot":cmd.append("--pilot")
            print(f"Reconstructing {c} at {s}; log: {log}",flush=True)
            start=time.monotonic();status="interrupted"
            try:
                with log.open("w") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
                status="ok" if r.returncode==0 else "failed"
            except subprocess.TimeoutExpired:status="timeout"
            finally:
                seconds=time.monotonic()-start;spent+=seconds
                ledger.append(dict(config=c,resolution=s,phase=args.phase,status=status,seconds=seconds));atomic_json(ledger_path,ledger)
            require(status=="ok",f"Reconstruction {status}: {log}. For larger-resolution CUDA OOM, prepare a NEW directory with --resolutions 384 512. Never silently mix resolutions.")
        if all(p.is_file() for p in pilot_records):
            estimate=spent+sum(2*max(read(record_path(dst,c,s,u))["elapsed_seconds"] for u in plan["expected_ids"][:2])*
                sum(not record_path(dst,c,s,u).exists() for u in plan["expected_ids"]) for c,s in cells)
            atomic_json(dst/"budget_estimate.json",dict(conservative_total_seconds=estimate,spent_seconds=spent,budget_seconds=budget,
                peak_pilot_cuda_bytes=max((read(p)["peak_cuda_bytes"] or 0) for p in pilot_records)))
            print(f"Conservative total estimate {estimate/3600:.2f}h; budget {args.max_hours:g}h. Pilot images are reused.")
        print("Inspect the preview overlays before interpreting regional measurements.")


def analyse(dst):
    dst=Path(dst).resolve();plan=get_plan(dst)
    records=[]
    for c in plan["configs"]:
        for s in plan["resolutions"]:
            for uid in plan["expected_ids"]:
                p=record_path(dst,c,s,uid)
                require(valid_record(p,plan),f"Incomplete reconstruction audit: {p}");records.append(read(p))
    out=dst/"analysis";out.mkdir(exist_ok=True)
    frames={}
    for kind in ("fidelity","regional","spectra","bands"):
        rows=[dict(image_uuid=r["image_uuid"],config=r["config"],resolution=r["resolution"],**v) for r in records for v in r[kind]]
        write_csv(out/(kind+"_per_image.csv"),rows);frames[kind]=pd.DataFrame(rows)
    summaries=[]
    for kind,metrics in (("fidelity",["psnr_db","ssim_reference","ssim_legacy_zero_border","ssim_scale_matched"]),
                         ("regional",["endplate_mse","centroid_mse","regional_difference","regional_ratio"])):
        df=frames[kind]
        keys=["config","resolution","sampling","channel"]+(["mapping"] if kind=="regional" else [])
        for values,g in df.groupby(keys,sort=True):
            for metric in metrics:
                v=g[metric].to_numpy(float);good=np.isfinite(v)
                require(good.all() or metric in ("psnr_db","regional_ratio"),f"Invalid metric {metric}")
                r=dict(zip(keys,values));r.update(kind=kind,metric=metric,n=len(v),n_finite=int(good.sum()))
                if good.sum()>=2:
                    ci=paired_summary(v[good]);r.update(mean=ci["mean_difference"],ci_low=ci["ci_low"],ci_high=ci["ci_high"])
                summaries.append(r)
    write_csv(out/"metric_summary.csv",summaries)
    pooled=[]
    for kind,keys in (("spectra",["config","resolution","sampling","channel","axis","bin"]),
                      ("bands",["config","resolution","sampling","channel","band"])):
        for values,g in frames[kind].groupby(keys,sort=True):
            s,r=g.signal_energy.sum(),g.residual_energy.sum();row=dict(zip(keys,values))
            row.update(kind=kind,n_images=len(g),pooled_signal_energy=float(s),pooled_residual_energy=float(r),pooled_ratio=float(r/s) if s>0 else None)
            if kind=="spectra":
                row.update(mean_of_ratios=float(g.ratio_legacy_floor.mean()),floored_images=int(g.floor_applied.sum()),empty_bins=int((g["count"]==0).sum()))
            else:row["mean_of_ratios"]=float(g.ratio.mean())
            pooled.append(row)
    write_csv(out/"power_summary.csv",pooled)
    # Fixed paired comparisons. Export intervals, not a collection of nominal p-values.
    contrasts=[]
    for kind,metric in (("fidelity","psnr_db"),("fidelity","ssim_reference"),("fidelity","ssim_scale_matched"),("regional","regional_difference")):
        df=frames[kind];df=df[(df.sampling=="posterior_mode")&(df.channel=="grayscale_mean")]
        if kind=="regional":df=df[df.mapping=="canvas_exact"]
        series={(c,s):df[(df.config==c)&(df.resolution==s)].set_index("image_uuid")[metric].reindex(plan["expected_ids"]).to_numpy(float) for c in plan["configs"] for s in plan["resolutions"]}
        a,b=plan["resolutions"]
        comparisons=[(f"resolution_{c}_{b}_minus_{a}",series[c,b]-series[c,a]) for c in plan["configs"]]
        for s in plan["resolutions"]:
            comparisons += [(f"{c}_minus_medvae_4_1_2d_at_{s}",series[c,s]-series["medvae_4_1_2d",s]) for c in plan["configs"][1:]]
        for name,d in comparisons:
            require(np.isfinite(d).all(),"A paired metric is undefined; inspect exact-match/region flags instead of dropping cases")
            ci=paired_summary(d);ci.pop("p_raw")
            contrasts.append(dict(kind=kind,metric=metric,comparison=name,**ci))
    write_csv(out/"descriptive_paired_contrasts.csv",contrasts)
    atomic_json(out/"analysis_complete.json",dict(plan_fingerprint=plan["fingerprint"],n_mode_reconstructions=len(records),
        n_sample_diagnostics=16,record_sha256={str(record_path(dst,r["config"],r["resolution"],r["image_uuid"]).relative_to(dst)):sha256(record_path(dst,r["config"],r["resolution"],r["image_uuid"])) for r in records},
        notes=["128 paired images for the primary deterministic audit; eight per checkpoint for the sampled diagnostic.",
            "Larger input resolution changes the reconstruction problem. This does not measure a residual against the native original.",
            "The geometric mask map is a separate analysis; legacy content-box maps are not silently replaced.",
            "The old frequency/regional source sampled the posterior by default. The new mode-based results are not an exact replication of that run.",
            "All reconstruction contrasts are descriptive with unadjusted image-bootstrap intervals. Do not infer equivalence from intervals containing zero."]))
    print(f"Reconstruction analysis complete: {out}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("prepare","pilot","run","analyse","worker"),required=True)
    p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--study-dir",type=Path);p.add_argument("--img-dir",type=Path)
    p.add_argument("--resolutions",type=int,nargs=2,default=[384,768]);p.add_argument("--device",default="cuda")
    p.add_argument("--max-hours",type=float,default=8);p.add_argument("--config",help=argparse.SUPPRESS)
    p.add_argument("--resolution",type=int,help=argparse.SUPPRESS);p.add_argument("--pilot",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.phase=="prepare":
        p.error("--study-dir and --img-dir are required for prepare") if not(a.study_dir and a.img_dir) else prepare(a)
    elif a.phase=="worker":worker(a)
    elif a.phase=="analyse":analyse(a.out_dir)
    else:run(a)
