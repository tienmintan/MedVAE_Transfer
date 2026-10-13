"""Figures 8.1 and 8.2 from saved generated and real latents.

No generator training or sampling. Labels mode trains the reference detector,
as the original diffusion_latent.py showlabels mode does. Writes only new figures.
Use --mode inspect first to check the generated archive's saved provenance.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_report_csv import save

def data(path):
    # Only load trusted project archives; UUIDs may be stored as object arrays.
    with np.load(path,allow_pickle=True) as z:
        return {k:z[k] for k in z.files}

def checked_generated(g):
    required=["latents","landmarks","aspects","guidance","ddim_steps","sample_sd"]
    missing=[k for k in required if k not in g]
    if missing:raise ValueError(f"Missing saved provenance/data: {missing}. Do not label this archive as guidance 3.")
    if not np.isclose(float(g["guidance"]),3):
        raise ValueError("This archive is not guidance 3. Regenerate in a separate directory or choose the correct saved archive.")
    if abs(float(g["sample_sd"])-1)>.3:
        raise ValueError("Sample SD fails the original utility-mode stability check.")
    if not np.isfinite(g["latents"]).all() or not np.isfinite(g["landmarks"]).all():
        raise ValueError("Non-finite generated data.")
    if len(g["latents"])!=len(g["landmarks"]) or len(g["aspects"])!=len(g["latents"]):
        raise ValueError("Generated arrays have inconsistent lengths.")

def decode(model,z,device):
    import torch
    outputs=[]
    # One image at a time keeps decoder-memory requirements modest.
    with torch.no_grad():
        for item in z:
            tensor=torch.tensor(item[None],dtype=torch.float32,device=device)
            answer=None;errors=[]
            for obj,name in [(model.model,"decode"),(model,"decode"),(model.model,"decode_first_stage")]:
                fn=getattr(obj,name,None)
                if fn is None:continue
                try:
                    answer=fn(tensor)
                    if isinstance(answer,(tuple,list)):answer=answer[0]
                    if torch.is_tensor(answer):break
                    answer=None
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
            if answer is None:raise RuntimeError("MedVAE decode failed: "+"; ".join(errors))
            outputs.append(answer.float().cpu().numpy()[0])
    return np.stack(outputs)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["inspect","samples","labels"],required=True)
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--generated",type=Path,required=True)
    p.add_argument("--latents-dir",type=Path)
    p.add_argument("--img-dir",type=Path)
    p.add_argument("--config",default="medvae_4_1_2d")
    p.add_argument("--out-dir",type=Path,default=Path("report_figures"))
    p.add_argument("--n-show",type=int,default=5)
    p.add_argument("--fid-epochs",type=int,default=100)
    p.add_argument("--detector-seed",type=int,default=0)
    a=p.parse_args()
    def local(path):
        return path if path.is_absolute() else a.root/path
    g=data(local(a.generated))
    print("Archive:",local(a.generated))
    print("Latents:",g["latents"].shape)
    for k in ["guidance","ddim_steps","sample_sd"]:
        print(k, g.get(k,"NOT SAVED"))
    print("Aspects saved:", "aspects" in g)
    if a.mode=="inspect":
        checked_generated(g)
        print("Metadata checks passed. This does not establish checkpoint identity or anatomical quality.")
        return
    checked_generated(g)
    if a.latents_dir is None:raise ValueError("--latents-dir is required for real test reconstructions.")
    te=data(local(a.latents_dir)/f"latents_{a.config}_test.npz")
    n=a.n_show
    if n<2 or n>6 or n>len(g["latents"]) or n>len(te["latents"]):
        raise ValueError("Use 2-6 samples, with enough real and generated examples.")
    if g["latents"].shape[1:]!=te["latents"].shape[1:]:
        raise ValueError("Generated/real latent shapes differ: check encoder config.")
    sys.path.insert(0,str(a.root.resolve()))
    import torch
    from medvae import MVAE
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=MVAE(model_name=a.config,modality="xray").to(device).eval()
    model.requires_grad_(False)
    gen=decode(model,g["latents"][:n],device).mean(1)
    real=decode(model,te["latents"][:n],device).mean(1)
    del model
    if device=="cuda":torch.cuda.empty_cache()
    fig,axes=plt.subplots(2,n,figsize=(2.2*n,6.5),squeeze=False)
    diagnostics=[]
    if a.mode=="labels":
        if a.img_dir is None:raise ValueError("--img-dir is required to map real landmarks to the padded canvas.")
        from diffusion_latent import _train_detector,_predict,_lm_err,to_canvas_centered
        from report_radiographs import locate
        from PIL import Image
        aspects=[]
        for u in te["uuids"][:n]:
            u=u.decode() if isinstance(u,bytes) else str(u)
            with Image.open(locate(local(a.img_dir),u)) as im:
                aspects.append(im.width/im.height)
        tr=data(local(a.latents_dir)/f"latents_{a.config}_train.npz")
        print("Training the real-data reference detector; this is not a plot-only operation.")
        det=_train_detector(tr["latents"].astype(np.float32),tr["landmarks"].astype(np.float32),
                            device,epochs=a.fid_epochs,seed=a.detector_seed)
        predictions=[_predict(det,g["latents"][:n].astype(np.float32),device),
                     _predict(det,te["latents"][:n].astype(np.float32),device)]
        targets=[g["landmarks"][:n],te["landmarks"][:n]]
        all_aspects=[g["aspects"][:n],np.asarray(aspects)]
        errors=[_lm_err(predictions[i],targets[i]) for i in range(2)]
    for row,images in enumerate([gen,real]):
        for i,pic in enumerate(images):
            ax=axes[row,i]
            ax.imshow(pic,cmap="gray",vmin=-1,vmax=1)
            title="Generated" if row==0 else "Real reconstruction"
            if a.mode=="labels":
                h,w=pic.shape
                for pts,col in [(targets[row][i],"#46DB8B"),(predictions[row][i],"#F15B53")]:
                    lm=to_canvas_centered(pts,all_aspects[row][i])
                    ax.scatter(lm[:68]*(w-1),lm[68:]*(h-1),s=6,c=col,edgecolors="black",linewidths=.2)
                title+=f"\ne = {errors[row][i]:.4f}"
                diagnostics.append({"row":"generated" if row==0 else "real","index":i,
                                    "normalised_detector_error":float(errors[row][i])})
            ax.set_title(title,fontsize=9);ax.axis("off")
    if a.mode=="labels":
        legend=[Line2D([],[],marker="o",ls="",color="#46DB8B",markersize=5,
                       label="Conditioning (generated) / ground truth (real)"),
                Line2D([],[],marker="o",ls="",color="#F15B53",markersize=5,label="Detector prediction")]
        fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,.045),ncol=2,frameon=False,fontsize=8)
        fig.text(.5,.015,"e: mean Euclidean landmark error in original-image normalised coordinates; detector agreement is not generated-image ground truth.",
                 ha="center",fontsize=7.5)
        stem="fig_8_2_conditioning_agreement"
        fig.tight_layout(rect=[0,.12,1,.95], h_pad=2.0)
    else:
        stem="fig_8_1_generated_vs_real"
        fig.text(.5,.025,"Same MedVAE decoder and display range for both rows; first saved samples, without quality-based selection.",
                 ha="center",fontsize=8)
        fig.tight_layout(rect=[0,.06,1,.95], h_pad=2.0)
    fig.suptitle(f"Guidance {float(g['guidance']):.1f}; {int(g['ddim_steps'])} DDIM steps",fontsize=11)
    save(fig,a.out_dir,stem)
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(a.out_dir/(stem+"_errors.csv"),index=False)
    print("Publication check: inspect anatomy, landmark mapping, and detector limitations; these are illustrative samples.")

if __name__=="__main__":
    main()
