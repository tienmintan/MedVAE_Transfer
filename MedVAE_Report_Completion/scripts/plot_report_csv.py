"""Generate report figures from saved CSVs. Does not train or load any model.

Run from the project directory containing outputs/:
  python MedVAE_Report_Completion/scripts/plot_report_csv.py --root . --out-dir report_figures
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Arc

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "savefig.facecolor": "white"})
BLUE, ORANGE, TEAL = "#276B96", "#BB633C", "#287B73"

def frame(root, name):
    d = pd.read_csv(root / name)
    if "config" in d:
        d["config"] = d.config.replace({k: "medvae_"+k+"_2d" for k in ["4_3","4_1","8_4","8_1"]})
    return d

def value(root, run, config, task):
    d = frame(root, f"outputs/{run}/frontier_summary.csv")
    s = d[(d.config == config) & (d.task == task)]
    if len(s) != 1 or int(s.iloc[0]["n"]) != 128:
        raise ValueError(f"Expected one summary over 128 images: {run}/{config}/{task}")
    return s.iloc[0]

def save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (stem+".pdf"), bbox_inches="tight")
    fig.savefig(out / (stem+".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out / (stem+".pdf"))

def annotate_bars(ax, bars, vals, tops=None, fmt=".3f", padding=.015):
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i,(bar,v) in enumerate(zip(bars, vals)):
        y = tops[i] if tops is not None else v
        ax.text(bar.get_x()+bar.get_width()/2, y+padding*span, format(v, fmt),
                ha="center", va="bottom", fontsize=9)

def pipeline(out):
    fig, ax = plt.subplots(figsize=(11.4,2.8))
    ax.set_xlim(-.2,15); ax.set_ylim(-.4,3.1); ax.axis("off")
    boxes = [
        (0,1.0,1.7,"Radiograph"),
        (2.1,1.0,1.9,"Aspect padding\n384 × 384"),
        (4.4,1.0,2.1,"Frozen\nMedVAE encoder"),
        (6.9,1.0,1.7,"Latent\n(C, H, W)"),
        (9.0,1.0,2.0,"Trainable\ntask head"),
        (12.1,1.85,2.3,"3 Cobb angles"),
        (12.1,.2,2.3,"68 landmarks"),
    ]
    for x,y,w,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,.65,boxstyle="round,pad=.08",
                                   facecolor="#EEF4F7",edgecolor=BLUE,linewidth=1.2))
        ax.text(x+w/2,y+.325,label,ha="center",va="center",fontsize=10)
    for left,right in zip(boxes[:4],boxes[1:5]):
        ax.annotate("",xy=(right[0]-.1,1.325),xytext=(left[0]+left[2]+.1,1.325),
                    arrowprops={"arrowstyle":"->","color":"#40505C","lw":1.5})
    for target in boxes[5:]:
        ax.annotate("",xy=(target[0]-.1,target[1]+.325),xytext=(11.15,1.325),
                    arrowprops={"arrowstyle":"->","color":"#40505C","lw":1.5})
    ax.add_patch(Rectangle((5.29,2.02),.33,.25,facecolor=BLUE))
    ax.add_patch(Arc((5.455,2.28),.24,.35,theta1=0,theta2=180,color=BLUE,lw=1.5))
    ax.text(5.45,2.79,"Swap encoder configuration",ha="center",fontsize=9,color=BLUE)
    ax.text(7.3,-.22,"Tasks are trained separately. Within each task, head design and training protocol are fixed; input channels match the latent.",
            ha="center",va="center",fontsize=8.5)
    save(fig,out,"fig_3_2_pipeline")

def ladder(path,out):
    d = pd.read_csv(path).set_index("rung").loc[["position","orientation","apex_orientation"]]
    fig, ax=plt.subplots(figsize=(6.5,3.8))
    x=np.arange(3); vals=d.rel_pct.to_numpy()
    bars=ax.bar(x, vals, color=[BLUE,TEAL,ORANGE],width=.6)
    ax.set_xticks(x,["Landmark\nposition","Endplate\norientation","End-vertebra\norientation"])
    ax.set_ylabel("Increase in error: 4_1 → 8_1 (%)")
    ax.set_ylim(0,max(vals)*1.3); ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    for b,v,p in zip(bars,vals,d.p_paired):
        ax.text(b.get_x()+b.get_width()/2,v+.25,f"+{v:.1f}%\np = {p:.3g}",ha="center",va="bottom")
    fig.tight_layout()
    save(fig,out,"fig_5_2_granularity")

def dissociation(root,out):
    f=frame(root,"outputs/fidelity/recon_fidelity_summary.csv").set_index("config")
    cs=["medvae_4_1_2d","medvae_8_4_2d"]; co=[value(root,"frontier",c,"cobb") for c in cs]
    fig,axs=plt.subplots(1,2,figsize=(7.3,3.7))
    s=f.loc[cs]
    a=axs[0]; vals=s.ssim.to_numpy()
    b=a.bar([0,1],vals,yerr=np.array([vals-s.ssim_lo,s.ssim_hi-vals]),
            color=[BLUE,ORANGE],capsize=4,width=.6)
    a.set_ylim(0,1.08); a.set_ylabel("SSIM (higher is better)")
    annotate_bars(a,b,vals,s.ssim_hi.to_numpy())
    a=axs[1]; vals=np.array([r.error_mean for r in co])
    lo=np.array([r.error_lo for r in co]);hi=np.array([r.error_hi for r in co])
    b=a.bar([0,1],vals,yerr=[vals-lo,hi-vals],color=[BLUE,ORANGE],capsize=4,width=.6)
    a.set_ylim(0,10.4);a.set_ylabel("Cobb MAE (degrees; lower is better)")
    annotate_bars(a,b,vals,hi)
    for a in axs:
        a.set_xticks([0,1],["4_1\n36 KiB","8_4\n36 KiB"]);a.grid(axis="y",alpha=.18);a.set_axisbelow(True)
    fig.suptitle("Both configurations: 16× compression; 95% image-bootstrap intervals",fontsize=10)
    fig.tight_layout()
    save(fig,out,"fig_6_1_dissociation")

def encoder_comparison(root,out):
    arms=[("DINOv2 raw\n3072 KiB","dino","dino_aug","dinov2-base"),
          ("DINOv2 + PCA-9\n36 KiB","dino_pca",None,"dinov2-base_pca9"),
          ("MedVAE 4_1\n36 KiB","frontier","aug","medvae_4_1_2d")]
    fig,axs=plt.subplots(1,2,figsize=(10,4.3))
    for ax,task in zip(axs,["cobb","landmark"]):
        upper=10.3 if task=="cobb" else .053
        ax.set_ylim(0,upper)
        for i,(label,run,aug,c) in enumerate(arms):
            for offset,rr,color,legend in [(-.19,run,BLUE,"Unaugmented"),(.19,aug,ORANGE,"Augmented")]:
                if rr is None:
                    ax.text(i+.19,upper*.1,"Not run",rotation=90,ha="center",va="bottom",fontsize=8,color="#666666")
                    continue
                r=value(root,rr,c,task)
                bs=ax.bar(i+offset,r.error_mean,width=.34,color=color,hatch="//" if i==0 else None,
                          edgecolor="#FFFFFF",yerr=[[r.error_mean-r.error_lo],[r.error_hi-r.error_mean]],
                          capsize=3,label=legend if i==0 else None)
                annotate_bars(ax,bs,[r.error_mean],[r.error_hi],fmt=".3f" if task=="cobb" else ".4f")
        ax.set_xticks(range(3),[v[0] for v in arms]);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
        ax.set_ylabel("Cobb MAE (degrees)" if task=="cobb" else "Landmark error (normalised)")
    a=frame(root,"outputs/dino_aug/frontier_per_image.csv")
    b=frame(root,"outputs/aug/frontier_per_image.csv")
    a=a[(a.task=="landmark")&(a.config=="dinov2-base")]
    b=b[(b.task=="landmark")&(b.config=="medvae_4_1_2d")]
    m=a.merge(b,on="image_uuid",suffixes=("_a","_b"),validate="one_to_one")
    if len(m)!=128:raise ValueError("Augmented landmark pairing must have 128 images.")
    p=float(ttest_rel(m.error_a,m.error_b).pvalue)
    axs[1].text(.5,.94,f"Augmented DINOv2 vs MedVAE: paired p = {p:.2f}",
                transform=axs[1].transAxes,ha="center",fontsize=9)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(.5,1.025))
    fig.text(.5,-.005,"Hatching identifies the raw DINOv2 representation with greater storage. Error bars: 95% image-bootstrap intervals.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=[0,0,1,.94])
    save(fig,out,"fig_7_1_encoder_comparison")
    return p

def augmentation(root,out):
    settings=[("Conservative","aug_cons"),("Moderate","aug_mod"),("Aggressive","aug")]
    fig,ax=plt.subplots(figsize=(7.1,3.8));ax.set_ylim(0,19)
    rows=[]
    for k,(task,col,label) in enumerate([("cobb",BLUE,"Cobb"),("landmark",ORANGE,"Landmarks")]):
        baseline=value(root,"frontier","medvae_4_1_2d",task).error_mean
        vals=[100*(1-value(root,run,"medvae_4_1_2d",task).error_mean/baseline) for _,run in settings]
        bars=ax.bar(np.arange(3)+(k-.5)*.36,vals,width=.34,color=col,label=label)
        annotate_bars(ax,bars,vals,fmt=".1f")
        rows += [{"task":task,"setting":s,"reduction_pct":float(v)} for (s,_),v in zip(settings,vals)]
    ax.set_xticks(np.arange(3),[s for s,_ in settings])
    ax.set_ylabel("Error reduction vs unaugmented 4_1 (%)")
    ax.legend(frameon=False);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,out,"fig_7_2_augmentation")
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--out-dir",type=Path,default=Path("report_figures"))
    p.add_argument("--ladder-csv",type=Path,default=Path(__file__).resolve().parents[1]/"data/mechanism_ladder.csv")
    p.add_argument("--figures",nargs="+",choices=["3.2","5.2","6.1","7.1","7.2"],
                   default=["3.2","5.2","6.1","7.1","7.2"])
    a=p.parse_args(); report={}
    if "3.2" in a.figures:pipeline(a.out_dir)
    if "5.2" in a.figures:ladder(a.ladder_csv,a.out_dir)
    if "6.1" in a.figures:dissociation(a.root,a.out_dir)
    if "7.1" in a.figures:report["augmented_landmark_p"]=encoder_comparison(a.root,a.out_dir)
    if "7.2" in a.figures:report["augmentation_reductions"]=augmentation(a.root,a.out_dir)
    (a.out_dir/"csv_figure_values.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
