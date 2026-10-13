"""Figure 5.1: existing bootstrap statistics plus anatomy masks on a reconstruction.

Does not recompute or replace the saved endplate CSVs. Requires the actual project,
its MedVAE environment/checkpoints, an original image, and the test landmark file.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_report_csv import save, frame, BLUE, ORANGE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--img-dir",type=Path,required=True)
    p.add_argument("--landmarks",type=Path,required=True)
    p.add_argument("--lookup",type=Path,default=Path("landmark_row_lookup_test_fixed.csv"))
    p.add_argument("--csv",type=Path,default=Path("medvae/data/boostnet.csv"))
    p.add_argument("--summary",type=Path,default=Path("outputs/freq_loss/endplate_summary.csv"))
    p.add_argument("--config",choices=["medvae_4_1_2d","medvae_8_1_2d"],default="medvae_8_1_2d")
    p.add_argument("--uuid")
    p.add_argument("--band-hw",type=float,default=2.5)
    p.add_argument("--cent-r",type=float,default=4.0)
    p.add_argument("--out-dir",type=Path,default=Path("report_figures"))
    a=p.parse_args()
    sys.path.insert(0,str(a.root.resolve()))
    import torch
    from medvae import MVAE
    from freq_loss_endplate import (transform_image,reconstruct,content_bbox,
                                    parse_landmark_row,map_landmarks,endplate_centroid_masks)
    def local(path):
        return path if path.is_absolute() else a.root/path
    df=pd.read_csv(local(a.csv));df=df[df.split=="test"]
    lookup=pd.read_csv(local(a.lookup)).set_index("image_uuid")
    if not lookup.index.is_unique:raise ValueError("Duplicate lookup UUID.")
    uuid=a.uuid or next(str(u) for u in df.image_uuid if u in lookup.index)
    if uuid not in set(df.image_uuid):raise ValueError("Illustration must use a test image.")
    lmfile=pd.read_csv(local(a.landmarks),header=None).to_numpy(dtype=float)
    row=int(lookup.loc[uuid,"landmark_row"])
    if lmfile.shape[1]!=136:raise ValueError("Expected all-x then all-y: 136 landmark columns.")
    x=transform_image(local(a.img_dir)/(uuid+".jpg"))
    device="cuda" if torch.cuda.is_available() else "cpu";x=x.to(device)
    model=MVAE(model_name=a.config,modality="xray").to(device).eval()
    model.requires_grad_(False)
    reconstruction=reconstruct(model,x)
    bbox,cm=content_bbox(x)
    if bbox is None:raise ValueError("No content region found.")
    points=map_landmarks(parse_landmark_row(lmfile[row]),bbox)
    ep,ct=endplate_centroid_masks(points,x.shape[-2:],cm.cpu().numpy(),a.band_hw,a.cent_r)
    if not ep.any() or not ct.any():raise ValueError("Empty anatomy mask.")
    summary=frame(a.root,str(a.summary)).set_index("config")
    cs=["medvae_4_1_2d","medvae_8_1_2d"];summary=summary.loc[cs]
    fig,axs=plt.subplots(1,2,figsize=(8.6,5),gridspec_kw={"width_ratios":[1,1.6]})
    ax=axs[0];ax.imshow(reconstruction.squeeze().detach().cpu().numpy(),cmap="gray",vmin=-1,vmax=1)
    overlay=np.zeros((*ep.shape,4))
    overlay[ep]=[.74,.35,.20,.60];overlay[ct]=[.14,.50,.67,.68]
    ax.imshow(overlay)
    ax.set_xlim(max(0,points[:,0].min()-22),min(ep.shape[1],points[:,0].max()+22))
    ax.set_ylim(min(ep.shape[0],points[:,1].max()+10),max(0,points[:,1].min()-10))
    ax.axis("off");ax.set_title("Decoded anatomy with masks\n"+a.config.replace("medvae_","").replace("_2d",""),fontsize=10)
    ax=axs[1];xpos=np.arange(2);w=.35
    for offset,prefix,color,label in [(-w/2,"endplate",ORANGE,"Endplate bands"),
                                      (w/2,"centroid",BLUE,"Body-centroid discs")]:
        means=summary[prefix+"_mean"].to_numpy()
        lo=summary[prefix+"_lo"].to_numpy();hi=summary[prefix+"_hi"].to_numpy()
        ax.bar(xpos+offset,means,width=w,color=color,label=label,
               yerr=[means-lo,hi-means],capsize=4)
    ax.set_xticks(xpos,["4_1","8_1"]);ax.set_ylabel("Mean squared reconstruction error")
    ax.set_title("Existing measurements on the test set",fontsize=10)
    ax.legend(frameon=False,fontsize=9);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.015,"Error bars: 95% image-bootstrap intervals. Illustration uses the original band/disc construction.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1])
    save(fig,a.out_dir,"fig_5_1_endplate_concentration")
    print(f"Illustration: {uuid}, config {a.config}, landmark row {row}")
    print("Inspect mask alignment. The content-bounding-box mapping matches freq_loss_endplate.py.")

if __name__=="__main__":
    main()
