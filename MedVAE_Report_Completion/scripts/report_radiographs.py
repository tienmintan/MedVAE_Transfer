"""Figures 2.1 and 3.1 from original radiographs and saved ground-truth landmarks.

Only reads inputs. Uses the gt_landmarks/gt_cobb arrays, never model predictions.
Automatic example selection is a starting point; inspect it before publication.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from plot_report_csv import save, BLUE, ORANGE

NAMES = [f"T{i}" for i in range(1,13)] + [f"L{i}" for i in range(1,6)]

def locate(root, uuid):
    stem = Path(str(uuid)).stem
    for ext in [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".JPG"]:
        f=root/(stem+ext)
        if f.is_file():return f
    raise FileNotFoundError(f"Original image not found: {root}/{stem}.*")

def load_case(root, uuid, lm):
    path=locate(root,uuid)
    with Image.open(path) as im:
        arr=np.asarray(im.convert("L"))
    h,w=arr.shape
    lm=np.asarray(lm,dtype=float)
    if lm.shape!=(136,) or not np.isfinite(lm).all():
        raise ValueError(f"Invalid landmark array for {uuid}")
    pts=np.stack([lm[:68]*w,lm[68:]*h],axis=1)
    return arr,pts.reshape(17,4,2),path

def clinical_pair(vertices):
    top=vertices[:,1]-vertices[:,0]
    bot=vertices[:,3]-vertices[:,2]
    if np.any(np.linalg.norm(top,axis=1)<1e-8) or np.any(np.linalg.norm(bot,axis=1)<1e-8):
        raise ValueError("Degenerate endplate annotation; choose another illustrative case.")
    # Unoriented endplate lines: acute intersection angle, in pixel geometry.
    sup=np.degrees(np.arctan2(top[:,1],top[:,0]))
    inf=np.degrees(np.arctan2(bot[:,1],bot[:,0]))
    sup=(sup+90)%180-90;inf=(inf+90)%180-90
    d=np.abs((sup[:,None]-inf[None,:]+90)%180-90)
    ii,jj=np.triu_indices(17,k=1)
    k=np.argmax(d[ii,jj])
    u,l=int(ii[k]),int(jj[k])
    return u,l,float(d[u,l]),float(sup[u]),float(inf[l])

def plot_cobb(a,d,uuids):
    if a.uuids:
        i=list(uuids).index(a.uuids[0])
    else:
        major=d["gt_cobb"].max(1)
        i=int(np.argsort(major)[len(major)//2])
    arr,V,path=load_case(a.img_dir,uuids[i],d["gt_landmarks"][i])
    u,l,angle,theta1,theta2=clinical_pair(V)
    fig,axs=plt.subplots(1,2,figsize=(7.2,6.4),gridspec_kw={"width_ratios":[1.2,1]})
    ax=axs[0]; h,w=arr.shape
    ax.imshow(arr,cmap="gray");ax.set_xlim(0,w);ax.set_ylim(h,0)
    for idx,edge,color,label in [(u,V[u,:2],BLUE,"Upper end vertebra"),
                                  (l,V[l,2:],ORANGE,"Lower end vertebra")]:
        q=V[idx,[0,1,3,2,0]]
        ax.plot(q[:,0],q[:,1],color=color,lw=1.6)
        vec=edge[1]-edge[0]
        if abs(vec[0])<1e-8:raise ValueError("Vertical endplate; choose another example.")
        xx=np.array([0,w])
        yy=edge[0,1]+(xx-edge[0,0])*vec[1]/vec[0]
        ax.plot(xx,yy,color=color,lw=1.3,ls="--")
        ax.text(.02*w,float(V[idx,:,1].mean())-.045*h,f"{label}: {NAMES[idx]}",
                color="white",fontsize=8,ha="left",
                bbox={"facecolor":"black","alpha":.7,"edgecolor":"none","pad":2})
    ax.set_title("Ground-truth endplates in pixel space",fontsize=10)
    ax.axis("off")
    ax=axs[1];ax.set_aspect("equal");ax.set_xlim(-1.4,1.4);ax.set_ylim(-1.4,1.4);ax.axis("off")
    t1,t2=-theta1,-theta2
    while t2-t1>90:t2-=180
    while t2-t1< -90:t2+=180
    for t,color,label in [(t1,BLUE,"Superior endplate of upper vertebra"),
                            (t2,ORANGE,"Inferior endplate of lower vertebra")]:
        vec=np.array([np.cos(np.radians(t)),np.sin(np.radians(t))])
        ax.plot([-vec[0],vec[0]],[-vec[1],vec[1]],color=color,lw=2,label=label)
    lo,hi=sorted([t1,t2]);mid=np.radians((lo+hi)/2)
    ax.add_patch(Arc((0,0),1.0,1.0,theta1=lo,theta2=hi,color="#333333",lw=1.5))
    ax.text(.78*np.cos(mid),.78*np.sin(mid),f"{angle:.1f}°",ha="center",va="center",fontsize=13)
    ax.set_title("Same orientations, translated to intersect",fontsize=10)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.03),frameon=False,fontsize=8)
    ax.text(0,-1.15,f"Illustrated Cobb angle: {angle:.1f}°\nStored major angle: {d['gt_cobb'][i].max():.1f}°",
            ha="center",va="center",fontsize=10)
    fig.tight_layout()
    save(fig,a.out_dir,"fig_2_1_cobb_definition")
    pd.DataFrame([{"image_uuid":str(uuids[i]),"image":str(path),"upper":NAMES[u],"lower":NAMES[l],
                   "geometric_angle_deg":angle,"stored_major_deg":float(d["gt_cobb"][i].max())}]
                ).to_csv(a.out_dir/"fig_2_1_case.csv",index=False)
    print("The two angle values are reported separately; this illustration does not validate all annotation geometry.")

def lumbar_contrast(arr,V):
    # Robust intensity range inside the bounding box of L1-L5 annotations.
    q=V[12:].reshape(-1,2);h,w=arr.shape
    x0=max(0,int(np.floor(q[:,0].min())));x1=min(w,int(np.ceil(q[:,0].max()))+1)
    y0=max(0,int(np.floor(q[:,1].min())));y1=min(h,int(np.ceil(q[:,1].max()))+1)
    roi=arr[y0:y1,x0:x1]
    if roi.size<16:return np.nan
    return float(np.percentile(roi,90)-np.percentile(roi,10))

def examples(a,d,uuids):
    major=d["gt_cobb"].max(1);metadata=[]
    for i,u in enumerate(uuids):
        arr,V,path=load_case(a.img_dir,u,d["gt_landmarks"][i])
        metadata.append({"image_uuid":str(u),"row":i,"major_deg":float(major[i]),
                         "lumbar_p90_minus_p10":lumbar_contrast(arr,V),"path":str(path)})
    records=pd.DataFrame(metadata)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    records.to_csv(a.out_dir/"radiograph_candidates.csv",index=False)
    if a.uuids:
        if not 3<=len(a.uuids)<=4:raise ValueError("--uuids requires three or four UUIDs for examples.")
        selected=[list(uuids).index(u) for u in a.uuids]
        labels=["Selected example"]*len(selected)
    else:
        selected=[int(np.argmin(major)),int(np.argmax(major))]
        valid=records[~records.row.isin(selected)].dropna(subset=["lumbar_p90_minus_p10"])
        if len(valid)==0:raise ValueError("No valid low-contrast candidate.")
        selected.append(int(valid.sort_values("lumbar_p90_minus_p10").iloc[0].row))
        labels=["Lower-angle example","Higher-angle example","Low-contrast candidate"]
    fig,axes=plt.subplots(1,len(selected),figsize=(3.1*len(selected),6))
    for ax,i,title in zip(axes,selected,labels):
        arr,V,path=load_case(a.img_dir,uuids[i],d["gt_landmarks"][i])
        points=V.reshape(68,2)
        ax.imshow(arr,cmap="gray")
        ax.scatter(points[:,0],points[:,1],s=5,c="#39E2BC",edgecolors="black",linewidths=.15)
        ax.set_title(f"{title}\nStored major angle {major[i]:.1f}°",fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    save(fig,a.out_dir,"fig_3_1_dataset_examples")
    records.iloc[selected].to_csv(a.out_dir/"fig_3_1_selected_cases.csv",index=False)
    print("Review every overlay and the low-contrast candidate. The contrast score is only a selection heuristic.")
    print("Use --uuids with three or four chosen identifiers to replace the automatic selection.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["cobb","examples"],required=True)
    p.add_argument("--preds",type=Path,required=True)
    p.add_argument("--img-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,default=Path("report_figures"))
    p.add_argument("--uuids",nargs="+",help="Exact image identifiers from the prediction archive.")
    a=p.parse_args()
    # Load only trusted local project NPZ files: UUIDs can be object arrays.
    with np.load(a.preds,allow_pickle=True) as npz:
        d={k:npz[k] for k in ["uuids","gt_landmarks","gt_cobb"]}
    uuids=np.array([u.decode() if isinstance(u,bytes) else str(u) for u in d["uuids"]])
    if len(set(uuids))!=len(uuids):raise ValueError("Duplicate image identifiers.")
    if d["gt_landmarks"].shape!=(len(uuids),136) or d["gt_cobb"].shape!=(len(uuids),3):
        raise ValueError("Unexpected ground-truth dimensions.")
    if a.mode=="cobb":plot_cobb(a,d,uuids)
    else:examples(a,d,uuids)

if __name__=="__main__":
    main()
