"""Check the existing environment without installing/upgrading training packages."""
import argparse
import importlib
import importlib.metadata
from pathlib import Path

from common import *


def run(args):
    install_repo_on_path()
    modules={"numpy":"numpy","scipy":"scipy","pandas":"pandas","matplotlib":"matplotlib","PIL":"Pillow","torch":"torch"}
    if args.stage in ("all","reconstruction","augmentation"):
        modules.update(torchvision="torchvision",monai="monai",huggingface_hub="huggingface-hub",medvae="local repository")
    if args.stage in ("all","reconstruction"):modules["skimage.metrics"]="scikit-image"
    status={};missing=[]
    for name,dist in modules.items():
        try:
            mod=importlib.import_module(name)
            status[name]={"import":"ok","version":getattr(mod,"__version__",None)}
        except Exception as exc:
            status[name]={"import":"failed","error":str(exc)};missing.append(dist)
    trainer_path()
    m=old_manifest(args.study_dir)
    required=[]
    for v in VARIANTS:
        for s in ("train","val","test"):
            p=old_cache(args.study_dir,m,v,s);required.append(str(p))
    # Image identities/dimensions are checked during preparation; this verifies paths early.
    for uid in m["expected_ids"]["test"]:
        image_path(args.img_dir,uid)
    record=dict(stage=args.stage,imports=status,cache_files_checked=len(required),trainer_sha256=TRAINER_SHA,
        study=str(args.study_dir.resolve()),img_dir=str(args.img_dir.resolve()),missing=missing)
    if "torch" not in missing:
        import torch
        from projection_validation import precision_mode,runtime_info
        require(not args.device.startswith("cuda") or torch.cuda.is_available(),"CUDA is unavailable")
        with precision_mode(torch,"ieee"):
            current=environment(torch,args.device);record["runtime"]=runtime_info(torch,args.device)
        reference=read(args.study_dir/"training_environment.json")
        record["historical_control_runtime_matches"]=(current==reference)
        record["runtime_differences"]={k:dict(expected=reference.get(k),actual=current.get(k)) for k in set(current)|set(reference) if current.get(k)!=reference.get(k)}
    atomic_json(args.output,record)
    print(json.dumps(record,indent=2))
    require(not missing,"Missing/broken imports: "+", ".join(missing)+". Resolve only these dependencies in the existing medvae environment; do not upgrade Torch/NumPy to silence this check.")
    print("Preflight passed. If historical_control_runtime_matches is false, use --control-mode refit in a new annotation output directory.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir",type=Path,required=True);p.add_argument("--img-dir",type=Path,required=True)
    p.add_argument("--stage",choices=("heads","reconstruction","augmentation","all"),default="all")
    p.add_argument("--device",default="cuda");p.add_argument("--output",type=Path,default=Path("outputs/followup_preflight.json"))
    run(p.parse_args())
