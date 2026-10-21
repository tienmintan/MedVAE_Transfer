"""Return predictions and diagnostics; exclude latent caches, weights and original images."""
import argparse
import tempfile
import zipfile
from pathlib import Path

from common import *


def collect(args):
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    require(args.part_mb>=1,"Use a positive archive-part size")
    entries=[];inventory=[];names=set();plan_checks={}
    for root in args.studies:
        root=root.resolve();require(root.is_dir(),f"Study missing: {root}")
        require(root.name not in names,"Study directory basenames must be distinct in a return bundle");names.add(root.name)
        if (root/"plan.json").exists():
            try:
                plan=get_plan(root,check_source=False)
                plan_checks[root.name]=dict(plan_integrity="ok",current_scripts_match=plan["source_sha256"]==source_hashes())
            except (ValueError,KeyError,json.JSONDecodeError) as exc:
                # A diagnostic return must remain possible even when the broken plan is the problem.
                plan_checks[root.name]=dict(plan_integrity="failed",error=str(exc))
        for p in sorted(root.rglob("*")):
            if not p.is_file():continue
            rel=p.relative_to(root)
            if any(x in ("caches","__pycache__") for x in rel.parts):continue
            if p.suffix not in (".json",".csv",".npz",".png",".svg",".log",".txt",".md"):continue
            # Augmented per-image feature arrays are large; their audit JSON is sufficient.
            if "augmentation_parts" in rel.parts and p.suffix==".npz":continue
            # Native image files are excluded. Preview PNGs and prespecified example NPZs are included.
            if p.suffix in (".png",".svg") and "previews" not in rel.parts and "analysis" not in rel.parts:continue
            name="studies/"+root.name+"/"+str(rel)
            entries.append((p,name));inventory.append(dict(archive_path=name,bytes=p.stat().st_size,sha256=sha256(p)))
    for p in sorted(HERE.rglob("*")):
        if p.is_file() and p.suffix in (".py",".json",".md") and "__pycache__" not in p.parts:
            name="scripts/medvae_followup/"+str(p.relative_to(HERE))
            entries.append((p,name));inventory.append(dict(archive_path=name,bytes=p.stat().st_size,sha256=sha256(p)))
    for p in args.extra:
        require(p.is_file(),f"Extra diagnostic missing: {p}")
        require(p.suffix in (".json",".log",".txt"),"Only small diagnostic text files may be supplied via --extra")
        name="extra/"+p.name;entries.append((p,name));inventory.append(dict(archive_path=name,bytes=p.stat().st_size,sha256=sha256(p)))
    require(len({name for _,name in entries})==len(entries),"Duplicate archive name")
    manifest=out/"return_manifest.json"
    atomic_json(manifest,dict(files=inventory,plan_checks=plan_checks,study_completion={p.name:dict(heads=(p/"heads_complete.json").is_file(),
        analysis=(p/"analysis/analysis_complete.json").is_file()) for p in args.studies},
        note="This bundle can include incomplete studies for diagnosis. No latent caches, training checkpoints or source images are included."))
    entries.insert(0,(manifest,"return_manifest.json"))
    # Each part is an independent standard ZIP. No special split-archive software is required.
    cap=int(args.part_mb*1024*1024);parts=[];current=[];size=0
    for entry in entries:
        n=entry[0].stat().st_size
        require(n<=cap,f"A single file exceeds the requested part limit: {entry[0]}")
        if current and size+n>cap:parts.append(current);current=[];size=0
        current.append(entry);size+=n
    if current:parts.append(current)
    for i,part in enumerate(parts,1):
        target=out/f"medvae_followup_return_{i:02d}.zip"
        require(not target.exists(),f"Return archive already exists: {target}. Use a new --out-dir to preserve earlier returns.")
        with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p,name in part:z.write(p,name)
        with zipfile.ZipFile(target) as z:require(z.testzip() is None,"Return ZIP integrity failure")
        print(f"Return: {target} ({target.stat().st_size/1024/1024:.2f} MiB)")
    print("Upload every medvae_followup_return_*.zip from this directory. Incomplete runs can be returned for diagnosis too.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--studies",type=Path,nargs="+",required=True);p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--extra",type=Path,nargs="*",default=[]);p.add_argument("--part-mb",type=float,default=100)
    collect(p.parse_args())
