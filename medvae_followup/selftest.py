"""Small synthetic checks. These are software tests, never thesis results."""
import argparse
import tempfile
from pathlib import Path
import contextlib
import io

import numpy as np

from common import *


def run(args):
    checks=[]
    trainer_path();checks.append("byte-identical audited trainer")
    from analyse_heads import contrast_definitions
    definitions=contrast_definitions()
    assert len(definitions)==18 and sum(t=="landmark" for _,t,_ in definitions)==9
    assert all(sum(weights.values())==0 for _,_,weights in definitions)
    checks.append("18 primary and 9 secondary contrast definitions")
    true=np.zeros((2,136));true[:,68:]=np.linspace(0,1,68)
    pred=true.copy();pred[:,:68]+=.01
    assert np.allclose(errors("landmark",pred,true),.01)
    # W=100, H=200, 1 pixel lateral error / 200 pixel vertical extent = 0.5%.
    assert np.allclose(errors("landmark",pred,true,np.array([[100,200],[100,200]])),.5)
    assert paired_summary(np.zeros(128))["p_raw"]==1
    assert paired_summary(np.arange(128)-63.5)["mean_difference"]==0
    checks.append("paired image errors, isotropic units and null contrast")
    from reconstruction_metrics import map_points,masks,spectrum
    lm=[]
    for y in np.linspace(.12,.88,17):lm.extend([[.4,y-.01],[.6,y-.01],[.4,y+.01],[.6,y+.01]])
    pts=np.asarray(lm);lm=np.r_[pts[:,0],pts[:,1]].astype(np.float32)
    points=map_points(lm,200,400,384)
    assert np.allclose(points[:,0],(pts[:,0]*200+100)*384/400)
    e,c=masks(points,(384,384),np.ones((384,384),bool),384)
    assert e.any() and c.any() and not np.any(e&c)
    yy,xx=np.indices((384,384));signal=.2+.1*np.sin(2*np.pi*72*xx/384)
    spec,bands=spectrum(signal,signal,384)
    assert len(spec)==128 and len(bands)==3
    assert all(r["residual_energy"]==0 for r in spec)
    assert bands[1]["signal_energy"]>bands[2]["signal_energy"]
    checks.append("padding geometry, non-overlapping scaled regions and common-frequency bands")
    helper=import_source(HERE/"references/augment_and_extract.py","medvae_augmentation_selftest")
    from PIL import Image
    im=Image.fromarray(np.full((400,200),128,np.uint8))
    pix=helper.landmarks_to_points(lm,200,400)
    aug,p=helper.augment_once(im,pix,np.random.default_rng(0),**helper.PRESETS["moderate"])
    assert aug is not None and np.max(abs(helper.endplate_tilts_px(p)-helper.endplate_tilts_px(pix)))<1e-5
    checks.append("original moderate augmentation preserves endplate tilts")
    with tempfile.TemporaryDirectory(prefix="medvae_followup_test_") as temp:
        dst=Path(temp)/"study"
        plan=set_plan(dst,dict(kind="synthetic_software_test"))
        assert get_plan(dst)==plan
        try:set_plan(dst,dict(kind="changed"))
        except ValueError:pass
        else:raise AssertionError("Changed plan accepted")
        with StudyLock(dst):
            try:
                with StudyLock(dst):pass
            except ValueError:pass
            else:raise AssertionError("Concurrent lock accepted")
        assert not (dst/"run.lock").exists()
        checks.append("immutable plans and single-run locking")
    if args.metrics:
        from reconstruction_metrics import fidelity,ssim_zero_map
        x=np.random.default_rng(0).uniform(0,1,(96,50));y=np.clip(.85*x+.04,0,1)
        r=fidelity(x,y,384);same=fidelity(x,x,384)
        assert r["ssim_interior_max_abs"]<1e-8 and same["ssim_reference"]==1 and same["mse"]==0
        expected=-10*np.log10(np.mean((x-y)**2));assert abs(r["psnr_db"]-expected)<1e-10
        r2=fidelity(x,y,768);assert r2["ssim_scaled_window"]==23 and r2["ssim_scaled_sigma"]==3
        checks.append("scikit-image PSNR/SSIM reference, exact match, and scale-matched window")
    if args.with_torch:
        import torch
        torch.set_num_threads(2)
        from projection_validation import precision_mode,precision_snapshot
        before=precision_snapshot(torch)
        with precision_mode(torch,"ieee"):
            assert environment(torch,"cpu")["required_precision_mode"]=="ieee"
        assert before==precision_snapshot(torch)
        trainer=import_source(trainer_path(),"medvae_audited_trainer_selftest")
        rng=np.random.default_rng(42)
        for channels in (1,3):
            def data(n):
                return dict(latents=rng.normal(size=(n,channels,96,96)).astype(np.float32),
                    cobb=rng.uniform(10,50,(n,3)).astype(np.float32),landmarks=rng.uniform(.1,.9,(n,136)).astype(np.float32))
            train,val=data(6),data(3)
            for task in TASKS:
                with precision_mode(torch,"ieee"):
                    model,(mu,sd)=trainer.train_one(task,train,val,channels,seed=0,epochs=2,batch_size=3,lr=.001,weight_decay=.0001,patience=2,device="cpu")
                    pred=trainer.predict(model,mu,sd,val["latents"],"cpu")
                target=val["cobb" if task=="cobb" else "landmarks"]
                assert pred.shape==target.shape and np.isfinite(pred).all()
                assert np.allclose(errors(task,pred,target),trainer.per_image_error(task,pred,target),rtol=1e-6,atol=1e-7)
        checks.append("four real CPU training/prediction smoke fits: two tasks x one/three channels")
        if args.metrics:
            # Numerically compare the old Torch zero-padding formula with the audit implementation.
            import torch.nn.functional as F
            x=np.random.default_rng(3).uniform(.1,.9,(96,50)).astype(np.float32);y=.7*x+.05
            a,b=torch.from_numpy(x)[None,None],torch.from_numpy(y)[None,None]
            z=torch.arange(11,dtype=torch.float32)-5;g=torch.exp(-z*z/(2*1.5**2));g/=g.sum();win=(g[:,None]*g[None,:])[None,None]
            filt=lambda q:F.conv2d(q,win,padding=5)
            ma,mb=filt(a),filt(b);va,vb=filt(a*a)-ma*ma,filt(b*b)-mb*mb;cov=filt(a*b)-ma*mb
            old=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
            from reconstruction_metrics import ssim_zero_map
            assert np.max(abs(old[0,0].numpy()-ssim_zero_map(x.astype(float),y.astype(float))))<2e-5
            checks.append("old float32 Torch SSIM formula agrees with the float64 audit")
    print(json.dumps(dict(status="passed",checks=checks,scope="Synthetic software checks. No new research results or GPU MedVAE validation."),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics",action="store_true");p.add_argument("--with-torch",action="store_true")
    run(p.parse_args())
