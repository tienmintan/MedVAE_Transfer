"""CPU-only checks of pairing, cache integrity, geometry and driver bookkeeping.

The driver test uses a deterministic stand-in trainer. It does not exercise
PyTorch, MedVAE checkpoint loading, encoder inference or actual optimisation.
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

import analyse_projection_study as analysis
import run_projection_study as runner
from analyse_projection_geometry import geometry_errors, pixel_scale
from projection_common import (SHAPES, SINGLE, TASKS, align_to, atomic_json, atomic_npz,
    checked_manifest, error_vectors, import_source, load_cache, numeric_agreement,
    paired, read_json, require, sha256)


HERE = Path(__file__).parent


class ProjectionChecks(unittest.TestCase):
    def test_reference_statistics_are_paired_by_id(self):
        raw = pd.read_csv(HERE/"references/raw_frontier_per_image.csv")
        proj = pd.read_csv(HERE/"references/projected_frontier_per_image.csv")
        a = raw.query("config == 'medvae_4_3_2d' and task == 'cobb'").set_index("image_uuid").error
        b = proj.query("config == 'medvae_4_3_2d' and task == 'cobb'").sample(frac=1,random_state=3).set_index("image_uuid").error
        result = paired(a.to_numpy(), b.loc[a.index].to_numpy(), resamples=1000)
        self.assertAlmostEqual(result["p"], stats.ttest_rel(b.loc[a.index],a).pvalue, places=12)
        self.assertAlmostEqual(result["mean_b"], 7.7511363, places=5)
        self.assertGreater(result["p"],.05)

    def test_cache_alignment_and_changed_labels_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"cache.npz"; rng=np.random.default_rng(1)
            data = dict(latents=rng.normal(size=(2,1,96,96)).astype("f4"), uuids=np.array(["b","a"]),
                        cobb=np.array([[1,2,3],[4,5,6]],dtype="f4"),landmarks=rng.random((2,136),dtype=np.float32))
            atomic_npz(p,**data); loaded=load_cache(p,SINGLE[0],["a","b"])
            ref={k:v[::-1] for k,v in data.items()}; out=align_to(ref,loaded)
            np.testing.assert_array_equal(out["latents"],ref["latents"])
            loaded["cobb"][0,0] += 1
            with self.assertRaisesRegex(ValueError,"labels differ"):align_to(ref,loaded)
            data["uuids"] = np.array(["a","a"]); atomic_npz(p,**data)
            with self.assertRaisesRegex(ValueError,"Duplicate"):load_cache(p,SINGLE[0],["a","b"])

    def test_mean_channel_storage_and_noncommuting_projection(self):
        z=np.array([[[[2.,-2.]], [[-2.,2.]], [[2.,2.]]]],dtype="f4")
        project=lambda x: np.maximum(x,0)
        self.assertFalse(np.allclose(project(z).mean(1),project(z.mean(1))))
        one=z[:,:1]; np.testing.assert_array_equal(one.mean(1,keepdims=True),one)
        self.assertEqual(np.zeros((1,96,96),dtype="f4").nbytes/1024,36)
        self.assertEqual(np.zeros((1,48,48),dtype="f4").nbytes/1024,9)

    def test_incomplete_preparation_and_numeric_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            atomic_json(Path(tmp)/"manifest.json",dict(status="preparing"))
            with self.assertRaisesRegex(ValueError,"incomplete"):checked_manifest(tmp)
        with self.assertRaisesRegex(ValueError,"Numerical check failed"):
            numeric_agreement(np.ones(3),np.zeros(3),1e-3,1e-3,"test")

    def test_family_denominators_include_pending_tests(self):
        plan=analysis.comparison_plan()
        values={(v,c,t):np.arange(1.,129.) for v in ("raw_keep","projected_keep")
                for c in ("medvae_4_3_2d","medvae_8_4_2d") for t in TASKS}
        done,missing,families=analysis.evaluate_plan(values,plan,resamples=100)
        self.assertEqual(len(plan),38)
        self.assertEqual(families["projection_effect"],12)
        self.assertEqual(len(done),4)
        self.assertEqual(len(missing),34)
        self.assertTrue((done.family_size==12).all())
        self.assertTrue((done.p==1).all())

    def test_cross_task_bootstrap_reproduces_existing_followup(self):
        ref=pd.read_csv(HERE/"references/projected_frontier_per_image.csv")
        values={("projected_keep",c,t):sub.sort_values("image_uuid").error.to_numpy()
                for (c,t),sub in ref.groupby(["config","task"])}
        result=analysis.relative_task_changes(values).iloc[0]
        self.assertAlmostEqual(result.cobb_minus_landmark_pp,2.98165842000222,places=9)
        self.assertAlmostEqual(result.ci95_lo_pp,-3.54461521628672,places=9)
        self.assertAlmostEqual(result.ci95_hi_pp,9.640990632795326,places=9)

    def test_geometry_uses_pixels_and_fails_missing_dimensions(self):
        g=import_source(HERE/"references/cobb_from_landmarks.py","test_geometry_source")
        corners=np.array([[0.,0.],[10.,10.],[0.,4.],[10.,14.]])
        V=np.tile(corners,(1,17,1,1)); pts=V.reshape(1,68,2)
        pixels=np.concatenate([pts[:,:,0],pts[:,:,1]],axis=1)
        scale=np.array([[100]*68+[200]*68],dtype=float); normalised=pixels/scale
        true_theta=g.endplate_angles(g.to_vertebrae(normalised*scale),(0,1,2,3))
        wrong_theta=g.endplate_angles(g.to_vertebrae(normalised),(0,1,2,3))
        self.assertAlmostEqual(float(true_theta.mean()),45.)
        self.assertLess(float(wrong_theta.mean()),30.)
        gt=dict(landmarks=normalised,cobb=np.array([[1.,2.,3.]]))
        errors=geometry_errors(g,gt,gt,scale,"vmajor",(0,1,2,3),(1,0,2))
        for value in errors.values():np.testing.assert_array_equal(value,0.)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"Missing or ambiguous"):pixel_scale(["absent"],tmp)
            Image.new("L",(100,200)).save(Path(tmp)/"a.jpg")
            s,_=pixel_scale(["a"],tmp); np.testing.assert_array_equal(s,scale)

    def test_driver_ensembles_predictions_and_resumes(self):
        rng=np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            study=Path(tmp)
            manifest=dict(status="ready",expected_ids={s:[s+str(i) for i in range(n)]
                for s,n in (("train",4),("val",2),("test",6))},cells={"projected_keep":{}})
            for c in SINGLE:
                channels,h,w=SHAPES[c]; entry=dict(paths={},sha256={},channels=1,
                    fp32_kib=h*w*4/1024,compression_ratio=384**2/(h*w))
                for split,ids in manifest["expected_ids"].items():
                    path=study/f"{c}_{split}.npz"; n=len(ids)
                    atomic_npz(path,latents=rng.normal(size=(n,1,h,w)).astype("f4"),uuids=np.array(ids),
                        cobb=np.zeros((n,3),dtype="f4"),landmarks=np.zeros((n,136),dtype="f4"))
                    entry["paths"][split]=str(path);entry["sha256"][split]=sha256(path)
                manifest["cells"]["projected_keep"][c]=entry
            atomic_json(study/"manifest.json",manifest)
            calls=[]
            def train(task,train,val,in_ch,**kw):
                calls.append((task,kw["seed"]))
                return (task,kw["seed"]),(0,1)
            def predict(model,mu,sd,x,device):
                task,seed=model
                return np.full((len(x),3 if task=="cobb" else 136),seed-4.5,dtype="f4")
            def error(task,pred,true):
                if task=="cobb":return np.abs(pred-true).mean(1)
                d=pred-true;return np.sqrt(d[:,:68]**2+d[:,68:]**2).mean(1)
            fake=types.SimpleNamespace(train_one=train,predict=predict,per_image_error=error,
                N_LANDMARKS=68,TASK_OUT_DIM={"cobb":3,"landmark":136},
                TASK_SPEC={"cobb":{"kind":"reg","src":"cobb"},"landmark":{"kind":"reg","src":"landmarks"}})
            torch=types.SimpleNamespace(__version__="test-only",version=types.SimpleNamespace(cuda=None),
                cuda=types.SimpleNamespace(is_available=lambda:False),backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda:None)))
            args=argparse.Namespace(study_dir=study,trainer=HERE/"references/train_frontier.py",phase="core",
                seeds=list(range(10)),device="cpu",progress=0,dry_run=False,no_save_models=True)
            with patch.dict(sys.modules,{"torch":torch}),patch.object(runner,"import_source",return_value=fake),contextlib.redirect_stdout(io.StringIO()):
                runner.run(args); runner.run(args)
            self.assertEqual(len(calls),40)
            for c in SINGLE:
                path=study/"runs/projected_keep"/c
                analysis.checked_run(path)
                summary=pd.read_csv(path/"summary.csv")
                self.assertTrue((summary.error_mean==0).all())
                self.assertTrue((pd.read_csv(path/"per_seed.csv").error_mean>0).all())
            args.seeds=[10]
            with self.assertRaisesRegex(ValueError,"seeds 0 through 9"):runner.run(args)


if __name__=="__main__":unittest.main()
