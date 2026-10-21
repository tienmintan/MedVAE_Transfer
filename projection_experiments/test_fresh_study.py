"""Test paired data provenance and fit bookkeeping. GPU inference uses stand-ins."""
import argparse
from contextlib import nullcontext, redirect_stdout
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import prepare_fresh_projection_study as fresh
import run_projection_study as runner
import analyse_projection_study as analysis
from projection_common import (CONFIGS, COUNTS, MULTI, SHAPES, atomic_npz, cache_path,
    error_vectors, read_json, reference_configs, sha256, validate_cell)
from projection_validation import NumericalMismatch, comparison_report, precision_snapshot
from test_projection_validation import new_backend


class Tensor:
    def __init__(self, value): self.value=np.asarray(value, dtype=np.float32)
    def to(self, **kwargs): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def mode(self): return self
    def __add__(self, other): return Tensor(self.value+other.value)


class FreshStudyChecks(unittest.TestCase):
    def test_raw_and_projected_are_from_one_api_call(self):
        calls=[]
        def compute(x, sample_posterior):
            self.assertFalse(sample_posterior);calls.append(1)
            return x, None, Tensor(x.value+1)
        model=types.SimpleNamespace(compute_latent_proj=compute, encode=lambda x:x,
            channel_ds=lambda z:Tensor(np.ones_like(z.value)), channel_proj=lambda z:z)
        torch=types.SimpleNamespace(no_grad=nullcontext,float32="f4")
        x=Tensor(np.arange(12).reshape(1,3,2,2))
        with patch.dict(sys.modules,{"torch":torch}):
            raw,projected,checks=fresh.extract_pair(model,x,"cpu",check_api=True)
        self.assertEqual(len(calls),1)
        np.testing.assert_array_equal(projected,raw+1)
        self.assertTrue(all(r["passed"] for r in checks.values()))

    def test_fresh_plan_and_reference_exclusion(self):
        m=dict(schema=3,study_kind="fresh_paired",reference_reuse=dict(raw_keep=[],projected_keep=[]))
        self.assertEqual(reference_configs(m,"raw"),[])
        self.assertEqual(reference_configs(m,"projected"),[])
        for phase,n in (("primary",4),("core",8),("pooling",4),("all",12)):
            self.assertEqual(len(runner.planned_cells(phase,m)),n)
        self.assertTrue(set(runner.planned_cells("primary",m)).issubset(runner.planned_cells("core",m)))
        m["reference_reuse"]["raw_keep"]=[CONFIGS[0]]
        with self.assertRaisesRegex(ValueError,"cannot reuse"):reference_configs(m,"raw")

    def make_fixture(self,root):
        ids={s:[s+str(i) for i in range(n)] for s,n in (("train",4),("val",2),("test",6))}
        hashes={};rng=np.random.default_rng(11)
        for config in CONFIGS:
            for split,uuids in ids.items():
                n=len(uuids)
                values=rng.normal(size=(n,*SHAPES[config])).astype("f4")+100
                cobb=np.arange(n*3,dtype="f4").reshape(n,3)/10
                landmarks=np.arange(n*136,dtype="f4").reshape(n,136)/10000
                path=cache_path(root/"labels",config,split)
                atomic_npz(path,latents=values[::-1],uuids=np.array(uuids[::-1]),
                           cobb=cobb[::-1],landmarks=landmarks[::-1])
                hashes[path]=sha256(path)
        args=argparse.Namespace(out_dir=root/"study",label_cache_dir=root/"labels",split_csv=root/"split.csv",
            img_dir=root,device="cpu",precision="ieee",spot_checks=2)
        args.split_csv.write_text("fixture\n")
        return ids,hashes,args

    def prepare_fixture(self,root,fail_api=False):
        ids,hashes,args=self.make_fixture(root)
        torch=types.SimpleNamespace(__version__="test-only")
        def build(config,device):return config,dict(sha256="checkpoint"+config)
        def source(config,path):return (config,str(path.stem)),dict(file_sha256=str(path.stem))
        def pair(config,x,device,check_api):
            _,uuid=x;c,h,w=SHAPES[config]
            raw=(np.arange(c*h*w,dtype="f4").reshape(1,c,h,w)/(c*h*w)+int(uuid[-1])/10).astype("f4")
            projected=2*raw
            reports={}
            if check_api:
                reports=dict(formula_vs_api_same_fresh_z=comparison_report(projected+(1 if fail_api else 0),projected),
                             encode_mode_vs_api_raw_same_input=comparison_report(raw,raw))
            return raw,projected,reports
        with patch.object(fresh,"split_ids",return_value=ids), \
             patch.object(fresh,"runtime_info",return_value={}), \
             patch.object(fresh,"build_verified_model",side_effect=build), \
             patch.object(fresh,"load_source_image",side_effect=source), \
             patch.object(fresh,"extract_pair",side_effect=pair),redirect_stdout(io.StringIO()):
            fresh._prepare(args,torch)
        return ids,hashes,args

    def test_failed_api_still_blocks_fresh_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(NumericalMismatch):self.prepare_fixture(root,True)
            self.assertEqual(read_json(root/"study/manifest.json")["status"],"failed")
            self.assertFalse((root/"study/caches").exists())

    def test_fresh_data_fit_precision_resume_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids,hashes,args=self.prepare_fixture(root)
            m=read_json(args.out_dir/"manifest.json")
            self.assertEqual(m["status"],"ready")
            self.assertEqual(m["reference_reuse"],dict(raw_keep=[],projected_keep=[]))
            self.assertTrue(all(not check["fresh_raw_vs_historical_raw"]["passed"]
                               for row in m["validation"] for check in row["spot_checks"]))
            for config in CONFIGS:
                raw=validate_cell(m,"raw_keep",config);proj=validate_cell(m,"projected_keep",config)
                for split in COUNTS:
                    self.assertEqual(raw[split]["uuids"].tolist(),ids[split])
                    self.assertLess(raw[split]["latents"].max(),3)
                    np.testing.assert_array_equal(proj[split]["latents"],2*raw[split]["latents"])
                if config in MULTI:
                    mean=validate_cell(m,"projected_mean",config)
                    np.testing.assert_array_equal(mean["test"]["latents"],proj["test"]["latents"].mean(1,keepdims=True))
            for path,digest in hashes.items():self.assertEqual(sha256(path),digest)
            with self.assertRaisesRegex(ValueError,"No completed results"):
                analysis.load_results(args.out_dir,m)
            torch=new_backend();torch.__version__="test-only"
            torch.version=types.SimpleNamespace(cuda=None)
            torch.cuda=types.SimpleNamespace(is_available=lambda:False)
            torch.backends.cudnn.version=lambda:None
            original_precision=precision_snapshot(torch);calls=[]
            def train(task,train,val,in_ch,**kw):
                self.assertTrue(all(x=="ieee" for x in precision_snapshot(torch)["values"].values()))
                calls.append(kw["seed"])
                return (task,float(train["latents"].mean())+kw["seed"]),(0,1)
            def predict(model,mu,sd,x,device):
                return np.full((len(x),3 if model[0]=="cobb" else 136),model[1],dtype="f4")
            def error(task,pred,true):
                if task=="cobb":return np.abs(pred-true).mean(1)
                delta=pred-true;return np.sqrt(delta[:,:68]**2+delta[:,68:]**2).mean(1)
            trainer=types.SimpleNamespace(train_one=train,predict=predict,per_image_error=error,
                N_LANDMARKS=68,TASK_OUT_DIM=dict(cobb=3,landmark=136),
                TASK_SPEC=dict(cobb=dict(kind="reg",src="cobb"),landmark=dict(kind="reg",src="landmarks")))
            runargs=argparse.Namespace(study_dir=args.out_dir,trainer=Path(__file__).parent/"references/train_frontier.py",
                phase="primary",seeds=list(range(10)),device="cpu",progress=0,dry_run=False,no_save_models=True)
            with patch.dict(sys.modules,{"torch":torch}),patch.object(runner,"import_source",return_value=trainer),redirect_stdout(io.StringIO()):
                runner.run(runargs);self.assertEqual(len(calls),80)
                runargs.phase="core";runner.run(runargs);self.assertEqual(len(calls),160)
                runner.run(runargs);self.assertEqual(len(calls),160)
            self.assertEqual(precision_snapshot(torch),original_precision)
            rows,values,_=analysis.load_results(args.out_dir,m)
            self.assertEqual(len(rows),8*2*6)
            self.assertEqual(len(values),16)
            self.assertEqual(set(rows.image_uuid),set(ids["test"]))
            self.assertEqual(read_json(args.out_dir/"training_environment.json")["required_precision_mode"],"ieee")


if __name__=="__main__":unittest.main()
