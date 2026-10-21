"""CPU regression checks for mismatch diagnosis and failure persistence.

Backend and preparation-flow checks use stand-ins. They do not validate GPU
arithmetic, MedVAE checkpoint loading or the cause of the reported discrepancy.
"""
import argparse
from contextlib import nullcontext
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import prepare_projection_study as prep
from diagnose_projection_mismatch import summarise_modes
from projection_common import atomic_npz, checked_manifest, sha256
from projection_validation import (NumericalMismatch, comparison_report, compare_spot_arrays,
                                   enforce, precision_mode, precision_snapshot)


def old_backend():
    return types.SimpleNamespace(backends=types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=True,benchmark=False,deterministic=False)))


def new_backend():
    return types.SimpleNamespace(backends=types.SimpleNamespace(fp32_precision="ieee",
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(fp32_precision="ieee")),
        cudnn=types.SimpleNamespace(fp32_precision="ieee",benchmark=False,deterministic=False,
            conv=types.SimpleNamespace(fp32_precision="tf32"),rnn=types.SimpleNamespace(fp32_precision="tf32"))))


class ValidationTests(unittest.TestCase):
    def test_small_rms_does_not_override_failed_elementwise_check(self):
        n=96*96; maximum=.017161846160888672; rms=.0003213561708085434
        background=np.sqrt((n*rms*rms-maximum*maximum)/(n-1))
        a=np.full((1,96,96),background); a[0,17,21]=maximum
        report=comparison_report(a,np.zeros_like(a))
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["rms"],rms)
        self.assertEqual(report["failing_elements"],1)
        self.assertEqual(report["worst_positions"][0]["index"],[0,17,21])
        with self.assertRaises(NumericalMismatch):enforce(report,"historical cache")

    def test_api_check_uses_same_latent_and_keeps_drift_separate(self):
        fresh=np.array([1.,2.,3.]); cached=fresh+.01
        reports=compare_spot_arrays(fresh,cached,2*fresh,2*fresh,2*cached,.001,.001)
        self.assertTrue(reports["formula_vs_api_same_fresh_z"]["passed"])
        self.assertFalse(reports["fresh_raw_vs_cached_raw"]["passed"])
        self.assertFalse(reports["projected_drift_from_different_raw_inputs"]["passed"])
        enforce(reports["formula_vs_api_same_fresh_z"],"API")

    def test_nonfinite_shape_and_zero_cases(self):
        self.assertFalse(comparison_report(np.ones(2),np.ones(3))["passed"])
        self.assertFalse(comparison_report(np.array([np.inf]),np.array([np.inf]))["passed"])
        self.assertTrue(comparison_report(np.zeros(3),np.zeros(3))["passed"])
        report=comparison_report(np.array([100.05,0.0005]),np.array([100.,0.]))
        self.assertTrue(report["passed"])

    def test_old_and_new_precision_apis_restore_state_after_exception(self):
        for torch in (old_backend(),new_backend()):
            original=precision_snapshot(torch)
            with precision_mode(torch,"current") as settings:self.assertEqual(settings,original)
            with precision_mode(torch,"ieee") as settings:
                self.assertTrue(all(v in (False,"ieee") for v in settings["values"].values()))
            self.assertEqual(precision_snapshot(torch),original)
            with self.assertRaisesRegex(RuntimeError,"example"):
                with precision_mode(torch,"tf32") as settings:
                    self.assertTrue(all(v in (True,"tf32") for v in settings["values"].values()))
                    raise RuntimeError("example")
            self.assertEqual(precision_snapshot(torch),original)

    def test_mode_summary_requires_every_selected_case_and_required_check(self):
        def row(mode,raw=True):
            return dict(mode=mode,comparisons={key:dict(passed=raw if key=="fresh_raw_vs_cached_raw" else True)
                for key in ("fresh_raw_vs_cached_raw","formula_vs_api_same_fresh_z","encode_mode_vs_api_raw_same_input")})
        report=dict(requested_modes=["current","ieee","tf32"],expected_checks_per_mode=2,
            checks=[row("current"),row("ieee"),row("ieee"),row("tf32"),row("tf32",False)])
        self.assertEqual(summarise_modes(report),["ieee"])

    def test_failure_is_persisted_and_input_cache_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rawdir=root/"raw";rawdir.mkdir();out=root/"study"
            path=rawdir/"latents_medvae_4_1_2d_train.npz"
            data=dict(latents=np.random.default_rng(2).normal(size=(2,1,96,96)).astype("f4"),
                uuids=np.array(["a","b"]),cobb=np.zeros((2,3),dtype="f4"),landmarks=np.zeros((2,136),dtype="f4"))
            atomic_npz(path,**data);digest=sha256(path)
            ids=pd.read_csv(Path(__file__).parent/"references/raw_frontier_per_image.csv").image_uuid.unique().tolist()
            expected=dict(train=["a","b"],val=["c"],test=ids)
            reports=compare_spot_arrays(np.ones(3),np.zeros(3),np.ones(3),np.ones(3),np.zeros(3),.001,.001)
            reports["encode_mode_vs_api_raw_same_input"] = comparison_report(np.ones(3), np.ones(3))
            args=argparse.Namespace(out_dir=out,raw_dir=rawdir,split_csv=root/"splits.csv",device="cpu",
                img_dir=root,existing_projected_dir=root/"projected",spot_checks=2,batch_size=8,
                precision="current",atol=.001,rtol=.001)
            args.split_csv.write_text("test fixture\n")
            torch=types.SimpleNamespace(__version__="test-only",no_grad=nullcontext)
            with patch.dict(sys.modules,{"torch":torch}),patch.object(prep,"split_ids",return_value=expected),\
                 patch.object(prep,"runtime_info",return_value={"test_only":True}),\
                 patch.object(prep,"build_verified_model",return_value=(object(),{"sha256":"fixture"})),\
                 patch.object(prep,"load_source_image",return_value=(None,{})),\
                 patch.object(prep,"collect_spot",return_value=(reports,{})):
                with self.assertRaises(NumericalMismatch):prep._prepare_impl(args)
            manifest=json.loads((out/"manifest.json").read_text())
            failure=json.loads((out/"numerical_failure.json").read_text())
            self.assertEqual(manifest["status"],"failed")
            self.assertEqual(failure["active_check"]["image_uuid"],"a")
            self.assertFalse(failure["comparison"]["passed"])
            self.assertEqual(sha256(path),digest)
            self.assertFalse((out/"caches").exists())
            with self.assertRaises(ValueError):checked_manifest(out)


if __name__=="__main__":unittest.main()
