"""Cache selection/refit regression tests. GPU calculations use explicit stand-ins."""
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
import pandas as pd

import prepare_projection_study as prep
import analyse_projection_study as analysis
from run_projection_study import planned_cells
from projection_common import (CONFIGS, MULTI, COUNTS, SHAPES, atomic_npz, cache_path,
    read_json, sha256, projected_reference_configs, validate_cell)
from projection_validation import (historical_cache_decision, projected_reuse_decision,
    comparison_report, compare_spot_arrays)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value, dtype=np.float32)
    def __add__(self, other): return Tensor(self.value + other.value)
    def __mul__(self, value): return Tensor(self.value * value)
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def project(values): return (values * np.float32(.5) + values) * np.float32(2)


class RecoveryChecks(unittest.TestCase):
    def test_uploaded_current_diagnosis_passes_fixed_but_not_strict(self):
        # Both current-mode reports from the user's diagnosis, not invented outputs.
        for sd, rms, maximum in ((6.980359812730266, .0003213561708085434, .017161846160888672),
                                 (9.75750309404694, .00029491225276095397, .011945247650146484)):
            report = dict(passed=False, expected_sd=sd, rms=rms, max_abs=maximum)
            self.assertFalse(historical_cache_decision(report, "strict")["passed"])
            decision = historical_cache_decision(report, "fixed")
            self.assertTrue(decision["passed"])
            self.assertFalse(decision["elementwise_passed"])
            self.assertFalse(report["passed"])

    def test_local_outlier_is_rejected_even_when_global_error_is_small(self):
        expected = np.linspace(-1, 1, 9216).reshape(1, 96, 96)
        actual = expected.copy(); actual[0, 0, 0] += .04
        report = comparison_report(actual, expected)
        decision = historical_cache_decision(report, "fixed")
        self.assertLess(decision["rms_over_cached_sd"], .001)
        self.assertFalse(decision["passed"])
        for a, b in ((np.ones(3), np.ones(3)), (np.ones(2), np.ones(3)),
                     (np.array([np.inf]), np.array([1.]))):
            self.assertFalse(historical_cache_decision(comparison_report(a, b), "fixed")["passed"])

    def test_reuse_requires_all_splits_and_keeps_fit_plan_consistent(self):
        rows = [dict(split=s, existing_projected_agreement=dict(passed=True),
                     existing_projected_row_order_matches_raw=True) for s in COUNTS]
        self.assertTrue(projected_reuse_decision(rows)["reuse"])
        self.assertFalse(projected_reuse_decision(rows, "never")["reuse"])
        with self.assertRaises(ValueError): projected_reuse_decision(rows[:2])
        rows[0]["existing_projected_agreement"]["passed"] = False
        self.assertFalse(projected_reuse_decision(rows)["reuse"])
        rows[0]["existing_projected_agreement"]["passed"] = True
        rows[0]["existing_projected_row_order_matches_raw"] = False
        self.assertFalse(projected_reuse_decision(rows)["reuse"])
        for reused, ncore in ((list(MULTI), 2), ([MULTI[1]], 3), ([], 4)):
            manifest = dict(schema=2, reference_reuse=dict(projected_keep=reused))
            self.assertEqual(len(planned_cells("core", manifest)), ncore)
            self.assertEqual(len(planned_cells("pooling", manifest)), 4)
        with self.assertRaises(ValueError): projected_reference_configs(dict(schema=2))

    def test_preparation_selects_whole_config_source_and_excludes_old_scores(self):
        rng = np.random.default_rng(8)
        ids = pd.read_csv(Path(__file__).parent / "references/raw_frontier_per_image.csv").image_uuid.unique().tolist()
        expected = dict(train=["a", "b"], val=["c", "d"], test=ids)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rawdir = root/"raw"; olddir = root/"old"; out = root/"study"
            hashes = {}
            for config in CONFIGS:
                for split, uuids in expected.items():
                    z = rng.normal(size=(len(uuids), *SHAPES[config])).astype("f4")
                    z.reshape(len(z), -1)[:, 0] = 0
                    data = dict(uuids=np.array(uuids), cobb=np.zeros((len(z), 3), dtype="f4"),
                                landmarks=np.zeros((len(z), 136), dtype="f4"))
                    path = cache_path(rawdir, config, split)
                    atomic_npz(path, latents=z, **data); hashes[path] = sha256(path)
                    if config in MULTI:
                        values = project(z) + np.float32(.5 if config == MULTI[0] and split == "train" else 1e-5)
                        path = cache_path(olddir, config, split)
                        atomic_npz(path, latents=values, representation=np.array("projected"), **data)
                        hashes[path] = sha256(path)
            args = argparse.Namespace(out_dir=out, raw_dir=rawdir, existing_projected_dir=olddir,
                split_csv=root/"splits.csv", img_dir=root, device="cpu", precision="current",
                raw_cache_policy="fixed", projected_reuse="auto", spot_checks=2, batch_size=8,
                atol=.001, rtol=.001)
            args.split_csv.write_text("test fixture\n")
            fake_torch = types.SimpleNamespace(__version__="test-only", no_grad=nullcontext,
                as_tensor=lambda values, device: Tensor(values))
            model = types.SimpleNamespace(channel_ds=lambda z: z*.5, channel_proj=lambda z: z*2)
            def spot(model, x, cached, device, atol, rtol, check_encode):
                fresh = cached.copy(); fresh.reshape(-1)[0] += np.float32(.005)
                reports = compare_spot_arrays(fresh, cached, project(fresh), project(fresh), project(cached), atol, rtol)
                reports["encode_mode_vs_api_raw_same_input"] = comparison_report(fresh, fresh)
                return reports, {}
            with patch.dict(sys.modules, {"torch":fake_torch}), \
                 patch.object(prep, "split_ids", return_value=expected), \
                 patch.object(prep, "runtime_info", return_value={"test_only":True}), \
                 patch.object(prep, "build_verified_model", return_value=(model, {"sha256":"fixture"})), \
                 patch.object(prep, "load_source_image", return_value=(None, {})), \
                 patch.object(prep, "collect_spot", side_effect=spot), redirect_stdout(io.StringIO()):
                prep._prepare_impl(args)
            manifest = read_json(out/"manifest.json")
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["historical_reencoding_summary"]["elementwise_failures"], 24)
            self.assertEqual(projected_reference_configs(manifest), [MULTI[1]])
            self.assertEqual(len(planned_cells("core", manifest)), 3)
            for config in MULTI:
                prepared = validate_cell(manifest, "projected_keep", config)
                means = validate_cell(manifest, "projected_mean", config)
                for split in COUNTS:
                    with np.load(cache_path(rawdir, config, split)) as raw:
                        canonical = project(raw["latents"])
                    target = canonical if config == MULTI[0] else canonical + np.float32(1e-5)
                    np.testing.assert_array_equal(prepared[split]["latents"], target)
                    np.testing.assert_array_equal(means[split]["latents"], target.mean(1, keepdims=True, dtype=np.float32))
            for path, digest in hashes.items(): self.assertEqual(sha256(path), digest)
            rows, values, seed_sd = analysis.load_results(out, manifest)
            self.assertEqual(set(rows.loc[rows.variant == "projected_keep", "config"]), {MULTI[1]})
            self.assertNotIn(("projected_keep", MULTI[0], "cobb"), values)
            self.assertNotIn(("projected_keep", MULTI[0], "cobb"), seed_sd)


if __name__ == "__main__": unittest.main()
