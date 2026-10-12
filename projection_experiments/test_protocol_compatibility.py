"""Regression tests for portable source verification; no GPU or Torch needed."""
import ast
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import analyse_projection_geometry as geometry
import projection_common as common
import run_projection_study as runner


ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT / "references"
FILES = {"training": "train_frontier.py", "geometry": "cobb_from_landmarks.py"}
ORIGINAL_PARSE = ast.parse


def parse_without_type_params(*args, **kwargs):
    """Use the function/class field layout found in Python 3.10 and 3.11."""
    tree = ORIGINAL_PARSE(*args, **kwargs)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node._fields = tuple(field for field in node._fields if field != "type_params")
    return tree


class ProtocolCompatibilityTests(unittest.TestCase):
    def test_bundled_sources_pass(self):
        for family, filename in FILES.items():
            with self.subTest(family=family):
                protocol = common.verify_protocol_source(REFERENCES / filename, family)
                self.assertEqual(common.sha256(REFERENCES / filename), protocol["files_sha256"][filename])

    def test_old_saved_hashes_reproduce_every_reported_mismatch(self):
        protocol = common.read_json(REFERENCES / "protocol.json")
        with patch.object(common.ast, "parse", side_effect=parse_without_type_params):
            for family, filename in FILES.items():
                names = protocol[family + "_definitions"]
                actual = common.definition_hashes(REFERENCES / filename, names)
                self.assertEqual([name for name in names if actual[name] != names[name]], list(names))

    def test_reference_comparison_passes_with_older_ast_layout(self):
        with patch.object(common.ast, "parse", side_effect=parse_without_type_params):
            for family, filename in FILES.items():
                common.verify_protocol_source(REFERENCES / filename, family)

    def test_comment_and_line_ending_changes_are_accepted_for_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            for family, filename in FILES.items():
                candidate = Path(tmp) / filename
                text = "# Local comment only.\n\n" + (REFERENCES / filename).read_text(encoding="utf-8")
                candidate.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
                common.verify_protocol_source(candidate, family)

    def test_changes_to_architecture_training_and_metric_are_rejected(self):
        original = (REFERENCES / FILES["training"]).read_text(encoding="utf-8")
        changes = [
            ("POOL = 4", "POOL = 3", "LatentBackbone"),
            ("torch.manual_seed(seed)", "torch.manual_seed(seed + 1)", "train_one"),
            ("return np.abs(pred - true).mean(axis=1)", "return np.abs(pred - true).sum(axis=1)", "cobb_per_image_error"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "trainer.py"
            for old, new, name in changes:
                with self.subTest(definition=name):
                    self.assertIn(old, original)
                    candidate.write_text(original.replace(old, new, 1), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, name):
                        common.verify_protocol_source(candidate, "training")

    def test_changed_geometry_is_rejected(self):
        original = (REFERENCES / FILES["geometry"]).read_text(encoding="utf-8")
        old = "score = sum(abs(cm[a[j], j]) for j in range(3))"
        self.assertIn(old, original)
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "geometry.py"
            candidate.write_text(original.replace(old, old.replace("sum(", "-sum("), 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "best_assignment"):
                common.verify_protocol_source(candidate, "geometry")

    def test_modified_reference_is_rejected_even_when_candidate_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy2(REFERENCES / "protocol.json", root / "protocol.json")
            for family, filename in FILES.items():
                reference = root / filename
                reference.write_bytes((REFERENCES / filename).read_bytes() + b"\n# changed reference\n")
                with self.assertRaisesRegex(ValueError, "pinned SHA256"):
                    common.verify_protocol_source(reference, family, references_dir=root)

    def test_missing_and_duplicate_audited_definitions_are_rejected(self):
        original = (REFERENCES / FILES["training"]).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "trainer.py"
            candidate.write_text(original.replace("def predict(", "def renamed_predict(", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing protocol definitions"):
                common.verify_protocol_source(candidate, "training")
            candidate.write_text(original + "\ndef predict(*args):\n    return None\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate protocol definitions"):
                common.verify_protocol_source(candidate, "training")

    def test_source_verification_does_not_execute_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "trainer.py"
            candidate.write_bytes((REFERENCES / FILES["training"]).read_bytes()
                                  + b'\nraise RuntimeError("must not execute during source check")\n')
            common.verify_protocol_source(candidate, "training")

    def test_dry_run_uses_bundled_trainer_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            ids = {split: [split + "0", split + "1"] for split in common.COUNTS}
            manifest = dict(status="ready", schema=3, study_kind="fresh_paired",
                            reference_reuse=dict(raw_keep=[], projected_keep=[]), expected_ids=ids, cells={})
            rng = np.random.default_rng(5)
            for variant, config in runner.planned_cells("primary", manifest):
                shape = common.SHAPES[config]
                entry = dict(paths={}, sha256={}, channels=shape[0], fp32_kib=int(np.prod(shape)) / 256)
                manifest["cells"].setdefault(variant, {})[config] = entry
                for split, uuids in ids.items():
                    path = common.cache_path(study / "caches" / variant, config, split)
                    common.atomic_npz(path, latents=rng.normal(size=(2, *shape)).astype("f4"),
                                      uuids=np.array(uuids), cobb=np.zeros((2, 3), dtype="f4"),
                                      landmarks=np.zeros((2, 136), dtype="f4"))
                    entry["paths"][split] = str(path)
                    entry["sha256"][split] = common.sha256(path)
            common.atomic_json(study / "manifest.json", manifest)
            before = {p: common.sha256(p) for p in study.rglob("*") if p.is_file()}
            cmd = [sys.executable, str(ROOT / "run_projection_study.py"),
                   "--study-dir", str(study), "--phase", "primary", "--dry-run"]
            result = subprocess.run(cmd, cwd=tmp, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Requested 80 fits", result.stdout)
            self.assertFalse((study / "runs").exists())
            self.assertEqual(before, {p: common.sha256(p) for p in study.rglob("*") if p.is_file()})

    def test_geometry_cli_defaults_to_bundled_reference(self):
        argv = ["analyse_projection_geometry.py", "--study-dir", "unused", "--img-dir", "unused"]
        with patch.object(sys, "argv", argv), patch.object(geometry, "analyse") as analyse:
            geometry.main()
        self.assertEqual(analyse.call_args.args[0].geometry_source, REFERENCES / FILES["geometry"])


if __name__ == "__main__":
    unittest.main()
