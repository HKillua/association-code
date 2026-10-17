from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from association.dataset import make_synthetic_archive, prepare
from association.reference import train_reference
from association.runtime import load_records, read_json, write_json
from tools.experiments.cdf1_validation import ARM_ORDER, main, run_validation


class CDF1ValidationEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        config = read_json(Path(__file__).resolve().parents[1] / "configs/smoke.json")
        config["caps"] = dict.fromkeys(("T_core", "T_head", "V_select", "T_test"), 1)
        config["heads"].update(epochs=1, deltas=[1.], calibration_steps=2)
        for key in ("tcm_epochs", "tam_epochs", "warmup_epochs", "joint_completion"):
            config["reference"].pop(key)
        config["reference"].update(
            schedule="staged_updates_v2", purpose="fit", tcm_updates=1, tam_updates=1,
            joint_updates=1, validate_every_updates=1, completion_cache_mib=1,
            batch_tracks=4, pair_batch_size=4)
        cls.config_path = cls.root / "original-config.json"
        write_json(cls.config_path, config)
        archive = make_synthetic_archive(cls.root / "data.zip", n_scenes=12)
        manifest = prepare(archive, config, cls.root / "prepared", synthetic=True)
        cls.manifest_path = cls.root / "prepared/manifest.json"
        cls.reference = cls.root / "reference"
        train_reference(archive, manifest, config, "A0", "ce", 20, "cpu", cls.reference)
        # The entry must work using caches alone, with no source archive available.
        archive.unlink()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_completed_reference_produces_all_arms_without_test_or_archive(self):
        output = self.root / "validation"
        loaded_splits = []

        def cached(path, binding):
            loaded_splits.append(Path(path).name)
            self.assertIn(Path(path).name, ("T_head", "V_select"))
            return load_records(path, binding)

        with patch("tools.experiments.cdf1_validation.load_records", side_effect=cached), \
                patch("association.dataset.load_split", side_effect=AssertionError("No data loading")), \
                patch("association.pipeline.run_pilot", side_effect=AssertionError("No full pilot")):
            result = main(["--reference", str(self.reference), "--manifest", str(self.manifest_path),
                           "--config", str(self.config_path), "--device", "cpu", "--cpu-threads", "1",
                           "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual(loaded_splits, ["T_head", "V_select"])
        report = read_json(output / "validation.json")
        self.assertEqual(set(report["arms"]), set(ARM_ORDER))
        self.assertEqual(report["selection_split"], "V_select")
        self.assertFalse(report["test_opened"])
        self.assertIn("A_full", report["missing_comparisons"])
        expected = report["arms"]["B"]["validation"]["macro_f1"] - report["arms"]["A0"]["validation"]["macro_f1"]
        self.assertAlmostEqual(report["differences"]["B_minus_A0"]["macro_f1"], expected)
        self.assertEqual(read_json(output / "run.json")["status"], "COMPLETED")
        self.assertEqual(read_json(output / "heads/run.json")["status"], "COMPLETED")
        self.assertIn("not held-out test results", (output / "validation.md").read_text())
        with self.assertRaises(FileExistsError):
            run_validation(self.reference, self.manifest_path, self.config_path, "cpu", output)

    def test_unfinished_reference_is_rejected_before_head_training(self):
        incomplete = self.root / "incomplete"
        shutil.copytree(self.reference, incomplete)
        metadata = read_json(incomplete / "run.json")
        metadata["status"] = "RUNNING"
        write_json(incomplete / "run.json", metadata)
        output = self.root / "rejected"
        with patch("tools.experiments.cdf1_validation.train_heads") as heads:
            with self.assertRaisesRegex(ValueError, "not a completed run"):
                run_validation(incomplete, self.manifest_path, self.config_path, "cpu", output)
            heads.assert_not_called()
        self.assertEqual(read_json(output / "run.json")["status"], "FAILED")
        self.assertFalse((output / "heads").exists())


if __name__ == "__main__":
    unittest.main()
