from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.dataset import make_synthetic_archive, prepare, load_split
from association.profile import profile_reference
from association.runtime import BudgetExceeded, read_json


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(Path(__file__).resolve().parents[1] / "configs" / "smoke.json")
        self.config["caps"] = dict.fromkeys(("T_core", "T_head", "V_select", "T_test"), 2)
        self.archive = make_synthetic_archive(self.root / "data.zip", n_scenes=12)
        self.manifest = prepare(self.archive, self.config, self.root / "prepared", synthetic=True)

    def test_two_real_gradient_steps_are_finite_change_parameters_and_never_read_validation_test(self):
        reads = []

        def reader(path, manifest, split, length):
            reads.append(split)
            return load_split(path, manifest, split, length)

        with patch("association.profile.load_split", side_effect=reader):
            report = profile_reference(self.archive, self.manifest, self.config, "A0", "ce", 20,
                                       "cpu", self.root / "profile", steps=2)
        self.assertEqual(reads, ["T_core"])
        self.assertTrue(report["parameters_changed"])
        self.assertEqual(report["completed_steps"], 2)
        self.assertTrue(np.isfinite(report["losses"]).all())
        self.assertTrue(np.isfinite(report["gradient_norms"]).all())
        self.assertFalse(report["eligible_reference"])
        self.assertFalse(report["gpu_measured"])
        self.assertIsNone(report["peak_allocated_bytes"])
        self.assertEqual(report["cost_projection"]["status"], "warmup_only_no_projection")
        self.assertIsNone(report["cost_projection"]["full_reference_seconds"])
        self.assertEqual(report["normalizer"]["training_scene_ids"], self.manifest["splits"]["T_core"])
        self.assertFalse(list((self.root / "profile").glob("*.pt")))
        run = read_json(self.root / "profile" / "run.json")
        self.assertEqual(run["status"], "COMPLETED")
        self.assertEqual(run["kind"], "reference_profile_not_eligible")

    def test_full_role_uses_only_training_splits_and_reports_stage_counts(self):
        self.config["reference"]["joint_completion"] = False
        report = profile_reference(self.archive, self.manifest, self.config, "A_full", "contrastive", 20,
                                   "cpu", self.root / "profile", steps=6)
        expected = self.manifest["splits"]["T_core"] + self.manifest["splits"]["T_head"]
        self.assertEqual(report["training_scene_ids"], expected)
        self.assertEqual(report["measured_steady_steps"], 1)
        self.assertFalse(report["joint_completion"])
        self.assertGreater(report["tam_steps_per_epoch"], 0)
        self.assertGreater(report["tcm_steps_per_epoch"], 0)
        self.assertGreater(report["cost_projection"]["tam_training_only_seconds"], 0)
        self.assertIsNone(report["cost_projection"]["tcm_training_seconds"])

    def test_budget_failure_retains_evidence_and_never_saves_checkpoint(self):
        with self.assertRaises(BudgetExceeded):
            profile_reference(self.archive, self.manifest, self.config, "A0", "ce", 20,
                              "cpu", self.root / "profile", steps=2, max_minutes=1e-12)
        self.assertEqual(read_json(self.root / "profile" / "run.json")["status"], "FAILED")
        self.assertEqual(read_json(self.root / "profile" / "profile.json")["status"], "FAILED")
        self.assertFalse(list((self.root / "profile").glob("*.pt")))

    def test_probe_limits_are_not_silently_expanded(self):
        for kwargs in ({"steps": 201}, {"steps": 0}, {"max_minutes": 31}):
            with self.assertRaises(ValueError):
                profile_reference(self.archive, self.manifest, self.config, "A0", "ce", 20,
                                  "cpu", self.root / "invalid", **kwargs)
        self.assertFalse((self.root / "invalid").exists())

    def test_new_config_cannot_reuse_an_incompatible_old_split(self):
        self.config["caps"]["T_core"] += 1
        with self.assertRaisesRegex(ValueError, "Manifest seed/caps"):
            profile_reference(self.archive, self.manifest, self.config, "A0", "ce", 20,
                              "cpu", self.root / "stale", steps=2)
        self.assertFalse((self.root / "stale").exists())


if __name__ == "__main__":
    unittest.main()
