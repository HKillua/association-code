import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.dataset import make_synthetic_archive, prepare, load_split
from association.reference import (fit_normalizer, load_reference, score_reference,
                                   train_reference)
from association.runtime import BudgetExceeded, load_records, read_json


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(Path(__file__).resolve().parents[1] / "configs" / "smoke.json")
        self.config["caps"] = dict.fromkeys(("T_core", "T_head", "V_select", "T_test"), 1)
        self.config["reference"]["batch_tracks"] = 4
        self.archive = make_synthetic_archive(self.root / "data.zip", n_scenes=12)
        self.manifest = prepare(self.archive, self.config, self.root / "prepared", synthetic=True)

    def fit(self, role="A0", branch="ce", suffix="fit"):
        return train_reference(self.archive, self.manifest, self.config, role, branch,
                               20, "cpu", self.root / suffix)

    def test_cpu_train_reload_cache_and_training_split_isolation(self):
        reads = []

        def reader(path, manifest, split, length):
            reads.append(split)
            self.assertNotEqual(split, "T_test")
            return load_split(path, manifest, split, length)

        with patch("association.reference.load_split", side_effect=reader):
            summary = self.fit()
        self.assertEqual(summary["training_scene_ids"], self.manifest["splits"]["T_core"])
        self.assertEqual(summary["normalizer"]["training_scene_ids"], self.manifest["splits"]["T_core"])
        self.assertEqual(reads, ["T_core", "V_select", "T_head"])
        self.assertFalse(summary["test_opened"])
        self.assertEqual(read_json(self.root / "fit" / "run.json")["status"], "COMPLETED")
        model, normalizer, loaded = load_reference(self.root / "fit", self.manifest, self.config,
                                                  "A0", "ce", 20, "cpu")
        scenes = load_split(self.archive, self.manifest, "V_select", self.config["model"]["length"])
        rescored = score_reference(model, normalizer, scenes, "cpu", batch_tracks=4)
        cached = load_records(self.root / "fit" / "V_select", summary["cache_bindings"]["V_select"])
        for left, right in zip(rescored, cached):
            np.testing.assert_allclose(left["scores"], right["scores"], rtol=0, atol=0)
        self.assertEqual(loaded["checkpoint_sha256"], summary["checkpoint_sha256"])
        self.assertEqual(loaded["manifest_sha256"], self.manifest["manifest_sha256"])
        self.assertEqual(loaded["threshold"], summary["threshold"])

    def test_full_reference_training_and_normalizer_include_head_only_as_training(self):
        summary = self.fit(role="A_full", branch="contrastive")
        expected = self.manifest["splits"]["T_core"] + self.manifest["splits"]["T_head"]
        self.assertEqual(summary["training_scene_ids"], expected)
        self.assertEqual(summary["normalizer"]["training_scene_ids"], expected)
        self.assertEqual(summary["training_splits"], ["T_core", "T_head"])
        self.assertTrue(set(expected).isdisjoint(self.manifest["splits"]["V_select"]))
        self.assertTrue(set(expected).isdisjoint(self.manifest["splits"]["T_test"]))
        load_reference(self.root / "fit", self.manifest, self.config, "A_full", "contrastive", 20, "cpu")

    def test_normalization_excludes_missing_slots(self):
        values = np.array([[[1., 2.], [float("nan"), float("nan")], [3., 6.]]])
        scenes = [{"values": values, "observed": np.array([[True, False, True]]), "scene_id": "core"}]
        normalizer = fit_normalizer(scenes)
        self.assertEqual(normalizer["observed_point_count"], 2)
        self.assertEqual(normalizer["mean"], [2., 4.])
        self.assertEqual(normalizer["std"], [1., 2.])

    def test_stale_config_code_and_corrupt_checkpoint_rejected(self):
        self.fit()
        changed = copy.deepcopy(self.config)
        changed["reference"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            load_reference(self.root / "fit", self.manifest, changed, "A0", "ce", 20, "cpu")
        with patch("association.runtime.code_fingerprint", return_value="changed-code"):
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_reference(self.root / "fit", self.manifest, self.config, "A0", "ce", 20, "cpu")
        with (self.root / "fit" / "checkpoint.pt").open("ab") as handle:
            handle.write(b"modified")
        with self.assertRaisesRegex(ValueError, "checkpoint checksum"):
            load_reference(self.root / "fit", self.manifest, self.config, "A0", "ce", 20, "cpu")

    def test_budget_failure_preserves_terminal_run_and_cannot_be_loaded_or_overwritten(self):
        self.config["reference"]["max_hours"] = 1e-12
        with self.assertRaises(BudgetExceeded):
            self.fit()
        self.assertEqual(read_json(self.root / "fit" / "run.json")["status"], "FAILED")
        with self.assertRaisesRegex(ValueError, "not a completed"):
            load_reference(self.root / "fit", self.manifest, self.config, "A0", "ce", 20, "cpu")
        with self.assertRaises(FileExistsError):
            self.fit()
        self.assertEqual(read_json(self.root / "fit" / "run.json")["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
