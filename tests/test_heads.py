import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from association.heads import train_heads, load_heads, predict_heads
from association.metrics import evaluate_scenes


def scene(scene_id, n=4):
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)
    identities = np.array(["A" if i < 2 else "B" for i in range(n)])
    labels = np.array([identities[i] == identities[j] for i, j in pairs])
    # Ambiguous scores deliberately contain both false positive/negative edges.
    scores = np.resize(np.array([-.4, .7, -.2, .5, -.8, .3]), len(pairs))
    return {"scene_id": scene_id, "component_id": "component-" + scene_id,
            "pairs": pairs, "scores": scores, "labels": labels,
            "sources": np.arange(n) % 2, "identities": identities,
            "node_ids": [f"{scene_id}-{i}" for i in range(n)],
            "intervals": np.array([[i, i + 2] for i in range(n)])}


class HeadTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "successful"
        cls.head_records = [scene("head-a"), scene("head-b", n=2)]
        cls.val_records = [scene("val-a"), scene("val-b", n=2)]
        cls.test_records = [scene("test-a"), scene("test-b", n=2)]
        cls.config = {"epochs": 2, "calibration_steps": 3, "deltas": [.5, 1.],
                      "learning_rate": .01, "max_hours": .1}
        cls.binding = {"reference_sha256": "synthetic-reference", "manifest_sha256": "synthetic-manifest"}
        cls.selection = train_heads(cls.head_records, cls.val_records, cls.config, 20,
                                    "cpu", cls.output, cls.binding)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_arms_equal_trial_budget_and_parameters(self):
        expected = {"A_prime", "B", "B_shuffle", "PairMLP", "EndpointMarginal", "ScenePool", "FixedRule"}
        self.assertEqual(set(self.selection), expected)
        for arm in expected - {"A_prime", "FixedRule"}:
            item = self.selection[arm]
            self.assertEqual(len(item["trials"]), 2)
            self.assertTrue(all(len(trial["validation_curve"]) == 2 for trial in item["trials"]))
            self.assertLess(abs(item["parameter_count"] - 833) / 833, .05)
        run = json.loads((self.output / "run.json").read_text())
        self.assertEqual(run["status"], "COMPLETED")
        self.assertEqual(set(run["artifacts"]), {"calibrator.pt", "heads.pt", "selection.json"})

    def test_calibrator_weights_scenes_equally_not_edges(self):
        document = json.loads((self.output / "selection.json").read_text())
        expected = np.mean([F.binary_cross_entropy_with_logits(
            torch.tensor(record["scores"], dtype=torch.float32),
            torch.tensor(record["labels"], dtype=torch.float32)).item()
            for record in self.head_records])
        self.assertAlmostEqual(document["calibration"]["curve"][0]["scene_mean_bce"], expected, places=6)
        self.assertEqual(document["calibration"]["fit_split"], "T_head")

    def test_checkpoint_roundtrip_reproduces_validation_and_frozen_predictions(self):
        bundle = load_heads(self.output, "cpu", self.binding)
        duplicate = load_heads(self.output / "heads.pt", "cpu", self.binding)
        val = predict_heads(bundle, self.val_records, 20, split="V_select")
        test = predict_heads(bundle, self.test_records, 20)
        repeated = predict_heads(duplicate, self.test_records, 20)
        self.assertIn("B_inference_shuffle", test)
        for arm in self.selection:
            measured = evaluate_scenes(val[arm], self.selection[arm]["threshold"])
            self.assertAlmostEqual(measured["macro_f1"], self.selection[arm]["validation"]["macro_f1"])
            for predicted, restored in zip(test[arm], repeated[arm]):
                np.testing.assert_array_equal(predicted["scores"], restored["scores"])
                self.assertIn("intervals", predicted)
                self.assertIn("identities", predicted)
                self.assertIn("node_ids", predicted)
        self.assertTrue(all(not parameter.requires_grad for parameter in bundle["calibrator"].parameters()))

    def test_no_common_neighbor_is_identity_for_every_arm_after_training(self):
        bundle = load_heads(self.output, "cpu")
        predictions = predict_heads(bundle, self.test_records, 20)
        for arm in predictions:
            np.testing.assert_array_equal(predictions[arm][1]["scores"], predictions["A_prime"][1]["scores"])
        for arm, selection in self.selection.items():
            if arm == "A_prime":
                continue
            corrected = predictions[arm][0]["scores"] - predictions["A_prime"][0]["scores"]
            self.assertLessEqual(float(np.max(np.abs(corrected))), selection["delta"] + 1e-6)

    def test_integrity_wrong_binding_and_inference_overlap_fail(self):
        with self.assertRaises(ValueError):
            load_heads(self.output, "cpu", {"reference_sha256": "other"})
        bundle = load_heads(self.output, "cpu")
        with self.assertRaises(ValueError):
            predict_heads(bundle, self.test_records, 21)
        with self.assertRaises(ValueError):
            predict_heads(bundle, self.head_records, 20, split="T_test")
        target = Path(self.temporary.name) / "corrupted"
        shutil.copytree(self.output, target)
        with (target / "heads.pt").open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_heads(target, "cpu")

    def test_one_class_overlap_and_budget_failures_keep_reports(self):
        negative = copy.deepcopy(self.head_records)
        for item in negative:
            item["labels"][:] = 0
        destination = Path(self.temporary.name) / "one-class"
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            train_heads(negative, self.val_records, self.config, 20, "cpu", destination, self.binding)
        self.assertEqual(json.loads((destination / "run.json").read_text())["status"], "FAILED")
        overlap = copy.deepcopy(self.val_records)
        overlap[0]["component_id"] = self.head_records[0]["component_id"]
        with self.assertRaisesRegex(ValueError, "share identity components"):
            train_heads(self.head_records, overlap, self.config, 20, "cpu",
                        Path(self.temporary.name) / "overlap", self.binding)
        budget = {**self.config, "max_hours": 1e-12}
        destination = Path(self.temporary.name) / "budget"
        with self.assertRaises(RuntimeError):
            train_heads(self.head_records, self.val_records, budget, 20, "cpu", destination, self.binding)
        self.assertEqual(json.loads((destination / "run.json").read_text())["status"], "FAILED")
        with self.assertRaises(ValueError):
            load_heads(destination, "cpu")

    def test_pair_order_is_preserved(self):
        bundle = load_heads(self.output, "cpu")
        permuted = copy.deepcopy(self.test_records)
        order = np.array([4, 0, 5, 2, 1, 3])
        for key in ("pairs", "labels", "scores"):
            permuted[0][key] = permuted[0][key][order]
        expected = predict_heads(bundle, self.test_records, 20)
        actual = predict_heads(bundle, permuted, 20)
        for arm in actual:
            np.testing.assert_allclose(actual[arm][0]["scores"], expected[arm][0]["scores"][order], atol=1e-7)

    def test_keyboard_interrupt_marks_run_failed_and_is_reraised(self):
        destination = Path(self.temporary.name) / "interrupted"
        with patch("association.heads._validate_split_records", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                train_heads(self.head_records, self.val_records, self.config, 20,
                            "cpu", destination, self.binding)
        report = json.loads((destination / "run.json").read_text())
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error_type"], "KeyboardInterrupt")
        self.assertIn("elapsed_seconds", report)
        with self.assertRaises(ValueError):
            load_heads(destination, "cpu")


if __name__ == "__main__":
    unittest.main()
