"""Independent orchestration contracts; real native fixtures, mocked expensive fits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association import pipeline
from association.dataset import make_synthetic_archive, prepare


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = make_synthetic_archive(self.root / "synthetic.zip")
        config_path = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"
        self.config = json.loads(config_path.read_text())
        self.manifest = prepare(self.archive, self.config, self.root / "prepared", synthetic=True)

    def test_changed_split_seed_is_rejected_before_training(self):
        config = copy.deepcopy(self.config)
        config["split_seed"] += 1
        with patch.object(pipeline, "train_reference", side_effect=AssertionError("training was reached")):
            with self.assertRaisesRegex(ValueError, "[Ss]eed|[Mm]anifest|[Cc]onfig"):
                pipeline.run_pilot(self.archive, self.manifest, config, torch.device("cpu"), self.root / "run")

    def test_changed_caps_are_rejected_before_training(self):
        config = copy.deepcopy(self.config)
        config["caps"]["T_core"] += 1
        with patch.object(pipeline, "train_reference", side_effect=AssertionError("training was reached")):
            with self.assertRaisesRegex(ValueError, "[Cc]ap|[Mm]anifest|[Cc]onfig"):
                pipeline.run_pilot(self.archive, self.manifest, config, torch.device("cpu"), self.root / "run")

    def _fake_pipeline(self, *, replace_reference_after_freeze=False, replace_head_after_freeze=False):
        config = copy.deepcopy(self.config)
        config["seeds"] = [20, 21]
        output = self.root / "run"
        events = []
        evaluating = False
        names = ["A_prime", "B", "EndpointMarginal", "PairMLP", "ScenePool", "FixedRule", "B_shuffle"]
        record = {"scene_id": "test-fixture", "component_id": "isolated-test", "node_ids": ["a", "b", "c"],
                  "pairs": np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64),
                  "labels": np.asarray([1, 0, 0]), "scores": np.asarray([.9, .1, .2]),
                  "sources": ["9001", "9002", "9001"], "identities": ["a", "a", "b"],
                  "intervals": np.asarray([[0, 10], [1, 9], [0, 10]], dtype=float)}

        def train_reference(*args):
            events.append(("reference", args[3], args[4], args[5]))
            path = Path(args[-1])
            path.mkdir(parents=True)
            for name in ("run.json", "summary.json", "checkpoint.pt"):
                (path / name).write_text("frozen " + name)

        def load_reference(path, manifest, config, role, branch, seed, device):
            checksum = "changed-checkpoint" if evaluating and replace_reference_after_freeze else f"{role}-{branch}-{seed}"
            summary = {"checkpoint_sha256": checksum, "threshold": .5,
                       "validation": {"macro_f1": .6 if branch == "ce" else .5},
                       "cache_bindings": {"T_head": {"split": "T_head"}, "V_select": {"split": "V_select"}}}
            return None, None, summary

        def train_heads(head, val, config, seed, device, path, binding):
            events.append(("heads", seed))
            Path(path).mkdir(parents=True)
            (Path(path) / "selection.json").write_text(json.dumps({"seed": seed, "binding": binding}))
            for name in ("heads.pt", "calibrator.pt", "run.json"):
                (Path(path) / name).write_text("frozen " + name)
            return {name: {"threshold": .5} for name in names}

        def load_test(archive, manifest, split, length):
            nonlocal evaluating
            self.assertEqual(split, "T_test")
            self.assertEqual(len([event for event in events if event[0] == "reference"]), 6)
            self.assertEqual(len([event for event in events if event[0] == "heads"]), 2)
            frozen = json.loads((output / "selection_frozen.json").read_text())
            self.assertEqual(set(frozen["selected"]), {"20", "21"})
            self.assertTrue(frozen["content_sha256"])
            evaluating = True
            if replace_head_after_freeze:
                (output / "seed20" / "heads" / "heads.pt").write_text("changed frozen weights")
            events.append(("test_read",))
            return []

        def evaluate(records, thresholds, config, seed):
            return {"seed": seed, "metrics": {name: {"macro_f1": 1., "micro_f1": 1.} for name in records},
                    "comparisons": {"B_minus_A_full": {"paired_component_difference": {"difference": 0.}}}}

        with patch.multiple(pipeline, train_reference=train_reference, load_reference=load_reference,
                            train_heads=train_heads, load_split=load_test,
                            load_records=lambda *args: [record], load_heads=lambda *args, **kwargs: {},
                            score_reference=lambda *args, **kwargs: [record],
                            predict_heads=lambda *args, **kwargs: {name: [record] for name in names},
                            evaluate_locked=evaluate):
            result = pipeline.run_pilot(self.archive, self.manifest, config, torch.device("cpu"), output)
        return events, result

    def test_all_requested_seed_selection_finishes_before_test_is_opened(self):
        events, report = self._fake_pipeline()
        self.assertEqual(events[-1], ("test_read",))
        self.assertEqual(len(report["seeds"]), 2)

    def test_reference_replacement_after_freeze_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "[Ff]roz|[Cc]heckpoint|[Rr]eference"):
            self._fake_pipeline(replace_reference_after_freeze=True)

    def test_head_weight_replacement_after_freeze_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "[Ff]roz|[Aa]rtifact"):
            self._fake_pipeline(replace_head_after_freeze=True)

    def test_training_failure_retains_record_without_opening_test(self):
        output = self.root / "run"
        with patch.object(pipeline, "train_reference", side_effect=RuntimeError("synthetic fit failure")), \
                patch.object(pipeline, "load_split") as read_test:
            with self.assertRaisesRegex(RuntimeError, "synthetic fit failure"):
                pipeline.run_pilot(self.archive, self.manifest, self.config, torch.device("cpu"), output)
            read_test.assert_not_called()
        self.assertFalse((output / "selection_frozen.json").exists())
        status = json.loads((output / "run.json").read_text())
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(status["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
