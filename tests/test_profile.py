import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association import profile as profile_module
from association.dataset import make_synthetic_archive, prepare, load_split
from association.models import AssociationModel
from association.profile import _memory_snapshot, profile_reference
from association.runtime import BudgetExceeded, digest, read_json


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
        positives = []
        association_loss = AssociationModel.association_loss

        def reader(path, manifest, split, length):
            reads.append(split)
            return load_split(path, manifest, split, length)

        def record_batch(model, scores, labels):
            positives.append(int((labels == 1).sum()))
            return association_loss(model, scores, labels)

        with patch("association.profile.load_split", side_effect=reader), \
                patch.object(AssociationModel, "association_loss", record_batch):
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
        self.assertEqual(report["step_positive_pair_counts"], positives)
        self.assertEqual(report["phase"], "finalization")
        self.assertEqual(set(report["phase_seconds"]),
                         {"data_loading", "preparation", "model_initialization", "optimization", "finalization"})
        self.assertTrue(all(value >= 0 for value in report["phase_seconds"].values()))
        self.assertGreaterEqual(report["elapsed_seconds"], sum(report["step_seconds"]))
        for elapsed, phases, positive, total in zip(report["step_seconds"], report["step_phase_seconds"],
                                                   report["step_positive_pair_counts"], report["step_pair_counts"]):
            self.assertEqual(set(phases), {"batch_preparation", "encode_tracks", "association_loss",
                                          "completion_auxiliary", "backward_optimizer"})
            self.assertTrue(all(value >= 0 for value in phases.values()))
            self.assertAlmostEqual(sum(phases.values()), elapsed, places=10)
            self.assertGreaterEqual(positive, 0)
            self.assertLessEqual(positive, total)

    def test_memory_snapshot_reads_cuda_allocator_counters_without_synchronizing(self):
        with patch("torch.cuda.max_memory_allocated", return_value=123) as allocated, \
                patch("torch.cuda.max_memory_reserved", return_value=456) as reserved, \
                patch("torch.cuda.synchronize") as synchronize:
            device = torch.device("cuda:0")
            self.assertEqual(_memory_snapshot(device),
                             {"peak_allocated_bytes": 123, "peak_reserved_bytes": 456})
            allocated.assert_called_once_with(device)
            reserved.assert_called_once_with(device)
            synchronize.assert_not_called()

    def test_budget_failure_after_a_step_preserves_progress_and_mocked_memory_peaks(self):
        snapshots = []

        def stop_after_step(run):
            saved = read_json(run.path / "profile.json")
            if saved["completed_steps"]:
                snapshots.append(saved)
                raise BudgetExceeded("test budget reached after durable step")

        with patch("association.profile._memory_snapshot", return_value={
                "peak_allocated_bytes": 123, "peak_reserved_bytes": 456}), \
                patch("association.runtime.Run.check", stop_after_step):
            with self.assertRaisesRegex(BudgetExceeded, "durable step"):
                profile_reference(self.archive, self.manifest, self.config, "A0", "ce", 20,
                                  "cpu", self.root / "profile", steps=2)
        self.assertEqual(snapshots[0]["status"], "RUNNING")
        self.assertEqual(snapshots[0]["peak_allocated_bytes"], 123)
        self.assertEqual(snapshots[0]["peak_reserved_bytes"], 456)
        self.assertGreater(snapshots[0]["phase_seconds"]["optimization"], 0)
        report = read_json(self.root / "profile" / "profile.json")
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["completed_steps"], 1)
        self.assertEqual(len(report["step_phase_seconds"]), 1)
        self.assertEqual(len(report["step_positive_pair_counts"]), 1)
        self.assertNotIn("cost_projection", report)
        self.assertEqual(read_json(self.root / "profile" / "run.json")["status"], "FAILED")

    def test_failed_memory_read_preserves_previous_peak_and_original_training_error(self):
        update = profile_module._update
        calls = 0
        failed = False

        def fail_second_update(*args):
            nonlocal calls, failed
            calls += 1
            if calls == 2:
                failed = True
                raise RuntimeError("original training failure")
            return update(*args)

        def memory_snapshot(device):
            if failed:
                raise RuntimeError("allocator counter unavailable")
            return {"peak_allocated_bytes": 123, "peak_reserved_bytes": 456}

        with patch.object(profile_module, "_update", fail_second_update), \
                patch.object(profile_module, "_memory_snapshot", memory_snapshot):
            with self.assertRaisesRegex(RuntimeError, "original training failure"):
                profile_reference(self.archive, self.manifest, self.config, "A0", "ce", 20,
                                  "cpu", self.root / "profile", steps=2)
        report = read_json(self.root / "profile" / "profile.json")
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["failure"], "original training failure")
        self.assertEqual(report["completed_steps"], 1)
        self.assertEqual(report["peak_allocated_bytes"], 123)
        self.assertEqual(report["memory_snapshot_completed_steps"], 1)
        self.assertIn("allocator counter unavailable", report["memory_measurement_error"])

    def test_cuda_profile_failure_writes_terminal_run_when_allocator_telemetry_also_fails(self):
        # Exercise the actual cuda:0 provenance and finish branch, while mocking
        # hardware APIs and failing before model allocation; no GPU is required.
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(
                    name="mock CUDA", total_memory=1024)), \
                patch("torch.cuda.reset_peak_memory_stats"), \
                patch("association.runtime.seed_everything"), \
                patch("association.profile.load_split", side_effect=ValueError("original data failure")), \
                patch("torch.cuda.max_memory_allocated", side_effect=RuntimeError("allocator unavailable")):
            with self.assertRaisesRegex(ValueError, "original data failure"):
                profile_reference(self.archive, self.manifest, self.config, "A0", "ce", 20,
                                  "cuda:0", self.root / "profile", steps=2)
        report = read_json(self.root / "profile" / "profile.json")
        run = read_json(self.root / "profile" / "run.json")
        self.assertEqual(run["provenance"]["device"], "cuda:0")
        self.assertEqual(run["provenance"]["gpu_name"], "mock CUDA")
        self.assertEqual(run["status"], "FAILED")
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(run["failure_type"], "ValueError")
        self.assertEqual(run["failure"], "original data failure")
        self.assertEqual(run["profile_sha256"], digest(self.root / "profile" / "profile.json"))
        self.assertIn("allocator unavailable", run["cuda_memory_measurement_error"])
        self.assertIn("allocator unavailable", report["memory_measurement_error"])
        self.assertEqual(run["completed_steps"], 0)

    @unittest.skipUnless(os.name == "posix", "requires POSIX SIGTERM semantics")
    def test_sigterm_during_next_step_keeps_last_durable_progress_incomplete(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        # Real CPU gradients, injected allocator counters: no CUDA or timing
        # claims. TERM during step two bypasses Python exception handlers.
        script = '''
import json, os, signal, sys
from pathlib import Path
import torch
from association import profile
torch.set_num_threads(1)
archive, manifest_path, config_path, output = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text())
config = json.loads(Path(config_path).read_text())
profile._memory_snapshot = lambda device: {"peak_allocated_bytes": 123, "peak_reserved_bytes": 456}
update = profile._update
calls = 0
def terminate_on_second_update(*args):
    global calls
    calls += 1
    if calls == 2:
        os.kill(os.getpid(), signal.SIGTERM)
    return update(*args)
profile._update = terminate_on_second_update
profile.profile_reference(archive, manifest, config, "A0", "ce", 20, "cpu", output, steps=3)
'''
        result = subprocess.run([sys.executable, "-c", script, str(self.archive),
                                 str(self.root / "prepared" / "manifest.json"), str(config_path),
                                 str(self.root / "profile")], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, -signal.SIGTERM, result.stderr)
        report = read_json(self.root / "profile" / "profile.json")
        self.assertEqual(report["status"], "RUNNING")
        self.assertEqual(report["completed_steps"], 1)
        self.assertEqual(report["phase"], "optimization")
        self.assertEqual(report["peak_allocated_bytes"], 123)
        self.assertEqual(report["peak_reserved_bytes"], 456)
        self.assertGreater(report["elapsed_seconds"], 0)
        self.assertEqual(len(report["step_phase_seconds"]), 1)
        self.assertEqual(len(report["step_positive_pair_counts"]), 1)
        self.assertNotIn("cost_projection", report)
        self.assertEqual(read_json(self.root / "profile" / "run.json")["status"], "RUNNING")
        self.assertFalse(list((self.root / "profile").glob("*.pt")))

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
