"""Reuse legacy staged-fit artifacts after hashes are stale or absent."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.cli import _manifest
from association.dataset import make_synthetic_archive, prepare
from association.reference import train_reference
from association.runtime import read_json, semantic_binding, write_json
from tools.experiments.cdf1_validation import run_validation


def stale_hashes(value):
    if isinstance(value, dict):
        return {key: ("old-unrelated-hash" if key not in semantic_binding({key: item})
                      else stale_hashes(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [stale_hashes(item) for item in value]
    return value


class LegacyStagedCacheCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.config = read_json(Path(__file__).resolve().parents[1] / "configs/smoke.json")
        cls.config["caps"] = dict.fromkeys(("T_core", "T_head", "V_select", "T_test"), 1)
        cls.config["heads"].update(epochs=1, deltas=[1.], calibration_steps=2)
        for key in ("tcm_epochs", "tam_epochs", "warmup_epochs", "joint_completion"):
            cls.config["reference"].pop(key)
        cls.config["reference"].update(
            schedule="staged_updates_v2", purpose="fit", tcm_updates=1, tam_updates=1,
            joint_updates=1, validate_every_updates=1, completion_cache_mib=1,
            batch_tracks=4, pair_batch_size=4)
        cls.config_path = cls.root / "config.json"
        write_json(cls.config_path, cls.config)
        archive = make_synthetic_archive(cls.root / "data.zip", n_scenes=12)
        cls.manifest = prepare(archive, cls.config, cls.root / "prepared", synthetic=True)
        cls.reference = cls.root / "reference"
        train_reference(archive, cls.manifest, cls.config, "A0", "ce", 20, "cpu", cls.reference)
        # Matches 010's old bindings: role/branch/seed plus optional hashes, no
        # newly required config snapshot or scene metadata inside the binding.
        payload = torch.load(cls.reference / "checkpoint.pt", weights_only=True)
        assert set(semantic_binding(payload["binding"])) == {"role", "branch", "seed"}
        archive.unlink()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_old_staged_fit_supports_all_heads_with_stale_or_missing_hashes(self):
        for mode, transform in (("stale", stale_hashes), ("missing", semantic_binding)):
            with self.subTest(mode=mode):
                reference = self.root / mode
                shutil.copytree(self.reference, reference)
                for name in ("run.json", "summary.json", "T_head/index.json", "V_select/index.json"):
                    write_json(reference / name, transform(read_json(reference / name)))
                payload = torch.load(reference / "checkpoint.pt", weights_only=True)
                payload["binding"] = transform(payload["binding"])
                torch.save(payload, reference / "checkpoint.pt")
                manifest_path = self.root / f"manifest-{mode}.json"
                write_json(manifest_path, transform(self.manifest))
                with patch("association.runtime.code_fingerprint", return_value="new-unrelated-code"), \
                        patch("association.dataset.load_split", side_effect=AssertionError("No data reread")):
                    result = run_validation(reference, manifest_path, self.config_path,
                                            "cpu", self.root / f"heads-{mode}")
                self.assertEqual(len(result["arms"]), 8)
                self.assertFalse(result["test_opened"])

    def test_wrong_validation_scores_are_rejected_before_head_training_without_hashes(self):
        reference = self.root / "wrong-validation-scores"
        shutil.copytree(self.reference, reference)
        for name in ("run.json", "summary.json", "T_head/index.json", "V_select/index.json"):
            write_json(reference / name, semantic_binding(read_json(reference / name)))
        payload = torch.load(reference / "checkpoint.pt", weights_only=True)
        payload["binding"] = semantic_binding(payload["binding"])
        torch.save(payload, reference / "checkpoint.pt")
        summary = read_json(reference / "summary.json")
        self.assertIsNotNone(summary["threshold"])
        self.assertGreater(summary["validation"]["tp"], 0)
        for item in read_json(reference / "V_select/index.json")["records"]:
            path = reference / "V_select" / item["file"]
            with np.load(path, allow_pickle=False) as arrays:
                record = {key: arrays[key] for key in arrays.files}
            # Preserve valid scenes, candidates and labels, but replace the
            # checkpoint's scores with a different all-reject prediction.
            record["scores"] = np.full_like(record["scores"], summary["threshold"] - 100.)
            np.savez(path, **record)
        manifest_path = self.root / "wrong-scores-manifest.json"
        write_json(manifest_path, semantic_binding(self.manifest))
        output = self.root / "rejected-wrong-scores"
        with patch("tools.experiments.cdf1_validation.train_heads") as fit:
            with self.assertRaisesRegex(ValueError, "cache metrics differ from selected validation"):
                run_validation(reference, manifest_path, self.config_path, "cpu", output)
            fit.assert_not_called()
        self.assertEqual(read_json(output / "run.json")["status"], "FAILED")
        self.assertFalse((output / "heads").exists())

    def test_actual_cached_scene_component_or_split_errors_are_rejected(self):
        for mode in ("scene", "component", "split"):
            with self.subTest(mode=mode):
                reference = self.root / f"wrong-{mode}"
                shutil.copytree(self.reference, reference)
                index_path = reference / "T_head/index.json"
                index = semantic_binding(read_json(index_path))
                if mode == "split":
                    index["binding"]["split"] = "T_test"
                else:
                    index["records"][0][f"{mode}_id"] = "wrong-population"
                write_json(index_path, index)
                with patch("tools.experiments.cdf1_validation.train_heads") as fit:
                    with self.assertRaisesRegex(ValueError, "scenes differ|component differs|requested reference/split"):
                        run_validation(reference, self.root / "prepared/manifest.json", self.config_path,
                                       "cpu", self.root / f"rejected-{mode}")
                    fit.assert_not_called()

    def test_prepared_cli_accepts_hash_changes_but_preserves_status_and_registration(self):
        from types import SimpleNamespace
        prepared = self.root / "prepared-cli"
        shutil.copytree(self.root / "prepared", prepared)
        write_json(prepared / "manifest.json", semantic_binding(self.manifest))
        audit = read_json(prepared / "audit.json")
        audit["manifest_sha256"] = "unrelated"
        write_json(prepared / "audit.json", audit)
        args = SimpleNamespace(prepared=str(prepared))
        self.assertEqual(_manifest(args, self.config)["splits"], self.manifest["splits"])
        changed = copy.deepcopy(self.config)
        changed["model"]["length"] += 1
        with self.assertRaisesRegex(ValueError, "registration length"):
            _manifest(args, changed)
        audit["status"] = "RUNNING"
        write_json(prepared / "audit.json", audit)
        with self.assertRaisesRegex(ValueError, "not COMPLETED"):
            _manifest(args, self.config)


if __name__ == "__main__":
    unittest.main()
