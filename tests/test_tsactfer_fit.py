"""Exercise real training and sealed-split boundaries on native-format fixtures."""
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from association.data import MTADArchive
from association.dataset import make_synthetic_archive, prepare, load_split_from_archive
from association.runtime import BudgetExceeded, Run, load_records, read_json, write_json
from baselines.tsactfer import TSAConfig, TSActfer
from baselines.tsactfer_data import pack_scene
from tools.experiments.tsactfer_fit import (learning_rate, main, score_scenes, train,
                                          validate_tsa_config)


def long_track_archive(root):
    """Give the standard synthetic fixture real ITSA gaps and >2-minute overlaps."""
    raw = make_synthetic_archive(root / "short.zip", n_scenes=16)
    target = root / "long.zip"
    with zipfile.ZipFile(raw) as source, zipfile.ZipFile(target, "x") as destination:
        offsets = {}
        for name in source.namelist():
            if "关联结果" not in name:
                continue
            sid = Path(name).stem.split("-")[-1]
            rows = list(csv.reader(io.StringIO(source.read(name).decode("utf-8-sig"))))
            for row in rows[1:]:
                fragment = int(row[2].split("-")[-1])
                offsets[(sid, row[3], row[4])] = (fragment // 2) * 1000
        for name in source.namelist():
            sid = Path(name).stem.split("-")[-1]
            rows = list(csv.reader(io.StringIO(source.read(name).decode("utf-8-sig"))))
            for row in rows[1:]:
                if "关联结果" in name:
                    offset = offsets[(sid, row[3], row[4])]
                    row[0], row[1] = str(float(row[0]) * 30 + offset), str(float(row[1]) * 30 + offset)
                else:
                    row[2] = str(float(row[2]) * 30 + offsets[(sid, row[1], row[0])])
            stream = io.StringIO(newline="")
            csv.writer(stream).writerows(rows)
            destination.writestr(name, stream.getvalue().encode("utf-8-sig"))
    return target


class TSAFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        repo = Path(__file__).resolve().parents[1]
        cls.protocol = read_json(repo / "configs/smoke.json")
        cls.protocol["caps"] = {"T_core": 4, "T_head": 1, "V_select": 2, "T_test": 1}
        cls.protocol["model"]["length"] = 32
        cls.config = read_json(repo / "configs/tsactfer-first-fit.json")
        cls.config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
                                   dropout=0., time_bins=256)
        cls.config["training"].update(updates=4, warmup_updates=1, pair_batch_size=2,
                                      validate_every_updates=2, inference_batch_tracks=8)
        cls.archive = long_track_archive(cls.root)
        cls.manifest = prepare(cls.archive, cls.protocol, cls.root / "prepared", synthetic=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_cli_training_reload_cache_and_sealed_test(self):
        output = self.root / "fit"
        protocol_path, config_path = self.root / "protocol.json", self.root / "tsa.json"
        write_json(protocol_path, self.protocol)
        write_json(config_path, self.config)
        seen = []

        def limited_load(archive, manifest, split, length):
            self.assertIn(split, ("T_core", "V_select"))
            seen.append(split)
            return load_split_from_archive(archive, manifest, split, length)

        with patch("tools.experiments.tsactfer_fit.load_split_from_archive", side_effect=limited_load):
            result = main(["--archive", str(self.archive), "--manifest", str(self.root / "prepared/manifest.json"),
                           "--protocol-config", str(protocol_path), "--tsa-config", str(config_path),
                           "--device", "cpu", "--cpu-threads", "1", "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual(seen, ["T_core", "V_select"])
        summary = read_json(output / "summary.json")
        history = read_json(output / "history.json")
        self.assertEqual([row["task"] for row in history], ["ITSA", "MSTSA", "ITSA", "MSTSA"])
        self.assertEqual([row["update"] for row in history if "selection" in row], [2, 4])
        self.assertTrue(all(np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm"]) for row in history))
        self.assertFalse(summary["test_opened"])
        self.assertEqual(summary["training_splits"], ["T_core"])
        self.assertEqual(set(summary["normalizer"]["training_scene_ids"]), set(self.manifest["splits"]["T_core"]))
        self.assertEqual(read_json(output / "run.json")["status"], "COMPLETED")
        cached = load_records(output / "V_select", summary["cache_binding"])
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = TSActfer(TSAConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
        with MTADArchive(self.archive) as archive:
            original = load_split_from_archive(archive, self.manifest, "V_select", self.protocol["model"]["length"])
        packed = [pack_scene(scene, summary["normalizer"], length=128, time_bins=256) for scene in original]
        rescored = score_scenes(model, packed, "cpu", 8)
        for raw, saved, fresh in zip(original, cached, rescored):
            np.testing.assert_array_equal(raw["pairs"], saved["pairs"])
            np.testing.assert_array_equal(raw["labels"], saved["labels"])
            np.testing.assert_array_equal(saved["scores"], fresh["scores"])
        with self.assertRaises(FileExistsError):
            train(self.archive, self.manifest, self.protocol, self.config, "cpu", output)

    def test_budget_and_insufficient_identity_fail_without_shrinking_or_completion(self):
        limited = self.root / "expired"
        with patch.object(Run, "check", side_effect=BudgetExceeded("synthetic deadline")):
            with self.assertRaises(BudgetExceeded):
                train(self.archive, self.manifest, self.protocol, self.config, "cpu", limited)
        self.assertEqual(read_json(limited / "run.json")["status"], "FAILED")
        self.assertFalse((limited / "summary.json").exists())
        config = {**self.config, "training": {**self.config["training"], "pair_batch_size": 512}}
        rejected = self.root / "insufficient"
        with self.assertRaisesRegex(ValueError, "without batch reduction"):
            train(self.archive, self.manifest, self.protocol, config, "cpu", rejected)
        self.assertEqual(read_json(rejected / "run.json")["status"], "FAILED")
        self.assertFalse((rejected / "checkpoint.pt").exists())
        self.assertTrue((rejected / "data_audit.json").exists())

    def test_schedule_and_native_budget_contract(self):
        repo = Path(__file__).resolve().parents[1]
        config = read_json(repo / "configs/tsactfer-first-fit.json")
        model = validate_tsa_config(config)
        self.assertEqual((model.length, model.d_model, model.num_layers, model.nhead), (128, 64, 4, 4))
        training = config["training"]
        self.assertGreater(learning_rate(1, training), 0.)
        self.assertAlmostEqual(learning_rate(67, training), training["learning_rate"])
        self.assertEqual(learning_rate(1000, training), 0.)
        config["training"]["max_hours"] = 6.01
        with self.assertRaisesRegex(ValueError, "six-hour"):
            validate_tsa_config(config)


if __name__ == "__main__":
    unittest.main()
