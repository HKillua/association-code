import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from association.data.archive import DataError, MTADArchive
from association.utils.config import validate_manifest_config
from association.data.legacy_dataset import (load_split, load_split_from_archive, make_synthetic_archive,
                                 prepare, prepare_scene, validate_archive)


CONFIG = {"split_seed": 20261016, "caps": {"T_core": 8, "T_head": 4, "V_select": 4, "T_test": 4},
          "model": {"length": 6}, "protocol": "synthetic-smoke-only-not-research-result"}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = make_synthetic_archive(self.root / "synthetic.zip")

    def test_prepare_writes_manifest_and_defers_test_measurements(self):
        reads = []
        original = MTADArchive._rows

        def logged(archive, sid, kind):
            reads.append((str(sid), kind))
            return original(archive, sid, kind)

        with patch.object(MTADArchive, "_rows", logged):
            manifest = prepare(self.archive, CONFIG, self.root / "prepared", synthetic=True)
        self.assertEqual({key: len(value) for key, value in manifest["splits"].items()}, CONFIG["caps"])
        measurement_scenes = {sid for sid, kind in reads if kind == "measurements"}
        expected = set(sum((manifest["splits"][split] for split in ("T_core", "T_head", "V_select")), []))
        self.assertEqual(measurement_scenes, expected)
        self.assertFalse(measurement_scenes.intersection(manifest["splits"]["T_test"]))
        audit = json.loads((self.root / "prepared" / "audit.json").read_text())
        self.assertEqual(audit["status"], "COMPLETED")
        self.assertEqual(audit["test_measurements_read"], 0)
        self.assertEqual(audit["reference_members_read"], 0)
        self.assertEqual(audit["measurement_members_read"], 16)
        self.assertEqual(audit["label_members_read"], 32)
        self.assertEqual(audit["splits"]["T_test"]["status"], "deferred_until_frozen_evaluation")
        self.assertIsNone(audit["splits"]["T_test"]["valid_nodes"])
        self.assertGreater(audit["splits"]["T_core"]["candidate_edges"], 0)
        self.assertNotIn("positive_edges", json.dumps(audit))
        self.assertNotIn(str(self.root), json.dumps(audit))
        self.assertEqual(json.loads((self.root / "prepared" / "manifest.json").read_text()), manifest)

    def test_scene_record_contains_all_pairs_and_real_intervals(self):
        with MTADArchive(self.archive) as archive:
            record = prepare_scene(archive, "1", 6, "component-test")
        n = len(record["node_ids"])
        self.assertEqual(record["pairs"].shape, (n * (n - 1) // 2, 2))
        self.assertTrue((record["pairs"][:, 0] < record["pairs"][:, 1]).all())
        self.assertEqual(record["values"].shape, (n, 6, 4))
        self.assertEqual(record["intervals"].shape, (n, 2))
        self.assertEqual(record["component_id"], "component-test")
        self.assertTrue(np.isfinite(record["values"]).all())
        for index, mask in enumerate(record["observed"]):
            np.testing.assert_allclose(record["observation_times"][index, mask][[0, -1]], record["intervals"][index])
        for pair, label in zip(record["pairs"], record["labels"]):
            self.assertEqual(bool(label), record["identities"][pair[0]] == record["identities"][pair[1]])
        self.assertTrue(any(label and record["sources"][pair[0]] == record["sources"][pair[1]]
                            for pair, label in zip(record["pairs"], record["labels"])))

    def test_load_split_reads_only_requested_measurements(self):
        manifest = prepare(self.archive, CONFIG, self.root / "prepared", synthetic=True)
        with MTADArchive(self.archive) as archive:
            records = load_split_from_archive(archive, manifest, "T_head", 6)
            measurement_scenes = {row["scene_id"] for row in archive.members_read if row["kind"] == "measurements"}
            self.assertEqual(measurement_scenes, set(manifest["splits"]["T_head"]))
            self.assertFalse(any(row["kind"] == "reference" for row in archive.members_read))
            validate_archive(archive, manifest)
        self.assertEqual([record["scene_id"] for record in records], manifest["splits"]["T_head"])
        again = load_split(self.archive, manifest, "T_head", 6)
        np.testing.assert_array_equal(again[0]["values"], records[0]["values"])


    def test_actual_manifest_seed_and_inventory_are_checked(self):
        manifest = prepare(self.archive, CONFIG, self.root / "prepared", synthetic=True)
        changed = copy.deepcopy(manifest)
        changed["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed/caps"):
            validate_manifest_config(changed, CONFIG)
        with MTADArchive(self.archive) as archive, \
                patch.object(archive, "scene_ids", return_value=["wrong-scene"]):
            with self.assertRaisesRegex(DataError, "Archive scenes"):
                validate_archive(archive, manifest)


if __name__ == "__main__":
    unittest.main()
