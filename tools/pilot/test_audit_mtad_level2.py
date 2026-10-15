"""Behavior tests using tiny synthetic ZIPs; no real dataset fixtures are copied."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import audit_mtad_level2 as audit


class Level2AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, labels=None, reference=None, measurements=None, extra=None, omit=None):
        tables = {
            "labels": labels if labels is not None else "0,10,111-0-0,9001,1\n0,10,222-0-0,9002,1\n",
            "reference": reference if reference is not None else "111-0,0,20,130,1,2\n222-0,0,21,131,3,4\n",
            "measurements": measurements if measurements is not None else "1,9001,0,20,130,1,2\n1,9002,0,21,131,3,4\n",
        }
        path = self.root / "fixture.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            for kind, name in audit.scene_paths(0).items():
                if kind == omit:
                    continue
                content = ",".join(audit.HEADERS[kind]) + "\n" + tables[kind]
                zf.writestr(name, content.encode("utf-8-sig"))
            for name, data in (extra or {}).items():
                zf.writestr(name, data)
        return path

    def run_fixture(self, path, expected_md5=None):
        expected_md5 = expected_md5 or hashlib.md5(path.read_bytes()).hexdigest()
        return audit.run_audit(path, self.root / "result", [0], expected_md5=expected_md5, expected_scene_ids=[0])

    def test_same_batch_different_sources_remain_distinct(self):
        report = self.run_fixture(self.fixture())
        self.assertFalse(report["errors"])
        scene = report["scenes"][0]
        self.assertEqual(scene["stats"]["measurements"]["tracks"], 2)
        self.assertTrue(scene["label_key_identity_is_unique"])
        self.assertEqual(scene["source_comparisons"]["9001"]["exact_time_position_match_points"], 1)
        self.assertEqual(scene["source_comparisons"]["9002"]["exact_time_position_match_points"], 1)
        self.assertEqual(report["coverage"]["sample_csv_members_read"], 3)
        self.assertEqual(len(json.loads((self.root / "result/archive_index.json").read_text())), 3)

    def test_ambiguous_labels_do_not_silently_choose_an_identity(self):
        path = self.fixture(labels="0,10,111-0-0,9001,1\n0,10,222-0-0,9001,1\n0,10,222-0-0,9002,1\n")
        report = self.run_fixture(path)
        self.assertFalse(report["errors"])
        scene = report["scenes"][0]
        self.assertFalse(scene["label_key_identity_is_unique"])
        counts = scene["source_comparisons"]["9001"]
        self.assertEqual(counts["ambiguous_label_at_time_points"], 1)
        self.assertEqual(counts["exact_time_position_match_points"], 0)
        self.assertIsNone(counts["position_match_fraction_given_exact_reference_time"])

    def test_missing_reference_retains_all_measurement_denominator(self):
        report = self.run_fixture(self.fixture(reference="111-0,0,20,130,1,2\n"))
        self.assertFalse(report["errors"])
        counts = report["scenes"][0]["source_comparisons"]["9002"]
        self.assertEqual(counts["measurement_points"], 1)
        self.assertEqual(counts["missing_reference_identity_points"], 1)
        self.assertEqual(counts["position_match_fraction_of_all_measurements"], 0.0)
        self.assertIsNone(counts["position_match_fraction_given_exact_reference_time"])

    def test_nonfinite_numeric_is_fatal_and_reported(self):
        report = self.run_fixture(self.fixture(measurements="1,9001,0,NaN,130,1,2\n"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("non-finite lat", "\n".join(report["errors"]))
        self.assertTrue((self.root / "result/audit.md").exists())
        self.assertEqual(report["coverage"]["sample_csv_members_read"], 3)
        self.assertEqual(report["coverage"]["sample_csv_members_content_valid"], 2)
        self.assertEqual(report["scenes"], [])

    def test_existing_output_is_never_overwritten(self):
        path = self.fixture()
        output = self.root / "result"
        output.mkdir()
        sentinel = output / "audit.json"
        sentinel.write_text("human notes")
        with self.assertRaises(FileExistsError):
            self.run_fixture(path)
        self.assertEqual(sentinel.read_text(), "human notes")

    def test_unsafe_member_rejected_without_extraction(self):
        report = self.run_fixture(self.fixture(extra={"../escaped.csv": b"untrusted"}))
        self.assertEqual(report["status"], "failed")
        self.assertIn("Unsafe archive member", "\n".join(report["errors"]))
        self.assertFalse((self.root / "escaped.csv").exists())
        self.assertFalse((self.root / "数据集level2").exists())
        self.assertEqual(report["coverage"]["sample_csv_members_read"], 0)

    def test_missing_csv_is_fatal(self):
        report = self.run_fixture(self.fixture(omit="reference"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("Missing expected CSV", "\n".join(report["errors"]))

    def test_malformed_row_is_fatal(self):
        report = self.run_fixture(self.fixture(measurements="1,9001,0,20,130,1,2,extra\n"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("wrong column count", "\n".join(report["errors"]))

    def test_repeated_and_decreasing_times_are_data_warnings(self):
        path = self.fixture(measurements="1,9001,1,20,130,1,2\n1,9001,0,20,130,1,2\n1,9001,0,20,130,1,2\n")
        report = self.run_fixture(path)
        self.assertFalse(report["errors"])
        stats = report["scenes"][0]["stats"]["measurements"]
        self.assertEqual(stats["nonmonotonic_tracks"], 1)
        self.assertEqual(stats["duplicate_time_points_beyond_first"], 1)
        self.assertEqual(report["status"], "passed_with_data_warnings")

    def test_first_identity_component_does_not_collapse_reference_ids(self):
        report = self.run_fixture(self.fixture(labels="0,10,111-0-0,9001,1\n0,10,111-1-0,9002,1\n",
                                  reference="111-0,0,20,130,1,2\n111-1,0,21,131,3,4\n"))
        scene = report["scenes"][0]
        self.assertEqual(scene["reference_A_to_multiple_A_B"], [{"A": "111", "A_B_values": ["111-0", "111-1"]}])
        self.assertEqual(scene["source_comparisons"]["9002"]["exact_time_position_match_points"], 1)

    def test_interval_matching_is_inclusive_but_no_extrapolation(self):
        path = self.fixture(labels="0,0,111-0-0,9001,1\n0,10,222-0-0,9002,1\n",
                            measurements="1,9001,0,20,130,1,2\n1,9001,1,20,130,1,2\n")
        report = self.run_fixture(path)
        counts = report["scenes"][0]["source_comparisons"]["9001"]
        self.assertEqual(counts["unique_label_in_interval_points"], 1)
        self.assertEqual(counts["outside_all_label_intervals_points"], 1)

    def test_crc_corruption_is_detected_for_read_member(self):
        path = self.fixture()
        blob = path.read_bytes()
        self.assertIn(b"111-0,0,20", blob)
        path.write_bytes(blob.replace(b"111-0,0,20", b"111-0,0,29", 1))
        report = self.run_fixture(path)
        self.assertEqual(report["status"], "failed")
        self.assertIn("CRC", "\n".join(report["errors"]))

    def test_hash_mismatch_prevents_content_audit(self):
        report = self.run_fixture(self.fixture(), expected_md5="0" * 32)
        self.assertEqual(report["status"], "failed")
        self.assertIn("MD5 mismatch", "\n".join(report["errors"]))
        self.assertEqual(report["coverage"]["sample_csv_members_read"], 0)

    def test_cli_returns_nonzero_on_invalid_archive(self):
        path = self.fixture()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = audit.main(["--archive", str(path), "--output", str(self.root / "cli-result"), "--scene-ids", "0"])
        self.assertEqual(code, 1)

    def test_legacy_filename_decoder_preserves_chinese(self):
        expected = "数据集level2/关联表/关联结果-0.csv"
        info = zipfile.ZipInfo(expected.encode("gbk").decode("cp437"))
        self.assertEqual(audit.decoded_name(info), expected)


if __name__ == "__main__":
    unittest.main()
