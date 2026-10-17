import csv
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from association.data import DataError
from association.reference_states import REFERENCE_HEADER, read_reference_states


def payload(rows, encoding="utf-8-sig", header=REFERENCE_HEADER):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode(encoding)


def manifest():
    splits = {"T_core": ["0", "1"], "T_head": ["2"], "V_select": ["3"], "T_test": ["4"]}
    scenes = [sid for group in splits.values() for sid in group]
    return {"splits": splits, "all_scenes": scenes, "components": {sid: sid for sid in scenes},
            "component_members": {sid: [sid] for sid in scenes},
            "component_splits": {sid: split for split, group in splits.items() for sid in group},
            "label_members": {sid: {} for sid in scenes}, "caps": {k: len(v) for k, v in splits.items()},
            "development_scenes": ["0"]}


class ReferenceStatesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.zip"

    def write_zip(self, rows, extra=()):
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("数据集level2/真实场景/场景-0.csv", payload(rows))
            for name, content in extra:
                archive.writestr(name, content)

    def test_selected_reference_only_preserves_raw_order_time_and_four_states(self):
        rows = [["A-0", 19, 23, 124, 2, 360], ["A-0", 5, 22, 123, 1, -10]]
        self.write_zip(rows, [("数据集level2/真实场景/场景-1.csv", b"UNSELECTED TRAIN"),
                              ("数据集level2/真实场景/场景-4.csv", b"SEALED TEST"),
                              ("数据集level2/量测场景/场景-0.csv", b"OBSERVATIONS MUST NOT BE READ"),
                              ("数据集level2/关联表/关联结果-0.csv", b"LABELS MUST NOT BE READ")])
        original_bytes, reads = self.path.read_bytes(), []
        original_read = zipfile.ZipFile.read
        def read(archive, member, *args, **kwargs):
            reads.append(member.filename)
            return original_read(archive, member, *args, **kwargs)
        with patch.object(zipfile.ZipFile, "read", read):
            result = read_reference_states(self.path, manifest(), split="T_core", scene_ids=[0])
        self.assertEqual(reads, ["数据集level2/真实场景/场景-0.csv"])
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual([r["time"] for r in result[0]["rows"]], [19., 5.])
        self.assertEqual([r["state"] for r in result[0]["rows"]], [[23., 124., 2., 360.], [22., 123., 1., -10.]])
        self.assertEqual(result[0]["rows"][0]["raw"]["cou"], "360")
        self.assertEqual(result[0]["issues"], [])

    def test_sealed_cross_split_empty_and_implicit_requests_fail_before_zip_open(self):
        cases = [("T_test", [4]), ("T_core", [3]), ("T_core", [4]),
                 ("V_select", [0]), ("T_core", []), ("T_core", [0, "00"]), ("T_core", "0")]
        for split, scenes in cases:
            with self.subTest(split=split, scenes=scenes), \
                    patch("association.reference_states.zipfile.ZipFile", side_effect=AssertionError("No file opened")):
                with self.assertRaises(DataError):
                    read_reference_states(self.path, manifest(), split=split, scene_ids=scenes)

    def test_problems_are_reported_without_dropping_or_resolving_rows(self):
        rows = [["A-0", 7, 1, 2, 3, 4], ["A-0", 7, 1, 2, 3, 4],
                ["B-0", 9, 1, 2, 3, 4], ["B-0", 9, 1, 9, 3, 4],
                ["C-0", 11, "nan", 2, 3, 4], ["C-0", 11, 1, 2, 3, 4],
                ["D-0", "inf", 1, 2, 3, 4], ["E-0", 3, 1, 2, "bad", 4],
                ["A-0-1", 2, 1, 2, 3, 4]]
        self.write_zip(rows)
        result = read_reference_states(self.path, manifest(), split="T_core", scene_ids=[0])[0]
        self.assertEqual(len(result["rows"]), len(rows))
        codes = [issue["code"] for issue in result["issues"]]
        self.assertEqual(codes.count("nonfinite"), 2)
        for code in ["duplicate_identical", "duplicate_conflict", "duplicate_unresolved", "nonnumeric", "invalid_identity"]:
            self.assertIn(code, codes)
        conflict = next(i for i in result["issues"] if i["code"] == "duplicate_conflict")
        self.assertEqual(conflict["lines"], [4, 5])
        self.assertEqual(result["rows"][7]["state"][2], None)
        self.assertEqual(result["rows"][7]["raw"]["vel"], "bad")
        self.assertTrue(math.isnan(result["rows"][4]["state"][0]))
        self.assertTrue(math.isinf(result["rows"][6]["time"]))

    def test_legacy_filename_encoding_and_csv_decoding_match_existing_archive(self):
        class GBKInfo(zipfile.ZipInfo):
            def _encodeFilenameFlags(self):
                return self.filename.encode("gbk"), self.flag_bits & ~0x800
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr(GBKInfo("数据集level2/真实场景/场景-3.csv"),
                             payload([["甲-1", 99, 20, 130, 2, 0]], "gb18030"))
        result = read_reference_states(self.path, manifest(), split="V_select", scene_ids=[3])
        self.assertEqual(result[0]["rows"][0]["identity_id"], "甲-1")
        self.assertEqual(result[0]["rows"][0]["time"], 99.)

    def test_missing_ambiguous_or_malformed_reference_is_not_silently_guessed(self):
        self.write_zip([["A-0", 0, 1, 2, 3, 4]])
        with self.assertRaisesRegex(DataError, "Missing requested"):
            read_reference_states(self.path, manifest(), split="T_core", scene_ids=[1])
        with zipfile.ZipFile(self.path, "a") as archive:
            archive.writestr("other/真实场景/场景-00.csv", payload([["A-0", 0, 1, 2, 3, 4]]))
        with self.assertRaisesRegex(DataError, "Ambiguous"):
            read_reference_states(self.path, manifest(), split="T_core", scene_ids=[0])
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("真实场景/场景-0.csv", payload([["A-0", 1]], header=["MMSI", "time"]))
        with self.assertRaisesRegex(DataError, "unexpected reference CSV header"):
            read_reference_states(self.path, manifest(), split="T_core", scene_ids=[0])


if __name__ == "__main__":
    unittest.main()
