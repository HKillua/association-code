import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from association.data.archive import (DEFAULT_DEVELOPMENT_SCENES,
                              LEVEL1_PREVIEW_RELATED_LEVEL2_SCENES, DataError, MTADArchive,
                              Scene, Track, build_manifest, resample_scene,
                              validate_manifest)


def csv_payload(header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def fixture_zip(path, specs, extra=None):
    with zipfile.ZipFile(path, "w") as archive:
        for sid, (labels, measurements) in specs.items():
            archive.writestr("数据集level2/关联表/关联结果-%s.csv" % sid,
                             csv_payload(["t_s", "t_e", "mmsi", "source", "batch"], labels))
            archive.writestr("数据集level2/量测场景/场景-%s.csv" % sid,
                             csv_payload(["batch", "source", "time", "lat", "lon", "vel", "cou"], measurements))
            # Invalid reference content must be completely irrelevant.
            archive.writestr("数据集level2/真实场景/场景-%s.csv" % sid, b"NEVER READ THIS")
        for name, payload in extra or []:
            archive.writestr(name, payload)


def basic_scene(identity="A-0-0", batch="0", source="9001"):
    return [[0, 10, identity, source, batch]], [[batch, source, 0, 20, 130, 1, 350],
                                               [batch, source, 10, 21, 131, 2, 10]]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "data.zip"

    def test_labels_and_inputs_never_read_references(self):
        fixture_zip(self.path, {0: basic_scene()})
        with MTADArchive(self.path) as archive:
            self.assertEqual(archive.scene_ids(), ["0"])
            scene = archive.load_scene(0)
            self.assertEqual(scene.tracks[0].identity_id, "A-0")
            self.assertEqual(scene.tracks[0].ancestor_id, "A")
            self.assertEqual({x["kind"] for x in archive.members_read}, {"labels", "measurements"})
            self.assertEqual(scene.audit["reference_members_read"], 0)

    def test_legacy_gbk_member_names_and_gb18030_csv(self):
        class GBKInfo(zipfile.ZipInfo):
            def _encodeFilenameFlags(self):
                return self.filename.encode("gbk"), self.flag_bits & ~0x800
        labels, points = basic_scene("甲-0-0")
        with zipfile.ZipFile(self.path, "w") as archive:
            for name, header, rows in [
                ("关联表/关联结果-0.csv", ["t_s", "t_e", "mmsi", "source", "batch"], labels),
                ("量测场景/场景-0.csv", ["batch", "source", "time", "lat", "lon", "vel", "cou"], points),
            ]:
                payload = csv_payload(header, rows).decode("utf-8-sig").encode("gb18030")
                archive.writestr(GBKInfo("数据集level2/" + name), payload)
        with MTADArchive(self.path) as archive:
            scene = archive.load_scene(0)
            self.assertEqual(scene.tracks[0].identity_id, "甲-0")

    def test_same_source_positive_fragments_are_retained(self):
        a, b = basic_scene("A-0-0", "0"), basic_scene("A-0-1", "1")
        fixture_zip(self.path, {0: (a[0] + b[0], a[1] + b[1])})
        with MTADArchive(self.path) as archive:
            scene = archive.load_scene(0)
        self.assertEqual(len(scene.tracks), 2)
        self.assertEqual(scene.tracks[0].source, scene.tracks[1].source)
        self.assertEqual(scene.tracks[0].identity_id, scene.tracks[1].identity_id)
        self.assertEqual(scene.audit["positive_edges_after_quarantine"], 1)

    def test_exact_duplicate_removed_but_conflicting_time_quarantined(self):
        labels, points = basic_scene()
        fixture_zip(self.path, {0: (labels, points + [points[0]])})
        with MTADArchive(self.path) as archive:
            scene = archive.load_scene(0)
        self.assertEqual(scene.audit["exact_duplicate_rows_removed"], 1)
        self.assertEqual(len(scene.tracks), 1)
        conflict = list(points[0])
        conflict[3] += 1
        fixture_zip(self.path, {0: (labels, points + [conflict])})
        with MTADArchive(self.path) as archive:
            scene = archive.load_scene(0)
        self.assertEqual(scene.tracks, [])
        self.assertEqual(scene.audit["quarantine_reasons"]["conflicting_same_time"], 1)

    def test_invalid_whole_nodes_and_coverage_are_explicit(self):
        labels, points = [], []
        cases = [(3, "nan", "nonfinite_measurement"), (5, -1, "negative_velocity"),
                 (6, -1, "missing_course"), (6, 361, "course_out_of_range"),
                 (3, 91, "coordinate_out_of_range"), (2, "bad", "nonnumeric_measurement")]
        for index, (column, value, _) in enumerate(cases):
            local_labels, local_points = basic_scene("A-0-%d" % index, str(index))
            local_points[0][column] = value
            labels += local_labels
            points += local_points
        fixture_zip(self.path, {0: (labels, points)})
        with MTADArchive(self.path) as archive:
            scene = archive.load_scene(0)
        self.assertEqual(scene.tracks, [])
        self.assertEqual(scene.audit["quarantined_nodes"], 6)
        self.assertEqual(scene.audit["positive_edges_lost_to_quarantine"], 15)
        for _, _, reason in cases:
            self.assertEqual(scene.audit["quarantine_reasons"][reason], 1)

    def test_ambiguous_identity_is_not_arbitrarily_chosen(self):
        labels, points = basic_scene()
        labels += [[0, 10, "B-0-0", "9001", "0"]]
        fixture_zip(self.path, {0: (labels, points)})
        with MTADArchive(self.path) as archive:
            scene = archive.load_scene(0)
        self.assertEqual(scene.tracks, [])
        self.assertEqual(scene.audit["quarantine_reasons"]["ambiguous_label_identity"], 1)

    def test_same_ab_different_c_labels_for_single_node_still_ambiguous(self):
        labels, points = basic_scene()
        labels += [[0, 10, "A-0-1", "9001", "0"]]
        fixture_zip(self.path, {0: (labels, points)})
        with MTADArchive(self.path) as archive:
            self.assertEqual(archive.load_scene(0).audit["quarantined_nodes"], 1)

    def test_label_interval_does_not_crop_measurement_inputs(self):
        labels, points = basic_scene()
        labels[0][:2] = [2, 8]
        fixture_zip(self.path, {0: (labels, points)})
        with MTADArchive(self.path) as archive:
            scene = archive.load_scene(0)
        self.assertEqual(scene.tracks[0].times.tolist(), [0, 10])
        self.assertEqual(scene.audit["measurement_rows_outside_label_interval_diagnostic_only"], 2)


    def test_invalid_identity_prevents_leakage_manifest(self):
        fixture_zip(self.path, {0: basic_scene("invalid")})
        with MTADArchive(self.path) as archive:
            with self.assertRaisesRegex(DataError, "A-B-C"):
                build_manifest(archive, 20, {name: 10 for name in ("T_core", "T_head", "V_select", "T_test")}, [])

    def test_shared_ancestor_components_and_development_are_isolated(self):
        specs = {index: basic_scene("I%d-0-0" % index) for index in range(20)}
        specs[0] = basic_scene("SHARED-0-0")
        specs[1] = basic_scene("SHARED-1-2")
        a, b = basic_scene("SHARED-2-0"), basic_scene("CHAIN-0-0", "1")
        specs[2] = a[0] + b[0], a[1] + b[1]
        specs[3] = basic_scene("CHAIN-7-1")
        fixture_zip(self.path, specs)
        caps = {name: 100 for name in ("T_core", "T_head", "V_select", "T_test")}
        with MTADArchive(self.path) as archive:
            manifest = build_manifest(archive, 20261016, caps, ["1"])
            repeat = build_manifest(archive, 20261016, caps, [1])
            self.assertTrue(all(item["kind"] == "labels" for item in archive.members_read))
        self.assertEqual(manifest, repeat)
        self.assertEqual(len(manifest["label_members"]), 20)
        self.assertTrue(validate_manifest(manifest))
        self.assertEqual(len({manifest["components"][str(i)] for i in range(4)}), 1)
        self.assertTrue(set(map(str, range(4))).issubset(manifest["splits"]["T_core"]))
        split_for_scene = {sid: split for split, scenes in manifest["splits"].items() for sid in scenes}
        ancestor_splits = {}
        for sid, (labels, _) in specs.items():
            for row in labels:
                ancestor_splits.setdefault(row[2].split("-")[0], set()).add(split_for_scene[str(sid)])
        self.assertTrue(all(len(values) == 1 for values in ancestor_splits.values()))
        self.assertNotIn(str(self.tmp.name), json.dumps(manifest))


    def test_manifest_rejects_overlapping_split_roles(self):
        fixture_zip(self.path, {0: basic_scene()})
        with MTADArchive(self.path) as archive:
            manifest = build_manifest(archive, 20, {name: 1 for name in ("T_core", "T_head", "V_select", "T_test")}, [0])
        manifest["splits"]["T_test"] = ["0"]
        with self.assertRaises(DataError):
            validate_manifest(manifest)

    def test_cross_level_preview_pins_the_whole_component_and_rejects_stale_manifest(self):
        from association.utils.config import load_config, validate_manifest_config
        config = load_config(Path(__file__).resolve().parents[2] / 'configs/first-fit-v2.json')
        specs = {int(s): basic_scene('EXPOSED-0-0')
                 for s in LEVEL1_PREVIEW_RELATED_LEVEL2_SCENES}
        # An unlisted scene sharing A must be pinned too, before scene caps.
        specs[71] = basic_scene('EXPOSED-1-0')
        specs.update({s: basic_scene(f'PRIVATE{s}-0-0') for s in range(61, 69)})
        fixture_zip(self.path, specs)
        with MTADArchive(self.path) as archive:
            stale = build_manifest(archive, config['split_seed'], config['caps'], [])
            with self.assertRaisesRegex(ValueError, 'development scenes are not all pinned'):
                validate_manifest_config(stale, config)
            current = build_manifest(archive, config['split_seed'], config['caps'],
                                     LEVEL1_PREVIEW_RELATED_LEVEL2_SCENES)
            self.assertTrue(validate_manifest_config(current, config))
            for sid in (*LEVEL1_PREVIEW_RELATED_LEVEL2_SCENES, '71'):
                cid = current['components'][sid]
                self.assertEqual(current['component_splits'][cid], 'T_core')
            self.assertEqual({r['kind'] for r in archive.members_read}, {'labels'})
        self.assertEqual(len(set(DEFAULT_DEVELOPMENT_SCENES)), 71)


class ResamplingTests(unittest.TestCase):
    def track(self, name, start, stop, courses=(350, 10)):
        return Track(name, "9001", name, "A-0", "A", np.array([start, stop]),
                     np.array([[20, 130, 1, courses[0]], [22, 132, 3, courses[1]]]))

    def test_common_grid_and_short_support_preserve_endpoint_anchors(self):
        tracks = [self.track("long", 0, 1000), self.track("short", 50, 50.001)]
        packed = resample_scene(Scene("0", tracks, {}), 64)
        self.assertEqual(packed["values"].shape, (2, 64, 4))
        self.assertEqual(packed["grid"].shape, (64,))
        self.assertTrue((packed["observed"].sum(1) >= 2).all())
        for index, track in enumerate(tracks):
            mask = packed["observed"][index]
            actual = packed["observation_times"][index, mask]
            self.assertEqual(actual[0], track.times[0])
            self.assertEqual(actual[-1], track.times[-1])
            self.assertTrue(((actual >= track.times[0]) & (actual <= track.times[-1])).all())
            np.testing.assert_allclose(packed["values"][index, mask][[0, -1]], track.values)
        self.assertEqual(packed["audit"]["registration"]["short_support_tracks"], 1)
        self.assertGreater(packed["audit"]["registration"]["max_absolute_time_displacement"], 0)
        self.assertFalse(packed["audit"]["registration"]["coordinate_extrapolation"])
        self.assertTrue(np.isnan(packed["observation_times"][~packed["observed"]]).all())
        self.assertTrue((packed["values"][~packed["observed"]] == 0).all())

    def test_course_interpolation_takes_shortest_arc(self):
        packed = resample_scene(Scene("0", [self.track("a", 0, 10)], {}), 3)
        np.testing.assert_allclose(packed["values"][0, :, 3], [350, 0, 10], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
