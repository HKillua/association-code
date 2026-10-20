"""Scientific boundaries of the anonymous leave-two-nodes-out control."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analysis.anonymous_calibration_control import (
    BASES, _group_report, _translation_data, calibrate_scene, estimate_translation, load_records, run)


class AnonymousCalibrationTests(unittest.TestCase):
    @staticmethod
    def scene():
        positions = np.zeros((6, 64, 2))
        positions[:, :, 0] = 20.
        positions[[0, 3], :, 1] = 110.
        positions[[1, 4], :, 1] = 111.
        positions[[2, 5], :, 1] = 112.
        positions[3:] += [.01, -.02]
        return positions, np.array(['9001']*3+['9002']*3), np.ones((6, 64), bool)

    def test_known_shift_exclusion_short_support_and_same_source_invariance(self):
        positions, sources, support = self.scene()
        support[1] = False
        support[1, 3] = True  # One joint support query remains eligible.
        first = calibrate_scene(positions, sources, support)
        pair_index = np.flatnonzero(np.all(first['pairs'] == [0, 3], axis=1))[0]
        np.testing.assert_allclose(first['shifts_degrees'][pair_index], [[-.01, .02]]*2, atol=1e-12)
        self.assertLess(abs(first['scores']['once'][pair_index]), 1e-6)
        for arm in ('once', 'three'):
            np.testing.assert_array_equal(first['scores'][arm][~first['attempted']],
                                          first['scores']['uncalibrated'][~first['attempted']])
        changed = positions.copy()
        changed[[0, 3]] += [12., 70.]
        estimate = estimate_translation(_translation_data(changed, sources, support), [0, 3])[0]
        np.testing.assert_array_equal(estimate[[0, 2]], first['shifts_degrees'][pair_index])
        self.assertEqual(len(first['pairs']), 15)
        self.assertFalse(first['fallback'].any())
        empty_support = calibrate_scene(positions, sources, np.zeros_like(support))
        self.assertEqual(int(empty_support['fallback'].sum()), 9)
        for arm in ('once', 'three'):
            np.testing.assert_allclose(empty_support['scores'][arm], empty_support['scores']['uncalibrated'], atol=1e-7)
        # Degrees wrap at the antimeridian, not a spurious 360-degree shift.
        dateline = positions.copy()
        dateline[..., 1] = (dateline[..., 1]+69.99+180.) % 360.-180.
        wrapped = calibrate_scene(dateline, sources, support)
        np.testing.assert_allclose(wrapped['shifts_degrees'][pair_index], [[-.01, .02]]*2, atol=1e-12)

    def test_grouping_uses_repository_partition_not_all_accepted_neighbors(self):
        # Accepted chain 0--1--2 forms input-anchor groups [0,1], [2],
        # not a three-node component or overlapping edge neighbourhoods.
        record = dict(scene_id='chain', sources=np.array(['9001', '9002', '9001']),
            pairs=np.array([[0, 1], [0, 2], [1, 2]]), scores=np.array([1., -1., 1.]),
            labels=np.array([True, False, False]))
        result = _group_report(record, 0.)
        self.assertEqual(result['groups'], [[0, 1], [2]])
        self.assertEqual(result['helper_links'], 2)
        self.assertEqual(result['wrong_helper_links'], 0)
        self.assertEqual(result['mixed_anchors'], 0)

    def test_reader_never_loads_truth_and_identity_changes_only_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root/'predictions'
            directory.mkdir()
            ids = ['v']
            manifest = dict(splits={k: ids if k == 'V_select' else [] for k in
                ('T_core', 'T_head', 'V_select', 'T_test')}, all_scenes=ids,
                components={'v': 'c'}, component_members={'c': ids}, component_splits={'c': 'V_select'},
                label_members={'v': 'v'}, development_scenes=[],
                caps={k: 1 for k in ('T_core', 'T_head', 'V_select', 'T_test')})
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'run.json').write_text(json.dumps(dict(status='COMPLETED', scene_ids=ids,
                data=dict(length=64, max_observations=128, time_unit_seconds=20.),
                inference_uses_reference=False, inference_uses_identity=False, test_opened=False)))
            (directory/'index.json').write_text(json.dumps([dict(scene_id='v', path='v.npz', fragment_count=6)]))
            positions, sources, support = self.scene()

            def save(identities):
                np.savez_compressed(directory/'v.npz', scene_id=np.asarray('v'),
                    node_ids=np.arange(6), sources=sources, identities=np.asarray(identities),
                    region_inside_support=support, query_seconds=np.arange(64.),
                    **{base: positions for base in BASES},
                    target=np.asarray([object()], object), target_valid=np.asarray([object()], object))

            save(['a', 'b', 'c', 'a', 'b', 'c'])
            first, _, _ = load_records(directory, manifest)
            report = run(directory, root/'manifest.json', root/'out')
            self.assertEqual(report['population'], dict(scenes=1, components=1, nodes=6, pairs=15, positives=3))
            save(['a', 'b', 'c', 'd', 'e', 'f'])
            second, _, _ = load_records(directory, manifest)
            for method in first:
                np.testing.assert_array_equal(first[method][0]['scores'], second[method][0]['scores'])
                self.assertFalse(second[method][0]['labels'].any())
            with self.assertRaises(FileExistsError):
                run(directory, root/'manifest.json', root/'out')


if __name__ == '__main__':
    unittest.main()
