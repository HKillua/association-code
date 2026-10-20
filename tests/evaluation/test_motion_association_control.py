"""Distance comparison preserves complete population and excludes position truth."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analysis.motion_association_control import METHODS, auc, load_records, rms_haversine_scores, run


class MotionAssociationTests(unittest.TestCase):
    def test_spherical_rms_and_tie_auc(self):
        positions = np.zeros((3, 64, 2))
        positions[0, :, 1], positions[1, :, 1] = 179., -179.
        positions[2] = positions[0]
        positions[2, 32:, 0] = 2.
        scores = rms_haversine_scores(positions, np.array([[0, 1], [0, 2]]))
        np.testing.assert_allclose(scores, -6371000.*np.deg2rad(2)*np.array([1., 1/np.sqrt(2)]))
        self.assertEqual(auc([True, False, True, False], [2., 2., 1., 0.]), .625)
        self.assertIsNone(auc([], []))

    def test_complete_pairs_empty_scene_no_target_read_and_identity_only_changes_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root/'predictions'
            predictions.mkdir()
            ids = ['v', 'empty']
            manifest = dict(splits={k: ids if k == 'V_select' else [] for k in
                                   ('T_core', 'T_head', 'V_select', 'T_test')},
                all_scenes=ids, components={s: 'c' for s in ids},
                component_members={'c': ids}, component_splits={'c': 'V_select'},
                label_members={s: s for s in ids}, development_scenes=[],
                caps={k: 2 for k in ('T_core', 'T_head', 'V_select', 'T_test')})
            manifest_path = root/'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            (root/'run.json').write_text(json.dumps(dict(status='COMPLETED', scene_ids=ids,
                data=dict(length=64, max_observations=128, time_unit_seconds=20.),
                inference_uses_reference=False, inference_uses_identity=False, test_opened=False)))
            (predictions/'index.json').write_text(json.dumps([
                dict(scene_id=s, path=f'{s}.npz', fragment_count=3 if s == 'v' else 0) for s in ids]))
            support = np.zeros((3, 64), bool)
            support[0, :20], support[1, 44:], support[2, :20] = True, True, True
            positions = np.zeros((3, 64, 2))
            positions[1, :, 0], positions[2, :, 0] = .1, .2

            def save(identities):
                for sid, n in [('v', 3), ('empty', 0)]:
                    np.savez_compressed(predictions/f'{sid}.npz', scene_id=np.asarray(sid),
                        node_ids=np.asarray(['a', 'b', 'c'])[:n], sources=np.asarray(['1', '2', '1'])[:n],
                        identities=np.asarray(identities)[:n], query_seconds=np.arange(64.),
                        region_inside_support=support[:n], **{method: positions[:n] for method in METHODS},
                        # Access would fail with allow_pickle=False; neither key may be loaded.
                        target=np.asarray([object()], dtype=object), target_valid=np.asarray([object()], dtype=object))

            save(['id0', 'id0', 'id1'])
            first, _ = load_records(predictions, manifest)
            self.assertEqual(first['native_hold'][0]['pairs'].tolist(), [[0, 1], [0, 2], [1, 2]])
            self.assertEqual(first['native_hold'][0]['joint_support_queries'].tolist(), [0, 20, 0])
            self.assertTrue(first['native_hold'][0]['labels'][0])  # Disjoint-support positive retained.
            report = run(predictions, manifest_path, root/'out')
            self.assertEqual(report['population']['pairs'], 3)
            self.assertEqual(report['population']['no_joint_q_support_pairs'], 2)
            self.assertEqual(report['methods']['native_hold']['own_threshold']['coverage']['all_scenes'], 2)
            self.assertEqual(report['methods']['native_hold']['own_threshold']['coverage']['scenes_without_pairs'], 1)
            with self.assertRaises(FileExistsError):
                run(predictions, manifest_path, root/'out')
            save(['id0', 'id1', 'id2'])
            second, _ = load_records(predictions, manifest)
            for method in METHODS:
                np.testing.assert_array_equal(first[method][0]['scores'], second[method][0]['scores'])
            self.assertFalse(second['native_hold'][0]['labels'].any())
            manifest['splits']['V_select'] = ['v']
            with self.assertRaises(ValueError):
                load_records(predictions, manifest)


if __name__ == '__main__':
    unittest.main()
