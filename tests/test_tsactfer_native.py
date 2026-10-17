import copy
import json
import unittest

import numpy as np

from association.data import DataError
from baselines.tsactfer_data import sample_batch
from baselines.tsactfer_native import (build_native_positive_pools, fit_native_normalizer,
                                      pack_native_scene)


def native_scene(specs, scene_id='train', split='T_core'):
    tracks = []
    for node, (identity, source, timestamps) in enumerate(specs):
        times = np.asarray(timestamps, dtype=np.float64)
        values = np.column_stack([20 + times / 10000, 130 + times / 20000,
                                  np.full(len(times), 5), np.full(len(times), 180)])
        tracks.append({'node_id': f'{scene_id}:{node}', 'identity_id': identity,
                       'source': source, 'canonical_times': times, 'canonical_values': values,
                       # An earlier view intentionally has only endpoints. Native TSA
                       # must explicitly reselect canonical points at its declared cap.
                       'times': times[[0, -1]], 'values': values[[0, -1]],
                       'selected_indices': np.asarray([0, len(times)-1])})
    n = len(tracks)
    pairs = np.column_stack(np.triu_indices(n, 1)).astype(np.int64)
    identities = np.asarray([row[0] for row in specs], dtype=str)
    return {'scene_id': scene_id, 'split': split, 'component_id': scene_id,
            'tracks': tracks, 'node_ids': [t['node_id'] for t in tracks],
            'identities': identities, 'sources': [t['source'] for t in tracks],
            'intervals': np.asarray([[t['canonical_times'][0], t['canonical_times'][-1]]
                                     for t in tracks], dtype=np.float64).reshape(n, 2),
            'pairs': pairs, 'labels': (identities[pairs[:, 0]] == identities[pairs[:, 1]]).astype(np.float32),
            'audit': {'native_inputs': {'valid_fragments': n}}}


class TSACtferNativeTests(unittest.TestCase):
    def test_native_selection_retains_short_fragment_and_audits_real_point_loss(self):
        scene = native_scene([('A-0', '9001', list(range(0, 7201, 10))),
                              ('A-0', '9002', [6424, 6488])])
        norm = fit_native_normalizer([scene], max_points=128)
        packed = pack_native_scene(scene, norm, max_points=128)
        self.assertEqual(packed['positions'].shape, (2, 128, 2))
        self.assertEqual(packed['observed'].sum(axis=1).tolist(), [128, 2])
        np.testing.assert_array_equal(packed['times'][1, :2], [6424, 6488])
        self.assertEqual(packed['times'][0, 0], 0)
        self.assertEqual(packed['times'][0, 127], 7200)
        audit = packed['packing_audit']
        self.assertEqual((audit['canonical_points'], audit['sampled_points']), (723, 130))
        self.assertEqual(audit['discarded_observations'], 593)
        self.assertEqual(audit['discarded_at_padding'], 0)
        self.assertEqual(audit['fragments_preserved'], 2)
        self.assertEqual(audit['clipped_time_count'], 0)
        self.assertNotIn('grid', packed)
        self.assertNotIn('observation_times', packed)
        json.dumps(audit, allow_nan=False)

    def test_frozen_training_moments_use_selected_canonical_points_only(self):
        train = native_scene([('A-0', '9001', [0, 5, 20, 50, 100])])
        # Omitted index 1 must not enter capped training moments.
        train['tracks'][0]['canonical_values'][1, :2] = [50, 150]
        norm = fit_native_normalizer([train], max_points=3)
        expected = train['tracks'][0]['canonical_values'][[0, 2, 4]][:, [1, 0]]
        np.testing.assert_allclose(norm['mean'], expected.mean(axis=0))
        np.testing.assert_allclose(norm['std'], expected.std(axis=0))
        self.assertEqual(norm['observed_point_count'], 3)
        self.assertEqual(norm['training_splits'], ['T_core'])
        validation = native_scene([('V-0', '9002', [0, 20, 100])], 'validation', 'V_select')
        validation['tracks'][0]['canonical_values'][:, :2] += .1
        before = copy.deepcopy(norm)
        packed = pack_native_scene(validation, norm, max_points=3)
        self.assertEqual(norm, before)
        recovered = packed['positions'][0, :3] * norm['std'] + norm['mean']
        np.testing.assert_allclose(recovered, validation['tracks'][0]['canonical_values'][:, [1, 0]],
                                   rtol=0, atol=1e-7)

    def test_training_roles_reject_validation_test_and_implicit_split(self):
        for split in ['V_select', 'T_test', None]:
            scene = native_scene([('A-0', '9001', [0, 200])], split=split)
            with self.subTest(split=split), self.assertRaises(DataError):
                fit_native_normalizer([scene], max_points=128)
            with self.subTest(pool_split=split), self.assertRaises(DataError):
                build_native_positive_pools([scene])
        core = native_scene([('A-0', '9001', [0, 200])])
        head = native_scene([('B-0', '9002', [0, 200])], 'head', 'T_head')
        norm = fit_native_normalizer([core, head], max_points=128)
        self.assertEqual(norm['training_splits'], ['T_core', 'T_head'])
        sealed = native_scene([('X-0', '9001', [0, 200])], 'test', 'T_test')
        with self.assertRaises(DataError):
            pack_native_scene(sealed, norm, max_points=128)

    def test_origin_is_minimum_canonical_start_not_first_node_or_incoming_view(self):
        scene = native_scene([('A-0', '9001', [120, 160, 300]),
                              ('B-0', '9002', [100, 200])])
        scene['tracks'][1]['times'] = np.asarray([200.])  # Ignored earlier view.
        norm = fit_native_normalizer([scene], max_points=3)
        packed = pack_native_scene(scene, norm, max_points=3)
        self.assertEqual(packed['packing_audit']['time_origin'], 100)
        np.testing.assert_array_equal(packed['time_ids'][0, :3], [1, 3, 10])
        np.testing.assert_array_equal(packed['time_ids'][1, :2], [0, 5])

    def test_vocabulary_overflow_fails_without_clipping_or_rescaling(self):
        scene = native_scene([('A-0', '9001', [0, 20, 40, 100])])
        norm = fit_native_normalizer([scene], max_points=4)
        before = scene['tracks'][0]['canonical_times'].copy()
        with self.assertRaisesRegex(DataError, '2 timestamps exceed'):
            pack_native_scene(scene, norm, max_points=4, time_bins=2)
        np.testing.assert_array_equal(scene['tracks'][0]['canonical_times'], before)
        packed = pack_native_scene(scene, norm, max_points=4, time_bins=6)
        np.testing.assert_array_equal(packed['time_ids'][0, :4], [0, 1, 2, 5])

    def test_cap_is_explicit_and_compatible_with_packing_and_frozen_statistics(self):
        scene = native_scene([('A-0', '9001', [0, 20, 40, 100])])
        for cap in [None, True, 1, 2.5]:
            with self.subTest(cap=cap), self.assertRaises(DataError):
                fit_native_normalizer([scene], max_points=cap)
        norm = fit_native_normalizer([scene], max_points=4)
        with self.assertRaisesRegex(DataError, 'length'):
            pack_native_scene(scene, norm, max_points=4, length=3)
        with self.assertRaisesRegex(DataError, 'same point cap'):
            pack_native_scene(scene, norm, max_points=3)
        with self.assertRaisesRegex(DataError, 'repeated'):
            fit_native_normalizer([scene, scene], max_points=4)
        leaked = copy.deepcopy(scene)
        leaked['split'] = 'V_select'
        with self.assertRaisesRegex(DataError, 'included'):
            pack_native_scene(leaked, norm, max_points=4)

    def test_all_candidate_metadata_order_and_inputs_are_unchanged(self):
        scene = native_scene([('A-0', '9001', [0, 100, 200]),
                              ('A-0', '9001', [5000, 5100, 5200]),
                              ('B-0', '9002', [0, 7000])])
        before = copy.deepcopy(scene)
        norm = fit_native_normalizer([scene], max_points=2)
        packed = pack_native_scene(scene, norm, max_points=2)
        for key in ['pairs', 'labels', 'identities', 'sources', 'node_ids', 'intervals']:
            self.assertIs(packed[key], scene[key])
            np.testing.assert_array_equal(packed[key], before[key])
        self.assertEqual(len(packed['pairs']), 3)
        np.testing.assert_array_equal(packed['labels'], [1, 0, 0])
        for track, original in zip(scene['tracks'], before['tracks']):
            for key in ['times', 'values', 'canonical_times', 'canonical_values']:
                np.testing.assert_array_equal(track[key], original[key])
        self.assertEqual(scene['audit'], before['audit'])
        for key in ['positions', 'times', 'time_ids']:
            self.assertTrue((packed[key][~packed['observed']] == 0).all())

    def test_positive_pools_and_existing_sampler_use_original_intervals(self):
        specs = [(identity, source, times) for identity in ('A-0', 'B-0')
                 for source, times in [('9001', [0, 100, 200, 300, 400]),
                                       ('9002', [50, 150, 250, 350, 450])]]
        scene = native_scene(specs)
        norm = fit_native_normalizer([scene], max_points=3)
        packed = pack_native_scene(scene, norm, max_points=3)
        pools, audit = build_native_positive_pools([packed])
        self.assertEqual(audit['eligible_pair_counts'], {'ITSA': 0, 'MSTSA': 2})
        batch = sample_batch([packed], pools, 'MSTSA', 2, np.random.default_rng(1))
        self.assertEqual(batch['positions'].shape, (4, 128, 2))
        self.assertEqual(set(batch['identities']), {'A-0', 'B-0'})
        self.assertTrue(batch['observed'].sum(axis=1).min() >= 2)

    def test_corrupt_time_or_metadata_fails_instead_of_repairing_order(self):
        base = native_scene([('A-0', '9001', [0, 20, 100])])
        for change in ['times', 'interval', 'identity']:
            scene = copy.deepcopy(base)
            if change == 'times':
                scene['tracks'][0]['canonical_times'][1] = 0
            elif change == 'interval':
                scene['intervals'][0, 1] = 99
            else:
                scene['tracks'][0]['identity_id'] = 'wrong'
            with self.subTest(change=change), self.assertRaises(DataError):
                fit_native_normalizer([scene], max_points=3)

    def test_empty_scene_reports_no_points_without_inventing_observations(self):
        train = native_scene([('A-0', '9001', [0, 200])])
        norm = fit_native_normalizer([train], max_points=128)
        empty = native_scene([], 'empty', 'V_select')
        packed = pack_native_scene(empty, norm, max_points=128)
        self.assertEqual(packed['positions'].shape, (0, 128, 2))
        self.assertEqual(packed['packing_audit']['sampled_points'], 0)
        self.assertFalse(packed['packing_audit']['origin_has_observations'])
        self.assertIsNone(packed['packing_audit']['point_retention_fraction'])


if __name__ == '__main__':
    unittest.main()
