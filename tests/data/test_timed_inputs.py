import copy
import json
import unittest

import numpy as np

from association.training.common import fit_normalizer
from association.data.archive import DataError
from association.data.timed import fit_timed_normalizer, pack_timed_scene, prepare_timed_scene


def native_scene(specs, scene_id='train', split='T_core'):
    tracks = []
    for index, (identity, source, timestamps) in enumerate(specs):
        times = np.asarray(timestamps, dtype=np.float64)
        values = np.column_stack([20+times/100000, 130+times/200000,
                                 5+times/10000, np.full(len(times), 350.)])
        tracks.append(dict(node_id=f'{scene_id}:{index}', identity_id=identity, source=source,
                           canonical_times=times, canonical_values=values,
                           times=times[[0, -1]], values=values[[0, -1]]))
    n = len(tracks)
    pairs = np.column_stack(np.triu_indices(n, 1)).astype(np.int64)
    identities = np.asarray([t['identity_id'] for t in tracks], dtype=str)
    return dict(scene_id=scene_id, component_id=scene_id, split=split, tracks=tracks,
        node_ids=[t['node_id'] for t in tracks], identities=identities,
        sources=[t['source'] for t in tracks], pairs=pairs,
        labels=(identities[pairs[:, 0]] == identities[pairs[:, 1]]).astype(np.float32),
        intervals=np.asarray([[t['canonical_times'][0], t['canonical_times'][-1]]
                              for t in tracks]).reshape(n, 2), audit={'original': True})


def prepare(scene, length=64, cap=128, unit=20, **kwargs):
    return prepare_timed_scene(scene, length=length, max_observations=cap,
                               time_unit_seconds=unit, **kwargs)


class TimedInputsTests(unittest.TestCase):
    def test_short_fragment_survives_zero_query_support_and_all_pairs_are_preserved(self):
        original = native_scene([('A-0', '9001', range(0, 7201, 10)),
                                 ('A-0', '9002', [6424, 6488]),
                                 ('B-0', '9001', [7000, 7200])])
        scene = prepare(original)
        self.assertEqual(scene['native_observed'].sum(axis=1).tolist(), [128, 2, 2])
        self.assertEqual(scene['observed'][1].sum(), 0)
        self.assertEqual(scene['timed_input_audit']['fragments_without_query_support'], 1)
        for field in ('pairs', 'labels', 'identities', 'sources', 'node_ids', 'intervals'):
            self.assertIs(scene[field], original[field])
        np.testing.assert_array_equal(scene['labels'], [1, 0, 0])
        np.testing.assert_array_equal(scene['native_times'][1, :2], [6424, 6488])
        np.testing.assert_array_equal(scene['grid'], np.linspace(0, 7200, 64))
        self.assertEqual(scene['timed_input_audit']['omitted_native_points'], 593)
        json.dumps(scene['timed_input_audit'], allow_nan=False)

    def test_cap_and_interpolation_use_same_selected_points_not_omitted_raw_values(self):
        original = native_scene([('A-0', '9001', [0, 10, 20, 30, 40])])
        original['tracks'][0]['canonical_values'][1, 0] = 80
        scene = prepare(original, length=5, cap=3)
        np.testing.assert_array_equal(scene['native_times'][0], [0, 20, 40])
        self.assertAlmostEqual(scene['values'][0, 1, 0], 20.0001)
        self.assertEqual(scene['timed_input_audit']['per_track'][0]['selected_indices'], [0, 2, 4])
        self.assertEqual(scene['timed_input_audit']['per_track'][0]['interpolated_query_points'], 2)
        self.assertEqual(scene['timed_input_audit']['per_track'][0]['exact_native_query_points'], 3)
        norm = fit_timed_normalizer([scene])
        points = original['tracks'][0]['canonical_values'][[0, 2, 4]]
        np.testing.assert_allclose(norm['mean'], points.mean(axis=0))
        self.assertEqual(norm['observed_point_count'], 3)

    def test_packing_has_real_tau_native_padding_and_queries_never_marked_observed(self):
        original = native_scene([('A-0', '9001', [100, 120, 140, 160]),
                                 ('A-0', '9002', [120, 140])])
        scene = prepare(original, length=4, cap=4, unit=10)
        norm = fit_timed_normalizer([scene])
        x, mask = pack_timed_scene(scene, norm)
        self.assertEqual(x.shape, (2, 8, 5))
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_array_equal(mask[:, :4], scene['native_observed'])
        self.assertFalse(mask[:, 4:].any())
        np.testing.assert_array_equal(x[0, :4, 4], [0, 2, 4, 6])
        np.testing.assert_array_equal(x[1, :2, 4], [2, 4])
        np.testing.assert_array_equal(x[:, 4:, 4], [[0, 2, 4, 6]]*2)
        self.assertTrue((x[1, 2:4] == 0).all())
        self.assertTrue(np.isnan(x[1, [4, 7], :4]).all())
        recovered = x[1, 5:7, :4]*norm['std'] + norm['mean']
        np.testing.assert_allclose(recovered, scene['values'][1, 1:3], atol=1e-7)
        np.testing.assert_array_equal(scene['observation_times'][scene['observed']],
                                      np.broadcast_to(scene['grid'], scene['observed'].shape)[scene['observed']])

    def test_circular_interpolation_crosses_north_and_dateline_by_shortest_arc(self):
        original = native_scene([('A-0', '9001', [0, 10])])
        original['tracks'][0]['canonical_values'][:, 1] = [179, -179]
        original['tracks'][0]['canonical_values'][:, 3] = [350, 10]
        scene = prepare(original, length=3, cap=2)
        self.assertEqual(scene['values'][0, 1, 1], -180)
        self.assertEqual(scene['values'][0, 1, 3], 0)
        np.testing.assert_array_equal(scene['values'][0, [0, 2]], original['tracks'][0]['canonical_values'])
        self.assertEqual(scene['timed_input_audit']['ambiguous_half_turn_count'], 0)

    def test_half_turn_tie_uses_signed_raw_difference_and_reports_ambiguity(self):
        original = native_scene([('A-0', '9001', [0, 10]), ('A-0', '9002', [0, 10])])
        for track, longitude, heading in zip(original['tracks'], ([0, 180], [0, -180]),
                                             ([0, 180], [180, 0])):
            track['canonical_values'][:, 1] = longitude
            track['canonical_values'][:, 3] = heading
        scene = prepare(original, length=3, cap=2)
        np.testing.assert_array_equal(scene['values'][:, 1, 1], [90, -90])
        np.testing.assert_array_equal(scene['values'][:, 1, 3], [90, 90])
        self.assertEqual(scene['timed_input_audit']['ambiguous_half_turn_count'], 4)
        self.assertEqual(scene['timed_input_audit']['fragments_preserved'], 2)
        self.assertIn('not established by truth', scene['timed_input_audit']['half_turn_policy'])

    def test_long_gaps_are_reported_and_not_filtered_or_called_repaired(self):
        scene = prepare(native_scene([('A-0', '9001', [0, 10000])]), length=3, cap=2)
        audit = scene['timed_input_audit']['per_track'][0]
        self.assertEqual(audit['canonical_max_gap_seconds'], 10000)
        self.assertEqual(audit['max_interpolation_bracket_seconds'], 10000)
        self.assertEqual(audit['interpolated_query_points'], 1)
        self.assertIn('not claimed repaired', scene['timed_input_audit']['interpolation_gap_policy'])

    def test_training_roles_freeze_validation_statistics_and_forbid_test(self):
        train = prepare(native_scene([('A-0', '9001', [0, 10])]))
        head = prepare(native_scene([('B-0', '9002', [0, 10])], 'head', 'T_head'))
        validation = prepare(native_scene([('V-0', '9001', [0, 10])], 'validation', 'V_select'))
        norm = fit_timed_normalizer([train, head]); before = copy.deepcopy(norm)
        pack_timed_scene(validation, norm)
        self.assertEqual(norm, before)
        self.assertEqual(norm['training_splits'], ['T_core', 'T_head'])
        with self.assertRaises(DataError):
            fit_timed_normalizer([validation])
        with self.assertRaises(DataError):
            prepare(native_scene([('X-0', '9001', [0, 10])], split='T_test'))
        with self.assertRaises(DataError):
            fit_timed_normalizer([train, train])
        leaked = copy.deepcopy(train); leaked['split'] = 'V_select'
        with self.assertRaises(DataError):
            pack_timed_scene(leaked, norm)

    def test_empty_scene_requires_explicit_physical_queries_and_can_be_packed(self):
        empty = native_scene([], 'empty', 'V_select')
        with self.assertRaisesRegex(DataError, 'explicit query_times'):
            prepare(empty, length=3, cap=2)
        scene = prepare(empty, length=3, cap=2, query_times=[100, 150, 200])
        train = prepare(native_scene([('A-0', '9001', [0, 10])]), length=3, cap=2)
        x, mask = pack_timed_scene(scene, fit_timed_normalizer([train]))
        self.assertEqual(x.shape, (0, 5, 5))
        self.assertEqual(mask.shape, (0, 5))
        self.assertEqual(scene['time_origin_seconds'], 100)
        self.assertEqual(scene['timed_input_audit']['canonical_points'], 0)


    def test_float32_cannot_change_physical_query_support_or_collapse_native_times(self):
        scene = prepare(native_scene([('A-0', '9001', [0, 1]),
                                      ('A-0', '9002', [.50000001, .6])]), length=3, cap=2, unit=1)
        self.assertFalse(scene['observed'][1].any())
        with self.assertRaisesRegex(DataError, 'precision-boundary mismatch'):
            pack_timed_scene(scene, fit_timed_normalizer([scene]))
        collapsed = prepare(native_scene([('A-0', '9001', [0, 2]),
                                          ('A-0', '9002', [1, 1.00000001])]), length=2, cap=2, unit=1)
        with self.assertRaisesRegex(DataError, 'distinct native timestamps'):
            pack_timed_scene(collapsed, fit_timed_normalizer([collapsed]))


    def test_normalization_excludes_missing_slots(self):
        values = np.array([[[1., 2.], [float("nan"), float("nan")], [3., 6.]]])
        scenes = [{"values": values, "observed": np.array([[True, False, True]]), "scene_id": "core"}]
        normalizer = fit_normalizer(scenes)
        self.assertEqual(normalizer["observed_point_count"], 2)
        self.assertEqual(normalizer["mean"], [2., 4.])
        self.assertEqual(normalizer["std"], [1., 2.])

if __name__ == '__main__':
    unittest.main()
