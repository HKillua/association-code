"""Hand-calculated supervision alignment; no archive or native data access."""
import copy
import json
import unittest

import numpy as np

from association.data import DataError
from association.fusion_targets import align_reference_states


def scene(rows, split='T_core'):
    return {
        'scene_id': 'synthetic', 'split': split, 'member': 'synthetic-reference.csv',
        'state_fields': ['lat', 'lon', 'vel', 'cou'], 'issues': [],
        'rows': [{'line': i + 2, 'identity_id': identity, 'time': time,
                  'state': list(state), 'raw': {'MMSI': identity, 'time': str(time)}}
                 for i, (identity, time, state) in enumerate(rows)],
    }


def align(data, queries, identity='ship-0', offset=0, gap=10):
    return align_reference_states(data, identity, queries,
                                  time_offset_seconds=offset, max_gap_seconds=gap)


class FusionTargetTests(unittest.TestCase):
    def test_explicit_time_offset_sorting_linear_values_and_native_velocity(self):
        data = scene([('ship-0', 110, [20, 40, 800, 20]),
                      ('other-0', 105, [89, 179, 1, 200]),
                      ('ship-0', 100, [10, 20, 400, 0])])
        result = align(data, [0, 5, 10], offset=100)
        np.testing.assert_allclose(result['states'], [[10, 20, 400, 0],
                                                     [15, 30, 600, 10],
                                                     [20, 40, 800, 20]])
        self.assertTrue(result['valid'].all())
        self.assertEqual(result['states'].dtype, np.float64)
        self.assertEqual(result['valid'].dtype, np.bool_)
        self.assertEqual(result['report']['identity_rows'], 2)
        self.assertEqual(result['report']['exact_queries'], 2)
        self.assertEqual(result['report']['interior_queries'], 1)
        self.assertEqual(result['report']['velocity_units'], 'native; no conversion')

    def test_missing_identity_does_not_create_zero_targets(self):
        result = align(scene([('other-0', 0, [1, 2, 3, 4])]), [0, 1])
        self.assertTrue(np.isnan(result['states']).all())
        self.assertFalse(result['valid'].any())
        self.assertTrue(result['report']['missing_identity'])
        self.assertEqual(result['report']['identity_rows'], 0)
        self.assertEqual(result['report']['no_reference_time_queries'], 2)
        self.assertEqual(result['report']['unavailable_queries'], 2)
        self.assertEqual(result['report']['valid_per_feature'], dict.fromkeys(('lat','lon','vel','cou'), 0))

    def test_bad_feature_remains_on_axis_and_cannot_be_bridged(self):
        data = scene([('ship-0', 0, [0, 0, 2, 10]),
                      ('ship-0', 5, [5, 10, -1, 20]),
                      ('ship-0', 10, [10, 20, 6, 30])])
        result = align(data, [0, 2.5, 5, 7.5, 10], gap=20)
        np.testing.assert_allclose(result['states'][:, :2], [[0,0],[2.5,5],[5,10],[7.5,15],[10,20]])
        np.testing.assert_array_equal(result['valid'][:, 2], [True,False,False,False,True])
        self.assertTrue(np.isnan(result['states'][1:4, 2]).all())
        self.assertTrue(result['valid'][:, [0,1,3]].all())
        self.assertEqual(result['report']['invalid_feature_rows']['vel'], 1)
        self.assertIn({'code':'out_of_range','lines':[3],'field':'vel'}, result['report']['issues'])

    def test_invalid_coordinates_nonfinite_nonnumeric_and_missing_course_are_masked(self):
        data = scene([('ship-0', 0, [91, -181, float('nan'), -1]),
                      ('ship-0', 1, [None, float('inf'), -0.01, 361]),
                      ('ship-0', 2, [-90, 180, 0, 360])])
        result = align(data, [0, 1, 2])
        self.assertFalse(result['valid'][:2].any())
        self.assertTrue(np.isnan(result['states'][:2]).all())
        np.testing.assert_equal(result['states'][2], [-90, 180, 0, 0])
        self.assertTrue(result['valid'][2].all())
        self.assertEqual(result['report']['invalid_feature_rows'], dict.fromkeys(('lat','lon','vel','cou'), 2))
        self.assertEqual(result['report']['course_360_reference_times'], 1)
        codes = {item['code'] for item in result['report']['issues']}
        self.assertTrue({'nonnumeric','nonfinite','out_of_range'}.issubset(codes))

    def test_duplicate_identical_and_feature_conflicts_are_explicit(self):
        data = scene([('ship-0', 0, [0, 10, 2, 20]),
                      ('ship-0', 0, [0, 10, 2, 20]),
                      ('ship-0', 5, [5, 20, 4, 40]),
                      ('ship-0', 5, [5, 21, 4, 40]),
                      ('ship-0', 10, [10, 30, 6, 60])])
        result = align(data, [0, 2.5, 5, 7.5, 10])
        self.assertTrue(result['valid'][0].all())
        self.assertFalse(result['valid'][1:4, 1].any())
        self.assertTrue(result['valid'][:, [0,2,3]].all())
        self.assertEqual(result['report']['duplicate_time_groups'], 2)
        self.assertEqual(result['report']['duplicate_extra_rows'], 2)
        self.assertEqual(result['report']['conflicting_duplicate_features']['lon'], 1)
        self.assertIn({'code':'duplicate_conflict','field':'lon','time':5.0,'lines':[4,5]},
                      result['report']['issues'])

    def test_invalid_duplicate_does_not_silently_take_the_valid_row(self):
        data = scene([('ship-0', 0, [1, 2, 3, 0]),
                      ('ship-0', 0, [1, 2, None, 360]),
                      ('ship-0', 10, [2, 3, 4, 10])])
        result = align(data, [0, 5, 10])
        self.assertTrue(result['valid'][:, :2].all())
        self.assertFalse(result['valid'][:2, 2:].any())
        self.assertTrue(result['valid'][2].all())
        self.assertEqual(result['report']['conflicting_duplicate_features']['cou'], 1)
        self.assertEqual(result['report']['invalid_feature_rows']['vel'], 1)

    def test_course_uses_shortest_arc_and_masks_180_degree_ambiguity(self):
        data = scene([('ship-0', 0, [1, 2, 3, 350]),
                      ('ship-0', 10, [1, 2, 3, 10]),
                      ('ship-0', 20, [1, 2, 3, 190])])
        result = align(data, [0, 5, 10, 15, 20])
        np.testing.assert_allclose(result['states'][[0,1,2,4], 3], [350, 0, 10, 190])
        self.assertFalse(result['valid'][3, 3])
        self.assertTrue(np.isnan(result['states'][3, 3]))
        self.assertTrue(result['valid'][:, :3].all())
        self.assertEqual(result['report']['ambiguous_heading_queries'], 1)
        reverse = align(scene([('ship-0',0,[0,0,1,10]),('ship-0',10,[0,0,1,350])]), [5])
        self.assertEqual(reverse['states'][0,3], 0)

    def test_exact_only_gap_limit_boundary_and_no_extrapolation(self):
        data = scene([('ship-0', 0, [1, 2, 3, 4]), ('ship-0', 10, [5, 6, 7, 8])])
        exact = align(data, [-1, 0, 5, 10, 11], gap=0)
        np.testing.assert_array_equal(exact['valid'].all(1), [False,True,False,True,False])
        self.assertEqual(exact['report']['outside_support_queries'], 2)
        self.assertEqual(exact['report']['gap_exceeded_queries'], 1)
        self.assertFalse(align(data, [5], gap=9.999)['valid'].any())
        self.assertTrue(align(data, [5], gap=10)['valid'].all())
        self.assertFalse(align(data, [-1,11], gap=100)['valid'].any())
        one = align(scene([('ship-0',5,[1,2,3,4])]), [4,5,6], gap=100)
        np.testing.assert_array_equal(one['valid'].all(1), [False,True,False])

    def test_unlocatable_reference_times_are_reported_not_interpolated(self):
        data = scene([('ship-0',float('nan'),[1,2,3,4]),
                      ('ship-0',float('inf'),[1,2,3,4]),
                      ('ship-0',None,[1,2,3,4]),
                      ('ship-0',0,[1,2,3,4])])
        result = align(data, [0,1])
        self.assertEqual(result['report']['unlocatable_time_rows'], 3)
        self.assertEqual(result['report']['reference_times'], 1)
        self.assertTrue(result['valid'][0].all())
        self.assertFalse(result['valid'][1].any())
        self.assertEqual({x['code'] for x in result['report']['issues']}, {'nonfinite_time','nonnumeric_time'})
        overflow = align(scene([('ship-0',1e308,[1,2,3,4])]), [0], offset=-1e308)
        self.assertFalse(overflow['valid'].any())
        self.assertEqual(overflow['report']['issues'][0]['code'], 'nonfinite_aligned_time')

    def test_reader_inputs_queries_and_raw_strings_remain_unchanged(self):
        data = scene([('ship-0',10,[10,20,30,360]),('ship-0',0,[None,10,20,350])])
        data['issues'] = [{'code':'existing', 'lines':[3]}]
        before = json.dumps(data, sort_keys=True)
        query = np.asarray([0.,5.,10.])
        original_query = query.copy()
        result = align(data, query)
        result['states'][:] = -999
        result['report']['issues'].clear()
        self.assertEqual(json.dumps(data, sort_keys=True), before)
        np.testing.assert_array_equal(query, original_query)

    def test_only_declared_training_development_splits_are_allowed(self):
        data = scene([('ship-0',0,[1,2,3,4])])
        for split in ('T_core','T_head','V_select'):
            data['split'] = split
            self.assertTrue(align(data, [0])['valid'].all())
        for split in ('T_test','test','unknown',None):
            data['split'] = split
            with self.subTest(split=split), self.assertRaisesRegex(DataError, 'split'):
                align(data, [0])

    def test_queries_and_explicit_policy_arguments_are_validated(self):
        data = scene([('ship-0',0,[1,2,3,4])])
        for queries in ([0,0], [1,0], [0,float('nan')], [0,float('inf')], [[0]], ['0'], [True]):
            with self.subTest(queries=queries), self.assertRaises(DataError):
                align(data, queries)
        for offset, gap in ((float('inf'),1),(float('nan'),1),(True,1),(0,-1),(0,float('inf')),(0,False)):
            with self.subTest(offset=offset,gap=gap), self.assertRaises(DataError):
                align(data, [0], offset=offset, gap=gap)
        with self.assertRaises(TypeError):
            align_reference_states(data, 'ship-0', [0])
        wrong_fields = copy.deepcopy(data)
        wrong_fields['state_fields'] = ['lon','lat','vel','cou']
        with self.assertRaises(DataError):
            align(wrong_fields, [0])
        empty = align(data, [])
        self.assertEqual(empty['states'].shape, (0,4))
        self.assertEqual(empty['valid'].shape, (0,4))
        self.assertEqual(empty['report']['fully_valid_queries'], 0)

    def test_invalid_reader_identity_cannot_become_supervision(self):
        result = align(scene([('broken',0,[1,2,3,4])]), [0], identity='broken')
        self.assertFalse(result['valid'].any())
        self.assertEqual(result['report']['invalid_identity_rows'], 1)
        self.assertEqual(result['report']['issues'][0]['code'], 'invalid_identity')


if __name__ == '__main__':
    unittest.main()
