"""Hand-calculated supervision alignment; no archive or native data access."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from association.data import DataError, Scene, Track
from association.dataset import prepare_scene
from association.fusion_targets import (
    align_reference_states, build_fusion_targets, build_scene_fusion_targets,
)


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


class FusionTargetAssemblyTests(unittest.TestCase):
    def setUp(self):
        tracks = []
        for index, (identity, start, stop) in enumerate((('ship-0', 0, 4), ('ship-0', 2, 8),
                                                       ('other-0', 1, 7), ('other-0', 6, 10))):
            tracks.append(Track(node_id=f'node-{index}', source=str(index), batch=str(index),
                                identity_id=identity, ancestor_id=identity.split('-')[0],
                                times=np.asarray([start, stop], dtype=float),
                                values=np.asarray([[10 + index, 20, 4, 350],
                                                   [12 + index, 24, 6, 10]], dtype=float)))
        raw = Scene('synthetic', tracks, {})
        # Execute real preparation/registration without opening any archive.
        archive = SimpleNamespace(load_scene=lambda scene_id: raw)
        self.prepared = prepare_scene(archive, 'synthetic', 6, 'synthetic-component')
        self.positive = np.flatnonzero(self.prepared['labels'] == 1)
        self.reference = scene([('ship-0', 100, [10, 20, 4, 350]),
                                ('ship-0', 110, [20, 40, 8, 10]),
                                ('other-0', 100, [30, 50, 10, 20]),
                                ('other-0', 110, [40, 70, 20, 40])])

    def build(self, indices=None, **overrides):
        arguments = {'split': 'T_core', 'nominal_grid': self.prepared['grid'],
                     'time_offset_seconds': 100, 'max_gap_seconds': 10}
        arguments.update(overrides)
        return build_fusion_targets(self.prepared, self.reference,
                                    self.positive if indices is None else indices, **arguments)

    def test_real_preparation_repeated_pair_order_and_one_alignment_per_identity(self):
        indices = self.positive[[1, 0, 1]]
        original = indices.copy()
        with patch('association.fusion_targets.align_reference_states', wraps=align_reference_states) as aligner:
            result = self.build(indices)
        self.assertEqual(aligner.call_count, 2)
        self.assertEqual([call.args[1] for call in aligner.call_args_list], ['other-0', 'ship-0'])
        np.testing.assert_array_equal(result['pairs'], self.prepared['pairs'][indices])
        np.testing.assert_array_equal(result['query_times'], [0, 2, 4, 6, 8, 10])
        np.testing.assert_array_equal(indices, original)
        self.assertEqual(result['identity_ids'], ['other-0', 'ship-0', 'other-0'])
        self.assertEqual(result['states'].shape, (3, 6, 4))
        np.testing.assert_allclose(result['states'][1, 1], [12, 24, 4.8, 354])
        np.testing.assert_array_equal(result['states'][0], result['states'][2])
        self.assertTrue(result['valid'].all())
        self.assertEqual(result['report']['pair_count'], 3)
        self.assertEqual(result['report']['unique_identity_count'], 2)
        self.assertEqual(result['report']['repeated_selection_count'], 1)
        self.assertEqual(result['report']['valid_per_feature'], dict.fromkeys(('lat','lon','vel','cou'), 18))
        result['states'][0] = -999
        self.assertFalse((result['states'][2] == -999).any())

    def test_partial_and_missing_supervision_retain_every_selected_pair(self):
        self.reference = scene([('ship-0', 102, [10, 20, 4, 350]),
                                ('ship-0', 108, [16, 32, None, 10])])
        result = self.build(self.positive[[0, 1, 1]])
        self.assertEqual(result['pairs'].shape, (3, 2))
        np.testing.assert_array_equal(result['valid'][0, :, 0], [False, True, True, True, True, False])
        np.testing.assert_array_equal(result['valid'][0, :, 2], [False, True, False, False, False, False])
        self.assertFalse(result['valid'][1:].any())
        self.assertTrue(np.isnan(result['states'][1:]).all())
        self.assertTrue(np.isnan(result['states'][~result['valid']]).all())
        self.assertEqual(result['report']['valid_per_pair_feature'], [[4,4,1,4],[0,0,0,0],[0,0,0,0]])
        self.assertEqual(result['report']['unavailable_pairs'], 2)
        self.assertEqual(result['report']['missing_identity_ids'], ['other-0'])

    def test_inputs_and_returned_metadata_do_not_alias(self):
        before = copy.deepcopy(self.prepared)
        reference_before = json.dumps(self.reference, sort_keys=True)
        grid = self.prepared['grid'].copy()
        result = self.build(nominal_grid=grid)
        result['pairs'][:] = 0
        result['query_times'][:] = -1
        result['identity_ids'][0] = 'changed-0'
        result['report']['per_identity']['ship-0']['issues'].append({'code': 'changed'})
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_array_equal(self.prepared[key], before[key])
            else:
                self.assertEqual(self.prepared[key], before[key])
        np.testing.assert_array_equal(grid, before['grid'])
        self.assertEqual(json.dumps(self.reference, sort_keys=True), reference_before)

    def test_scene_and_split_mismatches_are_rejected(self):
        for split in ('T_core', 'T_head', 'V_select'):
            self.reference['split'] = split
            self.prepared['split'] = split
            self.assertTrue(self.build(split=split)['valid'].all())
        self.reference['split'] = 'T_core'
        self.prepared['split'] = 'T_core'
        for split in ('T_head', 'T_test', None):
            with self.subTest(split=split), self.assertRaises(DataError):
                self.build(split=split)
        self.prepared['split'] = 'T_head'
        with self.assertRaisesRegex(DataError, 'split'):
            self.build()
        self.prepared['split'] = 'T_core'
        self.reference['scene_id'] = 'another-scene'
        with self.assertRaisesRegex(DataError, 'scene_id'):
            self.build()
        self.reference['scene_id'] = 'synthetic'
        del self.prepared['scene_id']
        with self.assertRaisesRegex(DataError, 'scene_id'):
            self.build()

    def test_negative_contradictory_self_and_invalid_candidate_pairs_fail(self):
        negative = int(np.flatnonzero(self.prepared['labels'] == 0)[0])
        with self.assertRaisesRegex(DataError, 'label 1'):
            self.build([negative])
        self.prepared['labels'][negative] = 1
        with self.assertRaisesRegex(DataError, 'identities'):
            self.build([negative])
        positive = int(self.positive[0])
        for pair in ([0,0], [-1,1], [0,len(self.prepared['identities'])]):
            self.prepared['pairs'][positive] = pair
            with self.subTest(pair=pair), self.assertRaises(DataError):
                self.build([positive])
        self.prepared['pairs'][positive] = [0,1]
        self.prepared['identities'][:2] = ['invalid', 'invalid']
        with self.assertRaisesRegex(DataError, 'A-B'):
            self.build([positive])

    def test_indices_are_integer_rows_without_wrapping_or_coercion(self):
        for indices in ([-1], [len(self.prepared['pairs'])], [0.0], [True], [False,0],
                        [[0]], 0, ['0'], np.asarray([2**64 - 1], dtype=np.uint64)):
            with self.subTest(indices=indices), self.assertRaises(DataError):
                self.build(indices)

    def test_only_matching_nominal_grid_and_scene_shapes_are_accepted(self):
        actual = self.prepared['observation_times'][2]
        self.assertFalse(np.array_equal(actual, self.prepared['grid']))
        for grid in (actual, self.prepared['grid'] + 1, [0,2,4], [0,2,4,6,8,float('nan')],
                     [0,2,4,6,8,float('inf')], [0,2,4,4,8,10], [0,2,4,6,10,8], [[0,2,4,6,8,10]]):
            with self.subTest(grid=grid), self.assertRaisesRegex(DataError, 'grid'):
                self.build(nominal_grid=grid)
        self.prepared['observed'] = self.prepared['observed'][:, :-1]
        with self.assertRaisesRegex(DataError, 'observed'):
            self.build()
        self.prepared['values'] = self.prepared['values'][:, :, :3]
        with self.assertRaisesRegex(DataError, 'values'):
            self.build()

    def test_empty_selection_retains_schema_and_still_validates_policy(self):
        with patch('association.fusion_targets.align_reference_states') as aligner:
            result = self.build([])
        aligner.assert_not_called()
        self.assertEqual(result['pairs'].shape, (0,2))
        self.assertEqual(result['states'].shape, (0,6,4))
        self.assertEqual(result['valid'].shape, (0,6,4))
        self.assertEqual(result['report']['pair_count'], 0)
        self.assertEqual(result['report']['unavailable_pairs'], 0)
        for arguments in ({'time_offset_seconds': float('inf')}, {'max_gap_seconds': -1},
                          {'time_offset_seconds': True}, {'max_gap_seconds': float('nan')}):
            with self.subTest(arguments=arguments), self.assertRaises(DataError):
                self.build([], **arguments)
        self.reference['state_fields'] = ['lon','lat','vel','cou']
        with self.assertRaises(DataError):
            self.build([])

    def test_prepared_targets_connect_tcm_fusion_and_masked_loss_on_cpu(self):
        import torch
        from association.fusion import FusionLoss, PairFusion, PairFusionConfig
        from association.models import AssociationModel, ModelConfig
        from association.reference import fit_normalizer, normalize_scene

        # The selected pair remains present with partial reference support. Only
        # observed measurements fit the normalizer; reference states stay native.
        self.reference = scene([('ship-0', 102, [10, 20, 4, 350]),
                                ('ship-0', 108, [16, 32, 6, 10])])
        assembled = self.build([int(self.positive[0])])
        normalized, observed = normalize_scene(self.prepared, fit_normalizer([self.prepared]))
        self.assertTrue((~observed[:2]).any())
        self.assertTrue(np.isnan(assembled['states']).any())
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20)
                association = AssociationModel(ModelConfig(length=6, d_model=8, nhead=2,
                                               num_layers=1, dim_feedforward=16, dropout=0))
                fusion = PairFusion(PairFusionConfig(fc_hidden=12, length=6, d_model=8,
                                     nhead=2, num_layers=1, dim_feedforward=16, dropout=0))
                objective = FusionLoss(position_loss_mode='standard_smooth_l1')
                complete = association.complete(torch.from_numpy(normalized), torch.from_numpy(observed))
                complete.retain_grad()
                prediction = fusion.forward_pairs(complete, torch.from_numpy(assembled['pairs']))['states']
                target = torch.as_tensor(assembled['states'], dtype=prediction.dtype)
                mask = torch.from_numpy(assembled['valid'])
                total = objective(prediction, target, mask)['total']
                self.assertTrue(torch.isfinite(total))
                total.backward()
                for module in (association.tcm_encoder, association.tcm_decoder,
                               fusion.left_encoder, fusion.right_encoder, fusion.decoder,
                               fusion.state_projection, objective):
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                self.assertGreater(complete.grad[:2][~torch.from_numpy(observed[:2])].abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None for p in association.tam_encoder.parameters()))
        finally:
            torch.set_num_threads(previous_threads)


class SceneFusionTargetGroupingTests(unittest.TestCase):
    def prepared(self, identities):
        return {'scene_id': 'synthetic', 'split': 'T_core',
                'identities': list(identities), 'grid': np.asarray([0., 5., 10., 15., 20.])}

    def build(self, data, identities, **overrides):
        options = {'time_offset_seconds': 100, 'max_gap_seconds': 10}
        options.update(overrides)
        return build_scene_fusion_targets(self.prepared(identities), data,
                                         target_identity_ids=identities, **options)

    def assert_matches_public(self, data, identities):
        query = self.prepared(identities)['grid']
        expected = {identity: align_reference_states(data, identity, query,
                    time_offset_seconds=100, max_gap_seconds=10) for identity in identities}
        actual = self.build(data, identities)
        self.assertEqual(list(actual), identities)
        for identity in identities:
            for key in ('states', 'valid', 'query_times'):
                np.testing.assert_array_equal(actual[identity][key], expected[identity][key])
            self.assertEqual(actual[identity]['report'], expected[identity]['report'])
        return actual

    def test_grouped_scene_matches_public_values_masks_reports_and_issue_order(self):
        data = scene([('other-0', 120, [30, 40, 7, 190]),
                      ('ship-0', 110, [10, 20, None, 0]),
                      ('other-0', 100, [10, 20, 3, 350]),
                      ('ship-0', 100, [0, 10, 2, 360]),
                      ('ship-0', 110, [10, 21, 4, 360]),
                      ('other-0', 110, [20, 30, 5, 10]),
                      ('ship-0', None, [91, 22, -1, 370]),
                      ('unrequested-0', 100, [1, 2, 3, 4])])
        # An unrequested malformed state was historically ignored, not repaired.
        data['rows'][-1]['state'] = ['malformed']
        data['issues'] = [{'code': 'original-first'}, {'code': 'original-second'}]
        before = json.dumps(data, sort_keys=True)
        actual = self.assert_matches_public(data, ['missing-0', 'ship-0', 'other-0'])
        for item in actual.values():
            self.assertEqual(item['report']['reference_rows'], 8)
            self.assertEqual(item['report']['source_issue_count'], 2)
        self.assertTrue(actual['missing-0']['report']['missing_identity'])
        self.assertGreater(len(actual['ship-0']['report']['issues']), 4)
        actual['ship-0']['states'][:] = -999
        actual['ship-0']['report']['issues'].clear()
        actual['other-0']['query_times'][:] = -1
        self.assertEqual(json.dumps(data, sort_keys=True), before)
        np.testing.assert_array_equal(actual['missing-0']['query_times'], [0, 5, 10, 15, 20])

    def test_whole_reference_scan_count_is_independent_of_target_count(self):
        class CountedRows(list):
            iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        identities = [f'ship-{index}' for index in range(8)]
        data = scene([(identity, 100, [1, 2, 3, 4]) for identity in identities])
        data['rows'] = CountedRows(data['rows'])
        result = self.build(data, identities)
        self.assertEqual(len(result), 8)
        self.assertEqual(data['rows'].iterations, 2)
        self.assertTrue(all(item['valid'][0].all() for item in result.values()))

    def test_illegal_rows_and_states_keep_public_error_order_and_ignore_forged_index(self):
        examples = []
        malformed_first = scene([('ship-0', 100, [1, 2, 3, 4])])
        malformed_first['rows'][0]['state'] = [1]
        malformed_first['rows'].append(None)
        examples.append(malformed_first)
        nonmapping_first = scene([('ship-0', 100, [1, 2, 3, 4])])
        nonmapping_first['rows'].insert(0, None)
        examples.append(nonmapping_first)
        malformed_later_target = scene([('ship-0', 100, [1, 2, 3, 4]),
                                        ('other-0', 100, [1, 2, 3, 4])])
        malformed_later_target['rows'][1]['state'] = [1]
        examples.append(malformed_later_target)
        for data in examples:
            data['rows_by_identity'] = {'ship-0': [], 'other-0': []}
            with self.subTest(data=data):
                with self.assertRaises(DataError) as expected:
                    for identity in ('ship-0', 'other-0'):
                        align(data, [0, 5, 10, 15, 20], identity=identity, offset=100)
                with self.assertRaises(DataError) as actual:
                    self.build(data, ['ship-0', 'other-0'])
                self.assertEqual(str(actual.exception), str(expected.exception))

    def test_unusual_reader_identity_values_preserve_public_comparison_behavior(self):
        data = scene([('ship-0', 100, [1, 2, 3, 4]),
                      (['unhashable'], 100, [1, 2, 3, 4]),
                      (np.asarray('other-0'), 100, [5, 6, 7, 8])])
        actual = self.assert_matches_public(data, ['ship-0', 'other-0', 'missing-0'])
        self.assertTrue(actual['other-0']['valid'][0].all())

    def test_empty_scene_still_checks_reference_rows_and_alignment_policy(self):
        data = scene([])
        self.assertEqual(self.build(data, []), {})
        with self.assertRaisesRegex(DataError, 'max_gap_seconds'):
            self.build(data, [], max_gap_seconds=-1)
        data['rows'].append(None)
        with self.assertRaisesRegex(DataError, 'mapping structure'):
            self.build(data, [])


if __name__ == '__main__':
    unittest.main()
