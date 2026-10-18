"""Hand-calculated partitions/recurrence and real CPU TCM/TFM inference."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from association.data.archive import DataError, Scene, Track
from association.data.legacy_dataset import prepare_scene
from association.models.fusion import PairFusion, PairFusionConfig
from association.evaluation.grouping import (fuse_track_groups, group_predicted_tracks,
                                          group_tracks_by_identity)
from association.models.tmtaff import AssociationModel, ModelConfig
from association.training.common import fit_normalizer, normalize_scene


def predicted(node_count, pairs, scores, threshold=0.5):
    return group_predicted_tracks(node_count, pairs, scores, threshold=threshold,
                                  grouping_policy='anchor_direct', node_order='input')


class PredictedGroupTests(unittest.TestCase):
    def test_anchor_neighbors_do_not_form_a_transitive_closure(self):
        # 0-1-2 and 0-3-4 are accepted paths, but only direct 0 neighbors
        # enter its group. Nodes already taken by 0 cannot enter later groups.
        pairs = np.asarray([[0,1], [1,2], [0,3], [3,4], [2,5]])
        groups = predicted(6, pairs, [0.5, 0.9, 0.8, 0.9, 0.5])
        self.assertEqual(groups, [[0,1,3], [2,5], [4]])
        flattened = [index for group in groups for index in group]
        self.assertEqual(sorted(flattened), list(range(6)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_threshold_ties_and_absent_edges_have_exact_meaning(self):
        self.assertEqual(predicted(4, np.asarray([[0,1], [0,2]]), [0.5, 0.499]),
                         [[0,1], [2], [3]])
        self.assertEqual(predicted(3, np.asarray([[0,1], [1,2]]), [-2., -1.], threshold=-2.),
                         [[0,1], [2]])

    def test_existing_all_reject_and_float64_threshold_contract(self):
        pairs = np.asarray([[0,1]])
        scores = np.asarray([0.5], dtype=np.float32)
        self.assertEqual(predicted(2, pairs, scores, threshold=None), [[0], [1]])
        # Do not round a float64 threshold down to the float32 score before
        # applying >=; this must match association.evaluation.association.predict exactly.
        just_above = float(np.nextafter(0.5, 1.0))
        self.assertEqual(predicted(2, pairs, scores, threshold=just_above), [[0], [1]])
        self.assertEqual(predicted(2, pairs, scores, threshold=0.5), [[0,1]])

    def test_candidate_order_and_orientation_do_not_change_node_order(self):
        pairs = np.asarray([[0,3], [0,1], [2,4], [1,2]])
        scores = np.asarray([0.8, 0.9, 0.7, 0.9])
        before_pairs, before_scores = pairs.copy(), scores.copy()
        expected = [[0,1,3], [2,4]]
        self.assertEqual(predicted(5, pairs, scores), expected)
        order = [3, 1, 0, 2]
        self.assertEqual(predicted(5, pairs[order, ::-1], scores[order]), expected)
        np.testing.assert_array_equal(pairs, before_pairs)
        np.testing.assert_array_equal(scores, before_scores)

    def test_empty_scene_and_all_singletons_are_retained(self):
        empty = np.empty((0,2), dtype=np.int64)
        self.assertEqual(predicted(0, empty, []), [])
        self.assertEqual(predicted(3, empty, []), [[0], [1], [2]])
        self.assertEqual(predicted(3, np.asarray([[0,1], [1,2]]), [0.1,0.2]), [[0], [1], [2]])


    def test_identity_groups_are_a_separate_oracle_comparison(self):
        identities = ['ship-0', 'other-0', 'ship-0', 'lonely-0']
        self.assertEqual(group_tracks_by_identity(identities, node_order='input'), [[0,2], [1], [3]])
        # Prediction has no identity argument and must retain this wrong merge.
        actual = predicted(4, np.asarray([[0,1], [0,2]]), [0.8,0.9])
        self.assertEqual(actual, [[0,1,2], [3]])
        identities[1] = 'ship-0'
        self.assertEqual(predicted(4, np.asarray([[0,1], [0,2]]), [0.8,0.9]), actual)
        self.assertEqual(group_tracks_by_identity([], node_order='input'), [])
        for invalid in ([None], [''], ['ship-0', 2], [['ship-0']]):
            with self.subTest(identities=invalid), self.assertRaises(DataError):
                group_tracks_by_identity(invalid, node_order='input')


class ArithmeticFusion(PairFusion):
    """Distinct FC memory and decoder outputs expose a wrong feedback path."""
    def __init__(self):
        super().__init__(PairFusionConfig(fc_hidden=4, length=2, d_model=4,
                                         nhead=2, num_layers=1, dim_feedforward=8, dropout=0))
        self.calls = []

    def forward(self, left, right):
        self.calls.append((left.clone(), right.clone(), torch.is_grad_enabled()))
        memory = 10 * left + right
        return {'fused_features': memory, 'states': memory + 1000}


class RecursiveFusionTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        torch.manual_seed(20)
        self.fusion = ArithmeticFusion().eval()
        self.completed = torch.arange(1, 6, dtype=torch.float32)[:,None,None].expand(5,2,4).clone()

    def tearDown(self):
        self.rng.__exit__(None, None, None)
        torch.set_num_threads(self.previous_threads)

    def fuse(self, groups, **kwargs):
        arguments = {'feedback_policy': 'fc_memory', 'singleton_policy': 'missing'}
        arguments.update(kwargs)
        return fuse_track_groups(self.fusion, self.completed, groups, **arguments)

    def test_hand_calculated_fc_memory_recurrence_and_final_decoder_states(self):
        records = self.fuse([[0,2,3], [1], [4]])
        self.assertEqual([record['node_indices'] for record in records], [[0,2,3], [1], [4]])
        self.assertEqual([record['fusion_steps'] for record in records], [2,0,0])
        self.assertEqual([record['status'] for record in records], ['fused','singleton_missing','singleton_missing'])
        # (10*1 + 3)*10 + 4 = 134; the first decoded state 1013 must
        # never replace FC memory 13 as the next left input.
        torch.testing.assert_close(records[0]['fused_features'], torch.full((2,4), 134.))
        torch.testing.assert_close(records[0]['states'], torch.full((2,4), 1134.))
        self.assertEqual(len(self.fusion.calls), 2)
        for (left, right, grad_enabled), expected in zip(self.fusion.calls, ((1.,3.), (13.,4.))):
            self.assertFalse(grad_enabled)
            torch.testing.assert_close(left, torch.full((1,2,4), expected[0]))
            torch.testing.assert_close(right, torch.full((1,2,4), expected[1]))
        for singleton in records[1:]:
            self.assertIsNone(singleton['states'])
            self.assertIsNone(singleton['fused_features'])

    def test_singletons_and_empty_scene_never_call_the_fusion_model(self):
        records = self.fuse([[0], [1], [2], [3], [4]])
        self.assertEqual(len(records), 5)
        self.assertTrue(all(item['states'] is None for item in records))
        self.assertEqual(self.fusion.calls, [])
        empty = fuse_track_groups(self.fusion, self.completed[:0], [],
                                  feedback_policy='fc_memory', singleton_policy='missing')
        self.assertEqual(empty, [])
        self.assertEqual(self.fusion.calls, [])


    def test_real_prepared_tcm_and_tfm_match_manual_ordered_recurrence(self):
        tracks = []
        identities = ['ship-0', 'other-0', 'ship-0', 'lonely-0']
        for index, (start, stop) in enumerate(((0,2), (2,4), (3,6), (1,5))):
            tracks.append(Track(node_id=f'node-{index}', source=str(index), batch=str(index),
                                identity_id=identities[index], ancestor_id=identities[index].split('-')[0],
                                times=np.asarray([start,stop], dtype=float),
                                values=np.asarray([[10+index,20,3,350], [12+index,22,5,10]], dtype=float)))
        raw = Scene('synthetic', tracks, {})
        prepared = prepare_scene(SimpleNamespace(load_scene=lambda sid: raw), 'synthetic', 4, 'component')
        values, observed = normalize_scene(prepared, fit_normalizer([prepared]))
        self.assertTrue((~observed).any())
        association = AssociationModel(ModelConfig(length=4, d_model=8, nhead=2,
                                       num_layers=1, dim_feedforward=16, dropout=.1)).eval()
        fusion = PairFusion(PairFusionConfig(fc_hidden=12, length=4, d_model=8, nhead=2,
                             num_layers=1, dim_feedforward=16, dropout=.1)).eval()
        # Deliberately retain a false positive (0,1) in the predicted group.
        groups = predicted(4, np.asarray([[0,1], [0,2]]), [0.8,0.9])
        self.assertEqual(groups, [[0,1,2], [3]])
        self.assertNotEqual(groups, group_tracks_by_identity(prepared['identities'], node_order='input'))
        with torch.inference_mode():
            completed = association.complete(torch.from_numpy(values), torch.from_numpy(observed))
            first = fusion(completed[0:1], completed[1:2])
            expected = fusion(first['fused_features'], completed[2:3])
        actual = fuse_track_groups(fusion, completed, groups,
                                   feedback_policy='fc_memory', singleton_policy='missing')
        repeated = fuse_track_groups(fusion, completed, groups,
                                     feedback_policy='fc_memory', singleton_policy='missing')
        self.assertEqual(actual[0]['states'].shape, (4,4))
        self.assertEqual(actual[0]['fused_features'].shape, (4,8))
        self.assertTrue(torch.isfinite(actual[0]['states']).all())
        self.assertTrue((actual[0]['states'][:,2] >= 0).all())
        torch.testing.assert_close(actual[0]['states'], expected['states'][0], rtol=0, atol=0)
        torch.testing.assert_close(actual[0]['fused_features'], expected['fused_features'][0], rtol=0, atol=0)
        torch.testing.assert_close(actual[0]['states'], repeated[0]['states'], rtol=0, atol=0)
        self.assertEqual(actual[1]['node_indices'], [3])
        self.assertIsNone(actual[1]['states'])


if __name__ == '__main__':
    unittest.main()
