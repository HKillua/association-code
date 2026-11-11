from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools.analysis import same_source_swap_pilot as pilot
from tools.analysis.same_source_swap_pilot import build_pool, delta, dense_edges, swapped


class SameSourceSwapTest(unittest.TestCase):
    def test_fixed_anchor_excludes_an_incoming_earlier_member(self):
        membership = np.array([0, 0, 1, 1, 2, 3])
        sources = np.array(['a', 'a', 'b', 'a', 'b', 'b'])
        pairs = np.column_stack(np.triu_indices(len(membership), 1))
        weights = dense_edges(pairs, np.ones(len(pairs)), len(membership))
        pool = build_pool(membership, sources, pairs, weights)
        actions = {tuple(action): bool(keep)
                   for action, keep in zip(pool['actions'], pool['keep'], strict=True)}
        self.assertIn((1, 3), actions)
        self.assertFalse(actions[1, 3])
        self.assertNotIn((4, 5), actions)
        np.testing.assert_array_equal(pool['selected'], [-1, -1])

    def test_incremental_score_and_true_pairs_match_full_partition(self):
        membership = np.array([0, 1, 0, 1, 0, 1, 2])
        sources = np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b'])
        identities = np.array(['A', 'B', 'B', 'A', 'A', 'B', 'C'])
        pairs = np.array(list(combinations(range(len(membership)), 2)))
        left, right = pairs.T
        same = sources[left] == sources[right]
        truth = identities[left] == identities[right]
        weights = np.where(truth, 1.0, -1.0)
        pool = build_pool(membership, sources, pairs, dense_edges(pairs, weights, len(membership)))
        self.assertTrue(pool['keep'][pool['selected'][0]])
        np.testing.assert_array_equal(pool['actions'][pool['selected'][0]], [2, 3])
        base = membership[left] == membership[right]
        for scope in (same, ~same):
            true_matrix = dense_edges(pairs, truth & scope, len(membership))
            truth_deltas = delta(true_matrix, membership, pool['actions'])
            for index, action in enumerate(pool['actions']):
                changed = swapped(membership, action)
                pred = changed[left] == changed[right]
                self.assertEqual((pred & scope).sum(), (base & scope).sum())
                self.assertEqual(truth_deltas[index], (truth & scope & pred).sum() - (truth & scope & base).sum())
                self.assertEqual(pool['gain'][index], weights[pred].sum() - weights[base].sum())

    def test_every_scene_is_saved_before_the_first_identity_read(self):
        nodes = np.array(['a', 'b', 'c', 'd'])
        sources = np.array(['r1', 'r2', 'r1', 'r2'])
        support = np.ones((4, 64), dtype=bool)
        pairs = np.column_stack(np.triu_indices(4, 1))
        scores = np.arange(len(pairs), dtype=float)
        membership = np.array([0, 0, 1, 1])
        scene = (nodes, sources, support, pairs, scores, dict.fromkeys(pilot.STARTS, membership))
        scenes = ['1', '2']
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / 'science'

            def load_identities(runs, sid, node_ids):
                self.assertEqual(sorted(p.name for p in (output / 'private').glob('[12].npz')),
                                 ['1.npz', '2.npz'])
                return np.array(['A', 'B', 'B', 'A'])

            with (patch.object(pilot, 'load_population', return_value=(scenes, {'1': 'x', '2': 'y'})),
                  patch.object(pilot, 'load_scene', return_value=scene),
                  patch.object(pilot, 'load_identities', side_effect=load_identities) as reads,
                  patch.object(pilot.os, 'sched_getaffinity', return_value={0}, create=True),
                  patch.object(pilot.resource, 'setrlimit')):
                pilot.run(Path(tmp), output)
            self.assertEqual(reads.call_count, 2)
            self.assertEqual(pilot.read(output / 'run.json')['status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
