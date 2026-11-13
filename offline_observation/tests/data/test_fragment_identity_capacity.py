"""Operator checks on tiny constructed labels, never MTAD effectiveness evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from tools.analysis.fragment_identity_capacity import analyze_scene, pool_scenes, pool_source_pairs, run


def record(identity, sources, absent=()):
    n = len(identity)
    pairs = np.column_stack(np.triu_indices(n, 1))
    ids = np.asarray(identity)
    support = np.full(len(pairs), 64, dtype=np.int64)
    for pair in absent:
        support[np.all(pairs == pair, axis=1)] = 0
    return dict(node_ids=np.asarray([f'private-node-{i}' for i in range(n)], dtype=str),
        sources=np.asarray(sources, dtype=str), pairs=pairs,
        labels=(ids[pairs[:, 0]] == ids[pairs[:, 1]]),
        joint_support_queries=support, scores=np.zeros(len(pairs)))


class FragmentIdentityCapacityTests(unittest.TestCase):
    def test_multifragment_bound_and_no_support_keep_all_identity_source_rows(self):
        row = analyze_scene(**record([0, 0, 0, 0, 0, 1], ['a', 'a', 'b', 'b', 'b', 'c'],
                                   absent=[(0, 2), (1, 4), (2, 3)]))
        self.assertEqual((row['positive_edges'], row['same_source_positive_edges'],
                          row['cross_source_positive_edges']), (10, 4, 6))
        self.assertEqual(row['unit_capacity_true_edges'], 2)
        self.assertEqual(row['unrepresentable_direct_edges_lower_bound'], 4)
        self.assertEqual(row['restricted_identities'], 1)
        self.assertEqual(row['zero_edge_identity_source_pairs'], 5)
        self.assertAlmostEqual(row['direct_edge_recall_upper_bound'], 1 / 3)
        self.assertEqual(row['no_joint_support_positive_edges'], 3)
        self.assertEqual(row['no_joint_support_same_source_positive_edges'], 1)
        self.assertEqual(row['no_joint_support_restricted_positive_edges'], 2)
        self.assertEqual(len(row['identities']), 2)  # One singleton is retained.
        self.assertEqual(len(row['source_pairs']), 3)
        self.assertNotIn('private-node', json.dumps(row))

    def test_single_track_and_three_source_constraints_are_distinct(self):
        one = analyze_scene(**record([0, 0], ['a', 'b'], absent=[(0, 1)]))
        self.assertEqual(one['direct_edge_recall_upper_bound'], 1)
        self.assertEqual(one['restricted_identities'], 0)
        self.assertEqual(one['no_joint_support_cross_source_positive_edges'], 1)
        multi = analyze_scene(**record([0] * 5, ['a', 'a', 'b', 'b', 'c']))
        self.assertEqual([r['true_edges'] for r in multi['source_pairs']], [4, 2, 2])
        self.assertEqual([r['unit_capacity_true_edges'] for r in multi['source_pairs']], [2, 1, 1])
        self.assertEqual(multi['direct_edge_recall_upper_bound'], .5)

    def test_zero_denominators_and_no_pair_scenes_are_not_dropped(self):
        local = analyze_scene(**record([0, 0, 1], ['a', 'a', 'a']))
        empty = analyze_scene(**record([], []))
        self.assertIsNone(local['direct_edge_recall_upper_bound'])
        self.assertEqual(local['identity_count'], 2)
        self.assertEqual(local['same_source_positive_edges'], 1)
        self.assertEqual(empty['node_count'], 0)
        pooled = pool_scenes([local, empty])
        self.assertEqual(pooled['scene_count'], 2)
        self.assertEqual(pooled['scenes_without_cross_source_true_edges'], 2)
        self.assertIsNone(pooled['micro_direct_edge_recall_upper_bound'])
        one = analyze_scene(**record([0, 0], ['a', 'b']))
        mixed = pool_scenes([one, local])
        self.assertEqual(mixed['defined_scene_equal_upper_bound'], 1)
        self.assertIsNone(mixed['strict_all_scene_equal_upper_bound'])
        self.assertEqual(pool_source_pairs([one, local])[0]['scenes_with_both_sources'], 1)

    def test_nontransitive_labels_and_incomplete_pairs_fail_without_repair(self):
        example = record([0, 0, 0], ['a', 'b', 'c'])
        example['labels'] = np.array([1, 0, 1])  # AB and BC true, AC false.
        with self.assertRaisesRegex(ValueError, 'equivalence'):
            analyze_scene(**example)
        example = record([0, 0, 1], ['a', 'b', 'c'])
        example['pairs'] = example['pairs'][:-1]
        with self.assertRaisesRegex(ValueError, 'complete canonical'):
            analyze_scene(**example)
        example = record([0, 0], ['a', 'b'])
        example['joint_support_queries'] = np.array([65])
        with self.assertRaisesRegex(ValueError, '0,64'):
            analyze_scene(**example)

    def test_capacity_does_not_use_score_or_node_identity_strings(self):
        original = record([0, 0, 0, 1], ['a', 'a', 'b', 'b'])
        first = analyze_scene(**original)
        changed = {**original, 'scores': np.full(len(original['pairs']), 100.),
                   'node_ids': np.array(['z:9009:1', 'y:1001:4', 'x:2002:5', 'w:3003:7'])}
        second = analyze_scene(**changed)
        self.assertNotEqual(first['raw_binding_counts'], second['raw_binding_counts'])
        for key in ('identities', 'source_pairs', 'unit_capacity_true_edges',
                    'restricted_identities', 'direct_edge_recall_upper_bound'):
            self.assertEqual(first[key], second[key])

    def test_failure_is_saved_and_existing_outputs_cannot_be_overwritten(self):
        with TemporaryDirectory() as root:
            args = SimpleNamespace(scores=str(Path(root) / 'missing.npz'),
                summary=str(Path(root) / 'missing.json'), manifest=str(Path(root) / 'manifest.json'),
                output=str(Path(root) / 'attempt'))
            with self.assertRaises(FileNotFoundError):
                run(args)
            saved = (Path(args.output) / 'run.json').read_bytes()
            self.assertEqual(json.loads(saved)['status'], 'FAILED')
            self.assertEqual(json.loads(saved)['model_forwards'], 0)
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual((Path(args.output) / 'run.json').read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
