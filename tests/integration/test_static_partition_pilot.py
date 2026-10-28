"""114 operators and conditional workload; synthetic graphs are software only."""
import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.analysis import static_partition_pilot as p
from tests.integration.test_frozen_score_decision_diagnostic import record


def slow_clusters(n, pairs, scores, method):
    weights = np.zeros((n, n))
    weights[pairs[:, 0], pairs[:, 1]] = scores-p.THRESHOLD
    weights[pairs[:, 1], pairs[:, 0]] = scores-p.THRESHOLD
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        choices = []
        for i, a in enumerate(clusters):
            for j in range(i+1, len(clusters)):
                b = clusters[j]
                values = [weights[x, y] for x in a for y in b]
                gain = min(values) if method == 'CL' else sum(values)
                if gain >= 0 if method == 'CL' else gain > 0:
                    choices.append((-gain, min(a), min(b), i, j))
        if not choices:
            break
        _, _, _, i, j = min(choices)
        clusters[i] = sorted(clusters[i]+clusters.pop(j))
    return p.membership_from_clusters(clusters, n)


class StaticPartitionTests(unittest.TestCase):
    def test_clique_and_signed_surrogate_are_distinct_and_obey_zero_boundary(self):
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        scores = p.THRESHOLD+np.array([3., -1., 2.])
        cl, stats, _ = p.agglomerate(3, pairs, scores, 'CL')
        ws, signed, trace = p.agglomerate(3, pairs, scores, 'WS')
        np.testing.assert_array_equal(cl, [0, 0, 1])
        np.testing.assert_array_equal(ws, [0, 0, 0])
        self.assertEqual(stats['merge_count'], 1)
        self.assertEqual(signed['merge_count'], 2)
        self.assertAlmostEqual(sum(item['gain'] for item in trace), 4.)
        zeros = np.full(3, p.THRESHOLD)
        cl, tied, trace = p.agglomerate(3, pairs, zeros, 'CL')
        ws, none, _ = p.agglomerate(3, pairs, zeros, 'WS')
        np.testing.assert_array_equal(cl, [0, 0, 0])
        np.testing.assert_array_equal(ws, [0, 1, 2])
        self.assertEqual(tied['tie_decisions'], 1)
        self.assertEqual((trace[0]['left_min_node'], trace[0]['right_min_node']), (0, 1))
        self.assertEqual(none['merge_count'], 0)

    def test_heap_updates_match_independent_complete_cross_pair_definition(self):
        pairs = np.column_stack(np.triu_indices(7, 1))
        for seed in range(4):
            # Discrete scores exercise ties and stale-candidate invalidation.
            scores = np.random.default_rng(seed).integers(-2, 4, len(pairs)).astype(float)
            for method in ('CL', 'WS'):
                with self.subTest(seed=seed, method=method):
                    fast, stats, _ = p.agglomerate(7, pairs, scores, method)
                    np.testing.assert_array_equal(fast, slow_clusters(7, pairs, scores, method))
                    self.assertLessEqual(stats['merge_count'], 6)
        for n in (0, 1):
            actual, stats, _ = p.agglomerate(n, np.empty((0, 2), int), np.empty(0), 'CL')
            self.assertEqual(len(actual), n)
            self.assertEqual(stats['merge_count'], 0)

    def test_sources_only_control_orders_and_labels_or_node_names_never_construct_partitions(self):
        r = record('12')
        r['sources'] = np.array(['a', 'a', 'b'])
        expected = p.frozen.metric_rows(p.frozen.evaluate_scenes([r], p.THRESHOLD))
        original = p.frozen.decode(3, r['pairs'], r['scores'])
        a, private = p.analyze_scene(r, original, expected)
        other = dict(r, labels=np.array([0, 0, 0], bool), node_ids=np.array(['z', 'y', 'x']))
        expected_other = p.frozen.metric_rows(p.frozen.evaluate_scenes([other], p.THRESHOLD))
        _, second = p.analyze_scene(other, original, expected_other)
        self.assertEqual(private['memberships'], second['memberships'])
        self.assertEqual(private['source_orders_new_to_original'], second['source_orders_new_to_original'])
        self.assertEqual(private['source_orders_new_to_original']['SRC_REVERSE'], [2, 0, 1])
        within = private['source_orders_new_to_original']['SRC_WITHIN']
        self.assertEqual(set(within[:2]), {0, 1})
        self.assertEqual(within[-1], 2)
        self.assertEqual(a['arms']['CL']['raw_rejected_within_group'], 0)
        self.assertEqual(a['arms']['CC']['raw_accepted_across_groups'], 0)

    def test_pilot_selection_does_not_depend_on_manifest_order(self):
        ids = [str(i) for i in range(1, 301)]
        self.assertEqual(p.pilot_selection(ids), p.pilot_selection(list(reversed(ids))))
        self.assertEqual(len(p.pilot_selection(ids)), 32)

    def test_gate_uses_all_required_decoded_metrics_and_rejects_null_or_controls_failure(self):
        def arm(f1=.5, recall=.9, false=.2, micro=.2):
            return dict(all=dict(macro_f1=f1, p_fa=micro), mstsa=dict(paper_metrics=dict(m_pca=recall, m_pfa=false)))
        data = dict(decoded={method: arm() for method in p.METHODS})
        data['decoded']['SRC_REVERSE'] = arm(f1=.99, recall=1., false=0., micro=0.)
        self.assertFalse(p.expansion_gate(data, True)['expand'])
        data['decoded']['CL'] = arm(f1=.505)
        self.assertTrue(p.expansion_gate(data, True)['methods']['CL']['passed'])
        self.assertFalse(p.expansion_gate(data, False)['expand'])
        for value in (arm(f1=.5049), arm(f1=.505, recall=.899), arm(f1=.505, false=.201),
                      arm(f1=.505, micro=.201), arm(f1=.505, recall=None)):
            with self.subTest(value=value):
                data['decoded']['CL'] = value
                self.assertFalse(p.expansion_gate(data, True)['expand'])

    def fixture(self, root):
        source = root/'original056'/'association'; source.mkdir(parents=True)
        ids = ['1', '2', '3', '4']
        rows = [record(sid) for sid in ids]
        raw = p.frozen.evaluate_scenes(rows, p.THRESHOLD)
        expected = {scope: raw['per_scene'] if scope == 'all' else raw[scope]['per_scene'] for scope in p.frozen.SCOPES}
        original = {sid: p.frozen.decode(3, r['pairs'], r['scores']) for sid, r in zip(ids, rows)}
        payload = {'scene_ids': np.array(ids)}
        payload.update({sid+'_'+key: r[key] for sid, r in zip(ids, rows) for key in p.frozen.FIELDS})
        np.savez_compressed(source/'final-scores.npz', **payload)
        p.frozen.write(root/'manifest.json', {})
        actual = dict(scenes=4, components=1, nodes=12, pairs=12, positives=4,
                      **{key: raw[key] for key in ('tp', 'fp', 'fn', 'tn')})
        metadata = (ids, {sid: 'c' for sid in ids}, original, expected)
        return source, metadata, actual

    def test_real_cache_flow_stops_at_pilot_or_expands_all_methods_without_repeating_pilot(self):
        for expand in (False, True):
            with self.subTest(expand=expand), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); source, metadata, actual = self.fixture(root)
                args = SimpleNamespace(source_association=source, output=root/'out', manifest=root/'manifest.json', cpu_threads=1)
                seen = []
                load = p.frozen.load_record
                def observe(*args):
                    seen.append(args[1]); return load(*args)
                # Zero practical-gain requirement is a test-only injection to
                # exercise the authorized continuation branch with tied CL=A0.
                with patch.object(p, 'PILOT_COUNT', 2), patch.object(p, 'GAIN_REQUIRED', 0. if expand else .005), \
                     patch.object(p.frozen, 'metadata', return_value=metadata), \
                     patch.object(p.frozen, 'EXPECTED', actual), patch.object(p.frozen, 'load_record', side_effect=observe):
                    result = p.run(args)
                self.assertEqual(result['expanded'], expand)
                self.assertEqual(len(seen), 4 if expand else 2)
                self.assertEqual(len(set(seen)), len(seen))
                self.assertEqual(result['unread_remainder_scene_count'], 0 if expand else 2)
                for sid in seen:
                    saved = p.frozen.read(root/'out/scenes'/f'scene-{sid}.json.gz')
                    self.assertEqual(tuple(saved['arms']), p.METHODS)
                self.assertIsNotNone(result['remainder'] if expand else result['pilot'])
                self.assertEqual(p.frozen.read(root/'out/run.json')['status'], 'COMPLETED')
                self.assertEqual(result['model_forwards'], 0)
                with self.assertRaises(FileExistsError):
                    p.run(args)

    def test_original_membership_mismatch_fails_without_any_conditional_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source, metadata, actual = self.fixture(root)
            ids, components, _, expected = metadata
            metadata = (ids, components, {sid: np.zeros(3, int) for sid in ids}, expected)
            args = SimpleNamespace(source_association=source, output=root/'failed', manifest=root/'manifest.json', cpu_threads=1)
            with patch.object(p, 'PILOT_COUNT', 2), patch.object(p.frozen, 'metadata', return_value=metadata), \
                 patch.object(p.frozen, 'EXPECTED', actual), self.assertRaisesRegex(ValueError, 'A0 differs'):
                p.run(args)
            state = p.frozen.read(root/'failed/run.json')
            self.assertEqual(state['status'], 'FAILED')
            self.assertEqual(state['completed_scenes'], 0)
            self.assertFalse((root/'failed/summary.json').exists())
