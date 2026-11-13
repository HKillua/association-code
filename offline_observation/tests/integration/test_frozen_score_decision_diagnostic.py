"""Small operator/cache checks; synthetic fixtures are not MTAD experiments."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.analysis import frozen_score_decision_diagnostic as d


def record(sid='1', *, n=3, scores=None, labels=None):
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], np.int64).reshape(-1, 2)
    return dict(scene_id=sid, component_id='c', node_ids=np.array([f'opaque-{i}' for i in range(n)], str),
        sources=np.arange(n) % 2, pairs=pairs,
        scores=np.array([2., 0., 2.] if scores is None else scores, float),
        labels=np.array([1, 0, 0] if labels is None else labels, bool),
        joint_support_queries=np.ones(len(pairs), np.int64))


class FrozenScoreDecisionTests(unittest.TestCase):
    def test_named_graph_permutation_changes_partition_not_raw_scores_and_labels_do_not_choose_order(self):
        r = record()
        result = d.analyze_scene(r, [7, 7, 2], random.Random(d.SEED))
        self.assertGreater(result['changed_permutation_count'], 0)
        self.assertEqual(result['arms'][0]['decoded']['fp'], 0)
        self.assertTrue(any(a['decoded']['fp'] > 0 for a in result['arms'][1:]))
        self.assertEqual(result['raw']['fp'], 1)
        self.assertTrue(all(a['raw_counts_unchanged'] for a in result['arms']))
        for item, membership in zip(result['arms'], result['memberships']):
            co = np.equal.outer(membership, membership)
            base = np.equal.outer(result['memberships'][0], result['memberships'][0])
            self.assertEqual(item['changed_comembership_pairs'], int(np.triu(co != base, 1).sum()))
        other = d.analyze_scene(dict(r, labels=np.zeros(3, bool)), [0, 0, 1], random.Random(d.SEED))
        self.assertEqual(result['memberships'], other['memberships'])
        self.assertEqual(result['permutations_new_to_original'], other['permutations_new_to_original'])
        with self.assertRaisesRegex(ValueError, 'A0 differs'):
            d.analyze_scene(r, [0, 0, 0], random.Random(d.SEED))

    def test_cliques_singletons_empty_population_and_inconsistent_labels_keep_denominators(self):
        clique = d.analyze_scene(record(scores=[2., 2., 2.], labels=[1, 1, 1]), [0, 0, 0], random.Random(d.SEED))
        self.assertEqual(clique['changed_permutation_count'], 0)
        empty = d.analyze_scene(record('2', n=0, scores=[], labels=[]), [], random.Random(d.SEED))
        single = d.analyze_scene(record('3', n=1, scores=[], labels=[]), [9], random.Random(d.SEED))
        self.assertEqual(empty['unique_partition_count'], 1)
        self.assertEqual(single['unique_permutation_count'], 1)
        self.assertIsNone(empty['arms'][0]['changed_comembership_fraction'])
        pooled = d.aggregate([clique, empty, single])
        self.assertEqual(pooled['scene_count'], 3)
        self.assertEqual(pooled['decoded']['A0']['all']['scene_count'], 1)
        self.assertEqual(pooled['eligible_node_ge2_scenes'], 1)
        self.assertEqual(pooled['per_scene_delta_signs']['all']['A1']['f1']['undefined'], 2)
        partition = pooled['decoded']['A0']['partition_diagnostics']
        self.assertEqual(partition['within_group_pair_count'] + partition['across_group_pair_count'], 3)
        broken = d.analyze_scene(record(labels=[1, 0, 1]), [0, 0, 1], random.Random(d.SEED))
        self.assertIsNone(broken['arms'][0]['mixed_group_count'])
        self.assertEqual(d.aggregate([clique, broken])['decoded']['A0']['partition_diagnostics']['identity_unavailable_scene_count'], 1)
        self.assertIsNone(d.aggregate([clique, broken])['decoded']['A0']['partition_diagnostics']['mixed_group_count'])

    def fixture(self, root):
        source = root / d.SOURCE_RUN / 'association'
        source.mkdir(parents=True)
        rows = [record(), record('2', n=1, scores=[], labels=[])]
        manifest = dict(dataset_level='Level1', split_unit='scene', splits={'V_select': ['1', '2']}, components={'1': 'c', '2': 'c'})
        final = d.evaluate_scenes(rows, d.THRESHOLD)
        groups = [dict(scene_id=r['scene_id'], membership=d.decode(len(r['sources']), r['pairs'], r['scores']).tolist(),
                       partition={'statistics_not_a_membership': True}) for r in rows]
        expected = dict(scenes=2, components=1, nodes=4, pairs=3, positives=1,
                        **{key: final[key] for key in ('tp', 'fp', 'fn', 'tn')})
        config = d.read(Path(d.__file__).resolve().parents[2] / 'configs/evidence-association-level1.json')
        common = dict(status='COMPLETED', synthetic=False, reference_read=False, test_opened=False, optimizer_updates=1000)
        d.write(source / 'run.json', dict(common, kind=config['scope'], config=config))
        summary = dict(common, scope=config['scope'], split_unit='scene',
            population={key: expected[key] for key in ('scenes', 'components', 'nodes', 'pairs', 'positives')},
            final_selection={'threshold': d.THRESHOLD}, selected={'selection': {'threshold': d.THRESHOLD}},
            final_result=dict(metrics=final, grouping=dict(policy='anchor_direct/input; every fragment is an anchor in helper diagnostics', per_scene=groups)))
        d.write(source / 'summary.json', summary)
        payload = {'scene_ids': np.array(['1', '2'])}
        payload.update({f"{r['scene_id']}_{key}": r[key] for r in rows for key in d.FIELDS})
        np.savez_compressed(source / 'final-scores.npz', **payload)
        d.write(root / 'manifest.json', manifest)
        return source, expected, summary

    def test_complete_cache_entrypoint_records_success_and_fail_closed_on_original_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, expected, summary = self.fixture(root)
            args = SimpleNamespace(source_association=source, manifest=root / 'manifest.json', output=root / 'good', cpu_threads=1)
            with patch.object(d, 'EXPECTED', expected):
                result = d.run(args)
                self.assertEqual(result['status'], 'COMPLETED')
                self.assertTrue(result['original_A0_membership_verified'])
                self.assertEqual(result['raw']['fp'], 1)
                self.assertEqual(result['raw']['itsa']['pair_count'] + result['raw']['mstsa']['pair_count'], 3)
                self.assertEqual(result['decoded']['A0']['partition_diagnostics']['input_identity_count'], 3)
                self.assertEqual(result['components']['c']['scene_count'], 2)
                self.assertEqual(d.read(root / 'good/run.json')['exit_code'], 0)
                self.assertTrue((root / 'good/scenes/scene-2.json.gz').exists())
                # The neighboring partition statistics dict must never be parsed as groups.
                summary['final_result']['grouping']['per_scene'][0]['membership'] = [0, 0, 0]
                d.write(source / 'summary.json', summary)
                args.output = root / 'bad'
                with self.assertRaisesRegex(ValueError, 'A0 differs'):
                    d.run(args)
                failed = d.read(root / 'bad/run.json')
                self.assertEqual(failed['status'], 'FAILED')
                self.assertEqual(failed['phase'], 'preflight')
                self.assertFalse((root / 'bad/summary.json').exists())
                self.assertFalse((root / 'bad/scenes').exists())
                self.assertEqual(failed['model_forwards'], 0)

    def test_cache_wrong_schema_or_incomplete_graph_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, expected, _ = self.fixture(root)
            args = SimpleNamespace(source_association=source, manifest=root / 'manifest.json', output=root / 'bad', cpu_threads=1)
            with np.load(source / 'final-scores.npz', allow_pickle=False) as z:
                payload = {key: z[key] for key in z.files}
            payload['1_pairs'] = payload['1_pairs'][:-1]
            for key in ('scores', 'labels', 'joint_support_queries'):
                payload['1_' + key] = payload['1_' + key][:-1]
            np.savez_compressed(source / 'final-scores.npz', **payload)
            with patch.object(d, 'EXPECTED', expected), self.assertRaisesRegex(ValueError, 'complete unordered'):
                d.run(args)
            self.assertEqual(d.read(root / 'bad/run.json')['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
