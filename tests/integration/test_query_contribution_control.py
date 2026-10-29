"""116 software regression checks using synthetic 115-shaped caches only."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration import test_static_partition_position_control as old_tests
from tools.analysis import query_contribution_control as p


def scene_fixture(sid='1'):
    args = old_tests.fixture(sid)
    public, private, _ = p.old.analyze_scene(*args)
    return args[1], {r: args[3][r] for r in p.PARTITIONS}, {
        r: private['artifacts'][r] for r in p.PARTITIONS}, public, private


def cache_fixture(root):
    root = root.resolve()
    source, previous, meta = old_tests.PositionControlTests().cache_fixture(root)
    control = root / p.CONTROL_RUN
    (control / 'scenes').mkdir(parents=True)
    (control / 'private').mkdir()
    for sid in meta['ids']:
        _, _, _, public, retained = scene_fixture(sid)
        p.old.frozen.write(control / 'scenes' / f'scene-{sid}.json.gz', public)
        # Match 115 run(), including candidate masks sharing the WS prefix.
        payload = {name + '_' + key: value for name, artifact in retained['artifacts'].items()
                   for key, value in artifact.items()}
        payload.update({name + '_' + key: value for name, masks in retained['pair_masks'].items()
                        for key, value in masks.items()})
        np.savez_compressed(control / 'private' / f'states-{sid}.npz', **payload)
    run = dict(status='COMPLETED', idea_id=p.old.IDEA, plan_version='1.0',
               source_system=str(source), partition_run=str(previous), scene_ids=meta['ids'],
               completed_scenes=len(meta['ids']), methods=list(p.old.METHODS))
    summary = dict(status='COMPLETED', idea_id=p.old.IDEA, plan_version='1.0',
                   scene_ids=meta['ids'], all_A0_exact_replays_verified=True, no_expansion=True)
    p.old.frozen.write(control / 'run.json', run)
    p.old.frozen.write(control / 'summary.json', summary)
    args = SimpleNamespace(source_system=source, partition_run=previous, control_run=control,
                           output=root / 'out', manifest=root / 'manifest.json', cpu_threads=1)
    return args, meta


class QueryContributionControlTests(unittest.TestCase):
    def test_eight_predictions_keep_fixed_R_population_and_pairwise_targets(self):
        fragment, memberships, retained, public, _ = scene_fixture()
        row, arrays, artifacts, weights = p.analyze_scene(fragment, memberships, retained, public)
        self.assertEqual(set(row['methods']), set(p.METHODS))
        self.assertEqual(len(row['pairwise']), len(p.COMPARISONS))
        for r, n in (('A0', 128), ('WS', 64)):
            baseline = artifacts[r + '_MEAN']
            for w in p.RULES:
                name = r + '_' + w
                self.assertEqual(row['methods'][name]['position']['all']['intersection_count'], n)
                self.assertEqual(row['methods'][name]['position']['all']['available_target_count'], 192)
                self.assertEqual(row['methods'][name]['population'], row['methods'][r + '_MEAN']['population'])
                np.testing.assert_array_equal(artifacts[name]['aligned_prediction_valid'], baseline['aligned_prediction_valid'])
                np.testing.assert_array_equal(artifacts[name]['node_group_indices'], baseline['node_group_indices'])
                self.assertEqual(set(row['methods'][name]['paper_position']), {'latitude', 'longitude'})
            # Unsupported query must retain the original first-member OLS state.
            for w in p.RULES:
                np.testing.assert_array_equal(artifacts[r + '_' + w]['group_states'][:, 0],
                                              fragment['native_states'][[g[0] for g in p.old.groups_from_membership(memberships[r], 3)], 0])
        self.assertAlmostEqual(artifacts['WS_MEAN']['group_states'][0, 1, 0], 1.002 / 3)
        self.assertAlmostEqual(artifacts['WS_SOURCE_BALANCED']['group_states'][0, 1, 0], .2505)
        np.testing.assert_array_equal(weights['WS_SOURCE_BALANCED'][:, 1], [.5, .25, .25])
        # Cross-R comparisons retain their own 64-point intersection; W does not
        # manufacture target coverage or discard the no-input target denominator.
        for w in p.RULES:
            pair = row['pairwise']['WS_' + w + '_vs_A0_' + w]
            self.assertEqual(pair['common_valid_position']['A0']['intersection_count'], 64)
            self.assertEqual(pair['coverage_transitions']['lost_target_points'], 64)
        self.assertEqual(len(arrays['methods']['A0_MEAN']['all']), 128)

    def test_bound_prediction_and_reference_corruption_fail_without_relaxation(self):
        for corrupt in ('states', 'membership', 'reference', 'summary', 'quantile'):
            fragment, memberships, retained, public, _ = scene_fixture()
            if corrupt == 'states':
                retained['WS']['group_states'] = retained['WS']['group_states'].copy()
                retained['WS']['group_states'][0, 1, 0] += 1e-12
            elif corrupt == 'membership':
                memberships['WS'] = np.array([0, 0, 1])
            elif corrupt == 'reference':
                fragment['target'][0, 1, 0] += 1e-12
            elif corrupt == 'summary':
                public['methods']['A0']['position']['all']['intersection_count'] += 1
            else:
                public['methods']['A0']['position']['all']['p99'] += 1.
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                p.analyze_scene(fragment, memberships, retained, public)

    def test_aggregation_keeps_point_weighted_physical_and_scene_equal_paper_metrics(self):
        outputs = [p.analyze_scene(*scene_fixture(sid)[:4]) for sid in ('1', '2')]
        rows, arrays = [o[0] for o in outputs], [o[1] for o in outputs]
        actual = p.aggregate(rows, arrays)
        for name in p.METHODS:
            all_values = np.concatenate([a['methods'][name]['all'] for a in arrays])
            self.assertAlmostEqual(actual['methods'][name]['position']['rmse'], np.sqrt(np.mean(all_values ** 2)))
            self.assertEqual(actual['methods'][name]['fixed_target_strata']['all']['available_target_points'], 384)
            own = 256 if name.startswith('A0_') else 128
            self.assertEqual(actual['methods'][name]['position']['intersection_count'], own)
            self.assertEqual(actual['methods'][name]['fixed_target_strata']['all']['point_count'], own)
            self.assertAlmostEqual(actual['methods'][name]['fixed_target_strata']['all']['coverage'], own / 384)
        changed = deepcopy(rows)
        changed[0]['methods']['WS_MEAN']['paper_position']['latitude']['target_mean']['ME'] = 1.
        changed[1]['methods']['WS_MEAN']['paper_position']['latitude']['target_mean']['ME'] = 9.
        changed[0]['methods']['WS_MEAN']['paper_position']['latitude']['defined_targets']['ME'] = 1
        changed[1]['methods']['WS_MEAN']['paper_position']['latitude']['defined_targets']['ME'] = 100
        paper = p.paper_aggregate([r['methods']['WS_MEAN'] for r in changed])['latitude']
        self.assertEqual(paper['scene_equal_target_mean']['ME'], 5.)
        self.assertEqual(paper['defined_targets']['ME'], 101)

    def test_original115_metadata_binds_completion_source_order_and_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, meta = cache_fixture(Path(temporary))
            with patch.object(p.old, 'metadata', return_value=meta):
                self.assertEqual(p.metadata(args.source_system, args.partition_run, args.control_run, {})['ids'], ['1', '2'])
                original = p.old.frozen.read(args.control_run / 'run.json')
                for key, value in (('status', 'FAILED'), ('source_system', str(Path(temporary) / 'wrong')),
                                   ('scene_ids', ['2', '1']), ('completed_scenes', 1), ('methods', ['A0', 'WS'])):
                    p.old.frozen.write(args.control_run / 'run.json', dict(original, **{key: value}))
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'original115'):
                        p.metadata(args.source_system, args.partition_run, args.control_run, {})
                p.old.frozen.write(args.control_run / 'run.json', original)
                original_summary = p.old.frozen.read(args.control_run / 'summary.json')
                for key in ('idea_id', 'plan_version'):
                    p.old.frozen.write(args.control_run / 'summary.json', dict(original_summary, **{key: 'different-plan'}))
                    with self.subTest(summary_key=key), self.assertRaisesRegex(ValueError, 'original115'):
                        p.metadata(args.source_system, args.partition_run, args.control_run, {})

    def test_cached_runner_consumes_real115_prefix_schema_and_preserves_partial_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                args, meta = cache_fixture(Path(temporary))
                if fail:
                    path = args.control_run / 'private' / 'states-2.npz'
                    with np.load(path, allow_pickle=False) as saved:
                        changed = {key: saved[key] for key in saved.files}
                    changed['WS_group_states'][0, 1, 0] += 1e-12
                    np.savez_compressed(path, **changed)
                with patch.object(p, 'metadata', return_value=meta):
                    if fail:
                        with self.assertRaisesRegex(ValueError, '115 exact WS Mean states'):
                            p.run(args)
                        state = p.old.frozen.read(args.output / 'run.json')
                        self.assertEqual((state['status'], state['exit_code'], state['completed_scenes']), ('FAILED', 1, 1))
                        self.assertTrue((args.output / 'private' / 'states-1.npz').exists())
                        self.assertFalse((args.output / 'summary.json').exists())
                    else:
                        result = p.run(args)
                        self.assertTrue(result['no_expansion'])
                        self.assertEqual(result['scene_ids'], ['1', '2'])
                        self.assertEqual(result['methods']['WS_MEAN']['position']['intersection_count'], 128)
                        self.assertTrue(result['fixed_R_population_invariants_verified'])
                        state = p.old.frozen.read(args.output / 'run.json')
                        self.assertEqual((state['status'], state['completed_scenes'], state['model_forwards']), ('COMPLETED', 2, 0))
                        with np.load(args.output / 'private' / 'states-2.npz', allow_pickle=False) as saved:
                            self.assertIn('WS_SOURCE_BALANCED_query_weights', saved.files)
                            self.assertIn('A0_MEAN_aligned_truth', saved.files)
                        with self.assertRaises(FileExistsError):
                            p.run(args)

    def test_budget_failure_at_start_and_after_summary_never_leaves_completed_state(self):
        for fail_at in ('start', 'final'):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as temporary:
                args, meta = cache_fixture(Path(temporary))
                def memory():
                    over = fail_at == 'start' or (args.output / 'summary.json').exists()
                    return p.old.frozen.RAM_BUDGET + 1 if over else 1024
                with patch.object(p, 'metadata', return_value=meta), patch.object(p.old.frozen, 'peak_rss_bytes', side_effect=memory):
                    with self.assertRaises(MemoryError):
                        p.run(args)
                state = p.old.frozen.read(args.output / 'run.json')
                self.assertEqual((state['status'], state['exit_code']), ('FAILED', 1))
                if (args.output / 'summary.json').exists():
                    self.assertEqual(p.old.frozen.read(args.output / 'summary.json')['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
