"""117 cache/reuse/failure checks on three synthetic scenes, not MTAD results."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration import test_query_contribution_control as prior_tests
from tools.analysis import query_contribution_full_control as p


def cache_fixture(root):
    """One imported pilot scene and two remaining scenes, in two components."""
    root = root.resolve()
    source, previous, meta = prior_tests.old_tests.PositionControlTests().cache_fixture(root)
    ids = ['1', '2', '3']
    components = dict(zip(ids, ('large', 'large', 'small')))
    payload = {'scene_ids': np.array(ids)}
    pilot = root / p.PILOT_RUN
    (pilot / 'scenes').mkdir(parents=True)
    (pilot / 'private').mkdir()
    pilot_artifacts = None
    for sid in ids:
        args = list(prior_tests.old_tests.fixture(sid))
        record, fragment, system, memberships, public114, _ = args
        payload.update({sid + '_' + key: record[key] for key in p.old.frozen.FIELDS})
        np.savez_compressed(source / 'predictions' / f'states-{sid}.npz', **fragment)
        np.savez_compressed(source / 'system-predictions' / f'states-{sid}.npz', **system)
        p.old.frozen.write(previous / 'private' / f'scene-{sid}.json.gz',
                           dict(memberships={k: v.tolist() for k, v in memberships.items()}))
        if sid == '1':
            public115, retained115, _ = p.old.analyze_scene(*args)
            public116, _, pilot_artifacts, weights = p.q.analyze_scene(
                fragment, {r: memberships[r] for r in p.q.PARTITIONS},
                {r: retained115['artifacts'][r] for r in p.q.PARTITIONS}, public115)
            public116['component_id'] = components[sid]
            p.old.frozen.write(pilot / 'scenes' / f'scene-{sid}.json.gz', public116)
            saved = {name + '_' + key: value for name, artifact in pilot_artifacts.items()
                     for key, value in artifact.items()}
            saved.update({name + '_query_weights': value for name, value in weights.items()})
            np.savez_compressed(pilot / 'private' / f'states-{sid}.npz', **saved)
        record['component_id'] = public114['component_id'] = components[sid]
        public114['node_count'] = len(record['node_ids'])
        public114['raw'] = p.old.frozen.evaluate_scenes([record], p.old.frozen.THRESHOLD)
        for name, membership in memberships.items():
            co = membership[record['pairs'][:, 0]] == membership[record['pairs'][:, 1]]
            public114['arms'][name]['decoded'] = p.old.frozen.evaluate_scenes(
                [dict(record, scores=co.astype(float))], .5)
        p.old.frozen.write(previous / 'scenes' / f'scene-{sid}.json.gz', public114)
    np.savez_compressed(meta['association'] / 'final-scores.npz', **payload)
    meta.update(ids=['1'], all_ids=ids, components=components)
    run = dict(status='COMPLETED', idea_id=p.q.IDEA, plan_version='1.0',
               source_system=str(source), partition_run=str(previous), scene_ids=['1'],
               completed_scenes=1, methods=list(p.q.METHODS))
    summary = dict(status='COMPLETED', idea_id=p.q.IDEA, plan_version='1.0', scene_ids=['1'],
                   original115_exact_replays_verified=True,
                   fixed_R_population_invariants_verified=True, no_expansion=True)
    p.old.frozen.write(pilot / 'run.json', run)
    p.old.frozen.write(pilot / 'summary.json', summary)
    args = SimpleNamespace(source_system=source, partition_run=previous, pilot_run=pilot,
                           output=root / 'out', manifest=root / 'manifest.json', cpu_threads=1)
    return args, meta, pilot_artifacts


class QueryContributionFullControlTests(unittest.TestCase):
    def test_three_scene_runner_reuses_six_arms_and_emits_only_new_private_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, meta, pilot_artifacts = cache_fixture(Path(temporary))
            with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)), \
                 patch.object(p, 'predict_group_outputs', wraps=p.predict_group_outputs) as predict, \
                 patch.object(p.old, 'evaluate_full_scene', wraps=p.old.evaluate_full_scene) as evaluate, \
                 patch.object(p.q, 'aggregate', wraps=p.q.aggregate) as aggregate:
                result = p.run(args)
            # These are constructed scene-arm counts, not physical data sizes.
            self.assertEqual(predict.call_count, 2 + 8 + 8)
            # The imported six predictions are evaluated for artifact/public
            # binding, but their physical predictions are never reconstructed.
            self.assertEqual(evaluate.call_count, 8 + 8 + 8)
            self.assertTrue(all(call.kwargs['method'] == 'nearest_observation'
                                for call in predict.call_args_list[:2]))
            self.assertTrue(all(call.kwargs == {'method_names': p.METHODS}
                                for call in aggregate.call_args_list))
            self.assertEqual(len(p.COMPARISONS), 16)
            self.assertEqual(len(set(p.COMPARISONS)), 16)
            self.assertEqual(result['scene_ids'], ['1', '2', '3'])
            self.assertEqual(result['scene_prediction_arms_imported'], 6)
            self.assertEqual(result['scene_prediction_arms_computed'], 18)
            for key, count in (('original32', 1), ('remaining268', 2), ('without_largest_component', 1)):
                self.assertEqual(result[key]['scene_count'], count)
                self.assertEqual(set(result[key]['methods']), set(p.METHODS))
                self.assertEqual(len(result[key]['pairwise']), 16)
            self.assertEqual(result['without_largest_component']['scene_ids'], ['3'])
            self.assertEqual(set(result['components']), {'large', 'small'})
            self.assertEqual(set(result['methods']), set(p.METHODS))
            self.assertEqual(len(result['pairwise']), 16)
            for name, report in result['methods'].items():
                own = 384 if name.startswith('A0_') else 192
                self.assertEqual(report['population']['target_identity_count'], 9)
                self.assertEqual(report['position']['available_target_count'], 576)
                self.assertEqual(report['position']['intersection_count'], own)
                self.assertEqual(report['fixed_target_strata']['no_input']['available_target_points'], 192)
                self.assertEqual(report['fixed_target_strata']['no_input']['point_count'], 0)
                self.assertIsNone(report['fixed_target_strata']['no_input']['mae'])
                self.assertEqual(report['paper_position']['latitude']['declared_targets'], 9)
            for r in p.q.PARTITIONS:
                mechanism = r + '_NEAREST_SUPPORTED_vs_' + r + '_NEAREST_ALL'
                self.assertIn(mechanism, result['pairwise'])
            for sid in ('1', '2', '3'):
                row = p.old.frozen.read(args.output / 'scenes' / f'scene-{sid}.json.gz')
                expected = {'A0_NEAREST_ALL', 'WS_NEAREST_ALL'} if sid == '1' else set(p.METHODS)
                self.assertEqual(set(row['newly_computed_prediction_methods']), expected)
                self.assertEqual(len(row['imported_prediction_methods']), 6 if sid == '1' else 0)
                self.assertTrue(row['original056_A0_exact_verified'])
                self.assertTrue(row['fixed_R_population_invariants_verified'])
                with np.load(args.output / 'private' / f'states-{sid}.npz', allow_pickle=False) as saved:
                    self.assertEqual({name for name in p.METHODS if name + '_group_states' in saved}, expected)
                    self.assertNotIn('WS_SOURCE_BALANCED_query_weights', saved.files)
                if sid == '1':
                    for name in row['imported_prediction_methods']:
                        np.testing.assert_array_equal(pilot_artifacts[name]['node_group_indices'],
                                                      np.array([0, 0, 1]) if name.startswith('A0_') else [0, 0, 0])
            # Public output remains summaries/counts: identifiers, coordinates,
            # memberships and per-node weights stay in the private artifacts.
            public_files = [args.output / 'summary.json', args.output / 'scene-summary.json.gz',
                            *sorted((args.output / 'scenes').glob('*.json.gz'))]
            forbidden = {'target_ids', 'target_identity_ids', 'node_ids', 'identities', 'sources',
                         'native_states', 'group_states', 'aligned_truth', 'aligned_prediction',
                         'node_group_indices', 'query_weights', 'weights'}
            def check_public(value):
                if isinstance(value, dict):
                    self.assertFalse(set(value) & forbidden)
                    for item in value.values():
                        check_public(item)
                elif isinstance(value, list):
                    for item in value:
                        check_public(item)
            for path in public_files:
                check_public(p.old.frozen.read(path))
            state = p.old.frozen.read(args.output / 'run.json')
            self.assertEqual((state['status'], state['completed_scenes'], state['exit_code']), ('COMPLETED', 3, 0))
            self.assertEqual((state['imported_prediction_scene_arms'], state['newly_computed_prediction_scene_arms']), (6, 18))
            for name in ('model_forwards', 'models_constructed', 'optimizer_updates'):
                self.assertEqual(state[name], 0)
            for name in ('gpu_used', 'scores_recomputed', 'groups_recomputed', 'threshold_reselected',
                         'normalizer_refitted', 'archive_read', 'csv_read', 'test_opened'):
                self.assertIs(state[name], False)
            self.assertIs(state['cached_reference_read'], True)
            with self.assertRaises(FileExistsError):
                p.run(args)

    def test_metadata_rejects_rebinding_or_incomplete_original116(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, meta, _ = cache_fixture(Path(temporary))
            frozen = p.old.frozen
            original_run = frozen.read(args.pilot_run / 'run.json')
            original_summary = frozen.read(args.pilot_run / 'summary.json')
            with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)):
                actual = p.metadata(args.source_system, args.partition_run, args.pilot_run, {})
                self.assertEqual((actual['ids'], actual['pilot_ids']), (['1', '2', '3'], ['1']))
                cases = [('run.json', key, value) for key, value in (
                    ('status', 'FAILED'), ('idea_id', 'other'), ('plan_version', 'other'),
                    ('source_system', str(Path(temporary) / 'wrong')),
                    ('partition_run', str(Path(temporary) / 'wrong')),
                    ('scene_ids', ['2']), ('completed_scenes', 0), ('methods', list(p.METHODS)))]
                cases += [('summary.json', key, value) for key, value in (
                    ('status', 'RUNNING'), ('idea_id', 'other'), ('plan_version', 'other'),
                    ('scene_ids', ['2']), ('original115_exact_replays_verified', False),
                    ('fixed_R_population_invariants_verified', False), ('no_expansion', False))]
                for filename, key, value in cases:
                    frozen.write(args.pilot_run / 'run.json', original_run)
                    frozen.write(args.pilot_run / 'summary.json', original_summary)
                    expected = original_run if filename == 'run.json' else original_summary
                    frozen.write(args.pilot_run / filename, dict(expected, **{key: value}))
                    with self.subTest(file=filename, key=key), self.assertRaisesRegex(ValueError, 'original116'):
                        p.metadata(args.source_system, args.partition_run, args.pilot_run, {})

    def test_predictions_finish_before_identity_or_reference_is_inspected(self):
        record, fragment, system, memberships, _, _ = prior_tests.old_tests.fixture()
        calls = []
        class GuardedFragment(dict):
            def __getitem__(self, key):
                if key in {'target_identity_ids', 'identities', 'target', 'target_valid'}:
                    if len(calls) != 8:
                        raise AssertionError('reference reached before all predictions were frozen')
                return super().__getitem__(key)
        real_predict = p.predict_group_outputs
        def predict(*args, **kwargs):
            self.assertEqual(len(args), 5)
            self.assertIs(args[0], fragment['native_states'])
            self.assertIs(args[1], fragment['nearest_seconds'])
            self.assertIs(args[2], fragment['support'])
            self.assertIs(args[3], fragment['sources'])
            calls.append(kwargs['method'])
            return real_predict(*args, **kwargs)
        with patch.object(p, 'predict_group_outputs', side_effect=predict):
            row, _, _ = p.analyze_scene(GuardedFragment(fragment), system,
                {r: memberships[r] for r in p.q.PARTITIONS}, record['scene_id'], 'c')
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(row['pairwise']), 16)

    def test_imported_predictions_and_public_reports_must_replay_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, retained = cache_fixture(Path(temporary))
            original = p.old.frozen.read(args.pilot_run / 'scenes' / 'scene-1.json.gz')
            _, fragment, system, memberships, _, _ = prior_tests.old_tests.fixture()
            for corruption in ('prediction', 'reference', 'membership', 'summary'):
                imported, public = deepcopy(retained), deepcopy(original)
                if corruption == 'prediction':
                    imported['WS_NEAREST_SUPPORTED']['group_states'][0, 1, 0] += 1e-12
                elif corruption == 'reference':
                    imported['WS_MEAN']['aligned_truth'][0, 1, 0] += 1e-12
                elif corruption == 'membership':
                    imported['WS_ANCHOR']['node_group_indices'][2] = 1
                else:
                    public['methods']['A0_MEAN']['position']['all']['mae'] += 1.
                with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, '116'):
                    p.analyze_scene(fragment, system, {r: memberships[r] for r in p.q.PARTITIONS},
                                    '1', 'large', pilot_artifacts=imported, pilot_row=public)

    def test_gate_conditions_partition_queries_and_keep_their_physical_common_points(self):
        record, fragment, system, memberships, _, _ = prior_tests.old_tests.fixture()
        # q0: no support, nearest_all is helper; q1: unsupported nearest anchor;
        # q2..63: both controls select the same member. No reference selects W.
        fragment['nearest_seconds'][:, 0] = [10., 0., 1.]
        fragment['support'][0, 1] = False
        fragment['nearest_seconds'][:, 1] = [0., 2., 3.]
        groups = p.old.groups_from_membership(memberships['A0'], len(fragment['native_states']))
        mean = p.old.ordinary_fusion(fragment['native_states'], fragment['nearest_seconds'],
                                   fragment['support'], groups)['predictions']['mean_supported']
        prefix = p.old.BASE + '_'
        targets = fragment['target_identity_ids']
        baseline = p.old.evaluate_full_scene(
            [dict(node_indices=g, states=mean[g[0]]) for g in groups],
            node_identities=fragment['identities'], target_identities=targets,
            target_states=system[prefix + 'aligned_truth'][:len(targets)],
            target_valid=system[prefix + 'aligned_truth_valid'][:len(targets)],
            node_observed=fragment['support'], split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        system.update({prefix + key: value for key, value in baseline['artifact'].items()})
        row, arrays, _ = p.analyze_scene(fragment, system,
            {r: memberships[r] for r in p.q.PARTITIONS}, record['scene_id'], 'c')
        self.assertEqual(row['gate_input_group_query_counts']['WS'], dict(
            has_support_choice_changed=1, no_support_all_not_anchor=1, same_choice=62))
        self.assertEqual(sum(row['gate_input_group_query_counts']['A0'].values()), 128)
        key = 'WS_NEAREST_SUPPORTED_vs_WS_NEAREST_ALL'
        pair = row['pairwise'][key]
        for region, count in (('gate_has_support_choice_changed', 1),
                              ('gate_no_support_all_not_anchor', 1), ('gate_same_choice', 62)):
            self.assertEqual(pair['strata'][region]['A0']['point_count'], count)
            self.assertEqual(pair['strata'][region]['candidate']['point_count'], count)
        self.assertEqual(pair['common_valid_position']['A0']['intersection_count'], 64)
        self.assertEqual(row['methods']['WS_NEAREST_ALL']['position']['all']['available_target_count'], 192)
        merged = p.aggregate([row, deepcopy(row)], [arrays, arrays])
        self.assertEqual(merged['gate_input_group_query_counts']['WS']['has_support_choice_changed'], 2)
        self.assertEqual(merged['pairwise'][key]['strata']['gate_no_support_all_not_anchor']['A0']['point_count'], 2)

    def test_last_scene_exact_failure_keeps_first_two_outputs_and_never_claims_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, meta, _ = cache_fixture(Path(temporary))
            path = args.source_system / 'predictions' / 'states-3.npz'
            with np.load(path, allow_pickle=False) as saved:
                changed = {key: saved[key] for key in saved.files}
            changed['native_states'][0, 1, 0] += 1e-12
            np.savez_compressed(path, **changed)
            with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)):
                with self.assertRaisesRegex(ValueError, '056 A0 Mean exact'):
                    p.run(args)
            state = p.old.frozen.read(args.output / 'run.json')
            self.assertEqual((state['status'], state['exit_code'], state['completed_scenes']), ('FAILED', 1, 2))
            self.assertEqual((state['imported_prediction_scene_arms'], state['newly_computed_prediction_scene_arms']), (6, 10))
            for sid in ('1', '2'):
                self.assertTrue((args.output / 'private' / f'states-{sid}.npz').exists())
                self.assertTrue((args.output / 'scenes' / f'scene-{sid}.json.gz').exists())
            self.assertFalse((args.output / 'private' / 'states-3.npz').exists())
            self.assertFalse((args.output / 'summary.json').exists())

    def test_budget_failure_at_start_or_final_checkpoint_preserves_failed_terminal_state(self):
        for phase in ('start', 'final'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                args, meta, _ = cache_fixture(Path(temporary))
                def memory():
                    exceeds = phase == 'start' or (args.output / 'summary.json').exists()
                    return p.old.frozen.RAM_BUDGET + 1 if exceeds else 1024
                with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)), \
                     patch.object(p.old.frozen, 'peak_rss_bytes', side_effect=memory):
                    with self.assertRaises(MemoryError):
                        p.run(args)
                state = p.old.frozen.read(args.output / 'run.json')
                self.assertEqual((state['status'], state['exit_code']), ('FAILED', 1))
                self.assertEqual(state['completed_scenes'], 0 if phase == 'start' else 3)
                if phase == 'final':
                    self.assertEqual(p.old.frozen.read(args.output / 'summary.json')['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
