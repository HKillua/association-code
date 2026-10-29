"""Small synthetic saved-array fixtures: software contracts, never MTAD results."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from association.evaluation import source_risk as risk
from association.utils import runtime
from tests.integration.test_source_risk_pilot import cached_input
from tools.analysis import observation_common as common
from tools.analysis import observation_conflict_probe as probe


def fixture(root):
    """Fabricate tiny prior caches, without any model fit or oracle grid search.

    Metadata fit/grid counts stand in for completed original runs. They do not
    describe computations performed here. Only fixture state arrays are built.
    """
    original = cached_input(root)
    source, quality, mixture = (root/name for name in ('fixture133', 'fixture136', 'fixture134'))
    for directory in (source, quality, mixture):
        (directory/'private').mkdir(parents=True)
    manifest = runtime.read_json(original.manifest)
    tids, vids = manifest['splits']['T_core'], manifest['splits']['V_select']
    common_state = dict(status='COMPLETED', exit_code=0, fixture_only=True, training_scene_ids=tids,
        scene_ids=vids, completed_training_scene_count=32, completed_scene_count=32,
        native_run=str(original.native_run), value_run=str(original.value_run), partition_run=str(original.partition_run))
    runtime.write_json(source/'run.json', dict(common_state, idea_id='IDEA-SOURCE-RISK-20261030', analytic_head_fits=39))
    runtime.write_json(quality/'run.json', dict(common_state, idea_id='IDEA-OBSERVATION-QUALITY-20261030',
        source_risk_run=str(source), manifest=str(original.manifest), T_methods=list(probe.METHODS)))
    runtime.write_json(mixture/'run.json', dict(status='COMPLETED', exit_code=0, fixture_only=True,
        idea_id='IDEA-MIXTURE-OPPORTUNITY-20261030', source_run=str(source), native_run=str(original.native_run),
        scene_ids=tids, completed_scene_count=32, reference_oracle_only=True, grid_size=4097))
    folds = []
    for component in sorted({manifest['components'][sid] for sid in tids}):
        selected = [sid for sid in tids if manifest['components'][sid] != component]
        folds.append(dict(heldout_component=component, training_scene_ids=selected,
                          training_components=sorted({manifest['components'][sid] for sid in selected})))
    runtime.write_json(source/'T-fold-index.json', folds)
    np.savez_compressed(mixture/'private/roles.npz', source_names=np.asarray(['source-z', 'source-a']),
                        indices=np.asarray([0, 1], np.int64))
    quality_rows, mixture_rows = [], []
    # Each block covers A/B/C/D, a Q tie and unknown LOO. Scores are synthetic
    # saved values, not trained predictions or proposed fusion policy results.
    preferences = np.asarray([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0],
                              [0, 1, 1, 1], [-1, -1, 0, -1], [-2, -2, 1, 0]])
    for sid in tids:
        component = manifest['components'][sid]
        motion_path = original.native_run/'private/native-motion/T_core'/f'scene-{sid}.npz'
        motion = common.load_npz(motion_path)
        chosen = preferences[np.arange(64) % len(preferences)]
        motion['nearest_seconds'][0] = np.where(chosen[:, 2] == 0, 2., 8.)
        motion['nearest_seconds'][1] = np.where(chosen[:, 2] == 1, 2., 8.)
        np.savez_compressed(motion_path, **motion)
        ev_path = original.native_run/'private/native-evaluation/T_core'/f'scene-{sid}.npz'
        ev = common.load_npz(ev_path)
        if int(sid) % 2:
            ev['identities'][1] = 't1'  # retain mixed first-anchor groups
            np.savez_compressed(ev_path, **ev)
        membership = np.asarray([0, 0, 1, 2], np.int64)
        motion, groups, mean, pack, saved = common.physical(original.native_run, 'T_core', sid, membership)
        labels = risk.source_labels(pack, groups, ev['identities'], ev['target_ids'], ev['target_states'], ev['target_valid'])
        saved.update({key: value for key, value in labels.items() if isinstance(value, np.ndarray)})
        saved['component_id'] = np.asarray(component)
        np.savez_compressed(source/'private'/f'training-{sid}.npz', **saved)
        policy = dict(saved, support=motion['support'])
        score = np.ones(pack['available'].shape)
        for side in (0, 1):
            score[side] = np.where(chosen[:, 3] == side, 0., 1.)
        np.savez_compressed(source/'private'/f'oof-{sid}.npz', QUADRATIC__scores=score)
        for column, label in ((0, 'loo'), (1, 'ols')):
            proxy = np.full(pack['available'].shape, 400.)
            known = np.ones(pack['available'].shape, bool)
            for side in (0, 1):
                proxy[side] = np.where(chosen[:, column] == side, 200., 400.)
                proxy[side, chosen[:, column] == -2] = np.nan
                known[side, chosen[:, column] == -2] = False
            policy[label+'__source_proxy_m2'], policy[label+'__source_known'] = proxy, known
        predictions = {}
        for name in probe.METHODS:
            if name == 'WS_SOURCE_EQUAL':
                local_score, kind = np.zeros_like(score), 'equal'
            elif name == 'WS_SOURCE_NEAREST':
                local_score, kind = motion['nearest_seconds'].copy(), 'hard'
            else:
                label = 'ols' if '_OLS_' in name else 'loo'
                # log(proxy) under softmax(-score) gives inverse proxy mass;
                # missing evidence explicitly falls back to equal for the group.
                local_score, kind = np.log(np.nan_to_num(policy[label+'__source_proxy_m2'], nan=400.)), 'soft'
                local_score[:2, ~policy[label+'__source_known'][:2].all(0)] = 0.
            recovered = risk.recover_source_policy(states=motion['states'], support=motion['support'], groups=groups,
                pack=pack, scores=local_score, kind=kind, mean_states=mean)
            predictions[name] = recovered['group_states']
            policy[name+'__source_weights'] = recovered['source_weights']
            policy[name+'__group_states'] = recovered['group_states']
        np.savez_compressed(quality/'private'/f'policy-T_core-{sid}.npz', **policy)
        np.savez_compressed(quality/'private'/f'diagnostic-T_core-{sid}.npz',
            **{key: labels[key] for key in ('errors_m', 'labeled', 'sample_weight')})
        row, _, artifacts = common.evaluate_predictions(motion, groups, pack, ev, predictions, {},
                                                       method_names=probe.METHODS, split='T_core')
        row.update(scene_id=sid, component_id=component)
        quality_rows.append(row)
        common.save_artifacts(quality/'private'/f'states-T_core-{sid}.npz', artifacts)
        qi = np.flatnonzero(labels['labeled'][:2].all(0))
        gi = np.zeros(len(qi), np.int64)
        sr = np.tile(np.asarray([0, 1], np.int64), (len(qi), 1))
        endpoint = labels['errors_m'][sr, qi[:, None]]
        hard = risk.recover_source_policy(states=motion['states'], support=motion['support'], groups=groups,
            pack=pack, scores=score, kind='hard', mean_states=mean)['group_states'][gi, qi, :2]
        truth = ev['target_states'][0, qi, :2]
        pair_score = score[sr, qi[:, None]]
        mass = (pair_score == pair_score.min(1, keepdims=True)).astype(float)
        mass /= mass.sum(1, keepdims=True)
        # A fabricated zero-gain cached grid diagnostic is sufficient for join
        # testing; 138 must never attempt to verify it by running a grid itself.
        grid = endpoint.min(1)
        bound = np.full(len(qi), .01); bound[::5] = np.nan
        np.savez_compressed(mixture/'private'/f'scene-{sid}.npz', node_ids=motion['node_ids'],
            query_seconds=motion['query_seconds'], membership=membership, group_indices=gi, query_indices=qi,
            source_rows=sr, group_pure=np.full(len(qi), int(sid) % 2 == 0, bool), truth=truth,
            endpoint_errors_m=endpoint, best_single_error_m=grid, grid_error_m=grid, gain_m=np.zeros(len(qi)),
            gap_bound_m=bound, min_resultant=np.ones(len(qi)), oof_QUADRATIC_scores=pair_score,
            QUADRATIC_hard_position=hard, QUADRATIC_hard_error_m=risk._position_distance(hard, truth),
            QUADRATIC_selection_regret_m=(mass*endpoint).sum(1)-grid,
            role_indices=np.tile(np.asarray([0, 1], np.int64), (len(qi), 1)))
        scope = dict(total_group_queries=192, source_support_zero=3, source_support_one=127, source_support_multi=62,
            complete_target_reference_valid_query_count=191, labeled_group_query_count=61,
            duplicate_anchor_group_count=1, unknown_anchor_group_count=0, first_anchor_target_count=2,
            target_without_anchor_count=1, first_anchor_reference_invalid_query_count=1,
            first_anchor_reference_valid_zero_source_count=2, first_anchor_reference_valid_one_source_count=64,
            first_anchor_reference_valid_multi_source_count=61, mixed_identity_group_count=int(sid) % 2,
            duplicate_group_queries=64, unknown_group_queries=0, reference_valid_points_without_anchor=64)
        mixture_rows.append(dict(scene_id=sid, component_id=component, scope=scope))
    common.old.frozen.write(quality/'training-scene-summary.json.gz', quality_rows)
    common.old.frozen.write(mixture/'scene-summary.json.gz', mixture_rows)
    runtime.write_json(mixture/'summary.json', dict(all_components=dict(query_count=32*61),
        scope={key: sum(row['scope'][key] for row in mixture_rows) for key in mixture_rows[0]['scope']}))
    return SimpleNamespace(quality_run=quality, source_run=source, mixture_run=mixture, native_run=original.native_run,
                           manifest=original.manifest, output=root/'new138', cpu_threads=1, time_policy='advisory')


class ObservationConflictTests(unittest.TestCase):
    def test_preferences_unknown_tie_and_endpoint_expectation_are_not_physical_mixture(self):
        desired = np.asarray([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0],
                              [0, 1, 1, 1], [-1, -1, 0, -1], [-2, -2, 1, 0]])
        values = np.ones((6, 4, 2))
        known = desired != -2
        for i, j in np.ndindex(desired.shape):
            if desired[i, j] >= 0:
                values[i, j, desired[i, j]] = 0.
            elif desired[i, j] == -2:
                values[i, j] = np.nan
        endpoint = np.tile([3., 7.], (6, 1))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = probe.preference_quantities(values, known, endpoint)
        np.testing.assert_array_equal(result['cell'], np.arange(6))
        self.assertEqual(result['selection_cost_m'][4, 0], 5.)
        self.assertEqual(result['selection_credit'][4, 0], .5)
        self.assertTrue(np.isnan(result['selection_regret_m'][5, 0]))
        np.testing.assert_array_equal(result['preference_mass'][5, 0], [0., 0.])
        swapped = probe.preference_quantities(values[:, :, ::-1], known, endpoint[:, ::-1])
        for key in ('cell', 'ols_subcell', 'selection_cost_m', 'selection_regret_m', 'selection_credit'):
            np.testing.assert_array_equal(result[key], swapped[key])
        # Two saved opposite endpoints may each be nonzero while their saved
        # midpoint is exact. Expected endpoint error cannot substitute for it.
        errors = probe.distance(np.asarray([[0., -1.], [0., 1.]]), np.zeros((2, 2)))
        self.assertGreater(errors.mean(), 100000.)
        self.assertEqual(probe.distance(np.zeros((1, 2)), np.zeros((1, 2)))[0], 0.)
        self.assertEqual(probe.delta_statistics([-5., 0., 1000., 1001.])['negative_count'], 1)
        self.assertIsNone(probe.statistics([])['mean'])

    def test_complete_chain_T_only_io_and_preserved_binding_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary).resolve())
            original_npz, original_json = probe.Reader.npz, probe.Reader.json
            accessed = []
            def guard(path):
                text = str(path)
                self.assertFalse(any(term in text for term in ('V_select', 'T_test', 'native-events', 'models.npz')))
                self.assertFalse(Path(path) in (args.source_run/'summary.json', args.quality_run/'summary.json'))
                accessed.append(text)
            def read_npz(reader, path, keys=None):
                guard(path)
                return original_npz(reader, path, keys)
            def read_json(reader, path):
                guard(path)
                return original_json(reader, path)
            with patch.object(probe.Reader, 'npz', read_npz), patch.object(probe.Reader, 'json', read_json):
                result = probe.run(args)
            self.assertEqual(result['all_components']['query_count'], 1952)
            self.assertEqual(result['all_components']['full_population']['point_count'], 4064)
            self.assertEqual(len(result['by_component']), 12)
            self.assertEqual(len(result['primary_bootstrap']), 4)
            self.assertEqual(sum(layer['query_count'] for layer in result['all_components']['main_cells'].values()), 1952)
            self.assertGreater(result['all_components']['main_cells']['U']['query_count'], 0)
            self.assertGreater(result['all_components']['overall']['oracle']['uncertified_count'], 0)
            state = runtime.read_json(args.output/'run.json')
            self.assertEqual(state['status'], 'COMPLETED')
            for key in ('new_fits', 'new_physical_predictions', 'new_grid_evaluations', 'model_forwards'):
                self.assertEqual(state[key], 0)
            self.assertTrue(accessed)
            # Source rows are joined by their actual member sets, not numbers
            # or anonymous role order. A permutation is legitimate; merging
            # different member sets is not.
            pack = common.load_npz(args.source_run/'private/training-0.npz')
            policy = common.load_npz(args.quality_run/'private/policy-T_core-0.npz')
            motion = common.load_npz(args.native_run/'private/native-motion/T_core/scene-0.npz')
            perm = np.asarray([1, 0, 2, 3])
            for key in ('features', 'source_states', 'available', 'supported_counts',
                        'source_group_indices', 'source_member_count'):
                policy[key] = policy[key][perm]
            policy['node_to_source_row'] = perm[policy['node_to_source_row']]
            _, mapping, _ = probe.structural_join(pack, policy, motion)
            np.testing.assert_array_equal(mapping, perm)
            policy['node_to_source_row'][0] = policy['node_to_source_row'][1]
            with self.assertRaisesRegex(ValueError, 'source members|identity join'):
                probe.structural_join(pack, policy, motion)
            # The old QOOF is admissible only with its original heldout
            # component excluded from the base fit, even though no fit runs now.
            folds_path = args.source_run/'T-fold-index.json'
            folds = runtime.read_json(folds_path)
            folds[0]['training_components'].append(folds[0]['heldout_component'])
            runtime.write_json(folds_path, folds)
            paths = {key: getattr(args, key) for key in ('quality_run', 'source_run', 'mixture_run', 'native_run', 'output')}
            with self.assertRaisesRegex(ValueError, 'OOF component leakage'):
                probe.metadata(paths, probe.Reader(), args.manifest)
            folds[0]['training_components'].pop()
            runtime.write_json(folds_path, folds)
            mixture_path = args.mixture_run/'private/scene-0.npz'
            mixture = common.load_npz(mixture_path)
            original_query = mixture['query_indices'].copy()
            mixture['query_indices'][1] = mixture['query_indices'][0]
            np.savez_compressed(mixture_path, **mixture)
            qrows = probe.Reader().json(args.quality_run/'training-scene-summary.json.gz')
            mrows = probe.Reader().json(args.mixture_run/'scene-summary.json.gz')
            with self.assertRaisesRegex(ValueError, '134 complete D2 query_indices'):
                probe.scene(paths, '0', 'T0', {'source-z': 0, 'source-a': 1}, probe.Reader(), qrows[0], mrows[0])
            mixture['query_indices'] = original_query
            np.savez_compressed(mixture_path, **mixture)
            path = args.native_run/'private/native-evaluation/T_core/scene-0.npz'
            saved = common.load_npz(path); saved['query_seconds'] += 1.
            np.savez_compressed(path, **saved)
            other = SimpleNamespace(**vars(args)); other.output = args.output.parent/'broken138'
            with self.assertRaisesRegex(ValueError, 'query_seconds'):
                probe.run(other)
            self.assertEqual(runtime.read_json(other.output/'run.json')['status'], 'FAILED')
            self.assertFalse((other.output/'summary.json').exists())
            self.assertEqual(runtime.read_json(args.output/'run.json')['status'], 'COMPLETED')

    def test_conditional_component_bootstrap_keeps_empty_draws_and_undefined_CI(self):
        def item(component, cell):
            return dict(component_id=component, data=dict(cell=np.asarray([cell]),
                actual_error_m=np.asarray([[3., 4., 3., 3., 2., 2., 3.]]),
                selection_regret_m=np.asarray([[1., np.nan, 4., 1.]])))
        result = probe.bootstrap([item('c0', 2), item('c1', 0)])
        primary = result['actual_LOO_minus_NEAREST']
        self.assertEqual(primary['point_estimate_m'], -2.)
        self.assertEqual(primary['contributing_component_count'], 1)
        self.assertIsNone(primary['percentile95_m'])
        self.assertGreater(primary['empty_resamples'], 0)
        self.assertEqual(primary['empty_resamples']+primary['valid_resamples'], 2000)
        unknown = result['selection_O_minus_N']
        self.assertIsNone(unknown['point_estimate_m'])
        self.assertEqual(unknown['valid_resamples'], 0)
        self.assertEqual(unknown['empty_resamples'], 2000)


if __name__ == '__main__':
    unittest.main()
