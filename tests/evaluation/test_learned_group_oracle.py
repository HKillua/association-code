"""061 frozen learned groups: real cache replay, population and choice contracts."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.learned_group_oracle import analyze_scene, aggregate, aggregation_arrays, METHODS
from association.evaluation.oracle_diagnostics import POPULATION_KEYS, select_candidates
from association.evaluation.system_attribution import ATTRIBUTES
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _replay
from tools.analysis.learned_group_oracle import CONFIG, SOURCE_RUN, run
from tools.analysis.system_attribution import read
from tools.experiments.level1_system_evaluate import evaluate_unique_population, ordinary_predictions
from tests.evaluation import test_association_duration as duration_fixture
from tests.evaluation import test_oracle_diagnostics as oracle_fixture


class LearnedGroupOracleTests(unittest.TestCase):
    @staticmethod
    def fixture(sparse_reference=False):
        """Reuse058's swallowed target/duplicate layout, expanded to the Q64 cache."""
        fragment, _, _ = oracle_fixture.OracleDiagnosticsTests.fixture()
        native = np.repeat(fragment['native_states'], 16, axis=1)
        truth = np.broadcast_to(np.array([[30., 120., 5., 1.], [30.01, 120., 6., 1.],
            [30.02, 120., 7., 2.], [30.03, 120., 8., 180.], [30.04, 120., 9., 90.]])[:, None],
            (5, 64, 4)).copy()
        valid = np.ones(truth.shape, bool)
        valid[1, 9, :2] = False
        valid[1, 10, 2] = False
        valid[1, 11, 3] = False
        valid[4] = False
        native[:, :, 1] = 120.
        native[2, :, 3], native[3, :, 3] = 359., 1.
        native[3, :, 0] = 30.  # Mixing helps A, but a later duplicate is better still.
        native[4] = truth[1]
        native[2, 3, 3], native[3, 3, 3] = 0., 180.  # Heading fallback.
        native[2, 4, 1], native[3, 4, 1] = 0., 180.  # Longitude fallback, latitude still mixes.
        native[3, 8] = native[2, 8]  # A supported helper can leave every candidate unchanged.
        support = np.ones((6, 64), bool)
        support[2, 5:7] = False
        support[3, 6:8] = False
        nearest = np.full((6, 64), 10.)
        nearest[2, 5:7] = 2000.
        nearest[3, 6:8] = 2000.
        if sparse_reference:
            truth[0, :, 0] += .2
            valid[:] = False
            valid[0, 0] = True
        truth[~valid] = np.nan
        nodes, targets = fragment['identities'], fragment['target_identity_ids']
        target_rows = [0, 1, 1, 2, 1, 4]
        fragment.update(native_states=native, target=truth[target_rows], target_valid=valid[target_rows],
            query_seconds=np.linspace(0., 2000., 64), support=support, nearest_seconds=nearest,
            exact_observation=np.zeros((6, 64), bool))
        groups = [[0, 1], [2, 3], [4], [5]]
        controls = ordinary_fusion(native, nearest, support, groups)
        arms = ordinary_predictions(controls, ('anchor', 'mean_supported', 'attribute_hybrid'))
        scene = dict(identities=nodes, target_identity_ids=targets.tolist(), target=fragment['target'],
            valid=fragment['target_valid'], motion=dict(states=native, support=support, nearest_seconds=nearest,
                                                       query_seconds=fragment['query_seconds']),
            fusion_targets={identity: dict(states=truth[i], valid=valid[i]) for i, identity in enumerate(targets)})
        reports, artifacts = evaluate_unique_population(scene, dict(ols=groups, learned_assoc=groups),
                                                       dict(ols=arms, learned_assoc=arms))
        for grouping in ('ols', 'learned_assoc'):
            fragment.update({grouping + '_' + arm: value for arm, value in arms.items()})
            fragment[grouping + '_membership'] = controls['membership']
            fragment[grouping + '_group_offsets'] = np.array([0, 2, 4, 5, 6])
            fragment[grouping + '_group_node_indices'] = np.arange(6)
        saved = {name + '_' + key: value for name, item in artifacts.items() for key, value in item.items()}
        expected = {name: dict(population={key: item[key] for key in POPULATION_KEYS},
                               full_target_metrics=item['full_target_metrics']) for name, item in reports.items()}
        return fragment, saved, expected

    @staticmethod
    def cache(root):
        """Use060's real synthetic six-arm cache with an actual056 provenance envelope."""
        original = duration_fixture.AssociationDurationTests.fixture(root)
        task = root/SOURCE_RUN
        task.mkdir()
        source, training = task/'system', task/'association'
        original.reference_system.rename(source)
        (root/'reference-fit').rename(training)
        inputs = {key: str(training/filename) for key, filename in (
            ('association_run', 'run.json'), ('association_summary', 'summary.json'),
            ('association_data_audit', 'data_audit.json'), ('association_checkpoint', 'final.pt'),
            ('association_scores', 'final-scores.npz'))}
        for filename in ('run.json', 'summary.json'):
            document = read_json(source/filename)
            document['inputs'] = inputs
            write_json(source/filename, document)
        config = read_json(CONFIG)
        write_json(training/'run.json', dict(status='COMPLETED', optimizer_updates=1000,
            kind=config['scope'], config=config, synthetic=False, reference_read=False, test_opened=False))
        return SimpleNamespace(source_system=source, manifest=original.manifest,
                               output=root/'oracle061', cpu_threads=1)

    def test_frozen_representative_population_and_three_cached_controls(self):
        fragment, saved, expected = self.fixture()
        replay = _replay(fragment, saved, expected)
        row, arrays = analyze_scene(fragment, replay)
        self.assertEqual(set(row['methods']), set(METHODS))
        population = row['methods']['anchor']['population']
        self.assertEqual(population['target_identity_count'], 5)
        self.assertEqual(population['predicted_group_count'], 4)
        self.assertEqual(population['duplicate_prediction_group_count'], 1)
        self.assertEqual(population['reference_unavailable_target_count'], 1)
        np.testing.assert_array_equal(arrays['unique_selected_group_indices'], [0, 1, -1, -1, 3])
        np.testing.assert_array_equal(arrays['group_anchor_node_indices'], [0, 2, 4, 5])
        for arm in METHODS:
            self.assertEqual(row['methods'][arm]['population'], population)
            self.assertEqual(row['methods'][arm]['full_target_metrics']['all']['position']['intersection_count'], 127)
        for arm in METHODS[:3]:
            self.assertEqual(row['methods'][arm]['full_target_metrics'],
                             replay[0]['learned_assoc_' + arm]['full_target_metrics'])
        # The later duplicate has zero error, but it must not replace A's first group.
        self.assertGreater(arrays['oracle_selector_position_error'][1, 0], 0.)
        np.testing.assert_array_equal(arrays['oracle_group_states'][2, 0, :2], fragment['native_states'][4, 0, :2])
        self.assertFalse(arrays['unique_common_valid'][2:4].any())  # B swallowed; C has no input.
        self.assertFalse(arrays['selector_mean_choice'][3].any())  # D's reference unavailable.
        for pair in row['pairwise'].values():
            self.assertEqual(pair['changed_representative_target_count'], 0)
            self.assertTrue(all(value == 0 for item in pair['coverage_transitions'].values() for value in item.values()))

    def test_choices_are_whole_candidates_and_opportunity_populations_are_distinct(self):
        fragment, saved, expected = self.fixture()
        replay = _replay(fragment, saved, expected)
        row, arrays = analyze_scene(fragment, replay)
        anchor = replay[1]['learned_assoc_anchor']['group_states']
        mean = replay[1]['learned_assoc_mean_supported']['group_states']
        choice, selected = arrays['selector_mean_choice'], arrays['oracle_group_states']
        for index, (_, columns) in enumerate(ATTRIBUTES.items()):
            for column in columns:
                np.testing.assert_array_equal(selected[..., column],
                    np.where(choice[..., index], mean[..., column], anchor[..., column]))
        self.assertTrue(choice[1, 0, 0])
        self.assertTrue(choice[1, 0, 2])  # 359/1 circular mean is closer to truth1 than 359.
        self.assertFalse(choice[1, 3, 2])  # Opposite headings fall back, hence tie->Anchor.
        self.assertFalse(choice[1, 9, 0])  # Invalid position reference does not select Mean.
        self.assertFalse(choice[1, 10, 1])
        self.assertFalse(choice[1, 11, 2])
        self.assertTrue(arrays['helper_supported'][1, 3])
        self.assertFalse(arrays['helper_participation'][1, 3, 3])
        self.assertTrue(arrays['helper_position_any'][1, 4])
        self.assertFalse(arrays['helper_position_both'][1, 4])
        self.assertTrue(arrays['helper_exists'][1, 7])
        self.assertFalse(arrays['helper_supported'][1, 7])
        self.assertTrue(arrays['helper_participation'][1, 8].all())
        self.assertFalse(arrays['candidate_nontrivial'][1, 8].any())
        position = row['selector_choices']['position']
        self.assertEqual(position['group']['eligible_point_count'], 190)
        self.assertEqual(position['unique']['eligible_point_count'], 127)
        self.assertEqual(position['group']['helper_exists_point_count'], 127)
        self.assertEqual(position['group']['helper_supported_point_count'], 125)
        self.assertEqual(position['group']['helper_participates_any_point_count'], 125)
        self.assertEqual(position['group']['helper_participates_all_point_count'], 124)
        self.assertEqual(position['unique']['nontrivial_point_count'], 124)
        strata = row['methods']['anchor']['common_target_strata']
        self.assertEqual(strata['anchor_outside_helper_supported']['position']['intersection_count'], 1)
        self.assertEqual(strata['predicted_group_no_support']['position']['intersection_count'], 1)
        self.assertEqual(strata['anchor_distance_gt_1800s']['position']['intersection_count'], 2)
        self.assertEqual(strata['no_predicted_group']['position']['intersection_count'], 0)

    def test_oracle_rejects_third_state_and_mismatched_prediction_masks(self):
        fragment, saved, expected = self.fixture()
        replay = _replay(fragment, saved, expected)

        def forged(anchor, mean, truth, valid, prediction_valid):
            states, mask, choice = select_candidates(anchor, mean, truth, valid, prediction_valid)
            states[1, 0, :2] = truth[1, 0, :2]
            return states, mask, choice

        with patch('association.evaluation.learned_group_oracle.select_candidates', side_effect=forged):
            with self.assertRaisesRegex(ValueError, 'candidate membership'):
                analyze_scene(fragment, replay)
        wrong = deepcopy(replay)
        wrong[1]['learned_assoc_mean_supported']['group_valid'][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, 'fixed ordinary group_valid'):
            analyze_scene(fragment, wrong)

    def test_pooled_quantiles_use_unique_points_and_empty_strata_are_null(self):
        rows, points = [], []
        for sparse in (False, True):
            fragment, saved, expected = self.fixture(sparse_reference=sparse)
            row, arrays = analyze_scene(fragment, _replay(fragment, saved, expected))
            rows.append(row)
            points.append(aggregation_arrays(arrays))
        result = aggregate(rows, points)
        values = np.concatenate([item['anchor_position_error'][np.isfinite(item['anchor_position_error'])]
                                 for item in points])
        self.assertEqual(len(values), 128)
        actual = result['methods']['anchor']['position_quantiles']['all']
        self.assertEqual(actual['point_count'], 128)
        self.assertEqual(actual['p95'], np.quantile(values, .95, method='linear'))
        self.assertEqual(actual['p99'], np.quantile(values, .99, method='linear'))
        scene_average = np.mean([row['methods']['anchor']['position_quantiles']['all']['p95'] for row in rows])
        self.assertNotAlmostEqual(actual['p95'], scene_average)
        self.assertIsNone(result['methods']['anchor']['position_quantiles']['no_predicted_group']['p99'])
        self.assertEqual(result['methods']['anchor']['position_quantiles']['no_predicted_group']['point_count'], 0)

    def test_real_cache_entry_and_ordinary_array_tamper(self):
        with TemporaryDirectory() as temporary:
            args = self.cache(Path(temporary))
            result = run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['scene_count'], 2)
            self.assertEqual(result['association_endpoint_step'], 1000)
            self.assertTrue(result['all_source_scenes_replayed'])
            self.assertFalse(result['test_opened'])
            self.assertFalse(result['gpu_used'])
            self.assertEqual(len(read(args.output/'scene-summary.json.gz')), 2)
            with np.load(args.output/'predictions'/'states-01.npz', allow_pickle=False) as arrays:
                self.assertIn('selector_mean_choice', arrays.files)
                self.assertNotIn('native_states', arrays.files)
            with self.assertRaises(FileExistsError):
                run(args)
            path = args.source_system/'predictions'/'states-01.npz'
            with np.load(path, allow_pickle=False) as saved:
                original = {key: saved[key] for key in saved.files}
            for arm in ('anchor', 'mean_supported', 'attribute_hybrid'):
                changed = deepcopy(original)
                changed['learned_assoc_' + arm][0, 0, 0] += .001
                np.savez_compressed(path, **changed)
                args.output = Path(temporary)/('tampered-' + arm)
                with self.assertRaisesRegex(ValueError, 'cached ordinary prediction'):
                    run(args)
                self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')
                self.assertFalse((args.output/'summary.json').exists())

    def test_wrong_source_run_and_interruption_do_not_publish_success(self):
        for case in ('different_run', 'unfinished', 'wrong_seed', 'selected_path',
                     'extra_sampling', 'interrupted'):
            with self.subTest(case=case), TemporaryDirectory() as temporary:
                args = self.cache(Path(temporary))
                if case == 'different_run':
                    changed = Path(temporary)/'LEVEL1-ASSOCIATION-20261022-059A-attempt-1'
                    args.source_system.parent.rename(changed)
                    args.source_system = changed/'system'
                    with self.assertRaisesRegex(ValueError, 'original056'):
                        run(args)
                elif case == 'unfinished':
                    path = args.source_system.parent/'association'/'run.json'
                    document = read_json(path)
                    document['status'] = 'RUNNING'
                    write_json(path, document)
                    with self.assertRaisesRegex(ValueError, 'training source'):
                        run(args)
                elif case in ('wrong_seed', 'extra_sampling'):
                    path = args.source_system.parent/'association'/'run.json'
                    document = read_json(path)
                    if case == 'wrong_seed':
                        document['config']['seed'] = 20261022
                    else:
                        document['config']['sampling'] = {'mode': 'uniform'}
                    write_json(path, document)
                    with self.assertRaisesRegex(ValueError, 'original training|original sibling'):
                        run(args)
                elif case == 'selected_path':
                    for filename in ('run.json', 'summary.json'):
                        path = args.source_system/filename
                        document = read_json(path)
                        document['inputs']['association_checkpoint'] = str(
                            args.source_system.parent/'association'/'selected.pt')
                        write_json(path, document)
                    with self.assertRaisesRegex(ValueError, 'original training|original sibling'):
                        run(args)
                else:
                    with patch('tools.analysis.learned_group_oracle.analyze_scene', side_effect=KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            run(args)
                self.assertEqual(read_json(args.output/'run.json')['status'],
                                 'INTERRUPTED' if case == 'interrupted' else 'FAILED')
                self.assertFalse((args.output/'summary.json').exists())
                args.output = args.source_system.parent/'new-output'
                with self.assertRaisesRegex(ValueError, 'entire immutable056 run'):
                    run(args)
                self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
