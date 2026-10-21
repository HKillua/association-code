"""054 attribution contracts using real grouping, fusion and physical evaluators."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from association.evaluation.fusion_controls import metrics, ordinary_fusion
from association.evaluation.physical import evaluate_aligned_fusion
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.evaluation.system_attribution import (
    ANCHOR, BASE, INDEPENDENT, analyze_scene, pool_metrics,
)
from association.utils.runtime import read_json, write_json
from tools.analysis.system_attribution import read as read_attribution, run as run_attribution
from tools.experiments.level1_system_evaluate import _pool_fragment


class SystemAttributionTests(unittest.TestCase):
    @staticmethod
    def fixture(*, invalid_first=False, disjoint_masks=False):
        """B is swallowed by A, C has no fragment, D has no reference.

        A's second group is a duplicate with a *better* estimate. It must not
        replace the first A output just because the reference makes it preferable.
        """
        q = 3
        target_ids = np.array(['a', 'b', 'c', 'd'])
        identities = np.array(['a', 'b', 'a', 'd'])
        truth = np.broadcast_to(np.array([
            [30., 120., 5., 1.], [31., 121., 6., 350.],
            [32., 122., 7., 180.], [33., 123., 8., 90.],
        ])[:, None], (4, q, 4)).copy()
        truth_valid = np.ones_like(truth, dtype=bool)
        truth_valid[3] = False
        truth[3] = np.nan
        if disjoint_masks:
            truth_valid[0, 1:, :2] = False
            truth_valid[0, [0, 2], 2] = False
        native = np.array([truth[0], truth[1], truth[0],
                           np.tile([33., 123., 8., 90.], (q, 1))])
        native[0, :, 0] += .002
        native[0, :, 2] += 4.
        native[0, :, 3] = 359.
        support = np.array([[True, False, False], [True, True, False],
                            [True, False, False], [True, True, True]])
        nearest = np.array([[0., 400., 2000.], [0., 0., 1900.],
                            [0., 500., 2200.], [0., 0., 0.]])
        groups = [[0, 1], [2], [3]]
        ordinary = ordinary_fusion(native, nearest, support, groups)
        baseline = ordinary['predictions']['mean_supported']
        if invalid_first:
            baseline = baseline.copy()
            baseline[0] = np.nan
        elif disjoint_masks:
            baseline = baseline.copy()
            baseline[0, [0, 2], :2] = np.nan
            baseline[0, [1, 2], 2] = np.nan
        node_target_indices = np.array([0, 1, 0, 3])
        fragment = dict(
            node_ids=np.array(['a-first', 'b-first', 'a-duplicate', 'd-first']),
            identities=identities, sources=np.array(['1', '2', '2', '1']),
            target_identity_ids=target_ids, query_seconds=np.array([0., 400., 2000.]),
            target=truth[node_target_indices], target_valid=truth_valid[node_target_indices],
            native_states=native, nearest_seconds=nearest, support=support,
            exact_observation=support.copy(), ols_mean_supported=baseline,
            ols_membership=ordinary['membership'],
            ols_group_offsets=np.array([0, 2, 3, 4]),
            ols_group_node_indices=np.array([0, 1, 2, 3]),
        )
        regions = dict(all=np.ones_like(support), inside_support=support,
                       outside_support=~support, far_gt_1800s=nearest > 1800.)
        fragment.update({'region_' + key: value for key, value in regions.items()})
        scene = evaluate_full_scene([
            dict(node_indices=group, states=baseline[group[0]],
                 prediction_valid=np.isfinite(baseline[group[0]]))
            for group in groups], node_identities=identities, target_identities=target_ids,
            target_states=truth, target_valid=truth_valid, node_observed=support,
            split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=support, nearest_seconds=nearest))
        saved = scene.pop('artifact')
        expected = dict(scene_id='fixture', component_id='fixture',
                        unique_target={BASE: scene}, group_count=dict(ols=len(groups)))
        if not invalid_first and not disjoint_masks:
            expected['fragment'] = {BASE: {
                region: metrics(baseline, fragment['target'], fragment['target_valid'], mask)
                for region, mask in regions.items()}}
        return fragment, saved, expected

    @staticmethod
    def metric(report, method, attr='position'):
        return report['methods'][method]['system']['full_target_metrics']['all'][attr]

    @classmethod
    def cached_source(cls, root):
        """Real two-scene 053 cache with different present sensor vocabularies."""
        source = root / 'original053'
        (source / 'predictions').mkdir(parents=True)
        (source / 'system-predictions').mkdir()
        scenes = []
        for sid in ('1', '2'):
            fragment, saved, expected = cls.fixture()
            if sid == '2':
                # A scene may contain none of the sensors seen in another scene.
                # Integer identifiers must join the same string-normalized strata.
                fragment['sources'] = np.full(len(fragment['sources']), 3)
            expected.update(scene_id=sid, component_id=sid)
            scenes.append(expected)
            np.savez_compressed(source / 'predictions' / f'states-{sid}.npz', **fragment)
            np.savez_compressed(source / 'system-predictions' / f'states-{sid}.npz',
                common_prediction_valid_on_target_rows=saved['aligned_prediction_valid'][:4],
                **{BASE + '_' + key: value for key, value in saved.items()})
        summary = dict(status='COMPLETED', dataset_level='Level1', split='V_select',
            scene_ids=['1', '2'], scene_count=2, fragment_count=8, method_names=[BASE],
            test_opened=False, optimizer_updates=0, per_scene=scenes,
            ols_working_point=dict(threshold=-1.95, selection_split='V_select',
                motion_observation_policy='canonical_all', validation_scene_ids=['1', '2']),
            fragment=_pool_fragment(scenes, [BASE]),
            unique_target={BASE: aggregate_fusion_reports([
                scene['unique_target'][BASE] for scene in scenes])})
        write_json(source / 'summary.json', summary)
        write_json(source / 'run.json', dict(status='COMPLETED', test_opened=False,
            optimizer_updates=0, completed_scenes=2, total_scenes=2,
            motion_observation_policy='canonical_all', threshold_reselected=False,
            normalizer_refitted=False))
        manifest = root / 'manifest.json'
        write_json(manifest, dict(dataset_level='Level1', split_unit='scene',
            splits=dict(T_core=['unused-training'], V_select=['1', '2'], T_test=['sealed'])))
        args = SimpleNamespace(source_system=source, manifest=manifest,
                               output=root / 'attribution054', cpu_threads=1)
        return args, summary

    def test_complete_registry_replay_and_duplicates_are_retained(self):
        fragment, saved, expected = self.fixture()
        report, arrays = analyze_scene(fragment, saved, expected)
        for method, covered, missing in ((BASE, 1, 2), (ANCHOR, 1, 2),
                                         (INDEPENDENT, 2, 1)):
            system = report['methods'][method]['system']
            self.assertEqual(system['target_identity_count'], 4)
            self.assertEqual(system['reference_unavailable_target_count'], 1)
            self.assertEqual(system['duplicate_prediction_group_count'], 1)
            metric = self.metric(report, method)
            self.assertEqual(metric['available_target_trajectory_count'], 3)
            self.assertEqual(metric['covered_target_trajectory_count'], covered)
            self.assertEqual(metric['missing_target_trajectory_count'], missing)
            self.assertEqual(metric['available_target_count'], 9)
            np.testing.assert_array_equal(arrays[method + '_target_ids'], saved['target_ids'])
        self.assertEqual(self.metric(report, BASE)['available_prediction_count'], 9)
        self.assertEqual(self.metric(report, BASE)['prediction_without_target_count'], 6)
        self.assertEqual(self.metric(report, INDEPENDENT)['prediction_without_target_count'], 6)

    def test_mutually_exclusive_causes_reconcile_with_original_missing_count(self):
        fragment, saved, expected = self.fixture()
        report, _ = analyze_scene(fragment, saved, expected)
        attr = report['attribution']['position']
        self.assertEqual(sum(attr['primary_counts'].values()), 4)
        self.assertEqual(attr['primary_counts']['covered'], 1)
        self.assertEqual(attr['primary_counts']['reference_unavailable'], 1)
        self.assertEqual(attr['primary_counts']['wrong_merge_no_representative'], 1)
        self.assertEqual(attr['primary_counts']['no_input_nodes'], 1)
        self.assertEqual(attr['missing_target_trajectory_count'], 2)
        rows = {row['target_id']: row for row in attr['target_rows']}
        self.assertEqual(rows['b']['input_node_indices'], [1])
        self.assertEqual(rows['b']['foreign_anchor_group_indices'], [0])
        self.assertEqual(rows['b']['mixed_group_indices'], [0])
        self.assertIsNone(rows['b']['selected_group'])
        self.assertEqual(rows['c']['input_node_indices'], [])
        self.assertEqual(rows['a']['anchor_group_indices'], [0, 1])
        self.assertEqual(rows['a']['mixed_group_indices'], [0])
        self.assertEqual(rows['a']['primary_reason'], 'covered')

    def test_invalid_first_representative_is_not_rescued_by_valid_duplicate(self):
        fragment, saved, expected = self.fixture(invalid_first=True)
        report, arrays = analyze_scene(fragment, saved)
        attr = report['attribution']['position']
        rows = {row['target_id']: row for row in attr['target_rows']}
        self.assertEqual(rows['a']['primary_reason'], 'selected_prediction_unavailable')
        self.assertEqual(rows['a']['anchor_group_indices'], [0, 1])
        self.assertEqual(rows['b']['primary_reason'], 'wrong_merge_no_representative')
        self.assertEqual(attr['missing_target_trajectory_count'], 3)
        self.assertEqual(attr['primary_counts']['wrong_merge_no_representative'], 1)
        self.assertFalse(arrays[BASE + '_aligned_prediction_valid'][0].any())
        self.assertTrue(arrays[BASE + '_group_valid'][1].all())
        self.assertEqual(self.metric(report, BASE)['covered_target_trajectory_count'], 0)

    def test_time_intersection_and_attributes_are_not_conflated(self):
        fragment, saved, expected = self.fixture(disjoint_masks=True)
        report, _ = analyze_scene(fragment, saved)
        for attr in ('position', 'speed'):
            rows = {row['target_id']: row for row in report['attribution'][attr]['target_rows']}
            self.assertEqual(rows['a']['primary_reason'], 'no_valid_time_intersection')
            self.assertEqual(self.metric(report, BASE, attr)['intersection_count'], 0)
            key = '__vs__'.join(sorted((BASE, INDEPENDENT)))
            for method in (BASE, INDEPENDENT):
                paired = report['pairwise'][key]['methods'][method]['all'][attr]
                self.assertEqual(paired['intersection_count'], 0)
                self.assertIsNone(paired['mae'])
        heading_rows = {row['target_id']: row
                        for row in report['attribution']['heading']['target_rows']}
        self.assertEqual(heading_rows['a']['primary_reason'], 'covered')
        self.assertEqual(self.metric(report, BASE, 'heading')['intersection_count'], 3)

    def test_anchor_control_keeps_group_order_without_reference_selection(self):
        fragment, saved, expected = self.fixture()
        report, arrays = analyze_scene(fragment, saved, expected)
        self.assertEqual(report['methods'][ANCHOR]['system']['selected_group_by_target'],
                         [0, None, None, 2])
        np.testing.assert_array_equal(arrays[ANCHOR + '_group_states'][0],
                                      fragment['native_states'][0])
        # The duplicate a fragment is exact, but it never replaces the first anchor.
        self.assertGreater(self.metric(report, ANCHOR)['mae'], 200.)
        self.assertEqual(self.metric(report, ANCHOR, 'heading')['mae'], 2.)
        self.assertEqual(self.metric(report, ANCHOR, 'speed')['mae'], 4.)
        self.assertGreater(self.metric(report, BASE)['mae'], self.metric(report, ANCHOR)['mae'])

    def test_fallback_fragment_view_is_separated_from_actual_group_output(self):
        fragment, saved, expected = self.fixture()
        report, arrays = analyze_scene(fragment, saved, expected)
        # No group member supports t=2000. The legacy mean-supported cache
        # falls back separately to each node, while the system emits only A.
        self.assertFalse(fragment['support'][:2, -1].any())
        np.testing.assert_array_equal(fragment[BASE][1, -1], fragment['native_states'][1, -1])
        np.testing.assert_array_equal(arrays[BASE + '_group_states'][0, -1],
                                      fragment['native_states'][0, -1])
        base = report['methods'][BASE]
        self.assertGreater(base['emitted_fragment']['all']['position']['mae'],
                           base['fragment']['all']['position']['mae'])
        for method in (ANCHOR, INDEPENDENT):
            row = report['methods'][method]
            self.assertEqual(row['fragment'], row['emitted_fragment'])

    def test_support_strata_do_not_change_population_when_grouping_changes(self):
        fragment, saved, expected = self.fixture()
        report, _ = analyze_scene(fragment, saved, expected)
        methods = report['methods']
        for region, metrics in methods[BASE]['fixed_target_strata'].items():
            for method in (ANCHOR, INDEPENDENT):
                for attr in ('position', 'speed', 'heading'):
                    with self.subTest(region=region, method=method, attr=attr):
                        self.assertEqual(metrics[attr]['available_target_count'],
                            methods[method]['fixed_target_strata'][region][attr]['available_target_count'])
        # C's three reference points are retained as no-input strata in every arm.
        for method in (BASE, ANCHOR, INDEPENDENT):
            self.assertEqual(methods[method]['fixed_target_strata']['no_input']
                             ['position']['available_target_count'], 3)

    def test_pairwise_common_points_preserve_original_denominators(self):
        fragment, saved, expected = self.fixture()
        report, arrays = analyze_scene(fragment, saved, expected)
        self.assertEqual(len(report['pairwise']), 3)
        key = '__vs__'.join(sorted((BASE, INDEPENDENT)))
        pair = report['pairwise'][key]['methods']
        for method in (BASE, INDEPENDENT):
            metric = pair[method]['all']['position']
            self.assertEqual(metric['intersection_count'], 3)
            self.assertEqual(metric['available_target_count'], 9)
            self.assertEqual(metric['coverage'], 1/3)
        # Independent output also covers b, but that cannot enter a paired MAE.
        self.assertEqual(self.metric(report, INDEPENDENT)['intersection_count'], 6)
        self.assertEqual(self.metric(report, BASE)['intersection_count'], 3)
        for pair_name, item in report['pairwise'].items():
            first, second = pair_name.split('__vs__')
            for attr in ('position', 'speed', 'heading'):
                with self.subTest(pair=pair_name, attr=attr):
                    change = item['error_change_second_minus_first']['all'][attr]
                    delta = arrays['delta_' + pair_name + '_' + attr]
                    finite = delta[np.isfinite(delta)]
                    self.assertEqual(change['common_point_count'], len(finite))
                    self.assertEqual(change['second_worse_point_count'], int((finite > 1e-9).sum()))
                    self.assertEqual(change['second_better_point_count'], int((finite < -1e-9).sum()))
                    self.assertEqual(change['tied_point_count'], int((np.abs(finite) <= 1e-9).sum()))
                    self.assertAlmostEqual(change['harm_sum'], finite[finite > 1e-9].sum(), delta=1e-9)
                    self.assertAlmostEqual(change['benefit_sum'], -finite[finite < -1e-9].sum(), delta=1e-9)
                    a = item['methods'][first]['all'][attr]
                    b = item['methods'][second]['all'][attr]
                    self.assertAlmostEqual(change['signed_error_change_sum'], finite.sum(), delta=1e-9)
                    self.assertAlmostEqual(finite.sum(), (b['mae'] - a['mae']) * len(finite), delta=1e-9)
        # Fusion harms position on two points but improves speed on those same
        # points. Attribute-specific changes cannot be inferred from a single MAE.
        fusion_pair = report['pairwise']['__vs__'.join(sorted((ANCHOR, BASE)))]
        changes = fusion_pair['error_change_second_minus_first']['all']
        self.assertEqual(changes['position']['second_worse_point_count'], 2)
        self.assertEqual(changes['position']['tied_point_count'], 1)
        self.assertEqual(changes['speed']['second_better_point_count'], 2)

    def test_changed_prediction_truth_mask_or_summary_cannot_silently_replay(self):
        fragment, saved, expected = self.fixture()
        for key in ('aligned_prediction', 'aligned_truth', 'aligned_truth_valid'):
            changed = deepcopy(saved)
            if key.endswith('_valid'):
                changed[key][0, 0, 0] = False
            else:
                changed[key][0, 0, 0] += .001
            with self.subTest(array=key), self.assertRaises(ValueError):
                analyze_scene(fragment, changed, expected)
        bad = deepcopy(expected)
        bad['unique_target'][BASE]['full_target_metrics']['all']['position']['mae'] += 1.
        with self.assertRaises(ValueError):
            analyze_scene(fragment, saved, bad)

    def test_pooling_distinguishes_point_weighted_and_scene_macro_errors(self):
        records = []
        for count, speed_error in ((1, 10.), (9, 2.)):
            truth = np.tile([30., 120., 5., 1.], (1, 9, 1))
            pred = truth.copy()
            pred[..., 2] += speed_error
            valid = np.ones_like(truth, bool)
            valid[:, count:] = False
            actual = evaluate_aligned_fusion(pred, truth, np.ones_like(truth, bool),
                prediction_valid=valid, region_mask=np.ones((1, 9), bool))
            records.append(dict(all=actual))
        pooled = pool_metrics(records)['all']['speed']
        self.assertAlmostEqual(pooled['mae'], 2.8)
        self.assertAlmostEqual(pooled['scene_macro_mae'], 6.)
        self.assertAlmostEqual(pooled['rmse'], np.sqrt(13.6))
        self.assertAlmostEqual(pooled['coverage'], 10/18)
        self.assertEqual(pooled['intersection_count'], 10)

    def test_cached_cli_replays_two_scenes_and_preserves_all_source_files(self):
        with TemporaryDirectory() as directory:
            args, original = self.cached_source(Path(directory))
            before = {path.relative_to(args.source_system): path.read_bytes()
                      for path in args.source_system.rglob('*') if path.is_file()}
            summary = run_attribution(args)
            self.assertEqual(summary['status'], 'COMPLETED')
            self.assertEqual(summary['scene_ids'], ['1', '2'])
            self.assertEqual(summary['fragment_count'], 8)
            self.assertTrue(summary['all_source_scenes_replayed'])
            self.assertEqual(summary['ols_working_point'], original['ols_working_point'])
            self.assertEqual(summary['attribution']['position']['missing_target_trajectory_count'], 4)
            self.assertEqual(summary['attribution']['position']['primary_counts']
                             ['wrong_merge_no_representative'], 2)
            base = summary['methods'][BASE]
            self.assertEqual(base['population']['duplicate_prediction_group_count'], 2)
            self.assertEqual(base['system']['full_target_metrics']['all']['position']
                             ['missing_prediction_count'], 12)
            self.assertEqual(base['fragment']['source_1']['position']['available_target_count'], 3)
            self.assertEqual(base['fragment']['source_2']['position']['available_target_count'], 6)
            self.assertEqual(base['fragment']['source_3']['position']['available_target_count'], 9)
            state = read_json(args.output / 'run.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertEqual(state['optimizer_updates'], 0)
            self.assertEqual(state['archive_members_read'], 0)
            self.assertEqual(state['reference_members_read'], 0)
            self.assertTrue(state['cached_reference_used'])
            self.assertFalse(state['gpu_used'])
            self.assertFalse(state['threshold_reselected'])
            self.assertFalse(state['test_opened'])
            self.assertEqual(read_json(args.output / 'summary.json')['scene_count'], 2)
            for sid in ('1', '2'):
                scene = read_attribution(args.output / 'scenes' / f'scene-{sid}.json.gz')
                self.assertEqual(len(scene['attribution']['position']['target_rows']), 4)
                self.assertTrue(scene['replay_verified'])
                with np.load(args.output / 'predictions' / f'states-{sid}.npz',
                             allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved[BASE + '_target_ids'], ['a', 'b', 'c', 'd'])
                    self.assertEqual(saved[BASE + '_group_states'].shape, (3, 3, 4))
                    key = 'common_' + '__vs__'.join(sorted((BASE, INDEPENDENT))) + '_valid'
                    self.assertEqual(saved[key].shape, (4, 3, 4))
            after = {path.relative_to(args.source_system): path.read_bytes()
                     for path in args.source_system.rglob('*') if path.is_file()}
            self.assertEqual(before, after)

    def test_cached_cli_rejects_missing_extra_or_misreported_source_population(self):
        for issue in ('missing_scene', 'extra_scene', 'aggregate_metric', 'input_policy'):
            with self.subTest(issue=issue), TemporaryDirectory() as directory:
                args, summary = self.cached_source(Path(directory))
                source_file = args.source_system / 'predictions' / 'states-2.npz'
                if issue == 'missing_scene':
                    source_file.unlink()
                elif issue == 'extra_scene':
                    (source_file.parent / 'states-3.npz').write_bytes(source_file.read_bytes())
                elif issue == 'aggregate_metric':
                    summary['unique_target'][BASE]['all']['position']['mae'] += 1.
                    write_json(args.source_system / 'summary.json', summary)
                else:
                    source_run = read_json(args.source_system / 'run.json')
                    source_run['motion_observation_policy'] = 'native128'
                    write_json(args.source_system / 'run.json', source_run)
                with self.assertRaises(ValueError):
                    run_attribution(args)
                if issue == 'aggregate_metric':
                    state = read_json(args.output / 'run.json')
                    self.assertEqual(state['status'], 'FAILED')
                    self.assertIn('source aggregate full target', state['error'])
                    self.assertFalse((args.output / 'summary.json').exists())
                else:
                    self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
