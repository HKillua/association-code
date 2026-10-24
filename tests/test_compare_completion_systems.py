"""096 cache comparison keeps missing targets, duplicates and exact point tails."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from association.evaluation.artifacts import ScenePredictionWriter
from association.evaluation.association import evaluate_scenes
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.utils import runtime
from tools.analysis import compare_completion_systems as comparison


class CompletionSystemComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, policy, *, counts=(3, 3), covered=None, separation=False, scene_change=False):
        directory = self.root / policy
        directory.mkdir()
        writer = ScenePredictionWriter(directory / 'predictions')
        records, reports = [], []
        methods = {key: [] for key in comparison.METHODS}
        threshold = .9 if separation else .5
        for index, q in enumerate(counts):
            sid = f's{index}'
            identities = np.array(['A', 'A', 'B'])
            truth = np.broadcast_to([30., 120., 5., 40.], (3, q, 4)).copy()
            observed = np.ones((3, q), bool)
            if q > 1:
                observed[:, 1] = False
            pairs = np.column_stack(np.triu_indices(3, 1))
            scene = dict(scene_id=sid, component_id=f'component{index}', split='V_select',
                input_adapter='native_time_v1', node_ids=np.array(['n0', 'n1', 'n2']),
                identities=identities, sources=np.array(['a', 'b', 'a']), grid=np.arange(q, dtype=float),
                pairs=pairs, labels=np.array([True, False, False]), intervals=np.tile([0., 2.], (3, 1)),
                observed=observed, values=truth.copy(), native_values=truth[:, :1].copy(),
                native_times=np.zeros((3, 1)), native_observed=np.ones((3, 1), bool))
            if scene_change and index == 0:
                scene['native_times'][0, 0] = 1.
            record = {key: scene[key] for key in ('scene_id', 'component_id', 'node_ids', 'identities',
                      'sources', 'intervals', 'labels', 'pairs')}
            record['scores'] = np.array([.8, .1, .1])
            records.append(record)
            predicted = [[0], [1], [2]] if separation else [[0, 1], [2]]
            artifacts, row = {}, dict(scene_id=sid, component_id=scene['component_id'], methods={})
            for method in comparison.METHODS:
                groups = [[0, 1], [2]] if method == 'oracle' else predicted
                outputs = []
                for group_index, group in enumerate(groups):
                    target = 0 if group[0] < 2 else 1
                    states = truth[target].copy()
                    # Unequal scene sizes plus nonconstant errors make an
                    # average-of-scene-quantiles implementation observably wrong.
                    lat_error = (index + 1) * (np.arange(q) + 1) * .001
                    if method == 'oracle':
                        lat_error = lat_error * .5
                    elif method in comparison.CONTROLS:
                        lat_error = lat_error * 2.
                    elif policy == 'state_hold':
                        lat_error = lat_error * 1.5
                    states[:, 0] += lat_error
                    mask = np.ones((q, 4), bool)
                    if method in comparison.CONTROLS:
                        mask = np.repeat(observed[group].any(0)[:, None], 4, axis=1)
                    elif covered is not None:
                        mask[:] = False
                        mask[np.asarray(covered), :] = True
                    outputs.append(dict(node_indices=group, states=states, prediction_valid=mask))
                report = evaluate_full_scene(outputs, node_identities=identities,
                    target_identities=['A', 'B', 'C'], target_states=truth, target_valid=np.ones_like(truth, bool),
                    node_observed=observed, split='V_select',
                    grouping_kind='oracle' if method == 'oracle' else 'predicted',
                    correspondence_policy='anchor_identity_first', return_artifact=True)
                artifacts[method] = report.pop('artifact')
                row['methods'][method] = report
                methods[method].append(report)
            writer(index, scene, artifacts)
            reports.append(row)
        binding = dict(completion_policy=policy, threshold=threshold,
            source_runs=dict(tcm='/original/088', tam='/original/094' + policy, tfm='/original/093' + policy),
            normalizer_equal=True, frozen_completion_equal=True, full_data_contract_equal=True,
            source_tam_update=4000, source_tfm_update=1000,
            validation_scene_ids=[row['scene_id'] for row in reports],
            validation_component_ids=[row['component_id'] for row in reports],
            normalizer=dict(input_adapter='native_time_v1', training_splits=['T_core']))
        evaluation = dict(oracle_only=False, selection=None, threshold=threshold, test_opened=False,
            records=records, association=evaluate_scenes(records, threshold), per_scene=reports,
            fusion={method: aggregate_fusion_reports(value) for method, value in methods.items()})
        runtime.write_json(directory / 'source-binding.json', binding)
        runtime.write_json(directory / 'run.json', dict(status='COMPLETED', mode=comparison.MODE,
            completion_policy=policy, optimizer_updates=0, threshold_reselected=False, test_opened=False,
            source_tam_run=binding['source_runs']['tam'], source_tfm_run=binding['source_runs']['tfm'],
            source_binding=binding, validation_scene_ids=[row['scene_id'] for row in reports]))
        runtime.write_json(directory / 'summary.json', dict(source_binding=binding, evaluation=evaluation))
        return directory

    def test_real_arrays_keep_duplicates_and_compute_exact_global_tails_without_weights(self):
        learned = self.write('learned', counts=(2, 5))
        hold = self.write('state_hold', counts=(2, 5), separation=True)
        with patch('torch.load', side_effect=AssertionError('comparison must never read weights')):
            result = comparison.compare(learned, hold, self.root / 'out')
        own = result['all']['full']['learned']['predicted']['all']['position']
        common = result['all']['common']['learned']['all']['position']
        self.assertEqual(result['scene_count'], 2)
        self.assertEqual(result['component_count'], 2)
        self.assertEqual(own['available_target_count'], 21)
        self.assertEqual(own['intersection_count'], 14)
        self.assertEqual(own['missing_prediction_count'], 7)
        self.assertEqual(common['coverage'], 2 / 3)
        self.assertEqual(result['all']['grouping']['state_hold']['duplicate_anchor_group_count'], 2)
        self.assertEqual(result['all']['full']['state_hold']['predicted']['all']['position']['prediction_without_target_count'], 7)
        self.assertEqual(result['all']['common']['state_hold']['all']['position']['prediction_without_target_count'], 0)
        errors = np.concatenate([np.tile((np.arange(q) + 1) * (i + 1) * .001, 2)
                                 for i, q in enumerate((2, 5))]) * 6371000 * np.pi / 180
        self.assertAlmostEqual(own['p95'], np.quantile(errors, .95), places=6)
        self.assertAlmostEqual(own['p99'], np.quantile(errors, .99), places=6)
        self.assertEqual(result['scene_wins']['learned_better'], 2)
        self.assertEqual(result['all']['controls']['learned']['predicted_mean']['predicted']['all']['position']['intersection_count'], 10)
        self.assertEqual(result['largest_component']['component_id'], 'component0')
        self.assertEqual(result['outside_largest_component']['scene_count'], 1)
        self.assertEqual(result['checkpoint_loads'], 0)
        for filename in ('summary.json.gz', 'scene-summary.json.gz', 'run.json', 'report.md'):
            self.assertTrue((self.root / 'out' / filename).is_file())
        with self.assertRaises(FileExistsError):
            comparison.compare(learned, hold, self.root / 'out')

    def test_disjoint_coverage_retains_population_and_null_common_error(self):
        learned = self.write('learned', covered=[0])
        hold = self.write('state_hold', covered=[2])
        result = comparison.compare(learned, hold, self.root / 'out')
        common = result['all']['common']['learned']['all']['position']
        self.assertEqual(common['available_target_count'], 18)
        self.assertEqual(common['intersection_count'], 0)
        self.assertIsNone(common['mae'])
        self.assertIsNone(common['p99'])
        self.assertEqual(result['scene_wins']['evaluable_units'], 0)
        self.assertEqual(result['all']['full']['learned']['predicted']['all']['position']['intersection_count'], 4)

    def test_real_input_change_rejected_and_failed_comparison_preserved(self):
        learned = self.write('learned')
        hold = self.write('state_hold', scene_change=True)
        with self.assertRaisesRegex(ValueError, 'scene/input population differs'):
            comparison.compare(learned, hold, self.root / 'out')
        self.assertEqual(runtime.read_json(self.root / 'out/run.json')['status'], 'FAILED')
        self.assertFalse((self.root / 'out/summary.json.gz').exists())

    def test_active_wrong_policy_or_reselected_threshold_rejected(self):
        learned = self.write('learned')
        hold = self.write('state_hold')
        source = runtime.read_json(hold / 'run.json')
        variants = [dict(status='RUNNING'), dict(completion_policy='learned'), dict(threshold_reselected=True)]
        for i, changes in enumerate(variants):
            runtime.write_json(hold / 'run.json', source | changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                comparison.compare(learned, hold, self.root / f'out{i}')

    def test_control_partition_change_and_false_original_metric_rejected(self):
        learned = self.write('learned')
        hold = self.write('state_hold')
        summary = runtime.read_json(hold / 'summary.json')
        summary['evaluation']['per_scene'][0]['methods']['predicted']['full_target_metrics']['all']['position']['mae'] += 1
        runtime.write_json(hold / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'cached replay mismatch'):
            comparison.compare(learned, hold, self.root / 'wrong_metric')
        summary['evaluation']['per_scene'][0]['methods']['predicted']['full_target_metrics']['all']['position']['mae'] -= 1
        runtime.write_json(hold / 'summary.json', summary)
        metadata = runtime.read_json(hold / 'predictions/000000.json')
        prefix = metadata['methods']['predicted_mean']['prefix']
        path = hold / 'predictions/000000.npz'
        with np.load(path, allow_pickle=False) as original:
            arrays = {key: original[key] for key in original.files}
        arrays[prefix + 'node_group_indices'][0] = 1
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'membership arrays disagree'):
            comparison.compare(learned, hold, self.root / 'wrong_groups')

    def test_source_binding_and_original_threshold_cannot_drift(self):
        learned = self.write('learned')
        hold = self.write('state_hold')
        binding = runtime.read_json(hold / 'source-binding.json')
        binding['threshold'] = .1
        runtime.write_json(hold / 'source-binding.json', binding)
        with self.assertRaisesRegex(ValueError, 'source binding differs'):
            comparison.compare(learned, hold, self.root / 'out')

    def test_matching_truncated_outputs_do_not_replace_the_original_source_population(self):
        learned, hold = self.write('learned'), self.write('state_hold')
        for directory in (learned, hold):
            index = runtime.read_json(directory / 'predictions/index.json')
            index['scenes'] = index['scenes'][:1]
            runtime.write_json(directory / 'predictions/index.json', index)
            run = runtime.read_json(directory / 'run.json')
            run['validation_scene_ids'] = run['validation_scene_ids'][:1]
            runtime.write_json(directory / 'run.json', run)
            summary = runtime.read_json(directory / 'summary.json')
            summary['evaluation']['per_scene'] = summary['evaluation']['per_scene'][:1]
            runtime.write_json(directory / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'original-source full prediction population differ'):
            comparison.compare(learned, hold, self.root / 'out')


if __name__ == '__main__':
    unittest.main()
