"""Cached-feature denominator/reuse checks; no model, real data or GPU."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.analysis import generated_support_score as diagnostic


def metric(count, evaluable, square, dimension):
    return dict(query_count=count, evaluable_query_count=evaluable,
        missing_reference_query_count=count-evaluable, feature_elements=evaluable*dimension,
        squared_error_sum=square, feature_mse=square/(evaluable*dimension) if evaluable else None,
        reference_coverage=evaluable/count if count else None)


def row(sid, node, *, length=4, dimension=2, empty_reference=False):
    values = {}
    for method in diagnostic.METHODS:
        inside = metric(1, 0 if empty_reference else 1, 0 if empty_reference else 4., dimension)
        before = metric(1, 0, 0., dimension)
        after = metric(length-2, 0 if empty_reference else 1,
                       0. if empty_reference else 8. if method == 'model' else 16., dimension)
        regions = dict(inside_support=inside, before_support=before, after_support=after)
        regions['all'] = diagnostic.pool(list(regions.values()))
        values[method] = dict(regions=regions)
    return dict(scene_id=sid, node_index=node, query=values)


def report(rows, ids):
    return dict(scene_count=len(ids), fragment_count=len(rows), test_opened=False,
        per_scene=[dict(scene_id=sid, fragment_count=sum(r['scene_id'] == sid for r in rows)) for sid in ids],
        per_fragment=rows, query={method: dict(regions={region: diagnostic.pool(
            [r['query'][method]['regions'][region] for r in rows]) for region in diagnostic.REGIONS})
            for method in diagnostic.METHODS}, target_policy='all four reference attributes valid',
        interpretation='frozen feature MSE, not metres')


class GeneratedSupportReportTests(unittest.TestCase):
    def test_missing_reference_and_empty_scene_remain_in_full_denominators(self):
        data = report([row('1', 0), row('1', 1, empty_reference=True), row('3', 0)], ['1', '2', '3'])
        result = diagnostic.summarize(data, ['1', '2'], ['1', '2', '3'], dimension=2, length=4)
        self.assertEqual(result['scene_count'], 2)
        self.assertEqual(result['fragment_count'], 2)
        self.assertEqual(result['declared_query_count'], 8)
        self.assertEqual(result['provided_query_count'], 2)
        self.assertEqual(result['generated_query_count'], 6)
        self.assertEqual(result['query_feature_metrics']['model']['all']['missing_reference_query_count'], 6)
        self.assertEqual(result['query_feature_metrics']['model']['generated']['feature_mse'], 4.)
        self.assertEqual(result['model_minus_hold_feature_mse']['generated'], -4.)
        self.assertEqual(result['per_scene'][1]['fragment_count'], 0)
        self.assertIsNone(result['per_scene'][1]['model_minus_hold_feature_mse']['generated'])
        self.assertIn('existing_source_output_only', result['full_V_source_aggregate']['status'])

    def test_corrupt_counts_population_or_metric_cannot_be_silently_dropped(self):
        baseline = report([row('1', 0)], ['1'])
        for defect in ('count', 'mse', 'duplicate', 'missing_scene'):
            data = copy.deepcopy(baseline)
            if defect == 'count':
                data['per_fragment'][0]['query']['model']['regions']['inside_support']['evaluable_query_count'] = 0
            elif defect == 'mse':
                data['per_fragment'][0]['query']['model']['regions']['all']['feature_mse'] = 999.
            elif defect == 'duplicate':
                data['per_fragment'].append(copy.deepcopy(data['per_fragment'][0]))
                data['fragment_count'] += 1
            else:
                data['per_scene'][0]['fragment_count'] = 2
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                diagnostic.summarize(data, ['1'], ['1'], dimension=2, length=4)

    def test_selection_is_fixed_before_results_and_independent_of_manifest_order(self):
        manifest = dict(dataset_level='Level1', split_unit='scene', splits=dict(
            T_core=[str(i) for i in range(1000, 3100)], V_select=[str(i) for i in range(1, 301)]))
        chosen = diagnostic.select_scenes(manifest)
        self.assertEqual(len(chosen), 16)
        manifest['splits']['V_select'].reverse()
        self.assertEqual(diagnostic.select_scenes(manifest), chosen)

    def test_cpu_cli_reuses_report_once_and_protects_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'tcm'; source.mkdir()
            ids = [str(i) for i in range(1, 301)]
            manifest = dict(dataset_level='Level1', split_unit='scene', splits=dict(
                T_core=[str(i) for i in range(1000, 3100)], V_select=ids))
            saved = report([row(sid, 0, length=64, dimension=128) for sid in ids], ids)
            run = dict(status='COMPLETED', phase='tcm', final_evaluation_status='COMPLETED', test_opened=False,
                cursor=dict(completed_updates=2424), config=dict(updates=dict(TAM_FROZEN=0, JOINT=0)),
                requested_model_config=dict(d_model=128, length=64))
            summary = dict(phase='tcm', normalizer=dict(input_adapter='native_time_v1',
                adapter_config=dict(length=64, max_observations=128, time_unit_seconds=20., speed_unit_policy=diagnostic.POLICY),
                physical_units=dict(policy=diagnostic.POLICY)))
            for path, data in ((root/'manifest.json', manifest), (source/'run.json', run),
                               (source/'summary.json', summary), (source/'completion-evaluation.json', saved)):
                path.write_text(json.dumps(data))
            argv = ['--tcm-run', str(source), '--manifest', str(root/'manifest.json'), '--output', str(root/'out')]
            self.assertEqual(diagnostic.main(argv), 0)
            result = json.loads((root/'out/summary.json').read_text())
            state = json.loads((root/'out/run.json').read_text())
            self.assertEqual(result['declared_query_count'], 16*64)
            self.assertEqual(state['source_report_reads'], 1)
            self.assertEqual(state['gpu_forwards'], 0)
            self.assertFalse(state['checkpoint_read'])
            before = (root/'out/run.json').read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(diagnostic.main(argv), 1)
            self.assertEqual((root/'out/run.json').read_bytes(), before)
