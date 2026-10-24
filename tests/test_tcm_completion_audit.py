"""095 real complete JSON aggregation, independent arithmetic and source refusal."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.analysis import tcm_completion_audit as audit


class TcmCompletionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.manifest = dict(dataset_level='Level1', splits=dict(T_core=['t'], V_select=['a', 'b', 'c'],
                                                               T_head=[], T_test=['sealed']),
                             components=dict(t='T', a='large', b='large', c='small', sealed='test'))
        audit.write(self.root / 'manifest.json', self.manifest)
        self.expected = dict(manifest=self.manifest, training=dict(updates=dict(TCM=1)),
                             model=dict(d_model=2, length=4), train_fragments=2,
                             validation_fragments=3, dimension=2, queries=4)
        coverage = {key: dict(track_presentations=0, pair_presentations=0) for key in
                    ('TCM', 'TAM_FROZEN', 'TFM_FROZEN', 'JOINT')}
        coverage['TCM'].update(track_presentations=2, unique_track_count=2, completed_epochs=1, scene_ids=['t'])
        audit.write(self.source / 'run.json', dict(status='COMPLETED', phase='tcm', final_evaluation_status='COMPLETED',
            training_stages=['TCM'], test_opened=False, config=self.expected['training'], selection_performed=False,
            cursor=dict(stage=1, update=0, completed_updates=1),
            checkpoint_cursor=dict(stage=0, update=1, completed_updates=1), requested_model_config=self.expected['model']))
        audit.write(self.source / 'summary.json', dict(phase='tcm', test_opened=False, selection_performed=False,
            completion_evaluation='completion-evaluation.json', coverage=coverage,
            normalizer=dict(training_splits=['T_core'], training_scene_ids=['t'])))
        audit.write(self.source / 'coverage.json', coverage)
        audit.write(self.source / 'history.json', [dict(stage='TCM', update=1, track_presentations=2)])
        # Invalid checkpoint content proves this audit does not need a loadable model.
        (self.source / 'checkpoint.pt').write_text('Never opened by095')
        fragments = [self.fragment('a', 'x', 0, 2, [3, 1, 5, 9], [4, 1, 5, 4], [True, True, False, True]),
                     self.fragment('a', 'y', 1, 3, [2, 3, 3, 4], [2, 3, 3, 4], [False] * 4),
                     self.fragment('c', 'z', 0, 5, [1, 2, 2, 5], [2, 2, 2, 3], [True] * 4)]
        self.complete = dict(scene_count=3, fragment_count=3, test_opened=False,
            no_complete_reference_fragment_count=1, per_fragment=fragments,
            per_scene=[dict(scene_id='a', fragment_count=2, no_complete_reference_fragment_count=1,
                            reference_targets_without_input=[]),
                       dict(scene_id='b', fragment_count=0, no_complete_reference_fragment_count=0,
                            reference_targets_without_input=['missing-input-target']),
                       dict(scene_id='c', fragment_count=1, no_complete_reference_fragment_count=0,
                            reference_targets_without_input=[])], one_step={}, query={})
        for method in audit.METHODS:
            self.complete['one_step'][method] = {direction: self.combine([row['one_step'][method][direction]
                for row in fragments]) for direction in ('all', 'forward', 'reverse')}
            self.complete['query'][method] = {family: {key: self.combine([row['query'][method][family][key]
                for row in fragments]) for key in keys} for family, keys in
                (('regions', audit.REGIONS), ('input_distance_metrics', audit.DISTANCES))}
        audit.write(self.source / 'completion-evaluation.json', self.complete)

    @staticmethod
    def measure(values, selected, valid):
        indices = [i for i in selected if valid[i]]
        n = len(indices)
        return dict(query_count=len(selected), evaluable_query_count=n,
            missing_reference_query_count=len(selected) - n, feature_elements=2 * n,
            squared_error_sum=2 * sum(values[i] for i in indices),
            feature_mse=sum(values[i] for i in indices) / n if n else None,
            reference_coverage=n / len(selected) if selected else None)

    @staticmethod
    def combine(rows):
        # Independent exact fixture arithmetic, not production pool().
        counts = {key: sum(row[key] for row in rows) for key in audit.COUNTS}
        sse = sum(row['squared_error_sum'] for row in rows)
        return dict(**counts, squared_error_sum=sse,
            feature_mse=sse / counts['feature_elements'] if counts['feature_elements'] else None,
            reference_coverage=counts['evaluable_query_count'] / counts['query_count'] if counts['query_count'] else None)

    def fragment(self, sid, node, index, native_count, model, hold, valid):
        row = dict(scene_id=sid, node_id=node, node_index=index, native_count=native_count,
                   has_complete_reference=any(valid), reference_declared=any(valid), one_step={}, query={})
        for method, values in [('model', model), ('state_hold', hold)]:
            step = self.measure([1 if method == 'model' else 2] * (native_count - 1),
                                list(range(native_count - 1)), [True] * (native_count - 1))
            row['one_step'][method] = dict(forward=step, reverse=copy.deepcopy(step), all=self.combine([step, step]))
            row['query'][method] = dict(regions={key: self.measure(values, selected, valid)
                for key, selected in dict(all=[0, 1, 2, 3], inside_support=[1, 2],
                                           before_support=[0], after_support=[3]).items()},
                input_distance_metrics={key: self.measure(values, selected, valid)
                for key, selected in dict(le_300s=[0, 1], gt_300_le_600s=[2],
                    gt_600_le_1800s=[], gt_1800s=[3], no_input=[]).items()})
        return row

    def args(self, name='output'):
        return argparse.Namespace(source_run=str(self.source), manifest=str(self.root / 'manifest.json'),
                                  output=str(self.root / name), cpu_threads=4)

    def test_real_full_artifact_regrouping_retains_empty_scene_and_missing_reference(self):
        # Main's real parsing/writing path, with only production population scaled.
        args = self.args()
        with patch.object(audit, 'profile', return_value=self.expected):
            code = audit.main(['--source-run', args.source_run, '--manifest', args.manifest,
                               '--output', args.output, '--cpu-threads', '4'])
        self.assertEqual(code, 0)
        result = audit.read(Path(args.output) / 'summary.json')
        self.assertEqual(result['fragment_count'], 3)
        self.assertEqual(result['query_count'], 12)
        self.assertEqual(result['component_count'], 2)
        self.assertEqual(result['reference_targets_without_input'], 1)
        total = result['metrics']['query/regions/all']
        self.assertAlmostEqual(total['model']['feature_mse'], 23 / 7)
        self.assertAlmostEqual(total['state_hold']['feature_mse'], 18 / 7)
        self.assertEqual(total['model']['missing_reference_query_count'], 5)
        self.assertAlmostEqual(result['metrics'][audit.OUTSIDE]['model']['feature_mse'], 4.5)
        outside = result['outside_largest_component']
        self.assertEqual(outside['scene_count'], 1)
        self.assertAlmostEqual(outside['metrics']['query/regions/all']['model']['feature_mse'], 2.5)
        with gzip.open(Path(args.output) / 'scene-summary.json.gz', 'rt') as handle:
            scenes = json.load(handle)
        self.assertEqual([row['scene_id'] for row in scenes], ['a', 'b', 'c'])
        self.assertEqual(scenes[1]['fragment_count'], 0)
        self.assertIsNone(scenes[1]['metrics']['query/regions/all']['model']['feature_mse'])
        wins = result['unit_diagnostics']['fragments']['query/regions/all']
        self.assertEqual((wins['model_worse'], wins['missing_units']), (2, 1))
        scene_wins = result['unit_diagnostics']['scenes']['query/regions/all']
        self.assertEqual((scene_wins['model_worse'], scene_wins['missing_units']), (2, 1))
        self.assertEqual(result['model_forwards'], 0)
        self.assertFalse(result['checkpoint_opened'])
        self.assertIn('unavailable', result['region_distance_joint'])
        self.assertEqual(audit.read(Path(args.output) / 'run.json')['status'], 'COMPLETED')
        with self.assertRaises(FileExistsError):
            audit.run(args, expected=self.expected)

    def test_rejects_excerpt_omitted_population_and_inconsistent_aggregate(self):
        for name in ('excerpt', 'omitted', 'aggregate'):
            bad = copy.deepcopy(self.complete)
            if name == 'excerpt':
                bad['not_a_replay_input'] = True
                bad.pop('per_fragment')
            elif name == 'omitted':
                bad['per_fragment'].pop()
            else:
                bad['query']['model']['regions']['all']['squared_error_sum'] += 3
            audit.write(self.source / 'completion-evaluation.json', bad)
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.run(self.args(name), expected=self.expected)
            self.assertEqual(audit.read(self.root / name / 'run.json')['status'], 'FAILED')

    def test_refuses_mixed_pair_population_and_invalid_source_without_relabeling(self):
        bad = copy.deepcopy(self.complete)
        bad['per_fragment'][0]['query']['state_hold']['regions']['after_support']['evaluable_query_count'] = 0
        audit.write(self.source / 'completion-evaluation.json', bad)
        with self.assertRaises(ValueError):
            audit.run(self.args('mixed'), expected=self.expected)
        audit.write(self.source / 'completion-evaluation.json', self.complete)
        run = audit.read(self.source / 'run.json')
        run['phase'] = 'full'
        audit.write(self.source / 'run.json', run)
        with self.assertRaisesRegex(ValueError, 'TCM endpoint'):
            audit.run(self.args('wrong_phase'), expected=self.expected)
        self.assertEqual(audit.read(self.source / 'run.json')['phase'], 'full')


if __name__ == '__main__':
    unittest.main()
