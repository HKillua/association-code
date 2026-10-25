"""Real native Level1 source -> four paired completion-value stage runs."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from association.data.archive import MTADArchive
from association.data.scene_manifest import build_fresh_scene_manifest
from association.evaluation.artifacts import load_scene_prediction
from association.models.timed import TimedAssociationModel
from association.training import tmtaff as training
from association.utils import runtime
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.analysis import compare_completion_policies as comparator
from tools.experiments import tmtaff_complete_fit as phase_cli
from tools.experiments import completion_value_fit as cli


class CompletionValueFitTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.fixture = make_native_zip_fixture(self)
        self.root = self.fixture.root

    def invoke(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue() + err.getvalue()

    def args(self, phase, policy, output=None):
        return ['--phase', phase, '--completion-policy', policy,
            '--source-tcm', str(self.root / 'source'),
            '--manifest', str(self.root / 'manifest.json'), '--alignment', str(self.root / 'alignment.json'),
            '--config', str(self.root / f'{phase}-{policy}.json'), '--archive', str(self.fixture.path),
            '--output', str(self.root / (output or f'{phase}-{policy}')),
            '--device', 'cpu', '--cpu-threads', '1']

    def prepare(self):
        # Optional keep is solely for independent review of real test artifacts.
        # A new fixture location avoids absolute endpoint paths becoming stale.
        if os.environ.get('COMPLETION_VALUE_TEST_KEEP'):
            self.root = Path(tempfile.mkdtemp(prefix='completion-value-real-', dir='/tmp'))
            self.fixture.root = self.root
            self.fixture.path = self.root / 'native.zip'
            print('REVIEW_ARTIFACTS=' + str(self.root), flush=True)
        self.fixture.write_zip(missing_reference=('2', 'b'))
        with zipfile.ZipFile(self.fixture.path) as archive:
            contents = {item.filename: archive.read(item) for item in archive.infolist()}
        contents['数据集level2/关联表/关联结果-3.csv'] = contents[
            '数据集level2/关联表/关联结果-1.csv'].replace(b'target1', b'target3')
        with zipfile.ZipFile(self.fixture.path, 'w') as archive:
            for name, payload in contents.items():
                archive.writestr(name.replace('数据集level2/', 'Level1/'), payload)
        with MTADArchive(self.fixture.path) as archive:
            self.fixture.manifest = build_fresh_scene_manifest(archive, seed=46,
                caps=dict(T_core=2, T_head=0, V_select=2, T_test=1))
        source = runtime.read_json(cli.ROOT / 'configs/tmtaff-tcm-level1.json')
        source['model'].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
        source['fusion'].update(fc_hidden=8, d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
        source['training'].update(batch_tracks=6, pair_batch_size=4, pair_microbatch=2,
            track_microbatch=2, completion_cache_mib=1, validation_batches=dict(tracks=4, pairs=16))
        source['training']['updates']['TCM'] = 2
        self.fixture.config = source
        self.source_config = copy.deepcopy(source)
        self.targets = {}
        for phase, stage in (('tam', 'TAM_FROZEN'), ('tfm', 'TFM_FROZEN')):
            for policy in cli.POLICIES:
                target = copy.deepcopy(source)
                target['model']['completion_policy'] = policy
                target['training']['updates'] = {key: (2 if key == stage else 0)
                                                for key in source['training']['updates']}
                target['training']['validate_every_updates'] = 2
                target['training']['fusion_monitor_updates'] = [] if phase == 'tam' else [0, 2]
                self.targets[phase, policy] = target
                runtime.write_json(self.root / f'{phase}-{policy}.json', target)
        self.fixture.data = dict(source['data'], length=64)
        original_read = zipfile.ZipFile.read

        def read(archive, member, *args, **kwargs):
            name = member.filename if isinstance(member, zipfile.ZipInfo) else str(member)
            self.assertFalse(name.endswith('-3.csv'), 'sealed T_test payload opened')
            return original_read(archive, member, *args, **kwargs)

        self.enterContext(patch.object(zipfile.ZipFile, 'read', side_effect=read, autospec=True))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = phase_cli.main(['--phase', 'tcm', '--device', 'cpu', *self.fixture.cli_args('source')])
        self.assertEqual(code, 0, out.getvalue() + err.getvalue())

        def profile(phase, policy):
            return dict(source=self.source_config, target=self.targets[phase, policy], train_scenes=2,
                        validation_scenes=2, train_fragments=8, validation_fragments=4)

        self.enterContext(patch.object(cli, '_experiment_contract', side_effect=profile))
        return torch.load(self.root / 'source/checkpoint.pt', map_location='cpu', weights_only=True)

    def test_active_source_does_not_load_weights_claim_or_create_output(self):
        source = self.root / 'source'
        source.mkdir()
        runtime.write_json(source / 'run.json', dict(status='RUNNING', phase='tcm'))
        with (patch.object(torch, 'load', side_effect=AssertionError('active checkpoint read')),
              patch.object(phase_cli, 'main', side_effect=AssertionError('pending run started'))):
            code, text = self.invoke(self.args('tfm', 'state_hold'))
        self.assertEqual(code, 75, text)
        self.assertIn('WAITING_DEPENDENCIES', text)
        self.assertFalse((self.root / 'tfm-state_hold').exists())

    def test_real_four_arm_pairing_full_endpoints_reload_and_artifact_rejections(self):
        source = self.prepare()
        results, initials = {}, {}
        for phase in ('tfm', 'tam'):
            for policy in cli.POLICIES:
                with ExitStack() as stack:
                    if policy == 'state_hold':
                        stack.enter_context(patch.object(TimedAssociationModel, '_extrapolate',
                            side_effect=AssertionError('hold policy called learned rollout')))
                    if phase == 'tfm':
                        for name in ('score_reference', 'select_threshold', 'calibrate_training_margin'):
                            stack.enter_context(patch.object(training, name,
                                side_effect=AssertionError('TFM called TAM/calibration')))
                    code, text = self.invoke(self.args(phase, policy))
                self.assertEqual(code, 0, text)
                output = self.root / f'{phase}-{policy}'
                saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
                initial = torch.load(output / 'pairing-initial.pt', map_location='cpu', weights_only=True)
                results[phase, policy], initials[phase, policy] = saved, initial
                self.assertEqual(saved['model_config']['completion_policy'], policy)
                self.assertTrue(saved['pairing_audit'])
                self.assertEqual(initial['optimizer']['state'], {})
                self.assertEqual(initial['cursor']['completed_updates'], 0)
                self.assertEqual(saved['normalizer'], source['normalizer'])
                self.assertEqual(saved['data_contract'], source['data_contract'])
                loaded = training.load_complete_snapshot(output / 'checkpoint.pt', selected=False, device='cpu')
                self.assertEqual(loaded['model'].config.completion_policy, policy)
                self.assertEqual(runtime.read_json(output / 'completion-value-verification.json')['status'], 'COMPLETED')
                self.assertEqual(cli.validate_completed_run(output, phase, policy)['actual_samples']['pair_presentations'], 8)
                for key, value in source['states'][0].items():
                    if key == 'positions' or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
                        torch.testing.assert_close(saved['states'][0][key], value, rtol=0, atol=0)
                summary = runtime.read_json(output / 'summary.json')
                if phase == 'tfm':
                    self.assertEqual([row['update'] for row in summary['fusion_evaluations']], [0, 2])
                    for endpoint in summary['fusion_evaluations']:
                        evaluation = runtime.read_json(endpoint['evaluation'])
                        self.assertTrue(evaluation['oracle_only'])
                        self.assertIsNone(evaluation['association'])
                        self.assertEqual([row['scene_id'] for row in evaluation['per_scene']], ['2', '4'])
                        index_path = Path(endpoint['predictions'])
                        index = runtime.read_json(index_path)
                        missing = load_scene_prediction(index_path.parent / index['scenes'][1]['metadata'])
                        self.assertEqual(missing['scene']['node_ids'].tolist(), [])
                        self.assertEqual(missing['methods']['oracle']['target_ids'].tolist(), ['target4a-0'])
                        self.assertFalse(missing['methods']['oracle']['aligned_prediction_valid'].any())
                else:
                    records = runtime.load_records(summary['endpoint_score_records'])
                    self.assertEqual([row['scene_id'] for row in records], ['2', '4'])
                    self.assertEqual([len(row['pairs']) for row in records], [6, 0])
                    self.assertFalse(summary['evaluation']['physical_fusion_evaluated'])
            a, b = (results[phase, policy] for policy in cli.POLICIES)
            ai, bi = (initials[phase, policy] for policy in cli.POLICIES)
            for ast, bst in zip(ai['states'], bi['states']):
                self.assertEqual(ast.keys(), bst.keys())
                for key in ast:
                    torch.testing.assert_close(ast[key], bst[key], rtol=0, atol=0)
            self.assertTrue(cli._same_tree(ai['rng'], bi['rng']))
            self.assertTrue(cli._same_tree(a['rng'], b['rng']))
            self.assertEqual([row['actual_samples'] for row in a['history']],
                             [row['actual_samples'] for row in b['history']])
            self.assertEqual([row['supervision_counts'] for row in a['history']],
                             [row['supervision_counts'] for row in b['history']])
            if phase == 'tam':
                for ac, bc in zip(a['margin_calibration']['samples'], b['margin_calibration']['samples']):
                    self.assertEqual(ac['groups'], bc['groups'])
                    self.assertEqual(ac['labels'], bc['labels'])

        # A successful phase record is insufficient if its saved policy, actual
        # samples, initial optimizer or frozen tensors were changed afterwards.
        output = self.root / 'tfm-state_hold'
        path = output / 'checkpoint.pt'
        original = results['tfm', 'state_hold']
        for case in ('policy', 'actual_sample', 'frozen'):
            saved = copy.deepcopy(original)
            if case == 'policy':
                saved['model_config']['completion_policy'] = 'learned'
            elif case == 'actual_sample':
                saved['history'][0]['actual_samples'][0]['labels'][0] = 0
            else:
                key = next(key for key in saved['states'][0] if key.startswith('tcm_encoder.'))
                saved['states'][0][key].add_(1.)
            torch.save(saved, path)
            with self.assertRaises(ValueError, msg=case):
                cli.validate_completed_run(output, 'tfm', 'state_hold')
        torch.save(original, path)

        # A stale/contradictory terminal run or journal cannot be accepted just
        # because checkpoint.pt has the desired endpoint and total exposures.
        for name, mutate in (
            ('run.json', lambda value: value['cursor'].update(completed_updates=1)),
            ('history.json', lambda value: value[0].update(loss=value[0]['loss'] + 1.)),
            ('coverage.json', lambda value: value['TFM_FROZEN'].update(pair_presentations=1)),
        ):
            artifact_path = output / name
            original_bytes = artifact_path.read_bytes()
            value = runtime.read_json(artifact_path)
            mutate(value)
            try:
                runtime.write_json(artifact_path, value)
                with self.assertRaises(ValueError, msg=name):
                    cli.validate_completed_run(output, 'tfm', 'state_hold')
            finally:
                artifact_path.write_bytes(original_bytes)

        tam_output = self.root / 'tam-state_hold'
        tam_summary = runtime.read_json(tam_output / 'summary.json')
        score_index_path = Path(tam_summary['endpoint_score_records']) / 'index.json'
        original_index = score_index_path.read_bytes()
        for field, wrong in (('update', 1), ('split', 'T_test'), ('test_opened', True)):
            try:
                index = runtime.read_json(score_index_path)
                index['binding'][field] = wrong
                runtime.write_json(score_index_path, index)
                with self.assertRaisesRegex(ValueError, 'provenance', msg=field):
                    cli.validate_completed_run(tam_output, 'tam', 'state_hold')
            finally:
                score_index_path.write_bytes(original_index)

        # Read the actual complete caches and saved weights; model construction,
        # checkpoint-to-model loading and training are forbidden in comparison.
        with (patch.object(comparator, '_expected_configuration',
                           side_effect=lambda phase, policy: self.targets[phase, policy]),
              patch.object(TimedAssociationModel, '__init__',
                           side_effect=AssertionError('CPU comparison constructed a model')),
              patch.object(training, 'load_complete_snapshot',
                           side_effect=AssertionError('CPU comparison restored a model')),
              patch.object(training, 'train_complete',
                           side_effect=AssertionError('CPU comparison trained'))):
            for phase in ('tfm', 'tam'):
                compared = comparator.compare(self.root / f'{phase}-learned',
                    self.root / f'{phase}-state_hold', self.root / f'compare-{phase}', phase)
                self.assertEqual(compared['status'], 'COMPLETED')
                self.assertEqual(compared['model_forward_count'], 0)
                self.assertTrue(compared['pairing']['all_initial_parameters_equal'])
                self.assertTrue(compared['pairing']['ordered_successful_samples_equal'])
                self.assertFalse(compared['pairing']['initial_predictions_required_equal'])
                self.assertEqual(compared['comparison']['scene_count'], 2)
                if phase == 'tam':
                    # The second V scene has a declared target but no nodes.
                    # It stays in the target denominator of the grouped report.
                    for side in cli.POLICIES:
                        totals = compared['comparison']['grouping_totals'][side]
                        self.assertEqual(totals['declared_target_count'], 4)
                        self.assertEqual(totals['node_count'], 4)
                        self.assertEqual(totals['anchor_matched_target_count'] + totals['unmatched_target_count'], 4)
                        self.assertGreaterEqual(totals['unmatched_target_count'], 1)
                        self.assertEqual(totals['anchor_matched_target_count'] + totals['duplicate_anchor_group_count'],
                                         totals['predicted_group_count'])

            # Same valid pair population/counts and matching checkpoint/history
            # are insufficient: changing the actual draw order breaks pairing.
            changed = copy.deepcopy(original)
            group = changed['history'][0]['actual_samples'][0]
            for key in ('pair_indices', 'pairs', 'labels'):
                group[key] = list(reversed(group[key]))
            self.assertNotEqual(group['pair_indices'],
                original['history'][0]['actual_samples'][0]['pair_indices'])
            history_path = output / 'history.json'
            original_history = history_path.read_bytes()
            try:
                torch.save(changed, path)
                runtime.write_json(history_path, changed['history'])
                self.assertEqual(cli.validate_completed_run(output, 'tfm', 'state_hold')['status'], 'COMPLETED')
                with self.assertRaisesRegex(ValueError, 'actual_samples'):
                    comparator.compare(self.root / 'tfm-learned', output,
                                       self.root / 'reject-sampling-order', 'tfm')
            finally:
                torch.save(original, path)
                history_path.write_bytes(original_history)

            # Keep the index, target counts and all array shapes unchanged, but
            # alter an actual cached input node. Metadata-only checks miss this.
            summary = runtime.read_json(output / 'summary.json')
            index_path = Path(summary['fusion_evaluations'][-1]['predictions'])
            index = runtime.read_json(index_path)
            cache_path = index_path.parent / index['scenes'][0]['npz']
            original_cache = cache_path.read_bytes()
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                node_ids = arrays['scene__node_ids'].tolist()
                node_ids[0] = 'changed-fragment-with-same-population-size'
                arrays['scene__node_ids'] = np.asarray(node_ids, dtype=str)
                np.savez_compressed(cache_path, **arrays)
                with self.assertRaises(ValueError):
                    comparator.compare(self.root / 'tfm-learned', output,
                                       self.root / 'reject-cache-population', 'tfm')
            finally:
                cache_path.write_bytes(original_cache)

    def test_wrong_fixed_policy_and_incomplete_source_are_rejected_before_fit(self):
        self.prepare()
        config_path = self.root / 'tfm-state_hold.json'
        changed = copy.deepcopy(self.targets['tfm', 'state_hold'])
        changed['model']['completion_policy'] = 'learned'
        runtime.write_json(config_path, changed)
        with patch.object(phase_cli, 'main', side_effect=AssertionError('invalid target started')):
            code, text = self.invoke(self.args('tfm', 'state_hold'))
        self.assertEqual(code, 1, text)
        self.assertFalse((self.root / 'tfm-state_hold').exists())
        runtime.write_json(config_path, self.targets['tfm', 'state_hold'])
        evaluation = runtime.read_json(self.root / 'source/completion-evaluation.json')
        evaluation['per_fragment'].pop()
        runtime.write_json(self.root / 'source/completion-evaluation.json', evaluation)
        with patch.object(phase_cli, 'main', side_effect=AssertionError('incomplete source accepted')):
            code, text = self.invoke(self.args('tfm', 'state_hold'))
        self.assertEqual(code, 1, text)
        self.assertIn('omits validation population', text)
        self.assertFalse((self.root / 'tfm-state_hold').exists())


if __name__ == '__main__':
    unittest.main()
