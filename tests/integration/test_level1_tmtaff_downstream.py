"""092 real Level1 phase entry, dependency waiting and frozen-source evidence."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import torch

from association.data.archive import MTADArchive
from association.data.scene_manifest import build_fresh_scene_manifest
from association.evaluation.artifacts import load_scene_prediction
from association.training import tmtaff as training
from association.utils import runtime
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.experiments import tmtaff_complete_fit as phase_cli
from tools.experiments import tmtaff_downstream_fit as cli


class Level1TmtaffDownstreamTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.fixture = make_native_zip_fixture(self)
        self.root = self.fixture.root

    def invoke(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            result = cli.main(argv)
        return result, out.getvalue() + err.getvalue()

    def args(self, phase, output='downstream'):
        return ['--phase', phase, '--source-tcm', str(self.root / 'source'),
            '--manifest', str(self.root / 'manifest.json'), '--alignment', str(self.root / 'alignment.json'),
            '--config', str(self.root / f'{phase}.json'), '--archive', str(self.fixture.path),
            '--output', str(self.root / output), '--device', 'cpu', '--cpu-threads', '1']

    def test_waiting_source_does_not_load_checkpoint_or_enter_training_or_create_output(self):
        source = self.root / 'source'
        source.mkdir()
        with (patch.object(torch, 'load', side_effect=AssertionError('active checkpoint opened')),
              patch.object(phase_cli, 'main', side_effect=AssertionError('pending task started'))):
            for status in (None, 'RUNNING'):
                if status is not None:
                    runtime.write_json(source / 'run.json', dict(status=status, phase='tcm'))
                code, text = self.invoke(self.args('tam'))
                self.assertEqual(code, 75, text)
                self.assertIn('WAITING_DEPENDENCIES', text)
                self.assertFalse((self.root / 'downstream').exists())
            runtime.write_json(source / 'run.json', dict(status='FAILED', phase='tcm'))
            code, text = self.invoke(self.args('tfm'))
            self.assertEqual(code, 1, text)
            self.assertNotIn('WAITING_DEPENDENCIES', text)
            runtime.write_json(source / 'run.json', dict(status='COMPLETED', phase='tcm',
                                                        final_evaluation_status='RUNNING'))
            code, text = self.invoke(self.args('tam'))
            self.assertEqual(code, 1, text)
            self.assertIn('without completed final evaluation', text)

    def prepare_level1(self):
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
            target = copy.deepcopy(source)
            target['training']['updates'] = {key: (2 if key == stage else 0)
                                              for key in source['training']['updates']}
            target['training']['validate_every_updates'] = 2
            target['training']['fusion_monitor_updates'] = [] if phase == 'tam' else [0, 2]
            self.targets[phase] = target
            runtime.write_json(self.root / f'{phase}.json', target)
        self.fixture.data = dict(source['data'], length=64)
        original_read = zipfile.ZipFile.read

        def read(archive, member, *args, **kwargs):
            name = member.filename if isinstance(member, zipfile.ZipInfo) else str(member)
            self.assertFalse(name.endswith('-3.csv'), 'T_test payload must stay sealed')
            return original_read(archive, member, *args, **kwargs)

        self.enterContext(patch.object(zipfile.ZipFile, 'read', side_effect=read, autospec=True))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = phase_cli.main(['--phase', 'tcm', '--device', 'cpu', *self.fixture.cli_args('source')])
        self.assertEqual(code, 0, out.getvalue() + err.getvalue())

        def profile(phase):
            return dict(source=self.source_config, target=self.targets[phase], train_scenes=2,
                        validation_scenes=2, train_fragments=8, validation_fragments=4)

        self.enterContext(patch.object(cli, '_experiment_contract', side_effect=profile))
        return torch.load(self.root / 'source/checkpoint.pt', map_location='cpu', weights_only=True)

    def assert_frozen(self, source, saved):
        self.assertEqual(saved['normalizer'], source['normalizer'])
        self.assertEqual(saved['normalizer']['training_scene_ids'], ['0', '1'])
        self.assertEqual(saved['data_contract'], source['data_contract'])
        for key, value in source['states'][0].items():
            if key == 'positions' or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
                torch.testing.assert_close(saved['states'][0][key], value, rtol=0, atol=0)

    def test_real_tam_stage_keeps_full_population_and_frozen_completion(self):
        source = self.prepare_level1()
        # A TCM checkpoint's unused TAM must not become the new trained TAM.
        for key, value in source['states'][0].items():
            if key.startswith('tam_encoder.'):
                value.add_(10.)
        torch.save(source, self.root / 'source/checkpoint.pt')
        code, text = self.invoke([*self.args('tam'), '--check-source-only'])
        self.assertEqual(code, 0, text)
        self.assertIn('READY_DEPENDENCIES', text)
        self.assertFalse((self.root / 'downstream').exists())
        original, initial = training._stage_parameters, {}

        def parameters(modules, stage):
            self.assertEqual(stage, 'TAM_FROZEN')
            result = original(modules, stage)
            self.assertTrue(all(not parameter.requires_grad for parameter in modules[0].completion_parameters()))
            initial.update({key: value.detach().clone() for key, value in modules[0].state_dict().items()})
            return result

        with (patch.object(training, '_stage_parameters', side_effect=parameters),
              patch.object(training, 'evaluate_complete', side_effect=AssertionError('untrained TFM forbidden')),
              patch.object(training, 'fusion_training_step', side_effect=AssertionError('wrong training phase'))):
            code, text = self.invoke(self.args('tam'))
        self.assertEqual(code, 0, text)
        output = self.root / 'downstream'
        saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
        self.assert_frozen(source, saved)
        self.assertTrue(any(not torch.equal(initial[key], value) for key, value in saved['states'][0].items()
                            if key.startswith('tam_encoder.')))
        self.assertTrue(all(not torch.equal(initial[key], value) for key, value in source['states'][0].items()
                            if key.startswith('tam_encoder.')))
        summary = runtime.read_json(output / 'summary.json')
        self.assertEqual(len(summary['validation_score_records']), 1)
        self.assertEqual(summary['evaluation']['association']['pair_count'], 6)
        records = runtime.load_records(summary['endpoint_score_records'])
        self.assertEqual([row['scene_id'] for row in records], ['2', '4'])
        self.assertEqual([len(row['pairs']) for row in records], [6, 0])
        self.assertEqual(runtime.read_json(output / 'downstream-verification.json')['status'], 'COMPLETED')
        # The independent postflight rejects a stale/fine-tuned completion.
        key = next(key for key in saved['states'][0] if key.startswith('tcm_encoder.'))
        saved['states'][0][key].add_(1.)
        torch.save(saved, output / 'checkpoint.pt')
        bound = cli.check_source(self.root / 'source', self.root / 'manifest.json',
            self.root / 'alignment.json', self.root / 'tam.json', 'tam')
        with self.assertRaisesRegex(ValueError, 'frozen TCM tensor changed'):
            cli.verify_endpoint(output, bound)

    def test_real_tfm_stage_saves_initial_and_final_oracle_full_population(self):
        source = self.prepare_level1()
        original, initial = training._stage_parameters, {}

        def parameters(modules, stage):
            self.assertEqual(stage, 'TFM_FROZEN')
            result = original(modules, stage)
            self.assertTrue(all(not parameter.requires_grad for parameter in modules[0].parameters()))
            initial.update({key: value.detach().clone() for key, value in modules[1].state_dict().items()})
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(training, '_stage_parameters', side_effect=parameters))
            for name in ('score_reference', 'select_threshold', 'calibrate_training_margin'):
                stack.enter_context(patch.object(training, name, side_effect=AssertionError('TAM forbidden')))
            evaluate = training.evaluate_complete

            def oracle_evaluate(model, *args, **kwargs):
                self.assertTrue(kwargs['oracle_only'])
                with patch.object(model.tam_encoder, 'forward', side_effect=AssertionError('TAM evaluation forbidden')):
                    return evaluate(model, *args, **kwargs)

            stack.enter_context(patch.object(training, 'evaluate_complete', side_effect=oracle_evaluate))
            code, text = self.invoke(self.args('tfm'))
        self.assertEqual(code, 0, text)
        output = self.root / 'downstream'
        saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
        self.assert_frozen(source, saved)
        self.assertTrue(any(not torch.equal(initial[key], value) for key, value in saved['states'][1].items()))
        summary = runtime.read_json(output / 'summary.json')
        self.assertEqual([row['update'] for row in summary['fusion_evaluations']], [0, 2])
        self.assertFalse(summary['selection_performed'])
        for endpoint in summary['fusion_evaluations']:
            evaluation = runtime.read_json(endpoint['evaluation'])
            self.assertTrue(evaluation['oracle_only'])
            self.assertIsNone(evaluation['association'])
            self.assertEqual([row['scene_id'] for row in evaluation['per_scene']], ['2', '4'])
            index = runtime.read_json(endpoint['predictions'])
            self.assertEqual([row['scene_id'] for row in index['scenes']], ['2', '4'])
            path = Path(endpoint['predictions']).parent
            empty_scene = load_scene_prediction(path / index['scenes'][1]['metadata'])
            self.assertEqual(empty_scene['scene']['node_ids'].tolist(), [])
            self.assertEqual(empty_scene['methods']['oracle']['target_ids'].tolist(), ['target4a-0'])
            self.assertFalse(empty_scene['methods']['oracle']['aligned_prediction_valid'].any())
        verification = runtime.read_json(output / 'downstream-verification.json')
        self.assertEqual(verification['status'], 'COMPLETED')
        self.assertEqual(verification['pair_presentations'], 8)
        self.assertFalse(verification['full_system_evaluated'])

    def test_completed_source_with_wrong_population_or_phase_config_is_blocked(self):
        self.prepare_level1()
        config = runtime.read_json(self.root / 'tam.json')
        config['training']['updates']['TCM'] = 1
        runtime.write_json(self.root / 'tam.json', config)
        with patch.object(phase_cli, 'main', side_effect=AssertionError('invalid phase started')):
            code, text = self.invoke(self.args('tam'))
        self.assertEqual(code, 1, text)
        self.assertFalse((self.root / 'downstream').exists())
        runtime.write_json(self.root / 'tam.json', self.targets['tam'])
        evaluation = runtime.read_json(self.root / 'source/completion-evaluation.json')
        evaluation['per_scene'].pop()
        runtime.write_json(self.root / 'source/completion-evaluation.json', evaluation)
        with patch.object(phase_cli, 'main', side_effect=AssertionError('incomplete population started')):
            code, text = self.invoke(self.args('tam'))
        self.assertEqual(code, 1, text)
        self.assertIn('omits validation population', text)
        self.assertFalse((self.root / 'downstream').exists())


if __name__ == '__main__':
    unittest.main()
