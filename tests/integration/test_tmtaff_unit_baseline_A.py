"""Synthetic endpoint gates and real short child lifecycle; no model/data/GPU."""
from dataclasses import asdict
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from association.data.physical_units import SPEED_UNIT_POLICY, unit_metadata
from association.models.tmtaff import ModelConfig
from association.models.fusion import PairFusionConfig
from association.training.tmtaff import _configuration
from association.utils import runtime
from tools.experiments import run_tmtaff_unit_baseline_A as chain


class UnitBaselineChainTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scene_ids = [str(i) for i in range(300)]
        for filename in ('a.zip', 'b.zip', 'c.zip', 'alignment.json'):
            (self.root/filename).write_text('{}')
        for name in ('056', '090c'):
            (self.root/name).mkdir()
        runtime.write_json(self.root/'manifest.json', dict(splits=dict(
            T_core=list(range(1000, 3100)), V_select=self.scene_ids)))
        self.args = SimpleNamespace(archive=[self.root/name for name in ('a.zip', 'b.zip', 'c.zip')],
            manifest=self.root/'manifest.json', alignment=self.root/'alignment.json',
            source_system=self.root/'056', reference090c_run=self.root/'090c',
            output=self.root/'output', device='cuda:0', cpu_threads=4)
        self.configs = {name: runtime.read_json(chain.ROOT/'configs'/f'tmtaff-{name}-unit-draft.json')
                        for name in ('tcm', 'tfm')}
        for name, config in self.configs.items():
            config['training'] = _configuration(config['training'], phase=name)
        self.calls = []

    def predictions(self, directory):
        directory.mkdir(parents=True)
        rows = []
        for index, sid in enumerate(self.scene_ids):
            entry = dict(scene_id=sid, scene_index=index, npz=f'{index}.npz', metadata=f'{index}.json')
            for key in ('npz', 'metadata'):
                (directory/entry[key]).write_text('synthetic')
            rows.append(entry)
        runtime.write_json(directory/'index.json', dict(test_opened=False, scenes=rows))

    def endpoint(self, name):
        directory = self.args.output/name
        directory.mkdir()
        evaluation = dict(per_scene=[dict(scene_id=sid) for sid in self.scene_ids], scene_count=300)
        if name == 'fixed056':
            runtime.write_json(directory/'run.json', dict(status='COMPLETED', test_opened=False,
                scenes=300, optimizer_updates=0, tam_scoring_forwards=0, threshold_reselected=False))
            runtime.write_json(directory/'summary.json', dict(test_opened=False, evaluation=evaluation))
            self.predictions(directory/'predictions')
            return
        config = self.configs[name]
        updates, stage, stage_index = (2424, 'TCM', 0) if name == 'tcm' else (1000, 'TFM_FROZEN', 2)
        cursor = dict(stage=stage_index, update=updates, completed_updates=updates)
        normalizer = dict(adapter_config=dict(speed_unit_policy=SPEED_UNIT_POLICY),
                          physical_units=unit_metadata(SPEED_UNIT_POLICY))
        history = [dict(stage=stage, update=i, fusion_valid_counts=[1]*4) for i in range(1, updates+1)]
        item = dict(completed_epochs=1, track_presentations=155134, unique_track_count=155134,
                    transition_presentations=35555236, pair_presentations=64000,
                    tracks=[(0, 1)])  # Checkpoint tuples serialize to JSON lists.
        coverage = {stage: item}
        payload = dict(format_version=1, phase=name, input_adapter='native_time_v1',
            config=config['training'], model_config=asdict(ModelConfig(**config['model'])),
            fusion_config=asdict(PairFusionConfig(**config['fusion'])), normalizer=normalizer,
            cursor=cursor, history=history, coverage=coverage, data_contract={}, selected=None,
            states=[dict(speed_unit_contract=torch.tensor(1), **{'input_embedding.weight': torch.tensor([1.])}),
                    dict(speed_unit_contract=torch.tensor(1))])
        run = dict(status='COMPLETED', phase=name, test_opened=False, final_evaluation_status='COMPLETED',
            selection_performed=False, config=config['training'], checkpoint_cursor=cursor,
            cursor=dict(stage=stage_index+1, update=0, completed_updates=updates))
        summary = dict(test_opened=False, normalizer=normalizer, coverage=coverage)
        if name == 'tcm':
            runtime.write_json(directory/'completion-evaluation.json', evaluation)
        else:
            payload.update(pairing_audit=True, initialization=dict(source_run=str(self.args.output/'tcm'),
                source_phase='tcm', source_tcm_updates=2424))
            (directory/'pairing-initial.pt').write_bytes(b'synthetic')
            (directory/'snapshots').mkdir()
            summary['fusion_evaluations'] = []
            for update in (0, 1000):
                (directory/'snapshots'/f'update-{update:06d}.pt').write_bytes(b'synthetic')
                endpoint = directory/'fusion-evaluations'/f'update-{update:06d}'
                self.predictions(endpoint/'predictions')
                runtime.write_json(endpoint/'evaluation.json', evaluation)
                summary['fusion_evaluations'].append(dict(update=update, status='COMPLETED'))
        for filename, value in (('run.json', run), ('summary.json', summary), ('history.json', history),
                                ('coverage.json', coverage), ('preparation.json', dict(configuration=config))):
            runtime.write_json(directory/filename, value)
        torch.save(payload, directory/'checkpoint.pt')

    def execute(self, row, logs, write_state):
        self.calls.append(row['name'])
        self.endpoint(row['name'])
        row['returncode'] = 0
        write_state()

    def test_success_requires_three_verified_endpoints_and_exact_argv(self):
        with patch.object(chain, '_execute_stage', side_effect=self.execute):
            result = chain.run(self.args)
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(self.calls, ['tcm', 'tfm', 'fixed056'])
        self.assertTrue(all(row['validation_status'] == 'COMPLETED' for row in result['stages']))
        tfm = result['stages'][1]['argv']
        self.assertIn('--pairing-audit', tfm)
        self.assertIn('--save-predictions', tfm)
        self.assertNotIn('--resume', tfm)
        self.assertIn('--reference090c-run', result['stages'][2]['argv'])

    def test_incomplete_missing_wrong_unit_or_failed_eval_never_start_tfm(self):
        for defect in ('short', 'missing', 'unit', 'evaluation'):
            with self.subTest(defect=defect):
                self.args.output = self.root/defect
                self.calls = []
                def execute(row, logs, write_state):
                    self.execute(row, logs, write_state)
                    directory = self.args.output/'tcm'
                    if defect == 'missing':
                        (directory/'checkpoint.pt').unlink()
                    elif defect == 'unit':
                        payload = torch.load(directory/'checkpoint.pt', weights_only=True)
                        payload['states'][0]['speed_unit_contract'] = torch.tensor(0)
                        torch.save(payload, directory/'checkpoint.pt')
                    else:
                        run = runtime.read_json(directory/'run.json')
                        if defect == 'short': run['cursor']['completed_updates'] = 2423
                        else: run['final_evaluation_status'] = 'FAILED'
                        runtime.write_json(directory/'run.json', run)
                with patch.object(chain, '_execute_stage', side_effect=execute):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        chain.run(self.args)
                self.assertEqual(self.calls, ['tcm'])
                state = runtime.read_json(self.args.output/'run.json')
                self.assertEqual((state['status'], state['stages'][0]['returncode']), ('FAILED', 0))
                self.assertEqual(state['stages'][0]['validation_status'], 'FAILED')

    def test_partial_tfm_prediction_population_stops_fixed056(self):
        def execute(row, logs, write_state):
            self.execute(row, logs, write_state)
            if row['name'] == 'tfm':
                path = self.args.output/'tfm/fusion-evaluations/update-001000/predictions/index.json'
                value = runtime.read_json(path); value['scenes'].pop(); runtime.write_json(path, value)
        with patch.object(chain, '_execute_stage', side_effect=execute):
            with self.assertRaisesRegex(ValueError, 'complete ordered V'):
                chain.run(self.args)
        self.assertEqual(self.calls, ['tcm', 'tfm'])
        self.assertFalse((self.args.output/'fixed056').exists())

    def test_real_nonzero_exit_prevents_successor_and_preserves_code(self):
        actual_execute = chain._execute_stage
        def dispatch(row, logs, write_state):
            self.calls.append(row['name'])
            row['argv'] = [sys.executable, '-c', 'print("out"); import sys; print("err",file=sys.stderr); sys.exit(7)']
            actual_execute(row, logs, write_state)
        with patch.object(chain, '_execute_stage', side_effect=dispatch):
            with self.assertRaisesRegex(ValueError, 'exited 7'):
                chain.run(self.args)
        state = runtime.read_json(self.args.output/'run.json')
        self.assertEqual(self.calls, ['tcm'])
        self.assertEqual(state['stages'][0]['returncode'], 7)
        self.assertEqual((self.args.output/'logs/tcm.stdout.log').read_text(), 'out\n')
        self.assertEqual((self.args.output/'logs/tcm.stderr.log').read_text(), 'err\n')

    def test_existing_output_is_not_changed(self):
        self.args.output.mkdir()
        marker = self.args.output/'keep'; marker.write_text('keep')
        with patch.object(chain, '_execute_stage') as execute:
            with self.assertRaises(FileExistsError): chain.run(self.args)
        execute.assert_not_called()
        self.assertEqual(list(self.args.output.iterdir()), [marker])
        self.assertEqual(marker.read_text(), 'keep')

    @unittest.skipUnless(os.name == 'posix', 'POSIX process group and TERM')
    def test_term_forwards_to_real_child_in_same_process_group(self):
        logs = self.root/'logs'; logs.mkdir()
        row = dict(name='short', argv=[sys.executable, '-c', 'import time; time.sleep(30)'], returncode=None)
        timer = threading.Timer(.2, os.kill, (os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            with self.assertRaises(InterruptedError):
                chain._execute_stage(row, logs, lambda: None)
        finally:
            timer.cancel(); timer.join()
        self.assertEqual(row['pgid'], os.getpgrp())
        self.assertEqual(row['returncode'], -signal.SIGTERM)
        with self.assertRaises(ProcessLookupError): os.kill(row['pid'], 0)


if __name__ == '__main__':
    unittest.main()
