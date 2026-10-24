"""088 executes native Level1 TCM only, with the real data/time conventions."""
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
from association.training import tmtaff as training
from association.training.common import fit_normalizer
from association.utils import runtime
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.experiments import tmtaff_complete_fit as cli


class Level1TcmQualificationTests(unittest.TestCase):
    def test_native_level1_cli_preserves_short_tail_population_and_frozen_projection(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        fixture = make_native_zip_fixture(self)
        fixture.write_zip(missing_reference=('2', 'b'))
        with zipfile.ZipFile(fixture.path) as archive:
            contents = {item.filename: archive.read(item) for item in archive.infolist()}
        # A fresh scene manifest reads all labels, but sealed measurement and
        # reference payloads remain deliberately invalid and cannot be consumed.
        contents['数据集level2/关联表/关联结果-3.csv'] = contents[
            '数据集level2/关联表/关联结果-1.csv'].replace(b'target1', b'target3')
        with zipfile.ZipFile(fixture.path, 'w') as archive:
            for name, payload in contents.items():
                archive.writestr(name.replace('数据集level2/', 'Level1/'), payload)
        with MTADArchive(fixture.path) as archive:
            fixture.manifest = build_fresh_scene_manifest(archive, seed=46,
                caps=dict(T_core=2, T_head=0, V_select=2, T_test=1))
        self.assertEqual(fixture.manifest['splits'],
            dict(T_core=['0', '1'], T_head=[], V_select=['2', '4'], T_test=['3']))
        production = runtime.read_json(Path(__file__).resolve().parents[1] /
                                       'configs/tmtaff-tcm-level1.json')
        self.assertEqual(production['training']['updates'],
            dict(TCM=2424, TAM_FROZEN=0, TFM_FROZEN=0, JOINT=0))
        self.assertEqual(divmod(155134, production['training']['batch_tracks']), (2423, 62))
        fixture.config = copy.deepcopy(production)
        fixture.config['model'].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
        fixture.config['fusion'].update(fc_hidden=8, d_model=8, nhead=2,
                                         num_layers=1, dim_feedforward=16)
        fixture.config['training'].update(batch_tracks=6)
        fixture.config['training']['updates']['TCM'] = 2
        fixture.data = dict(fixture.config['data'], length=fixture.config['model']['length'])
        self.assertEqual(fixture.data, dict(length=64, max_observations=128,
            time_unit_seconds=20., reference_longitude_policy='shortest_arc'))

        reads, initial = [], {}
        original_read, original_stage = zipfile.ZipFile.read, training._stage_parameters

        def guarded_read(archive, member, *args, **kwargs):
            name = member.filename if isinstance(member, zipfile.ZipInfo) else str(member)
            self.assertFalse(name.endswith('-3.csv'), 'sealed test payload must not be read')
            reads.append(name)
            return original_read(archive, member, *args, **kwargs)

        def stage_parameters(modules, stage):
            self.assertEqual(stage, 'TCM')
            parameters = original_stage(modules, stage)
            self.assertTrue(all(not p.requires_grad for p in modules[0].input_embedding.parameters()))
            initial.update({key: value.detach().clone() for key, value in modules[0].state_dict().items()})
            return parameters

        with ExitStack() as stack:
            stack.enter_context(patch.object(zipfile.ZipFile, 'read', guarded_read))
            stack.enter_context(patch.object(training, '_stage_parameters', side_effect=stage_parameters))
            for name in ('score_reference', 'select_threshold', 'evaluate_complete',
                         'calibrate_training_margin', 'fusion_training_step', 'FrozenCompletionCache'):
                stack.enter_context(patch.object(training, name,
                    side_effect=AssertionError('downstream forbidden: ' + name)))
            expected_normalizer = fit_normalizer(fixture.load()['train'])
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main(['--phase', 'tcm', '--device', 'cpu',
                    *fixture.cli_args('qualification'), '--time-policy', 'advisory'])
            self.assertEqual(code, 0, out.getvalue() + err.getvalue())
        output = fixture.root / 'qualification'
        saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
        self.assertEqual(saved['normalizer'], expected_normalizer)
        self.assertEqual(saved['normalizer']['training_scene_ids'], ['0', '1'])
        self.assertEqual(saved['cursor']['completed_updates'], 2)
        self.assertEqual(saved['training_stages'], ['TCM'])
        self.assertEqual([row['track_presentations'] for row in saved['history']], [6, 2])
        coverage = saved['coverage']['TCM']
        self.assertEqual(coverage['unique_track_count'], 8)
        self.assertEqual(coverage['track_presentations'], 8)
        self.assertEqual(coverage['transition_presentations'], 32)
        self.assertEqual(coverage['completed_epochs'], 1)
        for key, value in initial.items():
            if key.startswith('input_embedding.'):
                torch.testing.assert_close(saved['states'][0][key], value, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, saved['states'][0][key])
                            for key, value in initial.items() if key.startswith('tcm_encoder.')))
        run = runtime.read_json(output / 'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['final_evaluation_status'], 'COMPLETED')
        self.assertFalse(run['test_opened'])
        self.assertFalse(run['selection_performed'])
        report = runtime.read_json(output / 'completion-evaluation.json')
        self.assertEqual(report['scene_count'], 2)
        self.assertEqual(report['fragment_count'], 4)
        self.assertEqual([row['scene_id'] for row in report['per_scene']], ['2', '4'])
        self.assertEqual(report['per_scene'][1]['fragment_count'], 0)
        self.assertEqual(report['per_scene'][1]['reference_targets_without_input'], ['target4a-0'])
        self.assertEqual(report['no_complete_reference_fragment_count'], 2)
        self.assertEqual(report['query']['model']['regions']['all']['query_count'], 256)
        self.assertEqual(report['query']['model']['regions']['all']['missing_reference_query_count'], 128)
        self.assertFalse((output / 'predictions').exists())
        self.assertTrue(any('/真实场景/' in name for name in reads))
        self.assertEqual(runtime.read_json(output / 'preparation.json')['dataset']['data_config'], fixture.data)
