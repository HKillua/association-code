"""Explicit TCM endpoint uses the original updates and all native V scenes."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import unittest
from unittest.mock import patch

import torch

from association.models.fusion import PairFusionConfig
from association.models.tmtaff import ModelConfig
from association.training.calibration import MARGIN_POLICY
from association.training.tmtaff import train_complete, load_complete_snapshot
from association.utils import runtime
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.experiments import tmtaff_complete_fit as cli


class TcmFitCliTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.fixture = make_native_zip_fixture(self)
        self.root = self.fixture.root
        self.fixture.write_zip(missing_reference=('2', 'b'))
        self.fixture.config['model']['dropout'] = .1
        self.fixture.config['fusion']['dropout'] = .1
        self.fixture.config['training'].update(branch='contrastive', margin_policy=MARGIN_POLICY)
        self.fixture.config['training']['updates']['TCM'] = 2
        self.full_config = copy.deepcopy(self.fixture.config)
        self.fixture.config['training']['updates'] = dict(TCM=2, TAM_FROZEN=0, TFM_FROZEN=0, JOINT=0)

    def invoke(self, name, extra=()):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cli.main(['--phase', 'tcm', '--device', 'cpu',
                             *self.fixture.cli_args(name), *extra])

    def checkpoint(self, name):
        return torch.load(self.root/name/'checkpoint.pt', map_location='cpu', weights_only=True)

    def fit(self, name, *, full=False, **kwargs):
        dataset = self.fixture.load()
        config = self.full_config if full else self.fixture.config
        return train_complete(dataset['train'], dataset['validation'], ModelConfig(**config['model']),
            PairFusionConfig(**config['fusion']), config['training'], self.root/name,
            phase='full' if full else 'tcm', **kwargs)

    def assert_same_endpoint(self, left, right):
        self.assertEqual([row['loss'] for row in left['history']], [row['loss'] for row in right['history']])
        self.assertEqual(left['coverage'], right['coverage'])
        self.assertEqual(left['normalizer'], right['normalizer'])
        self.assertEqual(left['data_contract'], right['data_contract'])
        for a, b in zip(left['states'], right['states']):
            self.assertEqual(set(a), set(b))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        torch.testing.assert_close(left['rng']['torch'], right['rng']['torch'], rtol=0, atol=0)
        self.assertEqual(left['rng']['tracks'], right['rng']['tracks'])

    def test_native_cli_completes_without_downstream_and_matches_original_tcm(self):
        with ExitStack() as stack:
            for name in ('score_reference', 'select_threshold', 'evaluate_complete',
                         'calibrate_training_margin', 'fusion_training_step', 'FrozenCompletionCache'):
                stack.enter_context(patch('association.training.tmtaff.'+name,
                                          side_effect=AssertionError('downstream forbidden: '+name)))
            self.assertEqual(self.invoke('tcm'), 0)
            # Reaching the same TCM boundary in the default workflow must still
            # stop before its separately planned association/fusion updates.
            self.assertEqual(self.fit('full-prefix', full=True, max_updates_this_call=2)['status'], 'INTERRUPTED')
        saved = self.checkpoint('tcm')
        self.assert_same_endpoint(saved, self.checkpoint('full-prefix'))
        self.assertEqual(saved['phase'], 'tcm')
        self.assertEqual(saved['training_stages'], ['TCM'])
        self.assertEqual(saved['cursor'], dict(stage=0, update=2, completed_updates=2))
        self.assertIsNone(saved['selected'])
        self.assertIsNone(saved['margin_calibration'])
        self.assertIsNotNone(saved['optimizer'])
        loaded = load_complete_snapshot(self.root/'tcm'/'checkpoint.pt', selected=False)
        for key, expected in saved['states'][0].items():
            torch.testing.assert_close(loaded['model'].state_dict()[key], expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'no selected'):
            load_complete_snapshot(self.root/'tcm'/'checkpoint.pt')

        run = runtime.read_json(self.root/'tcm'/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['cursor']['completed_updates'], 2)
        self.assertEqual(run['final_evaluation_status'], 'COMPLETED')
        self.assertFalse(run['selection_performed'])
        self.assertIsNone(run['margin_calibration'])
        self.assertEqual(run['validation_timings'], [])
        self.assertFalse(run['test_opened'])
        report = runtime.read_json(self.root/'tcm'/'completion-evaluation.json')
        self.assertEqual(report['scene_count'], 2)
        self.assertEqual(report['fragment_count'], 4)
        self.assertEqual([row['scene_id'] for row in report['per_scene']], ['2', '4'])
        self.assertEqual(report['per_scene'][1]['fragment_count'], 0)
        self.assertEqual(report['per_scene'][1]['reference_targets_without_input'], ['target4a-0'])
        self.assertEqual(report['no_complete_reference_fragment_count'], 2)
        self.assertEqual(report['one_step']['model']['all']['query_count'], 16)
        self.assertEqual(report['query']['model']['regions']['all']['query_count'], 16)
        self.assertEqual(report['query']['model']['regions']['all']['missing_reference_query_count'], 8)
        for row in report['per_fragment']:
            if row['identity'] == 'target2b-0':
                self.assertIsNone(row['query']['model']['regions']['all']['feature_mse'])
        summary = runtime.read_json(self.root/'tcm'/'summary.json')
        self.assertEqual(summary['completion_evaluation'], 'completion-evaluation.json')
        self.assertNotIn('selection', summary)
        self.assertNotIn('threshold', summary)
        self.assertFalse((self.root/'tcm'/'predictions').exists())
        preparation = runtime.read_json(self.root/'tcm'/'preparation.json')
        self.assertEqual(preparation['phase'], 'tcm')

    def test_same_phase_resume_and_failed_diagnostic_cannot_be_called_complete(self):
        self.assertEqual(self.fit('once')['status'], 'COMPLETED')
        self.assertEqual(self.fit('partial', max_updates_this_call=1)['status'], 'INTERRUPTED')
        self.assertFalse((self.root/'partial'/'completion-evaluation.json').exists())
        self.assertEqual(self.invoke('resumed', ['--resume', str(self.root/'partial')]), 0)
        self.assert_same_endpoint(self.checkpoint('once'), self.checkpoint('resumed'))
        with self.assertRaisesRegex(ValueError, 'resume phase differs'):
            self.fit('cross-phase', full=True, resume_from=self.root/'partial')
        self.assertFalse((self.root/'cross-phase').exists())
        with patch('association.training.tmtaff.evaluate_tcm_completion',
                   side_effect=RuntimeError('diagnostic failed')):
            self.assertEqual(self.invoke('failed'), 1)
        failed = runtime.read_json(self.root/'failed'/'run.json')
        self.assertEqual(failed['status'], 'FAILED')
        self.assertEqual(failed['final_evaluation_status'], 'FAILED')
        self.assertEqual(failed['cursor']['completed_updates'], 2)
        self.assertEqual(self.checkpoint('failed')['cursor']['completed_updates'], 2)
        self.assertFalse((self.root/'failed'/'summary.json').exists())
        self.assertEqual(self.invoke('retry', ['--resume', str(self.root/'failed')]), 1)
        self.assertFalse((self.root/'retry').exists())

    def test_explicit_phase_and_zero_downstream_updates_are_required_before_data_read(self):
        with patch.object(cli, 'load_timed_dataset', side_effect=AssertionError('must reject before data')):
            self.assertEqual(self.invoke('predictions', ['--save-predictions']), 1)
            self.assertEqual(self.invoke('full-zero', ['--phase', 'full']), 1)
            self.fixture.config['training']['updates']['TAM_FROZEN'] = 1
            self.assertEqual(self.invoke('nonzero-downstream'), 1)
        for name in ('predictions', 'full-zero', 'nonzero-downstream'):
            self.assertFalse((self.root/name).exists())


if __name__ == '__main__':
    unittest.main()
