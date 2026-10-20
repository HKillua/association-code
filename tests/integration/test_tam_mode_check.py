"""Native observation-only selected-checkpoint CLI, including sealed ZIP reads."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
from dataclasses import asdict
import io
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from association.models.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.models.timed import TimedAssociationModel
from association.models.tmtaff import ModelConfig
from association.training.common import fit_normalizer
from association.training.tmtaff import STAGES, _scenes, _states, load_complete_snapshot
from association.utils import runtime
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.experiments import tam_mode_check as cli


class TamModeCheckCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.fixture = make_native_zip_fixture(self)
        self.root = self.fixture.root
        dataset = self.fixture.load()
        normalizer = fit_normalizer(dataset['train'])
        config = copy.deepcopy(self.fixture.config['training'])
        config.update(branch='contrastive', updates={s: 2 if s == 'TAM_FROZEN' else 0 for s in STAGES})
        mc = ModelConfig(**{**self.fixture.config['model'], 'dropout': .2, 'margin': 7.})
        fc = PairFusionConfig(**self.fixture.config['fusion'])
        model = TimedAssociationModel(mc, 'contrastive', max_observations=4, normalizer=normalizer)
        fusion = PairFusion(fc, normalizer)
        objective = FusionLoss(position_loss_mode=config['position_loss_mode'])
        states = _states((model, fusion, objective))
        # Synthetic completed TAM-only snapshot; deliberately not a full run.
        self.saved = dict(format_version=1, input_adapter='native_time_v1', max_observations=4,
            model_config=asdict(mc), fusion_config=asdict(fc), config=config, normalizer=normalizer,
            data_contract=_scenes(dataset['train'], dataset['validation'], 4),
            selected=dict(states=states, threshold=-.3, stage='TAM_FROZEN', update=1,
                          effective_margin=7., macro_f1=.5),
            cursor=dict(stage=1, update=2, completed_updates=2),
            history=[dict(stage='TAM_FROZEN', update=i) for i in (1, 2)],
            states=[{k: torch.zeros_like(v) for k, v in state.items()} for state in states])
        self.checkpoint = self.root/'checkpoint.pt'
        torch.save(self.saved, self.checkpoint)
        runtime.write_json(self.root/'manifest.json', self.fixture.manifest)
        # References become unreadable after preparing the checkpoint contract.
        with zipfile.ZipFile(self.fixture.path) as archive:
            members = [(name, b'NO REFERENCE READS' if '/真实场景/' in name else archive.read(name))
                       for name in archive.namelist()]
        with zipfile.ZipFile(self.fixture.path, 'w') as archive:
            for name, content in members:
                archive.writestr(name, content)

    def args(self, name='diagnostic'):
        return ['--archive', str(self.fixture.path), '--manifest', str(self.root/'manifest.json'),
            '--checkpoint', str(self.checkpoint), '--output', str(self.root/name), '--device', 'cpu',
            '--cpu-threads', '1', '--advisory-seconds', '.000001', '--expected-selected-update', '1']

    def invoke(self, args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cli.main(args)

    def test_native_zip_selected_checkpoint_no_reference_training_or_calibration(self):
        before = self.checkpoint.read_bytes()
        read_names = []
        original_read = zipfile.ZipFile.read

        def guarded_read(archive, name, *args, **kwargs):
            member = name.filename if isinstance(name, zipfile.ZipInfo) else name
            self.assertNotIn('/真实场景/', member)
            self.assertTrue(member.endswith('-0.csv') or member.endswith('-2.csv'))
            read_names.append(member)
            return original_read(archive, name, *args, **kwargs)

        original_rng = torch.get_rng_state().clone()
        with ExitStack() as stack:
            for forbidden in ('association.training.tmtaff.train_complete',
                    'association.training.tmtaff._optimizer',
                    'association.training.tmtaff.calibrate_training_margin',
                    'association.training.common.fit_normalizer',
                    'association.data.timed.fit_timed_normalizer',
                    'association.evaluation.association.select_threshold',
                    'association.data.archive.MTADArchive.archive_sha256',
                    'torch.optim.Adam', 'torch.optim.AdamW', 'torch.optim.SGD'):
                stack.enter_context(patch(forbidden, side_effect=AssertionError('forbidden: '+forbidden)))
            restored = stack.enter_context(patch('association.training.tmtaff.load_complete_snapshot',
                                                 wraps=load_complete_snapshot))
            stack.enter_context(patch.object(zipfile.ZipFile, 'read', guarded_read))
            self.assertEqual(self.invoke(self.args()), 0)
            restored.assert_called_once_with(str(self.checkpoint), device=torch.device('cpu'), selected=True)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(len(read_names), 4)  # Two labels and two measurements only.
        self.assertEqual(before, self.checkpoint.read_bytes())
        run = runtime.read_json(self.root/'diagnostic'/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['optimizer_updates'], 0)
        self.assertTrue(run['advisory_exceeded'])
        self.assertEqual(run['selected_scene_ids'], ['0', '2'])
        summary = runtime.read_json(self.root/'diagnostic'/'summary.json')
        self.assertEqual(summary['source_selection']['update'], 1)
        self.assertEqual(summary['repeats'], 8)
        self.assertEqual(summary['effective_margin'], 7.)
        self.assertEqual(summary['normalizer'], self.saved['normalizer'])
        self.assertEqual([row['shortfall'] for row in summary['preparation']['populations']], [1, 1])
        self.assertEqual([sample['candidate_pair_indices'] for sample in summary['samples']],
                         [[1, 4, 0, 2, 3, 5]]*2)
        self.assertTrue(all(summary['state_restoration'].values()))
        with np.load(self.root/'diagnostic'/'distances.npz', allow_pickle=False) as arrays:
            self.assertEqual(arrays['train_sample_0_distances'].shape, (8, 6))
            self.assertEqual(arrays['eval_default_fastpath_sample_0_distances'].shape, (6,))
            self.assertTrue(np.isfinite(arrays['eval_default_fastpath_sample_0_distances']).all())
            np.testing.assert_array_equal(arrays['sample_0_labels'], [1, 1, 0, 0, 0, 0])
            for mode in ('eval', 'train', 'train_all_dropout_zero'):
                d = arrays[f'{mode}_sample_0_distances']
                # Independent class-balanced original first-power CL recomputation.
                expected = .25*(d[:, :2].mean(1)+(np.maximum(7.-d[:, 2:], 0.)**2).mean(1))
                np.testing.assert_allclose(summary['modes'][mode]['by_split']['T_core'][
                    'class_balanced_loss_per_repeat'], expected, rtol=2e-7)

    def test_contract_mismatch_fails_without_success_summary(self):
        arguments = self.args('wrong-update')
        arguments[-1] = '2'
        self.assertEqual(self.invoke(arguments), 1)
        wrong = runtime.read_json(self.root/'wrong-update'/'run.json')
        self.assertEqual(wrong['status'], 'FAILED')
        self.assertIn('expected-selected-update', wrong['failure'])
        self.assertFalse((self.root/'wrong-update'/'summary.json').exists())
        self.saved['data_contract'][0][0]['node_ids'][0] = 'wrong-source-node'
        torch.save(self.saved, self.checkpoint)
        self.assertEqual(self.invoke(self.args('mismatch')), 1)
        run = runtime.read_json(self.root/'mismatch'/'run.json')
        self.assertEqual(run['status'], 'FAILED')
        self.assertIn('observation contract differs', run['failure'])
        self.assertFalse((self.root/'mismatch'/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
