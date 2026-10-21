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
        # Old checkpoints predate the explicit default representation/sampling fields.
        legacy = self.checkpoint('partial')
        legacy['model_config'].pop('heading_representation', None)
        legacy['config'].pop('tcm_sampling')
        torch.save(legacy, self.root/'partial'/'checkpoint.pt')
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

    def test_ordered_tcm_covers_each_fragment_keeps_tail_and_replays_epoch_boundary(self):
        self.fixture.config['training'].update(tcm_sampling='ordered_fragment', batch_tracks=3)
        self.fixture.config['training']['updates']['TCM'] = 4
        self.fit('ordered')
        expected = self.checkpoint('ordered')
        rows = expected['history']
        self.assertEqual([row['track_presentations'] for row in rows], [3, 1, 3, 1])
        self.assertEqual([row['tcm_sampling']['epoch'] for row in rows], [1, 1, 2, 2])
        self.assertEqual([row['tcm_sampling']['batch_in_epoch'] for row in rows], [1, 2, 1, 2])
        self.assertEqual([(row['tcm_sampling']['pool_start'], row['tcm_sampling']['pool_stop'])
                          for row in rows], [(0, 3), (3, 4), (0, 3), (3, 4)])
        # Native lengths 4,2,4,2 give 2*(3+1+3+1)=16 transitions per pass.
        self.assertEqual([row['transition_presentations'] for row in rows], [14, 2, 14, 2])
        coverage = expected['coverage']['TCM']
        self.assertEqual(coverage['unique_track_count'], 4)
        self.assertEqual(coverage['track_presentations'], 8)
        self.assertEqual(coverage['transition_presentations'], 32)
        self.assertEqual(coverage['completed_epochs'], 2)
        # Interrupt immediately before the short tail and at the epoch boundary.
        for cut in (1, 2):
            self.fit(f'ordered-part-{cut}', max_updates_this_call=cut)
            self.fit(f'ordered-resumed-{cut}', resume_from=self.root/f'ordered-part-{cut}')
            resumed = self.checkpoint(f'ordered-resumed-{cut}')
            self.assert_same_endpoint(expected, resumed)
            self.assertEqual([row['tcm_sampling'] for row in rows],
                             [row['tcm_sampling'] for row in resumed['history']])
        self.fixture.config['training'].pop('tcm_sampling')
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            self.fit('sampling-changed', resume_from=self.root/'ordered-part-1')
        self.assertFalse((self.root/'sampling-changed').exists())

    def test_explicit_phase_and_zero_downstream_updates_are_required_before_data_read(self):
        with patch.object(cli, 'load_timed_dataset', side_effect=AssertionError('must reject before data')):
            self.assertEqual(self.invoke('predictions', ['--save-predictions']), 1)
            self.assertEqual(self.invoke('full-zero', ['--phase', 'full']), 1)
            self.fixture.config['training']['updates']['TAM_FROZEN'] = 1
            self.assertEqual(self.invoke('nonzero-downstream'), 1)
        for name in ('predictions', 'full-zero', 'nonzero-downstream'):
            self.assertFalse((self.root/name).exists())

    def tam_config(self):
        self.fixture.config['training']['updates'] = dict(TCM=0, TAM_FROZEN=2, TFM_FROZEN=0, JOINT=0)
        return self.fixture.config

    def tam_fit(self, name, **kwargs):
        dataset, config = self.fixture.load(), self.tam_config()
        return train_complete(dataset['train'], dataset['validation'], ModelConfig(**config['model']),
            PairFusionConfig(**config['fusion']), config['training'], self.root/name,
            phase='tam', **kwargs)

    def test_tam_cli_transfers_only_completion_and_reuses_all_validation_scores(self):
        self.fit('source', initial_elapsed_seconds=50.)
        source = self.checkpoint('source')
        # A TCM run contains unused random TAM weights. Deliberately distinguish
        # them from a fresh initialization to catch accidentally loading all states.
        for key, value in source['states'][0].items():
            if key.startswith('tam_encoder.'):
                value.add_(10.)
        source['model_config'].pop('heading_representation', None)
        torch.save(source, self.root/'source'/'checkpoint.pt')
        legacy = load_complete_snapshot(self.root/'source'/'checkpoint.pt', selected=False)
        self.assertEqual(legacy['model'].config.heading_representation, 'linear_zscore')
        self.tam_config()
        from association.training import tmtaff as training
        initial, score_calls = {}, []
        original_parameters, original_score = training._stage_parameters, training.score_reference

        def parameters(modules, stage):
            initial.update({k: v.detach().clone() for k, v in modules[0].state_dict().items()})
            return original_parameters(modules, stage)

        def score(*args, **kwargs):
            records = original_score(*args, **kwargs)
            score_calls.append(records)
            return records

        with (patch.object(training, '_stage_parameters', side_effect=parameters),
              patch.object(training, 'score_reference', side_effect=score),
              patch.object(training, 'evaluate_complete', side_effect=AssertionError('physical fusion forbidden')),
              patch.object(training, 'fusion_training_step', side_effect=AssertionError('TFM/JOINT forbidden'))):
            self.assertEqual(self.invoke('tam', ['--phase', 'tam', '--initialize-tcm-from',
                                                 str(self.root/'source')]), 0)
        saved = self.checkpoint('tam')
        self.assertEqual(saved['training_stages'], ['TAM_FROZEN'])
        self.assertEqual(saved['cursor'], dict(stage=1, update=2, completed_updates=2))
        self.assertEqual([row['stage'] for row in saved['history']], ['TAM_FROZEN'] * 2)
        self.assertEqual(saved['normalizer'], source['normalizer'])
        self.assertEqual(saved['coverage']['TCM']['track_presentations'], 0)
        self.assertEqual(len(score_calls), 2)  # No third inference at the endpoint.
        for key, value in saved['states'][0].items():
            if key == 'positions' or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
                torch.testing.assert_close(value, source['states'][0][key], rtol=0, atol=0)
            if key.startswith('tam_encoder.'):
                self.assertFalse(torch.equal(initial[key], source['states'][0][key]))
        self.assertTrue(any(not torch.equal(initial[k], v)
                            for k, v in saved['states'][0].items() if k.startswith('tam_encoder.')))
        run = runtime.read_json(self.root/'tam'/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['margin_calibration']['status'], 'COMPLETED')
        self.assertIsNone(run['resume_from'])
        self.assertLess(run['elapsed_seconds'], 50.)
        summary = runtime.read_json(self.root/'tam'/'summary.json')
        self.assertFalse(summary['evaluation']['physical_fusion_evaluated'])
        association = summary['evaluation']['association']
        self.assertEqual(association['coverage']['all_scenes'], 2)
        self.assertEqual(association['pair_count'], 6)
        self.assertIn('paper_metrics', association['itsa'])
        self.assertIn('paper_metrics', association['mstsa'])
        self.assertEqual(len(summary['validation_score_records']), 2)
        for entry, expected in zip(summary['validation_score_records'], score_calls):
            self.assertEqual(entry['association']['coverage']['all_scenes'], 2)
            self.assertIn('paper_metrics', entry['association']['mstsa'])
            records = runtime.load_records(entry['records'])
            self.assertEqual([row['scene_id'] for row in records], ['2', '4'])
            for actual, original in zip(records, expected):
                self.assertEqual(actual['scores'].tolist(), original['scores'].tolist())
        self.assertFalse(run['test_opened'])

    def test_circular_tcm_transfers_frozen_representation_and_snapshot_rejects_mismatch(self):
        from association.training.common import score_reference
        self.fixture.config['model']['heading_representation'] = 'circular_sincos'
        self.assertEqual(self.fit('circular-source')['status'], 'COMPLETED')
        source = self.checkpoint('circular-source')
        self.assertEqual(source['states'][0]['input_embedding.weight'].shape[1], 5)
        self.assertIn('heading_mean_std', source['states'][0])
        self.assertEqual(self.tam_fit('circular-tam',
            initialize_tcm_from=self.root/'circular-source')['status'], 'COMPLETED')
        saved = self.checkpoint('circular-tam')
        self.assertIn('heading_mean_std', saved['initialization']['transferred_buffers'])
        for key, value in saved['states'][0].items():
            if key in ('positions', 'heading_mean_std') or key.startswith(
                    ('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
                torch.testing.assert_close(value, source['states'][0][key], rtol=0, atol=0)
        loaded = load_complete_snapshot(self.root/'circular-tam'/'checkpoint.pt')
        loaded['model'].validate_normalizer(source['normalizer'])
        dataset = self.fixture.load()
        batches = saved['config']['validation_batches']
        actual = score_reference(loaded['model'], loaded['normalizer'],
                                 dataset['validation'], 'cpu', batches['tracks'], batches['pairs'])
        summary = runtime.read_json(self.root/'circular-tam'/'summary.json')
        selected = next(row for row in summary['validation_score_records']
                        if row['update'] == saved['selected']['update'])
        expected = runtime.load_records(selected['records'])
        for left, right in zip(actual, expected):
            self.assertEqual(left['scores'].tolist(), right['scores'].tolist())

        changed = copy.deepcopy(loaded['normalizer'])
        changed['mean'][3] += 1.
        with self.assertRaisesRegex(ValueError, 'normalizer.*heading'):
            score_reference(loaded['model'], changed, dataset['validation'], 'cpu', 2, 3)
        corrupted = copy.deepcopy(saved)
        corrupted['normalizer'] = changed
        torch.save(corrupted, self.root/'circular-tam'/'mismatched.pt')
        with self.assertRaisesRegex(ValueError, 'normalizer.*heading'):
            load_complete_snapshot(self.root/'circular-tam'/'mismatched.pt')
        self.fixture.config['model']['heading_representation'] = 'linear_zscore'
        with self.assertRaisesRegex(ValueError, 'TCM initialization model'):
            self.tam_fit('different-representation', initialize_tcm_from=self.root/'circular-source')
        self.assertFalse((self.root/'different-representation').exists())

    def test_tam_resume_keeps_frozen_source_and_new_budget_and_rejects_changed_contract(self):
        self.fit('source')
        self.tam_fit('once', initialize_tcm_from=self.root/'source')
        self.tam_fit('partial', initialize_tcm_from=self.root/'source', max_updates_this_call=1)
        legacy = self.checkpoint('partial')
        legacy['model_config'].pop('tam_dropout')
        legacy['model_config'].pop('contrastive_negative_scale')
        legacy.pop('split_unit')
        torch.save(legacy, self.root/'partial'/'checkpoint.pt')
        self.assertEqual(self.tam_fit('resumed', resume_from=self.root/'partial')['status'], 'COMPLETED')
        expected, actual = self.checkpoint('once'), self.checkpoint('resumed')
        for left, right in zip(expected['states'], actual['states']):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        self.assertEqual([row['loss'] for row in expected['history']],
                         [row['loss'] for row in actual['history']])
        self.assertEqual(expected['initialization'], actual['initialization'])
        self.fixture.alignment['2']['time_offset_seconds'] += 1.
        with self.assertRaisesRegex(ValueError, 'data contract or normalizer differs'):
            self.tam_fit('different-data', initialize_tcm_from=self.root/'source')
        self.assertFalse((self.root/'different-data').exists())
        self.fixture.alignment['2']['time_offset_seconds'] -= 1.
        source_run = runtime.read_json(self.root/'source'/'run.json')
        source_run['status'] = 'FAILED'
        runtime.write_json(self.root/'source'/'run.json', source_run)
        with self.assertRaisesRegex(ValueError, 'COMPLETED TCM-only'):
            self.tam_fit('failed-source', initialize_tcm_from=self.root/'source')
        self.assertFalse((self.root/'failed-source').exists())

    def test_tam_zero_dropout_transfers_legacy_completion_and_preserves_resume_semantics(self):
        self.fit('source')
        source = self.checkpoint('source')
        source['model_config'].pop('tam_dropout')
        torch.save(source, self.root/'source'/'checkpoint.pt')
        self.fixture.config['model']['tam_dropout'] = 0.
        self.tam_fit('zero', initialize_tcm_from=self.root/'source')
        self.tam_fit('zero-partial', initialize_tcm_from=self.root/'source', max_updates_this_call=1)
        self.assertEqual(self.tam_fit('zero-resumed', resume_from=self.root/'zero-partial')['status'], 'COMPLETED')
        expected, resumed = self.checkpoint('zero'), self.checkpoint('zero-resumed')
        self.assert_same_endpoint(expected, resumed)
        self.assertEqual(expected['initialization']['source_tam_dropout'], .1)
        self.assertEqual(expected['initialization']['target_tam_dropout'], 0.)
        self.assertEqual(expected['margin_calibration']['tam_dropout'], 0.)
        self.assertEqual(expected['model_config']['dropout'], .1)
        for key, value in expected['states'][0].items():
            if key == 'positions' or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
                torch.testing.assert_close(value, source['states'][0][key], rtol=0, atol=0)
        for row in expected['history']:
            diagnostic = row['tam_train_batch']
            self.assertEqual(diagnostic['effective_margin'], expected['effective_margin'])
            count = diagnostic['positive_count'] + diagnostic['negative_count']
            weighted = sum(diagnostic[f'{kind}_class_loss_mean'] * diagnostic[f'{kind}_count']
                           for kind in ('positive', 'negative')) / count
            self.assertAlmostEqual(weighted, row['association_loss'], places=6)
            self.assertGreaterEqual(diagnostic['negative_fraction_inside_margin'], 0.)
            self.assertLessEqual(diagnostic['negative_fraction_inside_margin'], 1.)
        for selected in (True, False):
            loaded = load_complete_snapshot(self.root/'zero'/'checkpoint.pt', selected=selected)
            self.assertEqual(loaded['model'].config.effective_tam_dropout, 0.)
            self.assertEqual(loaded['model'].tam_encoder.layers[0].self_attn.dropout, 0.)
            self.assertEqual(loaded['model'].tcm_encoder.layers[0].self_attn.dropout, .1)
            self.assertEqual(loaded['fusion'].config.dropout, .1)
        self.fixture.config['model'].pop('tam_dropout')
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            self.tam_fit('changed-dropout-resume', resume_from=self.root/'zero-partial')
        self.assertFalse((self.root/'changed-dropout-resume').exists())
        self.fixture.config['model']['tam_dropout'] = 0.
        self.fixture.config['model']['dropout'] = 0.
        with self.assertRaisesRegex(ValueError, 'TCM initialization model'):
            self.tam_fit('changed-completion-dropout', initialize_tcm_from=self.root/'source')
        self.assertFalse((self.root/'changed-completion-dropout').exists())

    def test_inverse_margin_tam_transfer_resume_diagnostics_and_protocol_boundaries(self):
        self.fit('source')
        source = self.checkpoint('source')
        source['model_config'].pop('contrastive_negative_scale')
        source.pop('split_unit')
        torch.save(source, self.root/'source'/'checkpoint.pt')
        self.fixture.config['model'].update(tam_dropout=0., contrastive_negative_scale='inverse_margin')
        self.tam_fit('scaled', initialize_tcm_from=self.root/'source')
        self.tam_fit('scaled-partial', initialize_tcm_from=self.root/'source', max_updates_this_call=1)
        self.tam_fit('scaled-resumed', resume_from=self.root/'scaled-partial')
        expected, resumed = self.checkpoint('scaled'), self.checkpoint('scaled-resumed')
        self.assert_same_endpoint(expected, resumed)
        self.assertEqual(expected['effective_margin'], resumed['effective_margin'])
        self.assertEqual(expected['initialization']['source_contrastive_negative_scale'], 'paper')
        self.assertEqual(expected['initialization']['target_contrastive_negative_scale'], 'inverse_margin')
        self.assertEqual(expected['margin_calibration']['contrastive_negative_scale'], 'inverse_margin')
        for key, value in expected['states'][0].items():
            if key == 'positions' or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
                torch.testing.assert_close(value, source['states'][0][key], rtol=0, atol=0)
        for row in expected['history']:
            diagnostic = row['tam_train_batch']
            self.assertEqual(diagnostic['contrastive_negative_scale'], 'inverse_margin')
            count = diagnostic['positive_count'] + diagnostic['negative_count']
            weighted = sum(diagnostic[f'{kind}_class_loss_mean'] * diagnostic[f'{kind}_count']
                           for kind in ('positive', 'negative')) / count
            self.assertAlmostEqual(weighted, row['association_loss'], places=6)
        for selected in (True, False):
            loaded = load_complete_snapshot(self.root/'scaled'/'checkpoint.pt', selected=selected)
            self.assertEqual(loaded['model'].config.contrastive_negative_scale, 'inverse_margin')

        # Same scenes and tensors cannot silently change the split protocol on
        # resume or during a TCM transfer; different scene lists also remain bound.
        dataset = self.fixture.load()
        for scene in [*dataset['train'], *dataset['validation']]:
            scene['split_unit'] = 'scene'
        config = self.tam_config()
        for name, options in (
                ('protocol-resume', dict(resume_from=self.root/'scaled-partial')),
                ('protocol-transfer', dict(initialize_tcm_from=self.root/'source'))):
            with self.assertRaisesRegex(ValueError, 'resume configuration|TCM initialization model'):
                train_complete(dataset['train'], dataset['validation'], ModelConfig(**config['model']),
                    PairFusionConfig(**config['fusion']), config['training'], self.root/name,
                    phase='tam', **options)
            self.assertFalse((self.root/name).exists())
        self.fixture.config['model'].pop('contrastive_negative_scale')
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            self.tam_fit('changed-objective', resume_from=self.root/'scaled-partial')
        self.assertFalse((self.root/'changed-objective').exists())

    def tfm_config(self):
        self.fixture.config['training']['updates'] = dict(TCM=0, TAM_FROZEN=0, TFM_FROZEN=2, JOINT=0)
        self.fixture.config['training']['fusion_monitor_updates'] = [0, 1, 2]
        return self.fixture.config

    def tfm_fit(self, name, **kwargs):
        dataset, config = self.fixture.load(), self.tfm_config()
        return train_complete(dataset['train'], dataset['validation'], ModelConfig(**config['model']),
            PairFusionConfig(**config['fusion']), config['training'], self.root/name,
            phase='tfm', **kwargs)

    def test_tfm_transfers_frozen_completion_trains_true_pairs_and_reports_oracle_common_support(self):
        self.fit('source', initial_elapsed_seconds=50.)
        source = self.checkpoint('source')
        self.tfm_config()
        from association.training import tmtaff as training
        original_evaluate, original_step = training.evaluate_complete, training.fusion_training_step
        initial, steps = {}, []

        def evaluate(model, fusion, *args, **kwargs):
            self.assertTrue(kwargs['oracle_only'])
            self.assertFalse(kwargs['select_on_validation'])
            self.assertIsNone(kwargs['threshold'])
            if not initial:
                initial['model'] = copy.deepcopy(model.state_dict())
                initial['fusion'] = copy.deepcopy(fusion.state_dict())
            with patch.object(model.tam_encoder, 'forward', side_effect=AssertionError('TAM unused')):
                return original_evaluate(model, fusion, *args, **kwargs)

        def step(*args, **kwargs):
            self.assertEqual(kwargs['stage'], 'TFM_FROZEN')
            self.assertTrue((kwargs['labels'] == 1).all())
            if not steps:
                self.assertEqual(len(args[3].state), 0)  # A fresh optimizer, not the TCM state.
            steps.append(kwargs['stage'])
            return original_step(*args, **kwargs)

        with (patch.object(training, 'evaluate_complete', side_effect=evaluate),
              patch.object(training, 'fusion_training_step', side_effect=step),
              patch.object(training, 'score_reference', side_effect=AssertionError('no association scoring')),
              patch.object(training, 'calibrate_training_margin', side_effect=AssertionError('no calibration')),
              patch.object(training, 'select_threshold', side_effect=AssertionError('no selection'))):
            self.assertEqual(self.invoke('tfm', ['--phase', 'tfm', '--initialize-tcm-from',
                                               str(self.root/'source'), '--save-predictions']), 0)
        saved = self.checkpoint('tfm')
        self.assertEqual(saved['cursor'], dict(stage=2, update=2, completed_updates=2))
        self.assertEqual(saved['training_stages'], ['TFM_FROZEN'])
        self.assertEqual(saved['normalizer'], source['normalizer'])
        self.assertIsNone(saved['margin_calibration'])
        self.assertEqual(saved['selected']['endpoint_policy'], 'fixed_final')
        self.assertEqual(saved['selected']['update'], 2)
        self.assertEqual(steps, ['TFM_FROZEN', 'TFM_FROZEN'])
        for key, value in saved['states'][0].items():
            torch.testing.assert_close(value, initial['model'][key], rtol=0, atol=0)
            if key == 'positions' or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
                torch.testing.assert_close(value, source['states'][0][key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, initial['fusion'][key])
                            for key, value in saved['states'][1].items()))
        summary = runtime.read_json(self.root/'tfm'/'summary.json')
        self.assertFalse(summary['selection_performed'])
        self.assertEqual(summary['fixed_final_update'], 2)
        self.assertEqual([row['update'] for row in summary['fusion_evaluations']], [0, 1, 2])
        for row in summary['fusion_evaluations']:
            self.assertEqual(row['status'], 'COMPLETED')
            evaluation = runtime.read_json(row['evaluation'])
            self.assertEqual(len(evaluation['per_scene']), 2)
            self.assertNotIn('predicted', evaluation['fusion'])
            self.assertIsNone(evaluation['association'])
            common = evaluation['oracle_common_support']['fusion']
            for feature in ('position', 'speed', 'heading'):
                self.assertEqual(common['oracle']['all'][feature]['intersection_count'],
                                 common['simple_fusion_oracle']['all'][feature]['intersection_count'])
                self.assertIsNone(common['oracle']['missing'][feature]['mae'])
            index = runtime.read_json(row['predictions'])
            self.assertEqual(len(index['scenes']), 2)
            endpoint = torch.load(self.root/'tfm'/'snapshots'/f"update-{row['update']:06d}.pt",
                                  weights_only=True)
            self.assertEqual(endpoint['cursor']['update'], row['update'])
        run = runtime.read_json(self.root/'tfm'/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertLess(run['elapsed_seconds'], 50.)
        self.assertFalse(run['test_opened'])
        loaded = load_complete_snapshot(self.root/'tfm'/'checkpoint.pt')
        for key, value in loaded['fusion'].state_dict().items():
            torch.testing.assert_close(value, saved['states'][1][key], rtol=0, atol=0)

    def test_tfm_resume_replays_same_updates_without_repeating_zero_evaluation(self):
        self.fit('source')
        self.tfm_fit('once', initialize_tcm_from=self.root/'source')
        self.tfm_fit('partial', initialize_tcm_from=self.root/'source', max_updates_this_call=1)
        self.assertEqual(self.tfm_fit('resumed', resume_from=self.root/'partial')['status'], 'COMPLETED')
        expected, resumed = self.checkpoint('once'), self.checkpoint('resumed')
        self.assert_same_endpoint(expected, resumed)
        self.assertEqual([row['update'] for row in resumed['fusion_evaluations']], [0, 1, 2])
        self.fixture.config['training']['fusion_monitor_updates'] = [0, 1]
        with patch.object(cli, 'load_timed_dataset', side_effect=AssertionError('reject before data')):
            self.assertEqual(self.invoke('invalid', ['--phase', 'tfm', '--initialize-tcm-from',
                                                    str(self.root/'source')]), 1)
            self.assertEqual(self.invoke('missing-source', ['--phase', 'tfm']), 1)

    def test_tfm_failed_final_diagnostic_keeps_fixed_endpoint_without_claiming_completion(self):
        self.fit('source')
        from association.training import tmtaff as training
        original = training.evaluate_complete
        calls = 0
        def evaluate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('final physical diagnostic failed')
            return original(*args, **kwargs)
        with patch.object(training, 'evaluate_complete', side_effect=evaluate):
            with self.assertRaisesRegex(RuntimeError, 'final physical diagnostic failed'):
                self.tfm_fit('failed-tfm', initialize_tcm_from=self.root/'source')
        run = runtime.read_json(self.root/'failed-tfm'/'run.json')
        self.assertEqual(run['status'], 'FAILED')
        self.assertEqual(run['final_evaluation_status'], 'FAILED')
        self.assertEqual(self.checkpoint('failed-tfm')['cursor']['completed_updates'], 2)
        self.assertTrue((self.root/'failed-tfm'/'snapshots'/'update-000002.pt').is_file())
        self.assertFalse((self.root/'failed-tfm'/'summary.json').exists())

    def test_tam_cli_requires_explicit_new_initialization_and_zero_other_stages(self):
        self.tam_config()
        with patch.object(cli, 'load_timed_dataset', side_effect=AssertionError('must reject before data')):
            self.assertEqual(self.invoke('missing-source', ['--phase', 'tam']), 1)
            self.assertEqual(self.invoke('both', ['--phase', 'tam', '--initialize-tcm-from', 'source',
                                                '--resume', 'interrupted']), 1)
            self.fixture.config['training']['updates']['TCM'] = 1
            self.assertEqual(self.invoke('nonzero-tcm', ['--phase', 'tam',
                                                       '--initialize-tcm-from', 'source']), 1)


if __name__ == '__main__':
    unittest.main()
