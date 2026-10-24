"""The frozen hold control changes only completion inputs, with auditable draws."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.models.fusion import PairFusionConfig
from association.models.timed import TimedAssociationModel
from association.models.tmtaff import AssociationModel, ModelConfig
from association.training.batches import FrozenCompletionCache
from association.training.calibration import MARGIN_POLICY
from association.training.common import normalize_scene
from association.training.tmtaff import load_complete_snapshot, train_complete
from association.utils import runtime
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.experiments import tmtaff_complete_fit as cli


class CompletionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def config(self, policy='learned'):
        return ModelConfig(length=6, d_model=8, nhead=2, num_layers=1,
                           dim_feedforward=16, dropout=.2, completion_policy=policy)

    def packed(self):
        # The third native track is wholly between query samples: retaining it
        # exercises completion without an artificial observed query slot.
        histories = ((1., 2., 4.), (1.5, 2.5), (2.1, 2.2))
        query = torch.arange(6, dtype=torch.float32)
        values = torch.full((3, 11, 5), float('nan'))
        observed = torch.zeros(3, 11, dtype=torch.bool)
        for row, history in enumerate(histories):
            times = torch.tensor(history)
            count = len(times)
            observed[row, :count] = True
            values[row, :count, 4] = times
            values[row, :count, :4] = torch.stack(
                (times * .1 + row, times * .2, times * 0 + 1, times * .3), dim=-1)
            values[row, 5:, 4] = query
            inside = (query >= times[0]) & (query <= times[-1])
            values[row, 5:, :4][inside] = torch.stack(
                (query[inside] * .1 + row, query[inside] * .2,
                 query[inside] * 0 + 1, query[inside] * .3), dim=-1)
        return values, observed

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_hold_uses_endpoint_state_at_query_tau_and_exact_inside_embedding(self):
        torch.manual_seed(31)
        model = TimedAssociationModel(self.config('state_hold'), max_observations=5).eval()
        values, observed = self.packed()
        rng = torch.get_rng_state().clone()
        with (patch.object(model.tcm_encoder, 'forward', side_effect=AssertionError('TCM unused')),
              patch.object(model.tcm_decoder, 'forward', side_effect=AssertionError('TCM unused'))):
            actual = model.complete(values, observed)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        inside_features = model.embed_inputs(values, observed)[:, 5:]
        for row, count in enumerate(observed[:, :5].sum(1).tolist()):
            for column, tau in enumerate(values[row, 5:, 4]):
                if tau < values[row, 0, 4]:
                    endpoint = 0
                elif tau > values[row, count - 1, 4]:
                    endpoint = count - 1
                else:
                    torch.testing.assert_close(actual[row, column], inside_features[row, column],
                                               rtol=0, atol=0)
                    continue
                expected = model.embed_states(values[row, endpoint, :4], tau)
                # Scalar GEMV and batched GEMM may round the shared projection
                # differently; only the unchanged inside path is bit-exact.
                torch.testing.assert_close(actual[row, column], expected, rtol=1e-6, atol=1e-7)
        # The same endpoint state receives a new physical time signal.
        self.assertFalse(torch.equal(actual[2, 0], actual[2, 1]))
        old_latent = model.embed_states(values[0, 0, :4], values[0, 0, 4])
        self.assertFalse(torch.equal(actual[0, 0], old_latent))
        model.config = replace(model.config, completion_policy='learned')
        with torch.no_grad():
            learned = model.complete(values, observed)
        for row, count in enumerate(observed[:, :5].sum(1).tolist()):
            inside = ((values[row, 5:, 4] >= values[row, 0, 4]) &
                      (values[row, 5:, 4] <= values[row, count - 1, 4]))
            torch.testing.assert_close(actual[row, inside], learned[row, inside], rtol=0, atol=0)

    def test_hold_ignores_outside_query_contents_and_padding_without_gradients(self):
        model = TimedAssociationModel(self.config('state_hold'), max_observations=5).eval()
        values, observed = self.packed()
        expected = model.complete(values, observed)
        changed = values.clone()
        padding = ~observed[:, :5]
        changed[:, :5][padding] = float('inf')
        outside = ~torch.isfinite(values[:, 5:, :4]).all(-1)
        changed[:, 5:, :4][outside] = float('inf')
        changed.requires_grad_()
        actual = model.complete(changed, observed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(changed.grad).all())
        self.assertEqual(torch.count_nonzero(changed.grad[:, :5][padding]).item(), 0)
        self.assertEqual(torch.count_nonzero(changed.grad[:, 5:, :4][outside]).item(), 0)
        self.assertGreater(torch.count_nonzero(changed.grad[:, 0, :4]).item(), 0)

    def test_policy_preserves_parameter_initialization_and_rng(self):
        endpoints = []
        for policy in ('learned', 'state_hold'):
            torch.manual_seed(83)
            model = TimedAssociationModel(self.config(policy), max_observations=5)
            endpoints.append((copy.deepcopy(model.state_dict()), torch.get_rng_state().clone(), torch.rand(5)))
        self.assert_nested_equal(endpoints[0], endpoints[1])

    def test_hold_is_native_frozen_only_and_cache_rejects_policy_changes(self):
        with self.assertRaisesRegex(ValueError, 'completion_policy'):
            self.config('latent_hold')
        with self.assertRaisesRegex(ValueError, 'native-time'):
            AssociationModel(self.config('state_hold'))
        model = TimedAssociationModel(self.config('state_hold'), max_observations=5).eval()
        values, observed = self.packed()
        with self.assertRaisesRegex(ValueError, 'trainable TCM'):
            model.completion_loss(values, observed)
        with self.assertRaisesRegex(ValueError, 'frozen completion'):
            model.set_completion_trainable(True)
        model.set_completion_trainable(False)
        cache = FrozenCompletionCache(model, [(values.numpy(), observed.numpy())], 'cpu', 1_000_000, 2)
        actual = cache.get([(0, 0), (0, 2)])
        torch.testing.assert_close(actual, model.complete(values[[0, 2]], observed[[0, 2]]), rtol=0, atol=0)
        model.config = replace(model.config, completion_policy='learned')
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            cache.get([(0, 0)])

    def test_real_frozen_arms_record_fresh_state_actual_draws_and_reload_policy(self):
        fixture = make_native_zip_fixture(self)
        fixture.write_zip(missing_reference=('2', 'b'))
        fixture.config['model']['dropout'] = .1
        fixture.config['fusion']['dropout'] = .1
        fixture.config['training'].update(branch='contrastive', margin_policy=MARGIN_POLICY)
        fixture.config['training']['updates'] = dict(TCM=1, TAM_FROZEN=0, TFM_FROZEN=0, JOINT=0)
        dataset = fixture.load()
        model_config = ModelConfig(**fixture.config['model'])
        fusion_config = PairFusionConfig(**fixture.config['fusion'])

        def fit(name, phase, policy='learned', audit=False):
            return train_complete(dataset['train'], dataset['validation'],
                replace(model_config, completion_policy=policy), fusion_config,
                fixture.config['training'], fixture.root/name, phase=phase, pairing_audit=audit,
                initialize_tcm_from=fixture.root/'source' if phase != 'tcm' else None)

        def saved(name, filename='checkpoint.pt'):
            return torch.load(fixture.root/name/filename, map_location='cpu', weights_only=True)

        fit('source', 'tcm')
        source = saved('source')
        for phase, stage in (('tfm', 'TFM_FROZEN'), ('tam', 'TAM_FROZEN')):
            fixture.config['training']['updates'] = dict(TCM=0, TAM_FROZEN=0, TFM_FROZEN=0, JOINT=0)
            fixture.config['training']['updates'][stage] = 2
            fixture.config['training']['fusion_monitor_updates'] = [0, 2] if phase == 'tfm' else []
            initials, finals = [], []
            for policy in ('learned', 'state_hold'):
                name = phase + '-' + policy
                self.assertEqual(fit(name, phase, policy, True)['status'], 'COMPLETED')
                initial, final = saved(name, 'pairing-initial.pt'), saved(name)
                initials.append(initial)
                finals.append(final)
                self.assertEqual(initial['optimizer']['state'], {})
                self.assertEqual(initial['cursor']['update'], 0)
                self.assertEqual(initial['cursor']['completed_updates'], 0)
                self.assertEqual(initial['completion_policy'], policy)
                self.assertTrue(initial['pairing_audit'])
                self.assertEqual(initial['normalizer'], source['normalizer'])
                self.assertEqual(initial['data_contract'], source['data_contract'])
                self.assertEqual(initial['initialization']['target_completion_policy'], policy)
                self.assertEqual(initial['initialization']['source_completion_policy'], 'learned')
                self.assertEqual(initial['effective_margin'], final['effective_margin'])
                run = runtime.read_json(fixture.root/name/'run.json')
                self.assertEqual(run['completion_policy'], policy)
                self.assertEqual(run['pairing_initial'], str((fixture.root/name/'pairing-initial.pt').resolve()))
                for key, value in final['states'][0].items():
                    if key == 'positions' or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
                        torch.testing.assert_close(value, source['states'][0][key], rtol=0, atol=0)
                for row in final['history']:
                    self.assertEqual(row['stage'], stage)
                    actual = row['actual_samples']
                    labels = []
                    for group in actual:
                        scene = dataset['train'][group['scene_index']]
                        self.assertEqual(group['scene_id'], scene['scene_id'])
                        indices = np.asarray(group['pair_indices'])
                        self.assertEqual(group['pairs'], scene['pairs'][indices].astype(int).tolist())
                        self.assertEqual(group['labels'], scene['labels'][indices].astype(int).tolist())
                        labels.extend(group['labels'])
                    counts = row['supervision_counts']
                    self.assertEqual(counts['positive_labels'], labels.count(1))
                    self.assertEqual(counts['negative_labels'], labels.count(0))
                    self.assertEqual(counts['association_labels'], 4 if phase == 'tam' else 0)
                    self.assertEqual(counts['fusion_attributes'], row.get('fusion_valid_counts', [0, 0, 0, 0]))
                loaded = load_complete_snapshot(fixture.root/name/'checkpoint.pt', selected=False)
                self.assertEqual(loaded['model'].config.completion_policy, policy)
                self.assertEqual(loaded['model'].config.margin, final['effective_margin'])
                selected = load_complete_snapshot(fixture.root/name/'checkpoint.pt')
                self.assertEqual(selected['model'].config.completion_policy, policy)
                x, observed = normalize_scene(dataset['validation'][0], loaded['normalizer'])
                if policy == 'state_hold':
                    with patch.object(loaded['model'].tcm_encoder, 'forward', side_effect=AssertionError('TCM unused')):
                        self.assertTrue(torch.isfinite(loaded['model'].complete(torch.from_numpy(x),
                                                                              torch.from_numpy(observed))).all())
            self.assert_nested_equal(initials[0]['states'], initials[1]['states'])
            self.assert_nested_equal(initials[0]['optimizer'], initials[1]['optimizer'])
            self.assert_nested_equal(initials[0]['rng'], initials[1]['rng'])
            self.assert_nested_equal(finals[0]['rng'], finals[1]['rng'])
            self.assertEqual([row['actual_samples'] for row in finals[0]['history']],
                             [row['actual_samples'] for row in finals[1]['history']])
            self.assertEqual([row['supervision_counts'] for row in finals[0]['history']],
                             [row['supervision_counts'] for row in finals[1]['history']])
            if phase == 'tam':
                self.assertIsNotNone(initials[0]['margin_calibration'])
                self.assertIsNotNone(initials[1]['margin_calibration'])
            # Opt-in recording must not perturb the ordinary learned endpoint.
            fit(phase + '-plain', phase)
            plain = saved(phase + '-plain')
            self.assert_nested_equal(plain['states'], finals[0]['states'])
            self.assert_nested_equal(plain['rng'], finals[0]['rng'])
            self.assertFalse((fixture.root/(phase + '-plain')/'pairing-initial.pt').exists())
            self.assertFalse(any('actual_samples' in row for row in plain['history']))

    def test_cli_rejects_unsupported_policy_and_audit_before_input_read(self):
        fixture = make_native_zip_fixture(self)
        fixture.config['training']['updates'] = dict(TCM=1, TAM_FROZEN=0, TFM_FROZEN=0, JOINT=0)
        fixture.config['model']['completion_policy'] = 'state_hold'
        with (patch.object(cli, 'load_timed_dataset', side_effect=AssertionError('must reject before input')) as loader,
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            self.assertEqual(cli.main(['--phase', 'tcm', *fixture.cli_args('bad-policy')]), 1)
            loader.assert_not_called()
            fixture.config['model']['completion_policy'] = 'learned'
            self.assertEqual(cli.main(['--phase', 'tcm', '--pairing-audit',
                                       *fixture.cli_args('bad-audit')]), 1)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
