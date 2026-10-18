"""Train-only manual time scaling, compatibility and matched-arm regressions.

All inputs are tiny synthetic native scenes. No accelerator, MTAD archive,
reference fitting, or validation-based scale selection is performed.
"""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.cooperative import CooperativeConfig, CooperativeModel
from association import cooperative_training as training
from association.reference import fit_normalizer, normalize_scene
from tools.experiments.cooperative_fit import synthetic_configuration, synthetic_scenes


class CooperativeTimeScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.config = CooperativeConfig(length=4, max_observations=4, d_model=8,
            nhead=2, num_layers=1, dim_feedforward=16, head_hidden=8, dropout=0.)
        self.scenes, self.validation = synthetic_scenes(self.config)
        self.normalizer = fit_normalizer(self.scenes)
        self.packed = [normalize_scene(s, self.normalizer) for s in self.scenes]
        self.scaled_config = replace(self.config, support_time_policy='train_absmax_v1')

    def model(self, config=None, scale=None):
        torch.manual_seed(20)
        return CooperativeModel(config or self.config, self.normalizer,
                                support_time_scale=scale).eval()

    def encoded(self, model):
        values, observed = self.packed[0]
        return model.encode(torch.from_numpy(values), torch.from_numpy(observed))

    def test_legacy_preserves_manual_feature_formula_and_old_state_keys(self):
        model = self.model()
        self.assertEqual(model.config.support_time_policy, 'legacy')
        self.assertNotIn('support_time_scale', model.state_dict())
        encoded = self.encoded(model)
        pairs = torch.tensor([[0, 1], [0, 2]])
        first, last, count = model._support(encoded)
        support = torch.stack((first, last, last-first, count.log1p()), -1)
        left, right = encoded['pooled'][pairs[:, 0]], encoded['pooled'][pairs[:, 1]]
        sl, sr = support[pairs[:, 0]], support[pairs[:, 1]]
        explicit_input = torch.cat(((left-right).abs(), left*right, (sl-sr).abs(),
                                    sl*sr, left.new_zeros((len(pairs), 4))), -1)
        expected = model.identity_base(explicit_input).squeeze(-1)
        torch.testing.assert_close(model.score_pairs(encoded, pairs), expected, rtol=0, atol=0)
        q = encoded['queries']
        expected_support = torch.stack((q-first[:, None], last[:, None]-q,
            (last-first)[:, None].expand_as(q), count.log1p()[:, None].expand_as(q)), -1)
        torch.testing.assert_close(model._query_support(encoded), expected_support, rtol=0, atol=0)

    def test_new_policy_requires_explicit_legal_scale(self):
        for scale in (None, True, False, 0., .5, float('nan'), float('inf'), -float('inf')):
            with self.subTest(scale=scale):
                with self.assertRaises((TypeError, ValueError)):
                    self.model(self.scaled_config, scale)
        with self.assertRaises(ValueError):
            self.model(replace(self.config, support_time_policy='fit_on_validation'))
        for scale in (1., 1.5, 360.):
            model = self.model(self.scaled_config, scale)
            self.assertEqual(float(model.support_time_scale), scale)
            self.assertIn('support_time_scale', model.state_dict())
            self.assertFalse(model.support_time_scale.requires_grad)

    def test_scale_fits_only_valid_native_and_all_query_times(self):
        packed = copy.deepcopy(self.packed)
        values, observed = packed[0]
        m = self.config.max_observations
        # Max queries must be included even though query observed flags are false.
        values[:, -1, 4] = 200.
        # Valid native times can exceed queries; fit uses absolute magnitude.
        values[0, 0, 4] = -700.
        # All physical-state channels and reference contents are irrelevant to S.
        values[..., :4] = np.nan
        native_tau = values[:, :m, 4]
        native_tau[~observed[:, :m]] = np.inf
        scenes = copy.deepcopy(self.scenes)
        scenes[0]['fusion_targets'] = None
        result = training.fit_support_time_scaling(scenes, packed, m)
        self.assertEqual(result['policy'], 'train_absmax_v1')
        self.assertEqual(result['scale_tau'], 700.)
        values[0, 0, 4] = 0.
        result = training.fit_support_time_scaling(scenes, packed, m)
        self.assertEqual(result['scale_tau'], 200.)
        values[..., 4] = 0.25
        result = training.fit_support_time_scaling(scenes, packed, m)
        self.assertEqual(result['scale_tau'], 1.)

    def test_scale_rejects_nontraining_population_and_nonfinite_selected_time(self):
        m = self.config.max_observations
        for split in ('V_select', 'T_test', 'T_head'):
            scene = copy.deepcopy(self.scenes[0])
            scene['split'] = split
            with self.subTest(split=split), self.assertRaises(ValueError):
                training.fit_support_time_scaling([scene], self.packed, m)
        for native in (True, False):
            packed = copy.deepcopy(self.packed)
            packed[0][0][0, 0 if native else -1, 4] = np.nan
            with self.subTest(native=native), self.assertRaises(ValueError):
                training.fit_support_time_scaling(self.scenes, packed, m)
        def expired():
            raise TimeoutError('synthetic budget expired')
        with self.assertRaisesRegex(TimeoutError, 'synthetic budget expired'):
            training.fit_support_time_scaling(self.scenes, self.packed, m, check=expired)

    def test_only_manual_mlp_paths_change_not_encoding_physical_support_or_plain_decode(self):
        legacy, scaled = self.model(), self.model(self.scaled_config, 10.)
        for name, value in legacy.named_parameters():
            torch.testing.assert_close(value, dict(scaled.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in legacy.parameters()),
                         sum(p.numel() for p in scaled.parameters()))
        a, b = self.encoded(legacy), self.encoded(scaled)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        torch.testing.assert_close(legacy.summarize(a), scaled.summarize(b), rtol=0, atol=0)
        for left, right in zip(legacy._support(a), scaled._support(b)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for route in ('plain', 'support'):
            torch.testing.assert_close(legacy.decode_group(a, [0, 1], routing=route),
                                       scaled.decode_group(b, [0, 1], routing=route), rtol=0, atol=0)
        pairs = torch.tensor([[0, 1], [0, 2]])
        # Both identity heads, utility and ordinary gate must use the new rule.
        for use_utility in (False, True):
            self.assertFalse(torch.equal(legacy.score_pairs(a, pairs, use_utility),
                                         scaled.score_pairs(b, pairs, use_utility)))
        self.assertFalse(torch.equal(legacy.utility(a, pairs), scaled.utility(b, pairs)))
        members = torch.tensor([0, 1])
        self.assertFalse(torch.equal(legacy._route_weights(a, members, 'learned', None, None),
                                     scaled._route_weights(b, members, 'learned', None, None)))
        # Unscaled count features must not silently change with time scaling.
        torch.testing.assert_close(legacy._query_support(a)[..., 3],
                                   scaled._query_support(b)[..., 3], rtol=0, atol=0)

    def test_historical_checkpoint_without_policy_or_scale_loads_identically(self):
        model = self.model()
        config = asdict(self.config)
        config.pop('support_time_policy')
        payload = dict(format='cooperative-physical-v1', model_config=config,
                       model_state=model.state_dict(), normalizer=self.normalizer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'historical.pt'
            torch.save(payload, path)
            restored, _ = training.load_cooperative_checkpoint(path, device='cpu')
        self.assertEqual(restored.config.support_time_policy, 'legacy')
        self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
        for key in model.state_dict():
            torch.testing.assert_close(restored.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)
        a, b = self.encoded(model), self.encoded(restored)
        torch.testing.assert_close(model.decode_group(a, [0, 1]), restored.decode_group(b, [0, 1]), rtol=0, atol=0)
        torch.testing.assert_close(model.score_pairs(a, [[0, 1]]), restored.score_pairs(b, [[0, 1]]), rtol=0, atol=0)

    def test_native_zip_cli_assembles_scales_fits_and_exports_all_four_arms(self):
        from tests.test_timed_dataset import TimedDatasetTests
        from tools.experiments.cooperative_fit import main
        from association.prediction_artifacts import load_scene_prediction
        fixture = TimedDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config = dict(scope='cooperative-physical-v1',
            model=asdict(self.scaled_config), data={'time_unit_seconds': 10.},
            train_splits=['T_core'], training=synthetic_configuration())
        fitting_populations = []
        original_fit = training.fit_support_time_scaling
        def capture(scenes, packed, max_observations, check=None):
            fitting_populations.append([(s['scene_id'], s['split']) for s in scenes])
            return original_fit(scenes, packed, max_observations, check=check)
        stderr = io.StringIO()
        with patch.object(training, 'fit_support_time_scaling', side_effect=capture), \
             redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            code = main(['fit', '--device', 'cpu', '--save-predictions',
                         *fixture.cli_args('scaled-candidate')])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(fitting_populations, [[('0', 'T_core')]])
        output = fixture.root/'scaled-candidate'
        run = json.loads((output/'run.json').read_text())
        execution = json.loads((output/'execution.json').read_text())
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['completed_updates'], 6)
        self.assertEqual(run['support_time_scaling']['scale_tau'], 3.)
        self.assertEqual(run['model_config']['support_time_policy'], 'train_absmax_v1')
        self.assertFalse(run['test_opened'])
        self.assertFalse(execution['preparation']['test_opened'])
        self.assertEqual(execution['configuration']['model']['support_time_policy'], 'train_absmax_v1')
        for route in ('plain', 'identity', 'utility', 'learned'):
            model, payload = training.load_cooperative_checkpoint(output/f'adapt-{route}.pt', device='cpu')
            self.assertEqual(payload['support_time_scale_tau'], 3.)
            self.assertEqual(float(model.support_time_scale), 3.)
            evaluation = json.loads((output/(route+'-evaluation.json')).read_text())
            self.assertFalse(evaluation['test_opened'])
            self.assertEqual(evaluation['threshold'], run['threshold'])
            entries = json.loads((output/(route+'-predictions/index.json')).read_text())['scenes']
            self.assertEqual([entry['scene_id'] for entry in entries], ['2', '4'])
            for entry in entries:
                saved = load_scene_prediction(output/(route+'-predictions')/entry['metadata'])
                self.assertEqual(saved['scene']['split'], 'V_select')
                self.assertFalse(saved['metadata']['test_opened'])
                self.assertIn('predicted', saved['methods'])
                self.assertIn('oracle', saved['methods'])
                if entry['scene_id'] == '4':
                    # No-input V targets remain in the population, even with scaling.
                    artifact = saved['methods']['predicted']
                    self.assertGreater(artifact['aligned_truth_valid'].sum(), 0)
                    self.assertEqual(artifact['aligned_prediction_valid'].sum(), 0)

    def test_six_update_four_arm_fit_uses_one_training_scale_and_persists_it(self):
        initial, sequences, fitting_populations, batch_scales = {}, {}, [], []
        original_batch = training.train_batch
        original_fit = training.fit_support_time_scaling
        def fit_capture(scenes, packed, max_observations, check=None):
            fitting_populations.append([s['scene_id'] for s in scenes])
            return original_fit(scenes, packed, max_observations, check=check)
        def batch_capture(model, objective, optimizer, items, **kwargs):
            batch_scales.append(float(model.support_time_scale))
            if kwargs['stage'] == 'adapt':
                route = kwargs['routing']
                initial[route] = copy.deepcopy(model.state_dict())
                sequences[route] = [(x['scene_id'], x['pair_row'], x['removed'], x['label']) for x in items]
            return original_batch(model, objective, optimizer, items, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'fit'
            with patch.object(training, 'fit_support_time_scaling', side_effect=fit_capture), \
                 patch.object(training, 'train_batch', side_effect=batch_capture):
                result = training.train_cooperative(self.scenes, self.validation, self.scaled_config,
                    synthetic_configuration(), output, device='cpu')
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['completed_updates'], 6)
            self.assertEqual(result['completed_routes'], ['plain', 'identity', 'utility', 'learned'])
            self.assertEqual(fitting_populations, [[self.scenes[0]['scene_id']]])
            self.assertEqual(batch_scales, [1.5]*6)
            self.assertEqual(result['support_time_scaling']['policy'], 'train_absmax_v1')
            self.assertEqual(result['support_time_scaling']['scale_tau'], 1.5)
            stored_run = json.loads((output/'run.json').read_text())
            self.assertEqual(stored_run['support_time_scaling'], result['support_time_scaling'])
            for route in ('identity', 'utility', 'learned'):
                self.assertEqual(sequences[route], sequences['plain'])
                for key in initial['plain']:
                    torch.testing.assert_close(initial[route][key], initial['plain'][key], rtol=0, atol=0)
            base_scores = np.load(output/'plain-scores.npz', allow_pickle=False)
            for route in ('plain', 'identity', 'utility', 'learned'):
                with np.load(output/(route+'-scores.npz'), allow_pickle=False) as scores:
                    for key in base_scores.files:
                        np.testing.assert_array_equal(base_scores[key], scores[key])
            base_scores.close()
            for filename in ('teacher.pt', 'utility.pt', 'adapt-plain.pt', 'adapt-identity.pt',
                             'adapt-utility.pt', 'adapt-learned.pt'):
                restored, payload = training.load_cooperative_checkpoint(output/filename, device='cpu')
                self.assertEqual(payload['support_time_scale_tau'], 1.5)
                self.assertEqual(float(restored.support_time_scale), 1.5)
                self.assertEqual(restored.config.support_time_policy, 'train_absmax_v1')
                self.assertEqual(payload['model_state']['support_time_scale'].item(), 1.5)
            # Corrupted or inconsistent scalar/buffer pairs must not silently load.
            baseline = torch.load(output/'adapt-utility.pt', map_location='cpu', weights_only=True)
            corruptions = []
            for scalar in (None, .5, float('nan'), float('inf'), 2.):
                item = copy.deepcopy(baseline)
                if scalar is None:
                    item.pop('support_time_scale_tau')
                else:
                    item['support_time_scale_tau'] = scalar
                corruptions.append(item)
            for scalar in (float('nan'), 2.):
                item = copy.deepcopy(baseline)
                item['model_state']['support_time_scale'] = torch.tensor(scalar)
                corruptions.append(item)
            for index, item in enumerate(corruptions):
                path = Path(folder)/f'corrupt-{index}.pt'
                torch.save(item, path)
                with self.subTest(corruption=index), self.assertRaises((ValueError, RuntimeError)):
                    training.load_cooperative_checkpoint(path, device='cpu')


if __name__ == '__main__':
    unittest.main()
