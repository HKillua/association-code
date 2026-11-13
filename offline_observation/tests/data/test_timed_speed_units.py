"""Unit contract checks using tiny software fixtures, without fitting a model."""
import copy
from dataclasses import asdict
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.data.archive import DataError
from association.data.physical_units import KNOT_TO_MPS, SPEED_UNIT_POLICY
from association.data.timed import fit_timed_normalizer, pack_timed_scene
from association.evaluation.artifacts import ScenePredictionWriter, load_scene_prediction
from association.evaluation.tmtaff import evaluate_complete
from association.models.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.models.timed import TimedAssociationModel
from association.models.tmtaff import ModelConfig
from association.training.tmtaff import _scenes, load_complete_snapshot
from tests.fixtures.native_zip import NativeZipFixture


class TimedSpeedUnitsTests(NativeZipFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def physical(self, **kwargs):
        return self.load(data_config={**self.data, 'speed_unit_policy': SPEED_UNIT_POLICY}, **kwargs)

    def modules(self, normalizer):
        model = TimedAssociationModel(ModelConfig(**self.config['model']), 'ce',
                                      max_observations=4, normalizer=normalizer)
        fusion = PairFusion(PairFusionConfig(**self.config['fusion']), normalizer)
        with torch.no_grad():
            fusion.state_projection.weight.zero_()
            fusion.state_projection.bias.copy_(torch.tensor([20., 120., math.log(math.expm1(5.)), 45.]))
        return model, fusion

    def test_loader_normalizer_targets_and_legacy_values(self):
        old, physical = self.load(), self.physical()
        explicit_legacy = self.load(data_config={**self.data, 'speed_unit_policy': None})
        for legacy, current, unchanged in zip(old['train'] + old['validation'],
                                             physical['train'] + physical['validation'],
                                             explicit_legacy['train'] + explicit_legacy['validation']):
            self.assertNotIn('physical_units', legacy)
            self.assertEqual(legacy['adapter_config'], unchanged['adapter_config'])
            for key in ('native_values', 'values', 'grid', 'native_times', 'native_observed', 'pairs'):
                np.testing.assert_array_equal(legacy[key], unchanged[key])
            expected = legacy['native_values'].copy()
            expected[..., 2] *= KNOT_TO_MPS
            np.testing.assert_array_equal(current['native_values'], expected)
            expected = legacy['values'].copy()
            expected[..., 2] *= KNOT_TO_MPS
            np.testing.assert_allclose(current['values'], expected, equal_nan=True)
            for name, target in current['fusion_targets'].items():
                np.testing.assert_array_equal(target['states'], legacy['fusion_targets'][name]['states'])
                self.assertEqual(target['physical_units']['reference_speed_unit'], 'mps')
        normalizer = fit_timed_normalizer(physical['train'])
        legacy_normalizer = fit_timed_normalizer(old['train'])
        self.assertAlmostEqual(normalizer['mean'][2], 5 * KNOT_TO_MPS)
        self.assertEqual(normalizer['physical_units']['state_speed_unit'], 'mps')
        pack_timed_scene(physical['validation'][0], normalizer)
        for scene, wrong in ((physical['train'][0], legacy_normalizer), (old['train'][0], normalizer)):
            with self.assertRaisesRegex(DataError, 'unit'):
                pack_timed_scene(scene, wrong)
        _scenes(physical['train'], physical['validation'], 4)
        with self.assertRaisesRegex(ValueError, 'unit'):
            _scenes(physical['train'], old['validation'], 4)
        del physical['validation'][0]['fusion_targets']['target2a-0']['physical_units']
        with self.assertRaisesRegex(DataError, 'target units'):
            _scenes(physical['train'], physical['validation'], 4)

    def test_full_evaluator_singletons_controls_and_export_use_one_unit(self):
        data = self.physical()
        normalizer = fit_timed_normalizer(data['train'])
        model, fusion = self.modules(normalizer)
        records = []
        for scene in data['validation']:
            row = {key: scene[key] for key in ('scene_id', 'component_id', 'node_ids', 'identities',
                                              'sources', 'intervals', 'pairs', 'labels')}
            records.append({**row, 'scores': np.zeros(len(scene['pairs']))})
        writer = ScenePredictionWriter(self.root / 'predictions')
        result = evaluate_complete(model, fusion, normalizer, data['validation'], device='cpu',
            batch_tracks=4, pair_batch_size=4, select_on_validation=False, threshold=1.,
            singleton_policy='observed', association_records=records, predicted_controls=True,
            prediction_writer=writer)
        loaded = load_scene_prediction(self.root / 'predictions' / '000000.json')
        self.assertEqual(loaded['metadata']['physical_units']['state_speed_unit'], 'mps')
        for name in ('predicted', 'predicted_anchor', 'predicted_mean'):
            artifact = loaded['methods'][name]
            mask = artifact['aligned_prediction_valid'][..., 2]
            np.testing.assert_allclose(artifact['aligned_prediction'][..., 2][mask], 5 * KNOT_TO_MPS)
            self.assertEqual(str(artifact['state_speed_unit']), 'mps')
            metric = result['fusion'][name]
            self.assertEqual(metric['all']['speed']['unit'], 'mps')
            paper = metric['paper_fusion_metrics']['attributes']['speed']
            self.assertEqual(paper['unit'], 'kn')
        first = result['per_scene'][0]['methods']['predicted']
        self.assertAlmostEqual(first['full_target_metrics']['all']['speed']['mae'], 5 - 5*KNOT_TO_MPS)
        self.assertAlmostEqual(first['paper_fusion_metrics']['attributes']['speed']['target_mean']['ME'],
                               5 / KNOT_TO_MPS - 5)
        self.assertEqual(first['input_distance_metrics']['gt_1800s']['speed']['unit'], 'mps')
        self.assertEqual(result['fusion']['predicted']['all']['position']['unit'], 'metres')
        self.assertEqual(result['fusion']['predicted']['all']['heading']['unit'], 'degrees')
        self.assertEqual(result['fusion']['predicted']['all']['speed']['available_target_count'], 16)
        self.assertFalse(result['test_opened'])
        # The same call executes learned TCM completion and multi-member TFM
        # for both oracle groups, separately from the singleton bypass above.
        self.assertEqual(model.config.completion_policy, 'learned')
        self.assertGreater(result['fusion_forward_count'], 0)
        oracle = loaded['methods']['oracle']
        valid_speed = oracle['aligned_prediction_valid'][..., 2]
        np.testing.assert_allclose(oracle['aligned_prediction'][..., 2][valid_speed], 5., atol=1e-6)
        self.assertEqual(str(oracle['state_speed_unit']), 'mps')
        oracle_report = result['per_scene'][0]['methods']['oracle']
        self.assertAlmostEqual(oracle_report['full_target_metrics']['all']['speed']['mae'], 0., places=6)
        paper = oracle_report['paper_fusion_metrics']
        self.assertEqual(paper['attributes']['speed']['unit'], 'kn')
        self.assertAlmostEqual(paper['attributes']['speed']['target_mean']['ME'], 0., places=6)
        self.assertAlmostEqual(paper['metadata']['speed_units']['prediction_factor'], 1 / KNOT_TO_MPS)

    def test_checkpoint_and_normalizer_cannot_cross_unit_contracts(self):
        data, old = self.physical(), self.load()
        normalizer = fit_timed_normalizer(data['train'])
        old_normalizer = fit_timed_normalizer(old['train'])
        new_modules, old_modules = self.modules(normalizer), self.modules(old_normalizer)
        for new, legacy in zip(new_modules, old_modules):
            with self.assertRaisesRegex(ValueError, 'unit'):
                new.load_state_dict(legacy.state_dict(), strict=False)
            with self.assertRaisesRegex(ValueError, 'unit'):
                legacy.load_state_dict(new.state_dict(), strict=False)
            new.load_state_dict(new.state_dict())
        with self.assertRaisesRegex(ValueError, 'unit'):
            new_modules[0].validate_normalizer(old_normalizer)
        bad = copy.deepcopy(normalizer)
        bad['physical_units']['state_speed_unit'] = 'kn'
        with self.assertRaisesRegex(DataError, 'Normalizer'):
            self.modules(bad)
        objective = FusionLoss(position_loss_mode='standard_smooth_l1')
        payload = dict(format_version=1, input_adapter='native_time_v1',
            config=dict(branch='ce', position_loss_mode='standard_smooth_l1'),
            model_config=asdict(new_modules[0].config), fusion_config=asdict(new_modules[1].config),
            max_observations=4, normalizer=normalizer,
            states=[module.state_dict() for module in (*new_modules, objective)])
        checkpoint = self.root / 'units.pt'
        torch.save(payload, checkpoint)
        loaded = load_complete_snapshot(checkpoint, selected=False)
        self.assertEqual(loaded['normalizer'], normalizer)
        payload['states'][0] = old_modules[0].state_dict()
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(ValueError, 'unit'):
            load_complete_snapshot(checkpoint, selected=False)
        payload['normalizer'] = old_normalizer
        payload['states'] = [module.state_dict() for module in (*old_modules, objective)]
        torch.save(payload, checkpoint)
        legacy = load_complete_snapshot(checkpoint, selected=False)
        self.assertNotIn('speed_unit_contract', legacy['model'].state_dict())
        self.assertEqual(legacy['normalizer'], old_normalizer)

    def test_unsupported_routes_fail_before_archive_access(self):
        for changes in (dict(motion_observation_policy='canonical_all'), dict(retain_native_events=True)):
            with patch('association.data.dataset.MTADArchive', side_effect=AssertionError('archive opened')):
                with self.assertRaisesRegex(DataError, 'timed TMTAFF only'):
                    self.physical(**changes)
        with patch('association.data.dataset.MTADArchive', side_effect=AssertionError('archive opened')):
            with self.assertRaisesRegex(DataError, 'Unknown speed_unit_policy'):
                self.load(data_config={**self.data, 'speed_unit_policy': 'guess'})

    def test_cli_inspection_accepts_opt_in_without_model_or_training(self):
        from tools.experiments.tmtaff_complete_fit import main
        from association.utils import runtime
        self.config['data']['speed_unit_policy'] = SPEED_UNIT_POLICY
        with patch('association.training.tmtaff.train_complete', side_effect=AssertionError('no training')):
            self.assertEqual(main(self.cli_args('inspect') + ['--inspect-only']), 0)
        report = runtime.read_json(self.root / 'inspect' / 'preparation.json')
        self.assertEqual(report['configuration']['data']['speed_unit_policy'], SPEED_UNIT_POLICY)
        self.assertFalse(report['model_constructed'])
        self.assertFalse(report['gpu_training_started'])

    def test_opt_in_does_not_mutate_raw_input_or_accept_second_conversion(self):
        from tests.data.test_timed_inputs import native_scene, prepare
        raw = native_scene([('A-0', '9001', [0, 10, 20])])
        original = raw['tracks'][0]['canonical_values'].copy()
        current = prepare(raw, length=4, cap=4, speed_unit_policy=SPEED_UNIT_POLICY)
        np.testing.assert_array_equal(raw['tracks'][0]['canonical_values'], original)
        self.assertNotIn('tracks', current)
        with self.assertRaisesRegex(DataError, 'already adapted'):
            prepare(current, length=4, cap=4, speed_unit_policy=SPEED_UNIT_POLICY)

    def test_frozen_completion_cache_consumes_checked_physical_inputs(self):
        from association.training.batches import FrozenCompletionCache
        data = self.physical()
        normalizer = fit_timed_normalizer(data['train'])
        model, _ = self.modules(normalizer)
        model.set_completion_trainable(False, include_embedding=True)
        model.eval()
        normalized = [pack_timed_scene(scene, normalizer) for scene in data['train']]
        cache = FrozenCompletionCache(model, normalized, 'cpu', 4096, 4)
        actual = cache.get([(0, 0)])
        with torch.no_grad():
            expected = model.complete(torch.as_tensor(normalized[0][0][:1]),
                                      torch.as_tensor(normalized[0][1][:1]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(cache.get([(0, 0)]), expected, rtol=0, atol=0)
        self.assertEqual(cache.stats()['hits'], 1)
        self.assertEqual(model.state_dict()['speed_unit_contract'].item(), 1)
