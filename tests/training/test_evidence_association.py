"""Small real updates, isolated normalizer, checkpoint restoration and cancel."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.training.evidence_association import (
    _validate_partitions, fit_evidence_association, fit_normalizer, load_checkpoint, prepare_features, score_scenes)
from association.models.evidence_association import features_from_motion
from association.data.motion_evidence import prepare_motion_scenes
from association.utils import runtime
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.experiments.evidence_association_fit import synthetic_config, synthetic_scenes


class EvidenceTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_updates_complete_population_checkpoint_and_train_only_normalizer(self):
        train, validation = synthetic_scenes()
        packed = prepare_features(train, 'T_core', pair_chunk_size=2)
        for raw, cached in zip(train, packed):
            expected, base = features_from_motion(raw['motion'], raw['pairs'])
            np.testing.assert_array_equal(cached['features'], expected.astype(np.float32))
            np.testing.assert_array_equal(cached['base_scores'], base)
        frozen = fit_normalizer(packed)
        reference = np.concatenate([scene['features'] for scene in packed]).astype(np.float64)
        np.testing.assert_allclose(frozen['mean'], reference.mean(0), atol=1e-14, rtol=1e-14)
        expected_std = reference.std(0)
        np.testing.assert_allclose(frozen['std'], np.where(expected_std < 1e-6, 1., expected_std),
                                   atol=1e-14, rtol=1e-14)
        with self.assertRaises(ValueError):
            fit_normalizer(prepare_features(validation, 'V_select'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'check'
            report = fit_evidence_association(synthetic_config('full_evidence'), path, 'cpu',
                train_scenes=train, validation_scenes=validation, synthetic=True)
            self.assertEqual(report['optimizer_updates'], 3)
            self.assertEqual(report['population']['pairs'], 15)
            self.assertEqual(report['population']['nodes'], 6)
            self.assertFalse(report['test_opened'])
            final_model, checkpoint = load_checkpoint(path/'final.pt', 'cpu')
            self.assertGreater(torch.count_nonzero(final_model.network[-1].weight).item(), 0)
            records = score_scenes(final_model, prepare_features(validation, 'V_select'),
                                   checkpoint['normalizer'], 'cpu', 16)
            with np.load(path/'final-scores.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(records[0]['scores'], saved['validation_scores'])
                self.assertEqual(saved['validation_pairs'].shape, (15, 2))
            selected, selected_checkpoint = load_checkpoint(path/'selected.pt', 'cpu')
            self.assertEqual(selected_checkpoint['update'], report['selected']['update'])
            self.assertEqual(report['selected']['update'], 0)  # tie keeps initial perfect OLS
            baseline = runtime.read_json(path/'baseline.json')
            self.assertEqual(baseline['selected_threshold']['metrics']['fp'], 0)
            point = runtime.read_json(path/'ols-working-point.json')
            self.assertEqual(point['threshold'], baseline['selection']['threshold'])
            self.assertEqual(point['validation_scene_ids'], ['validation'])
            self.assertTrue(point['synthetic'])
            self.assertEqual(point['motion_observation_policy'], 'selected_cap128')
            self.assertEqual(runtime.read_json(path/'run.json')['status'], 'COMPLETED')

    def test_interrupt_is_not_completed_and_validation_components_cannot_train(self):
        train, validation = synthetic_scenes()
        config = synthetic_config('geometry_only')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'interrupted'
            with patch('association.training.evidence_association.prepare_features', side_effect=InterruptedError('cancel')):
                with self.assertRaises(InterruptedError):
                    fit_evidence_association(config, path, 'cpu', train_scenes=train,
                        validation_scenes=validation, synthetic=True)
            self.assertEqual(runtime.read_json(path/'run.json')['status'], 'INTERRUPTED')
            self.assertFalse((path/'summary.json').exists())
            validation[0]['component_id'] = train[0]['component_id']
            with self.assertRaisesRegex(ValueError, 'component_id overlap'):
                fit_evidence_association(config, Path(directory)/'overlap', 'cpu',
                    train_scenes=train, validation_scenes=validation, synthetic=True)

    def test_canonical_motion_training_reload_rejects_old_input_policy(self):
        fixture = make_native_zip_fixture(self)
        policy = 'canonical_all'
        config = synthetic_config('full_evidence')
        config['motion_observation_policy'] = policy
        config['training']['updates'] = 1
        # The fixture's sealed reference/test members remain unreadable.
        train = prepare_motion_scenes(fixture.path, fixture.manifest, 'T_core', motion_observation_policy=policy)
        validation = prepare_motion_scenes(fixture.path, fixture.manifest, 'V_select', motion_observation_policy=policy)
        output = fixture.root/'association'
        report = fit_evidence_association(config, output, 'cpu', train_scenes=train,
                                          validation_scenes=validation, synthetic=True)
        self.assertEqual(report['motion_observation_policy'], policy)
        model, checkpoint = load_checkpoint(output/'final.pt', 'cpu')
        packed = prepare_features(validation, 'V_select')
        score_scenes(model, packed, checkpoint['normalizer'], 'cpu', 16)
        del packed[0]['motion_observation_policy']
        with self.assertRaisesRegex(ValueError, 'motion_observation_policy'):
            score_scenes(model, packed, checkpoint['normalizer'], 'cpu', 16)
        checkpoint['config']['motion_observation_policy'] = 'selected_cap128'
        torch.save(checkpoint, output/'wrong-policy.pt')
        with self.assertRaisesRegex(ValueError, 'motion_observation_policy'):
            load_checkpoint(output/'wrong-policy.pt', 'cpu')

    def test_scene_protocol_permits_shared_components_but_never_overlapping_scenes(self):
        train, validation = synthetic_scenes()
        validation[0]['component_id'] = train[0]['component_id']
        for scene in train + validation:
            scene['split_unit'] = 'scene'
        self.assertEqual(_validate_partitions(train, validation), 'scene')
        cached = prepare_features(validation, 'V_select')
        self.assertEqual(cached[0]['split_unit'], 'scene')
        with tempfile.TemporaryDirectory() as directory:
            report = fit_evidence_association(synthetic_config('geometry_only'),
                Path(directory)/'scene-protocol', 'cpu', train_scenes=train,
                validation_scenes=validation, synthetic=True)
            self.assertEqual(report['split_unit'], 'scene')
            self.assertFalse(report['test_opened'])
        del validation[0]['split_unit']
        with self.assertRaisesRegex(ValueError, 'split_unit'):
            _validate_partitions(train, validation)
        validation[0]['split_unit'] = 'scene'
        validation[0]['scene_id'] = train[0]['scene_id']
        with self.assertRaisesRegex(ValueError, 'scene_id overlap'):
            _validate_partitions(train, validation)


if __name__ == '__main__':
    unittest.main()
