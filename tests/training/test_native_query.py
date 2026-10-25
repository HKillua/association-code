"""Three tiny updates exercise replay, frozen bases and checkpoint round trips."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from association.models.sequence_position import SequencePosition
from association.training.evidence_fusion import TRAIN_MODES, prepare_evidence_scenes, training_members
from association.training.native_query import NativeExperiment, SCOPE, fit, load_checkpoint
from tests.fixtures.learning import tiny_config, native_query_training_fixture


def _training_and_replay():
    scenes, replay = native_query_training_fixture()
    prepared = prepare_evidence_scenes(scenes, motion_observation_policy='canonical_all',
                                       retain_native_events=True)
    return scenes, prepared, replay


class NativeQueryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.scenes, cls.prepared, cls.replay = _training_and_replay()
        cls.base_config = dict(backbone='transformer', hidden_size=64, num_layers=2, initial_gate=.95)
        torch.manual_seed(17)
        cls.base = SequencePosition(**cls.base_config).eval()
        cls.base_state = {key: value.clone() for key, value in cls.base.state_dict().items()}
        cls.config = tiny_config()
        cls.config.update(scope=SCOPE, arm='mean', model=dict(hidden_dim=64, patches=8, heads=4))
        cls.results = {}
        for arm in ('mean', 'query', 'budget'):
            config = deepcopy(cls.config); config['arm'] = arm
            cls.results[arm] = fit(cls.scenes, config, cls.base, cls.base_config, cls.replay,
                cls.root/arm, source=dict(kind='unit-test-only', checkpoint='untrained-fixture'))

    def test_common_initialization_exact_replay_frozen_base_and_real_branch_updates(self):
        initials = {arm: torch.load(self.root/arm/'step-0.pt', weights_only=False)
                    for arm in ('mean', 'query')}
        shared = set(initials['mean']['branch_state']) & set(initials['query']['branch_state'])
        self.assertIn('decoder.in_proj_weight', shared)
        for key in shared:
            if not key.startswith('pooler.'):
                self.assertTrue(torch.equal(initials['mean']['branch_state'][key],
                                            initials['query']['branch_state'][key]), key)
        for arm in ('mean', 'query'):
            with self.subTest(arm=arm):
                report = self.results[arm]
                self.assertEqual(report['status'], 'COMPLETED')
                self.assertEqual(report['optimizer_updates'], 3)
                self.assertEqual(report['actual_successful_batches_matched'], 3)
                self.assertEqual(report['exposure']['sampled_anchors'], 6)
                expected_points = sum(sample['valid_position_points']
                    for row in self.replay for sample in row['batch'])
                self.assertEqual(report['exposure']['valid_position_points'], expected_points)
                self.assertTrue(report['frozen_base_unchanged'])
                self.assertFalse(report['test_opened'])
                self.assertEqual(report['training_scene_ids'], [str(s['scene_id']) for s in self.scenes])
                rows = [json.loads(line) for line in (self.root/arm/'training.jsonl').read_text().splitlines()]
                self.assertEqual([(row['step'], row['mode'], row['batch']) for row in rows],
                                 [(row['step'], row['mode'], row['batch']) for row in self.replay])
                model, saved = load_checkpoint(self.root/arm/'final.pt')
                for key, value in self.base_state.items():
                    self.assertTrue(torch.equal(value, saved['base_state'][key]), key)
                    self.assertTrue(torch.equal(value, model.base.state_dict()[key]), key)
                    self.assertTrue(torch.equal(value, self.base.state_dict()[key]), key)
                self.assertTrue(any(not torch.equal(value, saved['branch_state'][key])
                    for key, value in initials[arm]['branch_state'].items()))
                model.train()
                self.assertFalse(model.base.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in model.base.parameters()))

    def test_saved_loaded_predictions_match_and_budget_updates_only_its_trainable_copy(self):
        scene = self.prepared[0]
        members = training_members(scene, 0, 'clean_plus_wrong')
        for arm in ('mean', 'query', 'budget'):
            with self.subTest(arm=arm):
                loaded, saved = load_checkpoint(self.root/arm/'final.pt')
                reconstructed = NativeExperiment(self.base, arm, self.config['model'])
                reconstructed.branch.load_state_dict(saved['branch_state'], strict=True)
                loaded.eval(); reconstructed.eval()
                with torch.no_grad():
                    actual = loaded(scene['motion'], members)
                    expected = reconstructed(scene['motion'], members)
                for key in ('states', 'residual_m', 'frozen_base_states'):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                np.testing.assert_array_equal(actual['states'][:, 2:].numpy(),
                                               actual['frozen_base_states'][:, 2:].numpy())
                if arm == 'budget':
                    self.assertTrue(any(not torch.equal(value, saved['branch_state'][key])
                        for key, value in self.base_state.items()))
                    for key, value in self.base_state.items():
                        self.assertTrue(torch.equal(value, loaded.base.state_dict()[key]))

    def test_bad_members_fail_before_update_and_validation_input_is_rejected(self):
        broken = deepcopy(self.replay)
        sample = broken[0]['batch'][0]
        sample['members'] = sample['members'] + [sample['members'][0]]
        output = self.root/'bad-members'
        with self.assertRaisesRegex(ValueError, 'exposure differs before update 1'):
            fit(self.scenes, self.config, self.base, self.base_config, broken, output,
                source=dict(kind='unit-test-only'))
        report = json.loads((output/'run.json').read_text())
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertFalse((output/'final.pt').exists())
        saved = torch.load(output/'interrupted.pt', weights_only=False)
        self.assertEqual(saved['actual_updates'], 0)
        validation = deepcopy(self.scenes); validation[0]['split'] = 'V_select'
        with self.assertRaisesRegex(ValueError, 'only T_core'):
            fit(validation, self.config, self.base, self.base_config, self.replay,
                self.root/'bad-split', source=dict(kind='unit-test-only'))
        self.assertFalse((self.root/'bad-split').exists())


if __name__ == '__main__':
    unittest.main()
