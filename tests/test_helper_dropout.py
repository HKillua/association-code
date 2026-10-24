import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.training import sequence_position as trainer
from association.training.helper_dropout import drop_helper_fragments, dropout_audit
from association.models.sequence_position import sequence_inputs
from tests.test_sequence_position import tiny_configuration, tiny_training, motion_fixture


def dropout_configuration():
    config = tiny_configuration('transformer')
    config.update(scope=trainer.HELPER_DROPOUT_SCOPE,
        augmentation=dict(kind=trainer.HELPER_DROPOUT_KIND, probability=.5,
                          seed=config['training']['seed']+83000))
    return config


def rows(path):
    return [json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]


class HelperDropoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.train = tiny_training()
        cls.reference = cls.root/'reference'
        trainer.train_sequence_position(cls.train, tiny_configuration('transformer'), cls.reference)
        cls.output = cls.root/'dropout'
        cls.result = trainer.train_sequence_position(cls.train, dropout_configuration(), cls.output,
            paired_reference_fit=cls.reference)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_fragment_mask_preserves_anchor_order_empty_all_and_rng_independence(self):
        members = [3, 1, 7, 2]
        for probability, expected in ((0., members), (1., [3])):
            retained, mask = drop_helper_fragments(members, np.random.default_rng(2), probability)
            self.assertEqual(retained, expected)
            self.assertEqual(mask, [bool(probability)]*3)
        rng = np.random.default_rng(2); before = copy.deepcopy(rng.bit_generator.state)
        self.assertEqual(drop_helper_fragments([3], rng, .5), ([3], []))
        self.assertEqual(before, rng.bit_generator.state)
        plan_rng = np.random.default_rng(11); old = copy.deepcopy(plan_rng.bit_generator.state)
        drop_helper_fragments(members, np.random.default_rng(5), .5)
        self.assertEqual(old, plan_rng.bit_generator.state)
        a = drop_helper_fragments(members, np.random.default_rng(17), .5)
        b = drop_helper_fragments(members, np.random.default_rng(17), .5)
        self.assertEqual(a, b)
        for invalid in ([], [1, 1], [True, 2], [-1], [1.5]):
            with self.assertRaises(ValueError): drop_helper_fragments(invalid, rng, .5)

    def test_drop_recomputes_base_mean_features_without_reference_or_label_selection(self):
        motion = motion_fixture()
        original = sequence_inputs(motion, [0, 1, 2])
        retained, mask = drop_helper_fragments([0, 1, 2], np.random.default_rng(1), 1.)
        new = sequence_inputs(motion, retained)
        self.assertTrue(torch.equal(new[0], torch.tensor(motion['states'][0])))
        self.assertFalse(torch.equal(new[0], original[0]))
        self.assertFalse(torch.equal(new[1], original[1]))
        self.assertFalse(torch.equal(new[2], original[2]))
        scene = dict(motion=motion, identities=np.array(['a', 'a', 'b']))
        audit = dropout_audit(scene, [0, 1, 2], retained, mask, np.array([True, False, True]))
        self.assertEqual(audit['original_identity'], dict(correct_helpers=1, wrong_helpers=1))
        self.assertEqual(audit['retained_support']['valid_position_queries'], dict(zero=2, one=0, multiple=0))

    def test_actual_original_plan_pairing_and_successful_exposure_and_rng_replay(self):
        result = self.result
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['parameter_count'], 105154)
        self.assertTrue(result['paired_control']['initial_torch_rng_exact'])
        self.assertEqual(result['paired_control']['actual_successful_batches_matched'], 3)
        reference, augmented = rows(self.reference), rows(self.output)
        rng = np.random.default_rng(dropout_configuration()['augmentation']['seed'])
        total_dropped = 0
        for old, new in zip(reference, augmented):
            self.assertEqual((old['mode'], old['batch']), (new['mode'], new['batch']))
            self.assertEqual(new['augmentation_rng_before'], rng.bit_generator.state)
            for plan, audit in zip(new['batch'], new['augmentation_batch']):
                retained, mask = drop_helper_fragments(plan['members'], rng, .5)
                self.assertEqual(audit['retained_members'], retained)
                self.assertEqual(audit['drop_mask'], mask)
                self.assertEqual(audit['original_members'], plan['members'])
                self.assertEqual(sum(audit['retained_support']['valid_position_queries'].values()), plan['valid_position_points'])
                total_dropped += sum(mask)
            self.assertEqual(new['augmentation_rng_after'], rng.bit_generator.state)
        self.assertGreater(total_dropped, 0)
        exposure = result['augmentation_exposure']
        self.assertEqual(exposure['dropped_helpers'], total_dropped)
        self.assertEqual(exposure['successful_updates'], 3)
        self.assertEqual(exposure['sampled_anchors'], 6)
        self.assertEqual(exposure['original_helpers']-exposure['retained_helpers'], total_dropped)
        self.assertEqual(exposure['original_support']['valid_position_queries']['zero'] +
                         exposure['original_support']['valid_position_queries']['one'] +
                         exposure['original_support']['valid_position_queries']['multiple'], result['exposure']['valid_position_points'])

    def test_step0_final_loading_and_strict_config_rng_optimizer_rejections(self):
        initial, saved0 = trainer.load_checkpoint(self.output/'step-0.pt')
        reference, ref0 = trainer.load_checkpoint(self.reference/'step-0.pt')
        self.assertTrue(all(torch.equal(saved0['model'][key], value) for key, value in ref0['model'].items()))
        self.assertTrue(torch.equal(saved0['torch_rng'], ref0['torch_rng']))
        self.assertEqual(saved0['numpy_rng'], ref0['numpy_rng'])
        args = sequence_inputs(motion_fixture(), [0, 1, 2])
        self.assertTrue(torch.equal(initial(*args)['states'], reference(*args)['states']))
        _, final = trainer.load_checkpoint(self.output/'final.pt')
        self.assertEqual(final['step'], 3)
        self.assertEqual(final['augmentation_rng'], rows(self.output)[-1]['augmentation_rng_after'])
        for mutate in (lambda x: x.pop('augmentation_rng'),
                       lambda x: x.update(augmentation_rng={'seed': 1}),
                       lambda x: x.update(augmentation_applied_only_to_training=False),
                       lambda x: x.update(actual_updates=2),
                       lambda x: x['configuration']['augmentation'].update(probability=0.),
                       lambda x: x.update(augmentation=dict(kind='wrong', probability=.5, seed=20344023))):
            invalid = copy.deepcopy(final); mutate(invalid)
            path = self.root/'invalid.pt'; torch.save(invalid, path)
            with self.assertRaises(ValueError): trainer.load_checkpoint(path)
        bad = copy.deepcopy(ref0); bad['augmentation_rng'] = final['augmentation_rng']
        torch.save(bad, self.root/'invalid.pt')
        with self.assertRaises(ValueError): trainer.load_checkpoint(self.root/'invalid.pt')

    def test_zero_probability_lowlevel_full_training_regresses_original_exactly(self):
        # Production config remains frozen .5. Only this low-level draw is
        # replaced to prove the ordinary training path has no other intervention.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'zero'
            def zero(members, rng, probability): return drop_helper_fragments(members, rng, 0.)
            with patch.object(trainer, 'drop_helper_fragments', side_effect=zero):
                trainer.train_sequence_position(self.train, dropout_configuration(), path,
                    paired_reference_fit=self.reference)
            _, final = trainer.load_checkpoint(path/'final.pt')
            _, old = trainer.load_checkpoint(self.reference/'final.pt')
            self.assertTrue(all(torch.equal(final['model'][key], value) for key, value in old['model'].items()))
            self.assertEqual([r['loss'] for r in rows(path)], [r['loss'] for r in rows(self.reference)])

    def test_optimizer_failure_does_not_count_dropout_exposure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'failed'
            with patch.object(torch.optim.Adam, 'step', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    trainer.train_sequence_position(self.train, dropout_configuration(), path,
                        paired_reference_fit=self.reference)
            run = json.loads((path/'run.json').read_text())
            self.assertEqual(run['optimizer_updates'], 0)
            self.assertEqual(run['augmentation_exposure']['sampled_anchors'], 0)
            self.assertEqual(run['augmentation_exposure']['dropped_helpers'], 0)
            self.assertFalse((path/'training.jsonl').exists())
            self.assertFalse((path/'final.pt').exists())
            _, saved = trainer.load_checkpoint(path/'interrupted.pt')
            self.assertEqual(saved['step'], 0)

    def test_zero_supervision_is_finite_and_keeps_original_mask(self):
        train = tiny_training()
        for scene in train:
            for target in scene['fusion_targets'].values(): target['valid'][:, :2] = False
        with tempfile.TemporaryDirectory() as temp:
            control, output = Path(temp)/'control', Path(temp)/'drop'
            trainer.train_sequence_position(train, tiny_configuration('transformer'), control)
            result = trainer.train_sequence_position(train, dropout_configuration(), output,
                paired_reference_fit=control)
            self.assertEqual(result['exposure']['valid_position_points'], 0)
            self.assertEqual(result['augmentation_exposure']['retained_support']['valid_position_queries'], dict(zero=0, one=0, multiple=0))
            self.assertTrue(all(row['loss'] == 0 for row in rows(output)))

    def test_production_configuration_rejects_probability_seed_and_backbone_changes(self):
        for mutate in (lambda x: x['augmentation'].update(probability=0.),
                       lambda x: x['augmentation'].update(probability=True),
                       lambda x: x['augmentation'].update(seed=x['training']['seed']),
                       lambda x: x['model'].update(backbone='lstm'),
                       lambda x: x.update(position_loss='linear_relative_harm'),
                       lambda x: x.update(scope=trainer.SCOPE)):
            config = dropout_configuration(); mutate(config)
            with self.assertRaises(ValueError): trainer.configuration(config)


if __name__ == '__main__': unittest.main()
