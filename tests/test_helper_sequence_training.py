import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from association.models.helper_sequence import SCOPE, ROUTING_SEMANTICS, AGGREGATIONS
from association.training.helper_sequence import configuration, load_checkpoint, train_helper_sequence
from association.training.sequence_position import train_sequence_position
from tests.test_sequence_position import tiny_configuration, tiny_training


def helper_config(mode='post_pool', updates=5):
    config = tiny_configuration('transformer')
    config.update(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS)
    config['model'].pop('backbone')
    config['model']['aggregation'] = mode
    config['training']['updates'] = updates
    return config


class HelperTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_three_real_fits_match_original_sampling_and_initial_common_parameters(self):
        train = tiny_training()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = tiny_configuration('transformer')
            reference['training']['updates'] = 5
            train_sequence_position(train, reference, root/'reference')
            originals = torch.load(root/'reference'/'step-0.pt', weights_only=False)
            logs, initials = [], []
            for mode in AGGREGATIONS:
                out = root/mode
                result = train_helper_sequence(train, helper_config(mode), out,
                                               paired_reference_fit=root/'reference')
                self.assertEqual(result['status'], 'COMPLETED')
                self.assertEqual(result['paired_control']['actual_successful_batches_matched'], 5)
                self.assertTrue(result['paired_control']['sampling_exposure_exact'])
                counts = result['helper_support_exposure']
                self.assertEqual(sum(counts['all_queries'].values()), 5*2*64)
                self.assertEqual(sum(counts['valid_position_queries'].values()), result['exposure']['valid_position_points'])
                self.assertEqual(result['parameter_count'], 107618 if mode == 'weighted_pool' else 107602)
                _, final = load_checkpoint(out/'final.pt')
                _, initial = load_checkpoint(out/'step-0.pt')
                self.assertEqual(final['actual_updates'], 5)
                self.assertFalse(torch.equal(initial['model']['output.weight'], final['model']['output.weight']))
                for key, value in originals['model'].items():
                    actual = initial['model'][key]
                    if key == 'input_projection.0.weight':
                        self.assertEqual(torch.count_nonzero(actual[:, 76:]).item(), 0)
                        actual = actual[:, :76]
                    self.assertTrue(torch.equal(value, actual), key)
                initials.append(initial['model'])
                logs.append([json.loads(line)['batch'] for line in (out/'training.jsonl').read_text().splitlines()])
                invalid = copy.deepcopy(final)
                invalid['routing_semantics'] = 'old-schema'
                torch.save(invalid, out/'invalid.pt')
                with self.assertRaises(ValueError):
                    load_checkpoint(out/'invalid.pt')
            self.assertEqual(logs[0], logs[1])
            self.assertEqual(logs[1], logs[2])
            for key in initials[0]:
                self.assertTrue(torch.equal(initials[0][key], initials[1][key]), key)
                self.assertTrue(torch.equal(initials[1][key], initials[2][key]), key)

    def test_failed_optimizer_counts_no_helper_exposure_and_wrong_control_rejected(self):
        train = tiny_training()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = tiny_configuration('transformer')
            reference['training']['updates'] = 5
            train_sequence_position(train, reference, root/'reference')
            with patch.object(torch.optim.Adam, 'step', side_effect=RuntimeError('optimizer failure')):
                with self.assertRaisesRegex(RuntimeError, 'optimizer failure'):
                    train_helper_sequence(train, helper_config(), root/'failed', paired_reference_fit=root/'reference')
            record = json.loads((root/'failed'/'run.json').read_text())
            self.assertEqual(record['optimizer_updates'], 0)
            self.assertEqual(sum(record['helper_support_exposure']['all_queries'].values()), 0)
            _, saved = load_checkpoint(root/'failed'/'interrupted.pt')
            self.assertEqual(saved['step'], 0)
            bad = helper_config()
            bad['training']['seed'] += 1
            with self.assertRaises(ValueError):
                train_helper_sequence(train, bad, root/'wrong-seed', paired_reference_fit=root/'reference')
            self.assertFalse((root/'wrong-seed').exists())
            train[0]['split'] = 'V_select'
            with self.assertRaises(ValueError):
                train_helper_sequence(train, helper_config(), root/'validation', paired_reference_fit=root/'reference')

    def test_old_schema_and_hidden_architecture_changes_not_accepted(self):
        with self.assertRaises(ValueError):
            configuration(tiny_configuration('transformer'))
        for key, value in (('aggregation', 'max_pool'), ('hidden_size', 128), ('backbone', 'transformer')):
            config = helper_config()
            config['model'][key] = value
            with self.assertRaises(ValueError):
                configuration(config)


if __name__ == '__main__':
    unittest.main()
