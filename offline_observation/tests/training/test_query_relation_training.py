"""098 real warm-start fits, exact replay, checkpoint and failure boundaries."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from association.models.helper_sequence import helper_sequence_inputs
from association.models.query_relation import MODES, ROUTING_SEMANTICS, SCOPE
from association.models.sequence_position import sequence_inputs
from association.training import query_relation as training
from association.training import sequence_position as source_training
from association.utils.runtime import read_json
from tests.training.test_sequence_position import motion_fixture, tiny_configuration, tiny_training


def query_config(mode='generic', *, updates=5, seed=20261023):
    config = tiny_configuration('transformer')
    config.update(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS)
    config['model'].pop('backbone')
    config['model']['mode'] = mode
    config['training'].update(updates=updates, seed=seed)
    return config


def make_source(root, train, config):
    source_training.train_sequence_position(
        train, training.source_configuration(config), root/'reference')
    return root/'reference'


class QueryRelationTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_real_two_modes_match_final_source_batches_rng_and_all_parameter_training(self):
        train = tiny_training()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = make_source(root, train, query_config())
            source_model, source = source_training.load_checkpoint(source_path/'final.pt')
            initial_source = torch.load(source_path/'step-0.pt', weights_only=False)
            self.assertFalse(torch.equal(source['model']['output.weight'], initial_source['model']['output.weight']))
            originals = (source_path/'final.pt').read_bytes()
            old_rows = [json.loads(line) for line in (source_path/'training.jsonl').read_text().splitlines()]
            initials, results, final_rngs = [], [], []
            for mode in MODES:
                with patch.object(training, 'HEARTBEAT_INTERVAL', 2):
                    result = training.train_query_relation(train, query_config(mode), root/mode,
                        paired_reference_fit=source_path)
                self.assertEqual(result['status'], 'COMPLETED')
                self.assertEqual(result['optimizer_updates'], 5)
                self.assertEqual(result['total_stage_updates'], 10)
                self.assertEqual(result['branch_seed'], 20261023+98000)
                self.assertFalse(result['teacher_frozen'])
                self.assertEqual(result['parameter_count'], result['trainable_parameter_count'])
                self.assertTrue(result['paired_control']['source_final_parameters_exact'])
                self.assertTrue(result['paired_control']['initial_sampler_rng_exact'])
                self.assertTrue(result['paired_control']['final_sampler_rng_exact'])
                self.assertTrue(result['paired_control']['sampling_exposure_exact'])
                self.assertFalse(result['paired_control']['teacher_unchanged'])
                self.assertEqual(result['paired_control']['actual_successful_batches_matched'], 5)
                counts = result['helper_support_exposure']
                self.assertEqual(sum(counts['all_queries'].values()), 5*2*64)
                self.assertEqual(sum(counts['valid_position_queries'].values()), result['exposure']['valid_position_points'])
                self.assertEqual(result['helper_group_exposure']['sampled_anchors'], 10)
                self.assertGreater(result['helper_group_exposure']['any_multi_queries'], 0)
                self.assertGreater(result['helper_gradients']['branch_projection_nonzero_updates'], 0)
                self.assertGreater(result['helper_gradients']['phi_nonzero_updates'], 0)
                initial_model, initial = training.load_checkpoint(root/mode/'step-0.pt')
                model, final = training.load_checkpoint(root/mode/'final.pt')
                self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
                self.assertEqual(initial['optimizer']['state'], {})
                self.assertEqual(initial['numpy_rng'], initial_source['numpy_rng'])
                self.assertEqual(final['numpy_rng'], source['numpy_rng'])
                self.assertEqual(final['step'], 5)
                self.assertTrue(all(float(entry['step']) == 5 for entry in final['optimizer']['state'].values()))
                for name, tensor in source['model'].items():
                    self.assertTrue(torch.equal(initial['model'][name], tensor), name)
                    self.assertTrue(torch.equal(final['source_teacher']['model'][name], tensor), name)
                self.assertEqual(torch.count_nonzero(initial['model']['branch_projection.weight']), 0)
                self.assertTrue(any(not torch.equal(final['model'][name], tensor)
                                    for name, tensor in source['model'].items()))
                with torch.no_grad():
                    motion = motion_fixture()
                    args = helper_sequence_inputs(motion, [0, 1, 2])
                    expected = source_model(*sequence_inputs(motion, [0, 1, 2]))['states']
                    self.assertTrue(torch.equal(initial_model(*args)['states'], expected))
                    self.assertTrue(torch.equal(model(*args)['states'][:, 2:], args[2][:, 2:]))
                rows = [json.loads(line) for line in (root/mode/'training.jsonl').read_text().splitlines()]
                self.assertEqual([row['batch'] for row in rows], [row['batch'] for row in old_rows])
                self.assertEqual([row['mode'] for row in rows], [row['mode'] for row in old_rows])
                self.assertTrue(all(torch.isfinite(torch.tensor(row['gradient_norm'])) for row in rows))
                heartbeat = read_json(root/mode/'heartbeat.json')
                self.assertEqual((heartbeat['status'], heartbeat['phase'], heartbeat['optimizer_updates']),
                                 ('COMPLETED', 'completed', 5))
                self.assertNotIn('scenes', heartbeat)
                self.assertEqual(read_json(root/mode/'training-windows.json')['windows'][0]['successful_updates'], 5)
                initials.append(initial['model'])
                results.append(result)
                final_rngs.append(final['numpy_rng'])
            self.assertEqual(set(initials[0]), set(initials[1]))
            self.assertTrue(all(torch.equal(value, initials[1][name]) for name, value in initials[0].items()))
            self.assertEqual(results[0]['parameter_count'], results[1]['parameter_count'])
            self.assertEqual(results[0]['helper_support_exposure'], results[1]['helper_support_exposure'])
            self.assertEqual(results[0]['helper_group_exposure'], results[1]['helper_group_exposure'])
            self.assertEqual(final_rngs[0], final_rngs[1])
            self.assertEqual((source_path/'final.pt').read_bytes(), originals)

    def test_checkpoint_rejects_source_initialization_optimizer_and_rng_corruption(self):
        train = tiny_training()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = query_config(updates=3)
            source = make_source(root, train, config)
            training.train_query_relation(train, config, root/'fit', paired_reference_fit=source)
            _, final = training.load_checkpoint(root/'fit/final.pt')
            _, initial = training.load_checkpoint(root/'fit/step-0.pt')
            def reject(saved, message):
                torch.save(saved, root/'bad.pt')
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    training.load_checkpoint(root/'bad.pt')
            changed = copy.deepcopy(initial)
            changed['model']['output.bias'][0] += .1
            reject(changed, 'source-final')
            changed = copy.deepcopy(initial)
            changed['model']['branch_projection.weight'][0, 0] = 1.
            reject(changed, 'projection')
            changed = copy.deepcopy(initial)
            changed['model']['phi.0.weight'][0, 0] += .1
            reject(changed, 'initialization seed')
            changed = copy.deepcopy(final)
            changed['source_teacher']['model']['output.weight'][0, 0] = torch.nan
            reject(changed, 'finite')
            changed = copy.deepcopy(final)
            changed['model']['output.weight'] = changed['model']['output.weight'].double()
            reject(changed, 'dtypes')
            changed = copy.deepcopy(final)
            changed['source_teacher']['configuration']['training']['seed'] += 1
            reject(changed, 'source-final')
            changed = copy.deepcopy(final)
            next(iter(changed['optimizer']['state'].values()))['step'] -= 1
            reject(changed, 'successful update count')
            changed = copy.deepcopy(final)
            first = next(iter(changed['optimizer']['state']))
            changed['optimizer']['state'][first]['exp_avg'] = torch.zeros(1)
            reject(changed, 'moment')
            changed = copy.deepcopy(final)
            changed['numpy_rng'] = {'bit_generator': 'not-valid'}
            reject(changed, 'RNG')
            changed = copy.deepcopy(final)
            changed['torch_rng'] = torch.zeros(3, dtype=torch.uint8)
            reject(changed, 'RNG')
            changed = copy.deepcopy(final)
            changed['optimizer']['param_groups'][0]['params'][0] = changed['optimizer']['param_groups'][0]['params'][1]
            reject(changed, 'all model parameters')

    def test_optimizer_failure_keeps_zero_successful_exposure_and_real_recovery_state(self):
        train = tiny_training()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = query_config(updates=3)
            source = make_source(root, train, config)
            with patch.object(torch.optim.Adam, 'step', side_effect=RuntimeError('optimizer failure')):
                with self.assertRaisesRegex(RuntimeError, 'optimizer failure'):
                    training.train_query_relation(train, config, root/'failed', paired_reference_fit=source)
            record = read_json(root/'failed/run.json')
            self.assertEqual(record['status'], 'FAILED')
            self.assertEqual(record['optimizer_updates'], 0)
            self.assertEqual(record['exposure']['sampled_anchors'], 0)
            self.assertEqual(sum(record['helper_support_exposure']['all_queries'].values()), 0)
            self.assertEqual(record['helper_group_exposure']['sampled_anchors'], 0)
            self.assertEqual((root/'failed/training.jsonl').read_text(), '')
            self.assertFalse((root/'failed/final.pt').exists())
            _, saved = training.load_checkpoint(root/'failed/interrupted.pt')
            self.assertEqual(saved['step'], 0)
            self.assertEqual(saved['optimizer']['state'], {})
            self.assertEqual(read_json(root/'failed/heartbeat.json')['status'], 'FAILED')

    def test_failed_second_update_commits_only_first_actual_batch(self):
        train = tiny_training()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = query_config(updates=3)
            source = make_source(root, train, config)
            original_step = torch.optim.Adam.step
            calls = []
            def fail_second(optimizer, *args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('second update stopped')
                return original_step(optimizer, *args, **kwargs)
            with patch.object(torch.optim.Adam, 'step', new=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'second update stopped'):
                    training.train_query_relation(train, config, root/'failed', paired_reference_fit=source)
            record = read_json(root/'failed/run.json')
            self.assertEqual(record['optimizer_updates'], 1)
            self.assertEqual(record['exposure']['successful_updates'], 1)
            self.assertEqual(record['exposure']['sampled_anchors'], 2)
            self.assertEqual(sum(record['helper_support_exposure']['all_queries'].values()), 2*64)
            self.assertEqual(record['helper_group_exposure']['sampled_anchors'], 2)
            self.assertEqual(len((root/'failed/training.jsonl').read_text().splitlines()), 1)
            _, saved = training.load_checkpoint(root/'failed/interrupted.pt')
            self.assertEqual(saved['step'], 1)
            self.assertTrue(all(float(entry['step']) == 1 for entry in saved['optimizer']['state'].values()))

    def test_wrong_split_source_and_hidden_configuration_changes_fail_before_output(self):
        train = tiny_training()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = query_config(updates=3)
            source = make_source(root, train, config)
            changed = copy.deepcopy(config)
            changed['training']['seed'] += 1
            with self.assertRaises(ValueError):
                training.train_query_relation(train, changed, root/'seed', paired_reference_fit=source)
            self.assertFalse((root/'seed').exists())
            invalid_train = copy.deepcopy(train)
            invalid_train[0]['split'] = 'V_select'
            with self.assertRaisesRegex(ValueError, 'T_core'):
                training.train_query_relation(invalid_train, config, root/'split', paired_reference_fit=source)
            self.assertFalse((root/'split').exists())
            source_final = torch.load(source/'final.pt', weights_only=False)
            damaged = copy.deepcopy(source_final)
            next(iter(damaged['optimizer']['state'].values()))['exp_avg'].view(-1)[0] = torch.nan
            torch.save(damaged, source/'final.pt')
            with self.assertRaisesRegex(ValueError, 'moment'):
                training.train_query_relation(train, config, root/'bad-source', paired_reference_fit=source)
            self.assertFalse((root/'bad-source').exists())
            torch.save(source_final, source/'final.pt')
            source_record = read_json(source/'run.json')
            source_record['status'] = 'RUNNING'
            (source/'run.json').write_text(json.dumps(source_record))
            with patch.object(torch, 'load', side_effect=AssertionError('must not load active source')):
                with self.assertRaises(ValueError):
                    training.train_query_relation(train, config, root/'active', paired_reference_fit=source)
            self.assertFalse((root/'active').exists())
        with self.assertRaises(ValueError):
            training.configuration(tiny_configuration('transformer'))
        for key, value in (('mode', 'weighted_pool'), ('hidden_size', 128), ('num_layers', 1),
                           ('backbone', 'transformer'), ('initial_gate', True)):
            changed = query_config()
            changed['model'][key] = value
            with self.assertRaises(ValueError):
                training.configuration(changed)


if __name__ == '__main__':
    unittest.main()
