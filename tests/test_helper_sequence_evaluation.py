"""078 actual tiny fits, source replay, all-group inference and cached comparison."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.helper_sequence_position import (BUCKETS, POPULATIONS,
    evaluate_helper_sequence, helper_population)
from association.models.helper_sequence import HelperSequencePosition
from association.utils.runtime import read_json, write_json
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_train
from tools.experiments import helper_sequence_fit as entry


class HelperSequenceEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_groups_initial_and_helper_population_with_sparse_truth(self):
        for mode in entry.CONFIGS:
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root = Path(directory)
                source, metadata, validation = support_gate_cache_fixture(root, sparse=True)
                model = HelperSequencePosition(aggregation=mode)
                with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                    result = evaluate_helper_sequence(model, validation, source, metadata, root/'out', initial=True)
                self.assertEqual(result['model_forwards'], 4)  # Includes duplicate and missing truth.
                self.assertEqual(result['aggregation'], mode)
                self.assertEqual(result['helper_population']['all_output_group_queries']['total'], 256)
                self.assertEqual(result['helper_population']['unique_selected_valid_position_queries']['total'], 1)
                self.assertEqual(sum(result['helper_position_strata'][bucket]['learned_sequence']['count']
                                     for bucket in BUCKETS), 1)
                self.assertEqual(result['pairwise']['learned_sequence__minus__fixed_prior095']['all']['signed_error_change_sum'], 0.)
                with np.load(root/'out/predictions/states-3035.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['group_helper_offsets'], [0, 1, 2, 2, 2])
                    np.testing.assert_array_equal(saved['group_helper_node_indices'], [1, 3])
                    self.assertEqual(saved['learned_sequence_helper_feature_weights'].shape, (2, 64))
                with self.assertRaises(FileExistsError):
                    evaluate_helper_sequence(model, validation, source, metadata, root/'out')
                self.assertEqual(read_json(root/'out/run.json')['status'], 'COMPLETED')

    def test_population_three_buckets_duplicates_and_truth_validity(self):
        support = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]], bool)
        fragment = dict(group_supported_helper_count=np.array([[2, 1, 0], [0, 0, 0]]),
            group_helper_offsets=np.array([0, 2, 2]), group_helper_node_indices=np.array([1, 2]),
            group_anchor_node_indices=np.array([0, 3]),
            learned_sequence_helper_feature_weights=np.array([[.5, 1., 0.], [.5, 0., 0.]]),
            learned_assoc_group_offsets=np.array([0, 3, 4]),
            learned_assoc_group_node_indices=np.array([0, 1, 2, 3]), support=support)
        valid = np.ones((2, 3, 4), bool)
        valid[1] = False
        system = dict(anchor_target_ids=np.array(['a', 'b']),
            anchor_aligned_group_indices=np.array([0, -1, 1]),
            anchor_aligned_truth_valid=np.concatenate((valid, np.zeros((1, 3, 4), bool))),
            anchor_aligned_prediction_valid=np.ones((3, 3, 4), bool),
            anchor_group_anchor_target_indices=np.array([0, 0]), anchor_group_valid=np.ones((2, 3, 4), bool))
        result = helper_population(fragment, system)
        self.assertEqual(result['all_output_group_queries']['counts'], dict(zero=4, one=1, two_or_more=1))
        self.assertEqual(result['all_output_group_valid_position_queries']['total'], 6)
        self.assertEqual(result['unique_selected_valid_position_queries']['counts'], dict(zero=1, one=1, two_or_more=1))
        fragment['group_supported_helper_count'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'helper counts'):
            helper_population(fragment, system)

    def test_invalid_feature_weights_reject_and_no_output_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root)
            model = HelperSequencePosition()
            forward = model.forward
            def corrupt(*args):
                result = forward(*args)
                result['helper_weights'] = result['helper_weights'] + 1
                return result
            with patch.object(model, 'forward', side_effect=corrupt), \
                    patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                with self.assertRaisesRegex(ValueError, 'weights'):
                    evaluate_helper_sequence(model, validation, source, metadata, root/'out')
            self.assertEqual(read_json(root/'out/run.json')['status'], 'FAILED')

    def test_production_contract_exact_and_source_failure_before_archive(self):
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        for filename in entry.CONFIGS.values():
            config = read_json(entry.ROOT/'configs'/filename)
            entry.production_contract(config, manifest, alignment)
            altered = deepcopy(config)
            altered['training']['updates'] = 3999
            with self.assertRaisesRegex(ValueError, 'frozen'):
                entry.production_contract(altered, manifest, alignment)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(source_system=str(root/'source056/system'), paired_control=str(root/'old'), output=str(root/'out'),
                archive=['unread.zip'], manifest=str(entry.ROOT/'configs/data/mtad-level1-scene3000.json'),
                alignment=str(entry.ROOT/'configs/data/mtad-level1-alignment.json'),
                config=str(entry.ROOT/'configs'/entry.CONFIGS['post_pool']), device='cpu', cpu_threads=4, time_policy='advisory')
            with patch.object(entry, '_bind_source', side_effect=ValueError('source failure')), \
                    patch.object(entry, 'load_timed_dataset') as archive, patch.object(entry, 'train_helper_sequence') as train:
                with self.assertRaisesRegex(ValueError, 'source failure'):
                    entry.run(args)
                archive.assert_not_called(); train.assert_not_called()
            self.assertEqual(read_json(root/'out/run.json')['status'], 'FAILED')

    def test_replica_contract_fixed_seed_only_bc_and_no_mixed_profile(self):
        from tools.analysis import compare_helper_sequence as compare
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        for mode, filename in entry.REPLICA_CONFIGS.items():
            replica = read_json(entry.ROOT/'configs'/filename)
            original = read_json(entry.ROOT/'configs'/entry.CONFIGS[mode])
            expected = deepcopy(original)
            expected['training']['seed'] = 20261021
            self.assertEqual(replica, expected)
            entry.production_contract(replica, manifest, alignment, profile='replica081')
            for config, profile in ((replica, 'screen078'), (original, 'replica081')):
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    entry.production_contract(config, manifest, alignment, profile=profile)
                # This is a real metadata rejection, before weights or arrays
                # can be read. A config from another seed must not enter a table.
                with TemporaryDirectory() as directory:
                    path = Path(directory)
                    (path/'fit').mkdir()
                    for name in ('run.json', 'summary.json', 'fit/run.json'):
                        write_json(path/name, dict(configuration=config, profile=profile))
                    with self.assertRaisesRegex(ValueError, 'frozen'):
                        compare.helper_metadata(path, profile=profile)
            with TemporaryDirectory() as directory:
                path = Path(directory)
                (path/'fit').mkdir()
                for name in ('run.json', 'summary.json', 'fit/run.json'):
                    write_json(path/name, dict(configuration=replica, profile='screen078'))
                with self.assertRaisesRegex(ValueError, 'profile differs'):
                    compare.helper_metadata(path, profile='replica081')
        post = read_json(entry.ROOT/'configs'/entry.CONFIGS['post_pool'])
        post['training']['seed'] = 20261021
        with self.assertRaisesRegex(ValueError, 'frozen'):
            entry.production_contract(post, manifest, alignment, profile='replica081')

    def test_actual_control_three_helper_fits_endpoints_and_comparison(self):
        self._actual_control_fits_endpoints_and_comparison('screen078')

    def test_actual_replica_control_two_helper_fits_endpoints_and_comparison(self):
        self._actual_control_fits_endpoints_and_comparison('replica081')

    def _actual_control_fits_endpoints_and_comparison(self, profile):
        from tools.experiments import sequence_position_fit as original
        from tools.analysis import compare_helper_sequence as compare
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root)
            train = tiny_train()
            manifest = dict(splits={'T_core': [str(s['scene_id']) for s in train], 'V_select': ['3035']},
                            components={'3035': 'shared'})
            linear_profile, linear_filename, reference_arm = entry.PROFILE_REFERENCES[profile]
            modes = entry.PROFILE_CONFIGS[profile]
            linear = read_json(entry.ROOT/'configs'/linear_filename)
            linear['training'].update(updates=3, batch_size=2)
            for name, value in (('linear', linear), ('manifest', manifest), ('alignment', {})):
                write_json(root/(name+'.json'), value)
            args = Namespace(source_system=str(source), output=str(root/'linear'), paired_control=None,
                archive=['unread.zip'], manifest=str(root/'manifest.json'), alignment=str(root/'alignment.json'),
                config=str(root/'linear.json'), device='cpu', cpu_threads=4, time_policy='advisory', profile=linear_profile)
            dataset = lambda *a, **k: dict(train=train, validation=validation, report={'fixture': True})
            with patch.object(original, 'production_contract'), patch.object(original, '_bind_source'), \
                    patch.object(original, '_load_metadata', return_value=metadata), \
                    patch.object(original, 'load_timed_dataset', side_effect=dataset), \
                    patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                original.run(args)
            configs = {}
            args.profile = profile
            for mode, filename in modes.items():
                config = read_json(entry.ROOT/'configs'/filename)
                config['training'].update(updates=3, batch_size=2)
                configs[mode] = config
                write_json(root/(mode+'.json'), config)
                args.config, args.output, args.paired_control = str(root/(mode+'.json')), str(root/mode), str(root/'linear')
                with patch.object(entry, 'production_contract'), patch.object(entry, '_bind_source'), \
                        patch.object(entry, '_load_metadata', return_value=metadata), \
                        patch.object(entry, 'load_timed_dataset', side_effect=dataset), \
                        patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                    result = entry.run(args)
                self.assertEqual(result['status'], 'COMPLETED')
                self.assertEqual(result['profile'], profile)
                self.assertEqual(result['training_pairing']['actual_successful_batches_matched'], 3)
                self.assertTrue(read_json(root/mode/'endpoint-comparison.json')['population_and_masks_exact'])
                self.assertEqual(read_json(root/mode/'run.json')['completed_endpoints'], ['initial', 'final'])
            def meta(path):
                path = Path(path)
                final = read_json(path/'final/summary.json')
                return dict(path=path, method='learned_sequence', configuration=read_json(path/'run.json')['configuration'],
                    source_system=str(source), scene_ids=['3035'], components={'3035': 'shared'}, summary=final)
            # Only production-scale manifest/config registry binding is replaced.
            # Real log replay, initial tensor checks, cache correspondence, H/I,
            # helper cohorts and complete-target metric reconstruction all run.
            with patch.object(compare, '_metadata', side_effect=lambda path, name: meta(path)) as base_binding, \
                    patch.object(compare, 'helper_metadata', side_effect=lambda path, profile: (Path(path).name, meta(path))):
                result = compare.run(Namespace(helper_run=[str(root/mode) for mode in modes],
                    linear_run=str(root/'linear'), output=str(root/'comparison'), cpu_threads=4, profile=profile))
                if profile == 'replica081':
                    single = compare.run(Namespace(helper_run=[str(root/'pre_pool')],
                        linear_run=str(root/'linear'), output=str(root/'single-comparison'),
                        cpu_threads=4, profile=profile))
                    self.assertFalse(single['complete_profile'])
                    self.assertNotIn('weighted_pool__minus__pre_pool', single['relative_harm']['all'])
            self.assertEqual(base_binding.call_args.args[1], reference_arm)
            self.assertEqual(result['profile'], profile)
            self.assertTrue(result['complete_profile'])
            self.assertEqual(result['complete_abc'], profile == 'screen078')
            self.assertEqual(result['single_seed_screen'], profile == 'screen078')
            self.assertEqual(result['pairing']['matched_updates'], 3)
            self.assertEqual(result['position_population']['covered_target_points'], 127)
            self.assertEqual(sum(result['helper_cohorts'][bucket]['linear']['point_equal']['unit_count']
                                 for bucket in BUCKETS), 127)
            self.assertEqual('pre_pool__minus__post_pool' in result['relative_harm']['all'], profile == 'screen078')
            self.assertIn('weighted_pool__minus__pre_pool', result['relative_harm']['all'])
            self.assertEqual(result['helper_population']['all_output_group_queries']['total'], 256)
            # Tampering an actually used sample must fail even when metrics exist.
            log = root/'pre_pool/fit/training.jsonl'
            text = log.read_text()
            log.write_text(text.replace('"scene_id": "train-a"', '"scene_id": "invalid"'))
            # Deterministically alter the first batch member independent of fixture IDs.
            import json
            lines = [json.loads(line) for line in log.read_text().splitlines()]
            lines[0]['batch'][0]['node_id'] += '-tamper'
            log.write_text('\n'.join(json.dumps(line) for line in lines)+'\n')
            with self.assertRaises(ValueError):
                compare.verify_pairing({'linear': meta(root/'linear'), 'pre_pool': meta(root/'pre_pool')})


if __name__ == '__main__':
    unittest.main()
