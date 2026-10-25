"""098 real tiny training, completed budget binding and full cache comparison."""
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.utils.runtime import read_json, write_json
from association.evaluation import query_relation as evaluation
from association.models.query_relation import QueryRelationPosition
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_train
from tools.experiments import sequence_position_fit as original
from tools.experiments import frozen_position_adapter_fit as budget_entry
from tools.experiments import query_relation_fit as entry
from tools.analysis import compare_frozen_position_adapter as budget_compare
from tools.analysis import compare_query_relation as compare
from tools.analysis import summarize_query_relation as closeout


class QueryRelationPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_four_production_configs_and_unplanned_budget_rejected(self):
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        self.assertEqual(set(entry.EXPERIMENTS.values()), {'098A', '098B', '098C', '098D'})
        for key, filename in entry.CONFIGS.items():
            config = read_json(entry.ROOT/'configs'/filename)
            self.assertEqual(key, (config['training']['seed'], config['model']['mode']))
            entry.production_contract(config, manifest, alignment)
            config['training']['updates'] += 1
            with self.assertRaises(ValueError):
                entry.production_contract(config, manifest, alignment)

    def test_endpoint_postprocessing_failure_cannot_leave_completed_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'endpoint'
            output.mkdir()
            write_json(output/'run.json', dict(status='COMPLETED'))
            with patch.object(evaluation, 'evaluate_helper_sequence', return_value={}):
                with self.assertRaisesRegex(ValueError, 'incomplete scene traversal'):
                    evaluation.evaluate_query_relation(QueryRelationPosition(), [], root/'source',
                        {'scene_ids': ['3035']}, output,
                        source_teacher_checkpoint=root/'teacher/fit/final.pt')
            self.assertEqual(read_json(output/'run.json')['status'], 'FAILED')

    def test_real_source_budget_two_modes_full_endpoints_and_independent_cached_table(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root)
            write_json(source/'run.json', dict(status='COMPLETED'))
            train = tiny_train()
            manifest = dict(splits={'T_core': [str(s['scene_id']) for s in train], 'V_select': ['3035']},
                components={'3035': 'shared'})
            base = read_json(entry.ROOT/'configs/sequence-position-transformer-extended-level1.json')
            base['training'].update(updates=4, batch_size=2)
            for name, value in (('config', base), ('manifest', manifest), ('alignment', {})):
                write_json(root/(name+'.json'), value)
            args = Namespace(source_system=str(source), output=str(root/'linear'), paired_control=None,
                budget_control=None, archive=['fixture-no-archive.zip'], manifest=str(root/'manifest.json'),
                alignment=str(root/'alignment.json'), config=str(root/'config.json'),
                device='cpu', cpu_threads=4, time_policy='advisory', profile='extended4000')
            dataset = lambda *a, **k: dict(train=train, validation=validation, report={'tiny_fixture': True})

            def meta(path, arm=None):
                path = Path(path).resolve()
                return dict(path=path, method='learned_sequence', source_system=str(source.resolve()),
                    scene_ids=['3035'], components={'3035': 'shared'},
                    summary=read_json(path/'final/summary.json'))

            # Only source file loading, fixed-production size and raw-observation
            # reconstruction are fixture seams. All forwards, fits, checkpoint
            # binding, successful batches, group alignment and metrics are real.
            stack.enter_context(patch('association.evaluation.sequence_position.frozen_motion',
                side_effect=lambda raw, *_: raw))
            for module in (original, budget_entry, entry):
                stack.enter_context(patch.object(module, 'production_contract'))
                stack.enter_context(patch.object(module, '_bind_source'))
                stack.enter_context(patch.object(module, '_load_metadata', return_value=metadata))
                stack.enter_context(patch.object(module, 'load_timed_dataset', side_effect=dataset))
            for module in (budget_entry, entry, compare):
                stack.enter_context(patch.object(module, 'frozen_configuration', side_effect=lambda cfg: cfg))
            stack.enter_context(patch.object(budget_compare, '_metadata', side_effect=meta))
            stack.enter_context(patch.object(entry, 'source_metadata', side_effect=meta))
            stack.enter_context(patch.object(compare, '_metadata', side_effect=meta))
            for module in (budget_compare, compare, closeout):
                stack.enter_context(patch.object(module, 'declared_manifest', return_value=manifest))

            original.run(args)
            args.paired_control = str(root/'linear')
            config = read_json(entry.ROOT/'configs/frozen-position-adapter-full-finetune-level1.json')
            config['training'].update(updates=4, batch_size=2)
            write_json(root/'config.json', config)
            args.output = str(root/'budget')
            budget_entry.run(args)
            args.budget_control = str(root/'budget')
            for mode in ('generic', 'relation'):
                config = read_json(entry.ROOT/'configs'/entry.CONFIGS[(20261023, mode)])
                config['training'].update(updates=4, batch_size=2)
                write_json(root/'config.json', config)
                args.output = str(root/mode)
                check = entry.check_source(args)
                self.assertEqual(check['budget_pairing']['status'], 'PAIR_VERIFIED')
                self.assertFalse(Path(args.output).exists())
                result = entry.run(args)
                self.assertEqual(result['status'], 'COMPLETED')
                self.assertEqual(result['source_teacher_forwards'], 0)
                self.assertEqual(result['parameter_count'], result['trainable_parameter_count'])
                self.assertFalse(result['test_opened'])
                labels = ['initial', 'final'] + (['relation_off'] if mode == 'relation' else [])
                self.assertEqual(read_json(root/mode/'run.json')['completed_endpoints'], labels)
                for label in labels:
                    self.assertEqual(read_json(root/mode/label/'run.json')['model_forwards'], 4)
                with self.assertRaises(FileExistsError):
                    entry.run(args)

            params = Namespace(linear_run=str(root/'linear'), budget_run=str(root/'budget'),
                generic_run=str(root/'generic'), relation_run=str(root/'relation'),
                output=str(root/'comparison'), cpu_threads=4)
            result = compare.run(params)
            self.assertEqual(result['model_forwards'], 0)
            self.assertEqual(result['position_population']['covered_target_points'], 127)
            self.assertTrue(result['pairing']['generic_relation_initial_parameters_exact'])
            self.assertTrue(result['pairing']['generic_relation_parameter_count_equal'])
            self.assertTrue(result['no_multi_prediction_exact'])
            self.assertFalse(result['default_promoted'])
            loaded, decision = closeout.load_table(root/'comparison', 20261023)
            self.assertEqual(loaded['scene_ids'], ['3035'])
            self.assertEqual(decision, result['scientific_screen'])
            _, budget = budget_compare.adapter_metadata(root/'budget')
            _, generic = compare.relation_metadata(root/'generic')
            _, relation = compare.relation_metadata(root/'relation')
            arms = dict(linear=meta(root/'linear'), finetune=budget, generic=generic, relation=relation)
            _, arrays = compare.scene_pair(arms, '3035')
            errors = arrays['relation_position_error'][np.isfinite(arrays['relation_position_error'])]
            self.assertAlmostEqual(result['position_weightings']['all']['relation']['point_equal']['p99'],
                float(np.quantile(errors, .99)), places=8)
            # A real cache corruption must fail before it can affect rankings.
            cache = root/'relation/relation_off/predictions/states-3035.npz'
            with np.load(cache, allow_pickle=False) as values:
                damaged = {key: values[key].copy() for key in values.files}
            damaged['group_supported_helper_count'][0, 0] += 1
            np.savez_compressed(cache, **damaged)
            with self.assertRaises(ValueError):
                compare.scene_pair(arms, '3035')


if __name__ == '__main__':
    unittest.main()
