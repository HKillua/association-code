"""Real tiny source, budget, scratch P/V fit and three full-population endpoints."""
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.utils.runtime import read_json, write_json
from association.evaluation import deepaf_peer as evaluation
from association.models.deepaf_peer import DeepAFPeer
from association.training import deepaf_peer as peer_training
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_train
from tools.experiments import sequence_position_fit as original
from tools.experiments import frozen_position_adapter_fit as budget_entry
from tools.experiments import deepaf_peer_fit as entry
from tools.analysis import compare_frozen_position_adapter as budget_compare
from tools.analysis import compare_deepaf_peer as peer_compare


class DeepAFPeerPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_frozen_production_configs_and_budget_mutation(self):
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        for key, filename in entry.CONFIGS.items():
            config = read_json(entry.ROOT/'configs'/filename)
            self.assertEqual(key, (config['training']['seed'], config['training']['learning_rate']))
            entry.production_contract(config, manifest, alignment)
            config['training']['updates'] += 1
            with self.assertRaises(ValueError):
                entry.production_contract(config, manifest, alignment)

    def test_duplicate_endpoint_refusal_preserves_completed_record(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)/'completed'
            output.mkdir()
            write_json(output/'run.json', dict(status='COMPLETED'))
            before = (output/'run.json').read_bytes()
            with self.assertRaises(FileExistsError):
                evaluation.evaluate_deepaf_peer(DeepAFPeer(), [], Path(directory)/'source', {}, output)
            self.assertEqual(before, (output/'run.json').read_bytes())

    def test_real_fit_full_endpoints_and_cached_population(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root)
            for scene in validation:
                # This cached interface fixture has no original observations.
                # Supply explicit synthetic rates, never a query-state OLS refit.
                motion = scene['motion']
                motion['ols_slope_degrees_per_second'] = np.full((len(motion['states']), 2), 1e-5)
            write_json(source/'run.json', dict(status='COMPLETED'))
            train = tiny_train()
            manifest = dict(splits={'T_core': [str(s['scene_id']) for s in train], 'V_select': ['3035']},
                            components={'3035': 'shared'})
            config = read_json(entry.ROOT/'configs/sequence-position-transformer-extended-level1.json')
            config['training'].update(updates=4, batch_size=2)
            for name, value in (('config', config), ('manifest', manifest), ('alignment', {})):
                write_json(root/(name+'.json'), value)
            args = Namespace(source_system=str(source), output=str(root/'linear'), paired_control=None,
                budget_control=None, archive=['fixture-no-archive.zip'], manifest=str(root/'manifest.json'),
                alignment=str(root/'alignment.json'), config=str(root/'config.json'), selection_record=None,
                device='cpu', cpu_threads=4, time_policy='advisory', profile='extended4000')
            dataset = lambda *a, **k: dict(train=train, validation=validation, report={'tiny_fixture': True})

            def meta(path, arm=None):
                path = Path(path).resolve()
                return dict(path=path, method='learned_sequence', source_system=str(source.resolve()),
                    scene_ids=['3035'], components={'3035': 'shared'}, summary=read_json(path/'final/summary.json'))

            # Only raw fixture loading and production population sizes are seams;
            # sampling, optimizer, P/V forward, checkpoint, matching and metrics run.
            stack.enter_context(patch('association.evaluation.sequence_position.frozen_motion',
                                      side_effect=lambda raw, *_: raw))
            for module in (original, budget_entry, entry):
                stack.enter_context(patch.object(module, 'production_contract'))
                stack.enter_context(patch.object(module, '_bind_source'))
                stack.enter_context(patch.object(module, '_load_metadata', return_value=metadata))
                stack.enter_context(patch.object(module, 'load_timed_dataset', side_effect=dataset))
            stack.enter_context(patch.object(budget_entry, 'frozen_configuration', side_effect=lambda cfg: cfg))
            stack.enter_context(patch.object(budget_compare, '_metadata', side_effect=meta))
            stack.enter_context(patch.object(budget_compare, 'declared_manifest', return_value=manifest))
            stack.enter_context(patch.object(entry, 'source_metadata', side_effect=meta))
            original.run(args)
            args.paired_control = str(root/'linear')
            config = read_json(entry.ROOT/'configs/frozen-position-adapter-full-finetune-level1.json')
            config['training'].update(updates=4, batch_size=2)
            write_json(root/'config.json', config)
            args.output = str(root/'budget')
            budget_entry.run(args)
            args.budget_control = str(root/'budget')
            args.output = str(root/'peer')
            config = read_json(entry.ROOT/'configs/deepaf-peer-level1-lr1e3.json')
            config['training'].update(updates=8, source_updates=4, batch_size=2)
            write_json(root/'config.json', config)
            result = entry.run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['optimizer_updates'], 8)
            self.assertFalse(result['test_opened'])
            self.assertEqual(read_json(root/'peer/run.json')['completed_endpoints'], ['initial', 'mid', 'final'])
            # Changed bootstrap semantics must not silently reinterpret a v1
            # checkpoint, whether only its envelope or also its config is old.
            for legacy_config in (False, True):
                saved = torch.load(root/'peer/fit/final.pt', map_location='cpu', weights_only=False)
                saved['routing_semantics'] = 'fixed056-pv-closed-loop-tbptt1-v1'
                if legacy_config:
                    saved['configuration']['routing_semantics'] = saved['routing_semantics']
                torch.save(saved, root/'legacy-bootstrap.pt')
                with self.assertRaises(ValueError):
                    peer_training.load_checkpoint(root/'legacy-bootstrap.pt')
            for label in ('initial', 'mid', 'final'):
                summary = read_json(root/'peer'/label/'summary.json')
                self.assertEqual(summary['model_forwards'], 4)
                self.assertEqual(summary['published_method'], 'DeepAF')
                self.assertEqual(summary['position_weightings']['all']['learned_sequence']['point_equal']['unit_count'], 127)
                with np.load(root/'peer'/label/'predictions/states-3035.npz', allow_pickle=False) as cache:
                    self.assertEqual(cache['learned_sequence_group_states'].shape, (4, 64, 4))
                    np.testing.assert_array_equal(cache['learned_sequence_group_states'][..., 2:],
                                                  cache['mean_supported_group_states'][..., 2:])
                    self.assertTrue(np.isfinite(cache['group_deepaf_local_offset_m']).all())
            original_read = peer_compare.read_json
            tiny_config_path = (entry.ROOT/'configs/deepaf-peer-level1-lr1e3.json').resolve()

            def tiny_read(path):
                return config if Path(path).resolve() == tiny_config_path else original_read(path)

            # Production size/config checks are the only new seams. Actual fit
            # replay, checkpoint/endpoint binding and saved predictions remain.
            with patch.object(entry, 'frozen_configuration', side_effect=lambda value: value), \
                    patch.object(peer_compare, 'declared_manifest', return_value=manifest), \
                    patch.object(peer_compare, 'read_json', side_effect=tiny_read), \
                    patch.object(peer_compare, '_metadata', side_effect=meta):
                peer_arm = peer_compare.peer_metadata(root/'peer')
                ours_arm = peer_compare.ours_metadata(root/'budget')
                table, scenes = peer_compare.cached_comparison(dict(ours=ours_arm, deepaf=peer_arm))
                self.assertEqual(table['position_population']['covered_target_points'], 127)
                with np.load(root/'peer/final/predictions/states-3035.npz', allow_pickle=False) as cache:
                    error = cache['learned_sequence_position_error']
                    self.assertAlmostEqual(table['position_weightings']['all']['deepaf']['point_equal']['mae'],
                                           float(error[np.isfinite(error)].mean()), places=8)
                self.assertEqual(len(scenes), 1)
            with self.assertRaises(FileExistsError):
                entry.run(args)


if __name__ == '__main__':
    unittest.main()
