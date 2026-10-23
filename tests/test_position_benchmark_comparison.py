"""Actual gate/sequence cache interoperability and fail-closed common population."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.support_gate import evaluate_support_gate
from association.evaluation.sequence_position import evaluate_sequence_position
from association.models.support_gate import SupportGate
from association.models.sequence_position import SequencePosition
from association.training.sequence_position import train_sequence_position
from association.training.support_gate import train_support_gate
from association.utils.runtime import read_json, write_json
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.test_sequence_position import tiny_configuration, tiny_training
from tests.training.test_support_gate import tiny_config, tiny_train
from tools.analysis import compare_position_benchmarks as entry
from tools.experiments.support_gate_replication import annotate_endpoint


class PositionBenchmarkComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, root):
        source, metadata, validation = support_gate_cache_fixture(root)
        arms = {}
        for name in entry.ARM_CONFIGS:
            path = root/name; path.mkdir()
            sequence = name in ('071C_lstm', '071D_transformer')
            if sequence:
                model = SequencePosition(backbone='lstm' if name == '071C_lstm' else 'transformer')
                with torch.no_grad():
                    model.output.bias.fill_(.01)
                with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                    summary = evaluate_sequence_position(model, validation, source, metadata, path/'final')
            else:
                model = SupportGate()
                with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda raw, *_: raw):
                    summary = evaluate_support_gate(model, validation, source, metadata, path/'final')
                    if name == '071A_log':
                        evaluate_support_gate(model, validation, source, metadata, path/'initial')
            write_json(path/'run.json', {'source_system': str(source)})
            arms[name] = dict(path=path, method='learned_sequence' if sequence else 'learned_gate',
                source_system=str(source), scene_ids=metadata['scene_ids'], components=metadata['component_ids'], summary=summary)
        return arms

    @staticmethod
    def mutate(path, key, change):
        with np.load(path, allow_pickle=False) as saved:
            values = dict(saved)
        change(values[key])
        np.savez_compressed(path, **values)

    def test_actual_two_schemas_full_artifacts_four_weights_and_tail(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); arms = self.fixture(root)
            row, data = entry.scene_arrays(arms, '3035', arms['071A_log']['path']/'initial')
            pooled = entry._pool([row], [data])
            self.assertEqual(pooled['population']['target_identity_count'], 5)
            self.assertEqual(pooled['population']['duplicate_prediction_group_count'], 1)
            for method in entry.METHODS:
                metric = pooled['position_weightings']['all'][method]['point_equal']
                self.assertEqual(metric['unit_count'], 127)
                self.assertIsNotNone(metric['p99'])
                self.assertEqual(pooled['position_weightings']['all'][method]['scene_equal_point_mean']['unit_count'], 1)
            np.testing.assert_array_equal(data['fixed_prior095_position_error'], data['071A_log_position_error'])
            args = Namespace(log_run071A=str(arms['071A_log']['path']), linear_run071B=str(arms['071B_linear']['path']),
                lstm_run071C=str(arms['071C_lstm']['path']), transformer_run071D=str(arms['071D_transformer']['path']),
                source_070=str(arms['070_linear']['path']), output=str(root/'table'), cpu_threads=4)
            with patch.object(entry, '_metadata', side_effect=lambda path, arm: arms[arm]), \
                    patch.object(entry, 'verify_pairing', return_value={'status': 'PAIR_VERIFIED', 'matched_updates': 1000}), \
                    patch.object(entry, 'verify_sequence_sampling', return_value={'status': 'PAIR_VERIFIED'}):
                result = entry.run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['model_forwards'], 0)
            self.assertTrue((root/'table/report.md').exists())
            self.assertEqual(read_json(root/'table/run.json')['status'], 'COMPLETED')
            with self.assertRaises(FileExistsError):
                entry.run(args)

    def test_actual_sequence_batch_pairing_and_exposure_tamper_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('lstm', 'transformer'):
                train_sequence_position(tiny_training(), tiny_configuration(name), root/name)
            gate_config = tiny_config(); gate_config['training']['seed'] = 20261023
            train_support_gate(tiny_train(), gate_config, root/'gate')
            actual = entry.verify_sequence_sampling(root/'lstm', root/'transformer', root/'gate')
            self.assertEqual(actual['status'], 'PAIR_VERIFIED')
            self.assertEqual(actual['matched_updates'], 3)
            self.assertEqual(actual['matched_anchors'], 6)
            self.assertTrue(actual['gate_actual_batch_sequence_exact'])
            self.assertFalse(actual['initial_parameters_compared'])
            log = root/'transformer/training.jsonl'
            original = log.read_text()
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0]['batch'][0]['members'].append(999)
            log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'batch pairing'):
                entry.verify_sequence_sampling(root/'lstm', root/'transformer')
            log.write_text(original)
            saved = read_json(root/'transformer/run.json')
            changed = deepcopy(saved); changed['exposure']['valid_position_points'] += 1
            write_json(root/'transformer/run.json', changed)
            with self.assertRaisesRegex(ValueError, 'sampling exposure'):
                entry.verify_sequence_sampling(root/'lstm', root/'transformer')
            saved['status'] = 'FAILED'
            write_json(root/'transformer/run.json', saved)
            with self.assertRaisesRegex(ValueError, 'completed'):
                entry.verify_sequence_sampling(root/'lstm', root/'transformer')

    def test_metadata_real_checkpoint_binding_accepts_and_refuses_tampering(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, metadata, validation = support_gate_cache_fixture(root)
            out = root/'arm'; out.mkdir()
            config = tiny_config()
            config['position_loss'] = 'log1p'
            config['training'].update(seed=20261023, updates=1000, batch_size=1)
            # Real optimizer/checkpoint, with only the tiny data population and
            # config reads substituted. No checkpoint-loader mock is allowed.
            fit = train_support_gate(tiny_train(), config, out/'fit')
            model, _ = entry.gate_checkpoint(out/'fit/final.pt')
            with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda raw, *_: raw):
                evaluate_support_gate(model, validation, source, metadata, out/'final')
            annotate_endpoint(out/'final', config, 1000)
            record = dict(status='COMPLETED', configuration=config, optimizer_updates=1000,
                source_system=str(source), test_opened=False, validation_selection=False,
                completed_endpoints=['initial', 'final'])
            write_json(out/'run.json', record); write_json(out/'summary.json', record)
            manifest = dict(splits=dict(T_core=fit['training_scene_ids'], V_select=metadata['scene_ids']),
                            components=metadata['component_ids'])
            def fixture_read(path):
                if Path(path) == entry.ROOT/'configs'/entry.ARM_CONFIGS['071A_log']:
                    return config
                if Path(path) == entry.ROOT/'configs/data/mtad-level1-scene3000.json':
                    return manifest
                return read_json(path)
            with patch.object(entry, 'read_json', side_effect=fixture_read):
                result = entry._metadata(out, '071A_log')
                self.assertEqual(result['scene_ids'], metadata['scene_ids'])
                checkpoint_path = out/'fit/final.pt'
                original = checkpoint_path.read_bytes()
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                checkpoint['actual_updates'] = 999
                torch.save(checkpoint, checkpoint_path)
                with self.assertRaisesRegex(ValueError, 'actual_updates'):
                    entry._metadata(out, '071A_log')
                checkpoint_path.write_bytes(original)
                final = read_json(out/'final/summary.json')
                final['checkpoint']['path'] = str(out/'fit/step-0.pt')
                write_json(out/'final/summary.json', final)
                with self.assertRaisesRegex(ValueError, 'actual frozen endpoint'):
                    entry._metadata(out, '071A_log')

    def test_masks_groups_control_states_and_error_cache_mismatches_rejected(self):
        changes = [
            ('system-predictions', 'learned_sequence_aligned_prediction_valid', lambda x: x.__setitem__((0, 0, 0), ~x[0, 0, 0])),
            ('system-predictions', 'learned_sequence_group_node_indices', lambda x: x.__setitem__(0, x[1])),
            ('system-predictions', 'anchor_group_states', lambda x: x.__setitem__((0, 0, 0), x[0, 0, 0]+.01)),
            ('predictions', 'learned_sequence_position_error', lambda x: x.__setitem__(np.unravel_index(np.flatnonzero(np.isfinite(x))[0], x.shape), 999.)),
            ('predictions', 'learned_sequence_group_states', lambda x: x.__setitem__((0, 0, 0), x[0, 0, 0]+.01)),
            ('predictions', 'region_anchor_inside_support', lambda x: x.__setitem__((0, 0), ~x[0, 0])),
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); arms = self.fixture(root)
            for directory, key, change in changes:
                path = arms['071C_lstm']['path']/'final'/directory/'states-3035.npz'
                original = path.read_bytes()
                with self.subTest(key=key):
                    self.mutate(path, key, change)
                    with self.assertRaises(ValueError):
                        entry.scene_arrays(arms, '3035', arms['071A_log']['path']/'initial')
                path.write_bytes(original)

    def test_failed_actual_pairing_produces_no_successful_table(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            runs = []
            for name in entry.ARM_CONFIGS:
                path = root/name; path.mkdir()
                write_json(path/'run.json', {'source_system': str(root/'source056/system')})
                runs.append(str(path))
            args = Namespace(**dict(zip(('log_run071A', 'linear_run071B', 'lstm_run071C', 'transformer_run071D', 'source_070'), runs)),
                             output=str(root/'out'), cpu_threads=4)
            with patch.object(entry, 'verify_pairing', side_effect=ValueError('actual batch mismatch')), \
                    patch.object(entry, '_metadata') as metadata:
                with self.assertRaisesRegex(ValueError, 'actual batch mismatch'):
                    entry.run(args)
            metadata.assert_not_called()
            self.assertEqual(read_json(root/'out/run.json')['status'], 'FAILED')
            self.assertFalse((root/'out/summary.json.gz').exists())


if __name__ == '__main__':
    unittest.main()
