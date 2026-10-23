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
from tools.analysis import compare_position_replication as replication
from tools.experiments.support_gate_replication import annotate_endpoint


class PositionBenchmarkComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test073_actual_sampling_and_gate_budget_prefix(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = tiny_config(); cfg['position_loss'] = 'linear'
            cfg['training']['seed'] = 20261023
            cfg['training']['updates'] = 2
            train_support_gate(tiny_train(), cfg, root/'old')
            cfg['training']['updates'] = 3
            train_support_gate(tiny_train(), cfg, root/'new')
            seq = tiny_configuration('transformer')
            # Common scientific settings, including wall-time policy, match.
            seq['training'] = deepcopy(cfg['training'])
            train_sequence_position(tiny_training(), seq, root/'seq')
            result = replication.verify_gate_prefix(root/'new', root/'old')
            self.assertEqual(result['matched_prefix_updates'], 2)
            self.assertTrue(result['initial_parameters_exact'])
            pair = replication.verify_method_sampling(root/'new', root/'seq')
            self.assertEqual(pair['matched_updates'], 3)
            self.assertEqual(pair['matched_anchors'], 6)
            # Same exposure totals alone must not hide changed group membership.
            path = root/'seq/training.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['batch'][0]['members'].append(999)
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'actual gate/sequence batches'):
                replication.verify_method_sampling(root/'new', root/'seq')
            checkpoint = torch.load(root/'new/step-0.pt', weights_only=False)
            first = next(iter(checkpoint['model']))
            checkpoint['model'][first].add_(.01)
            torch.save(checkpoint, root/'new/step-0.pt')
            with self.assertRaisesRegex(ValueError, 'initial parameters'):
                replication.verify_gate_prefix(root/'new', root/'old')

    def test073_two_seed_table_reuses_real_cache_schemas(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); old = self.fixture(root)
            current = dict(zip(replication.ARMS, (old['071A_log'], old['071B_linear'],
                                                  old['071D_transformer'], old['071C_lstm'])))
            args = Namespace(gate_run073A=str(current[replication.ARMS[0]]['path']),
                gate_run073B=str(current[replication.ARMS[1]]['path']),
                transformer_run073C=str(current[replication.ARMS[2]]['path']),
                transformer_run072C=str(current[replication.ARMS[3]]['path']),
                source071B=str(old['071B_linear']['path']), source070=str(old['070_linear']['path']),
                output=str(root/'table073'), cpu_threads=4)
            with patch.object(replication, '_metadata', side_effect=lambda path, name: current[name]), \
                    patch.object(replication, 'verify_method_sampling', return_value={'status': 'PAIR_VERIFIED'}), \
                    patch.object(replication, 'verify_gate_prefix', return_value={'initial_parameters_exact': True,
                                                                               'matched_prefix_updates': 1000}):
                result = replication.run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(len(result['methods']), 9)
            self.assertEqual(set(result['pairing']), {'20261021', '20261023'})
            self.assertEqual(result['population']['duplicate_prediction_group_count'], 1)
            for name in replication.ARMS:
                self.assertEqual(result['position_weightings']['all'][name]['point_equal']['unit_count'], 127)
            self.assertEqual(result['model_forwards'], 0)
            self.assertFalse(result['test_opened'])

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

    def test072_optional_runs_are_all_or_none_and_table_uses_actual_caches(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); arms = self.fixture(root)
            args = Namespace(log_run071A=str(arms['071A_log']['path']), linear_run071B=str(arms['071B_linear']['path']),
                lstm_run071C=str(arms['071C_lstm']['path']), transformer_run071D=str(arms['071D_transformer']['path']),
                source_070=str(arms['070_linear']['path']), output=str(root/'table072'), cpu_threads=4,
                constant_run072A=str(arms['071A_log']['path']), lstm_run072B=None, transformer_run072C=None)
            with self.assertRaisesRegex(ValueError, 'all three'):
                entry.run(args)
            self.assertFalse((root/'table072').exists())
            args.lstm_run072B = str(arms['071C_lstm']['path'])
            args.transformer_run072C = str(arms['071D_transformer']['path'])
            arms['072B_lstm4000'] = arms['071C_lstm']
            arms['072C_transformer4000'] = arms['071D_transformer']
            with np.load(arms['071A_log']['path']/'final/predictions/states-3035.npz') as saved:
                logit, gate = float(saved['group_logits'][0, 0]), float(saved['group_gate'][0, 0])
            constant = dict(arms['071A_log'], selected_logit=logit, selected_gate=gate,
                            fit_source=str(root/'constant/fit.json'), train_only=True)
            with patch.object(entry, '_metadata', side_effect=lambda path, arm: arms[arm]), \
                    patch.object(entry, '_constant_metadata', return_value=constant), \
                    patch.object(entry, 'verify_pairing', return_value={'status': 'PAIR_VERIFIED'}), \
                    patch.object(entry, 'verify_sequence_sampling', return_value={'status': 'PAIR_VERIFIED'}), \
                    patch.object(entry, 'verify_extended_sampling', return_value={'status': 'PAIR_VERIFIED', 'matched_updates': 4000}):
                result = entry.run(args)
            self.assertEqual(len(result['methods']), 13)
            self.assertEqual(result['position_weightings']['all']['072A_constant']['point_equal']['unit_count'], 127)
            self.assertEqual(result['extended_sequence_pairing']['matched_updates'], 4000)
            self.assertIn('not a new loss-pair replication', result['interpretation'])
            # A valid route with the wrong scalar is still rejected: metadata
            # must describe every actual group, not only the aggregate metric.
            arms[entry.CONSTANT_METHOD] = dict(constant, selected_gate=.5)
            with self.assertRaisesRegex(ValueError, 'T-fitted constant gate'):
                entry.scene_arrays(arms, '3035', arms['071A_log']['path']/'initial')

    def test072_constant_fit_binds_training_selection_and_actual_replayed_samples(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, metadata, validation = support_gate_cache_fixture(root)
            old, out = root/'071B', root/'constant'
            (old/'fit').mkdir(parents=True); out.mkdir()
            config = {'scope': 'constant-gate-test-fixture'}
            scenes = ['train-scene']
            batch = [dict(scene_id='train-scene', anchor_index=0, node_id='node0', members=[0],
                          valid_position_points=64, eligible_query_points=0, eligible_valid_position_points=0)]*8
            rows = [dict(step=i, mode=entry.TRAIN_MODES[(i-1) % 3], batch=batch) for i in range(1, 1001)]
            (old/'fit/training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            write_json(old/'fit/run.json', {'training_scene_ids': scenes})
            write_json(out/'fit-sampling.json', rows)
            nominal = np.linspace(0., 1., 1001)
            logits = np.empty(1001, np.float32); logits[0], logits[-1] = -100., 100.
            logits[1:-1] = np.log(nominal[1:-1]/(1.-nominal[1:-1]))
            gates = torch.sigmoid(torch.as_tensor(logits)).numpy()
            record = dict(status='COMPLETED', scope='072-training-fitted-constant-gate',
                configuration=config, optimizer_updates=0, source071B=str(old), source_system=str(source),
                test_opened=False, validation_selection=False, completed_endpoints=['final'])
            fit = dict(record, train_only=True, candidate_count=1001, sampling_replay_verified=True,
                successful_updates_replayed=1000, sampled_anchors=8000, training_scene_ids=scenes,
                source_training_log=str(old/'fit/training.jsonl'), grid_logits=logits.astype(float).tolist(),
                effective_gates=gates.astype(float).tolist(), training_losses=np.abs(nominal-.95).tolist(),
                selected_index=950, selected_logit=float(logits[950]), selected_gate=float(gates[950]))
            write_json(out/'run.json', record); write_json(out/'summary.json', record); write_json(out/'fit.json', fit)
            model = SupportGate()
            with torch.no_grad():
                model.head[-1].bias.fill_(fit['selected_logit'])
            with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda raw, *_: raw):
                evaluate_support_gate(model, validation, source, metadata, out/'final')
            for name in ('run.json', 'summary.json'):
                endpoint = read_json(out/'final'/name)
                endpoint.update(configuration=config, fit_source=str(out/'fit.json'),
                                selected_logit=fit['selected_logit'], selected_gate=fit['selected_gate'])
                write_json(out/'final'/name, endpoint)
            manifest = dict(splits=dict(T_core=scenes, V_select=metadata['scene_ids']), components=metadata['component_ids'])
            def fixture_read(path):
                if Path(path) == entry.ROOT/'configs/constant-gate-level1.json': return config
                if Path(path) == entry.ROOT/'configs/data/mtad-level1-scene3000.json': return manifest
                return read_json(path)
            with patch.object(entry, 'read_json', side_effect=fixture_read):
                accepted = entry._constant_metadata(out, old)
                self.assertTrue(accepted['train_only'])
                self.assertEqual(accepted['selected_gate'], fit['selected_gate'])
                # A different grid winner cannot be disguised as the chosen
                # scalar, and a different training cohort cannot pass via counts.
                fit['selected_index'] = 949; write_json(out/'fit.json', fit)
                with self.assertRaisesRegex(ValueError, 'minimize'):
                    entry._constant_metadata(out, old)
                fit['selected_index'] = 950; write_json(out/'fit.json', fit)
                changed = deepcopy(rows); changed[-1]['batch'][0]['members'] = [0, 1]
                write_json(out/'fit-sampling.json', changed)
                with self.assertRaisesRegex(ValueError, 'actual replayed samples'):
                    entry._constant_metadata(out, old)

    def test072_extended_prefix_and_fresh_parameters_are_verified(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, rows = {}, {}
            for backbone in ('lstm', 'transformer'):
                old, new = root/('old-'+backbone), root/('new-'+backbone)
                # Genuine fresh checkpoints are sufficient for this audit;
                # long-run log parsing is separately exercised above.
                tiny = tiny_configuration(backbone)
                train_sequence_position(tiny_training(), tiny, old)
                original = torch.load(old/'step-0.pt', weights_only=False)
                for directory, count in ((old, 1000), (new, 4000)):
                    directory.mkdir(exist_ok=True)
                    checkpoint = deepcopy(original)
                    checkpoint['configuration']['training']['updates'] = count
                    torch.save(checkpoint, directory/'step-0.pt')
                    records[str(directory)] = dict(configuration=checkpoint['configuration'], training_scene_ids=['train'])
                    rows[str(directory)] = [dict(step=i, mode=entry.TRAIN_MODES[(i-1) % 3], batch=[{'node': 0}])
                                            for i in range(1, count+1)]
            def sampling(path): return records[str(path)], rows[str(path)], {}
            paths = [root/name for name in ('new-lstm', 'new-transformer', 'old-lstm', 'old-transformer')]
            with patch.object(entry, '_sequence_sampling', side_effect=sampling), \
                    patch.object(entry, 'verify_sequence_sampling', return_value={'status': 'PAIR_VERIFIED'}):
                result = entry.verify_extended_sampling(*paths)
                self.assertTrue(result['prefixes']['lstm']['initial_parameters_exact'])
                rows[str(paths[0])][999]['batch'] = [{'node': 1}]
                with self.assertRaisesRegex(ValueError, 'sampling prefix'):
                    entry.verify_extended_sampling(*paths)
                rows[str(paths[0])][999]['batch'] = [{'node': 0}]
                checkpoint = torch.load(paths[0]/'step-0.pt', weights_only=False)
                checkpoint['model']['output.bias'][0] += 1.
                torch.save(checkpoint, paths[0]/'step-0.pt')
                with self.assertRaisesRegex(ValueError, 'fresh parameters'):
                    entry.verify_extended_sampling(*paths)


if __name__ == '__main__':
    unittest.main()
