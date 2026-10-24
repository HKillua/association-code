"""084 log denominators, stable corruption groups and read-only CLI contracts."""
from argparse import Namespace
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from association.evaluation.helper_exposure_audit import (
    LOSS_SEMANTICS, MODES, audit_fit, validation_description)
from tools.analysis import helper_exposure_audit as entry


def _write(path, value):
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def _fixture(root, seed=20261023):
    fit = root/str(seed)/'fit'
    fit.mkdir(parents=True)
    config = json.loads((entry.ROOT/'configs/helper-sequence-pre-pool-level1.json').read_text())
    config['training'].update(seed=seed, updates=6, batch_size=2)
    def sample(sid, anchor, valid, all_counts, valid_counts, members):
        return (dict(scene_id=sid, anchor_index=anchor, node_id=f'{sid}:{anchor}', members=members,
                     valid_position_points=valid, eligible_query_points=0, eligible_valid_position_points=0),
                dict(all_queries=dict(zip(('zero', 'one', 'multiple'), all_counts)),
                     valid_position_queries=dict(zip(('zero', 'one', 'multiple'), valid_counts))))
    a = sample('0', 0, 1, [1, 63, 0], [1, 0, 0], [0, 1])
    ap = sample('0', 0, 1, [0, 64, 0], [0, 1, 0], [0, 1, 2])
    b = sample('0', 1, 4, [0, 64, 0], [0, 4, 0], [1, 0])
    z = sample('3', 0, 0, [64, 0, 0], [0, 0, 0], [0])
    rows = []
    for step in range(1, 7):
        batch = ((a, b), (ap, z), (b, z))[(step-1) % 3]
        rows.append(dict(step=step, mode=MODES[(step-1) % 3], batch=[deepcopy(s[0]) for s in batch],
            helper_support_batch=[deepcopy(s[1]) for s in batch], loss=1., gradient_norm=1., gradient_clipped=False,
            helper_gradients=dict(phi_nonzero_updates=True, scorer_nonzero_updates=False),
            objectives=dict(valid_position_points=sum(s[0]['valid_position_points'] for s in batch))))
    exposure = dict(successful_updates=6, sampled_anchors=12, valid_position_points=20, nonzero_gradient_updates=6,
        scenes={'0': 8, '3': 4}, anchors={'0:0': 4, '0:1': 4, '3:0': 4},
        modes={'clean': dict(updates=2, anchors=4, valid_position_points=10),
               'clean_plus_wrong': dict(updates=2, anchors=4, valid_position_points=2),
               'wrong_only': dict(updates=2, anchors=4, valid_position_points=8)})
    support = {p: dict(zero=0, one=0, multiple=0) for p in ('all_queries', 'valid_position_queries')}
    for row in rows:
        for counts in row['helper_support_batch']:
            for pop in support:
                for bucket in support[pop]:
                    support[pop][bucket] += counts[pop][bucket]
    record = dict(configuration=config, status='COMPLETED', test_opened=False, optimizer_updates=6,
        training_scene_ids=['0', '3'], scope=config['scope'], loss_semantics=LOSS_SEMANTICS,
        motion_observation_policy='canonical_all', helper_aggregation='pre_pool', final_checkpoint='final.pt',
        paired_control=dict(status='COMPLETED', initial_parameters_exact=True, initial_sampler_rng_exact=True,
            sampling_exposure_exact=True, actual_successful_batches_matched=6), exposure=exposure,
        helper_support_exposure=support, trained_scene_count=2,
        helper_gradients=dict(phi_nonzero_updates=6, scorer_nonzero_updates=0),
        gradient_clipping=dict(clipped_updates=0, successful_update_fraction=0.))
    _save(fit, record, rows)
    return fit, config, record, rows


def _save(fit, record, rows):
    _write(fit/'run.json', record)
    (fit/'training.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))


def _validation(root):
    path = root/'comparison'
    path.mkdir()
    path = path/'summary.json.gz'
    value = dict(status='COMPLETED', profile='replica081', test_opened=False, complete_profile=True,
        scene_ids=['1'], scene_count=1, population=dict(predicted_group_count=3),
        position_population=dict(covered_target_points=2, covered_target_trajectories=2),
        helper_population=dict(
            all_output_group_queries=dict(total=192, counts=dict(zero=160, one=32, two_or_more=0)),
            all_output_group_valid_position_queries=dict(total=3, counts=dict(zero=1, one=2, two_or_more=0)),
            unique_selected_valid_position_queries=dict(total=2, counts=dict(zero=1, one=1, two_or_more=0))),
        position_weightings={'all': {name: {key: dict(unit_count=count, mae=2., mse=4., rmse=2.)
            for key, count in zip(('point_equal', 'scene_target_equal', 'scene_equal_point_mean', 'scene_equal_target_mean'), (2, 2, 1, 1))}
            for name in ('linear', 'pre_pool', 'weighted_pool')}})
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream)
    return path, value


class HelperExposureAuditTests(unittest.TestCase):
    def test_loss_coefficients_zero_supervision_and_mode_dedup_are_distinct(self):
        with TemporaryDirectory() as temp:
            fit, config, _, _ = _fixture(Path(temp))
            result = audit_fit(fit, allowed_training_scene_ids=['0', '3'], expected_configuration=config)
            actual = result['actual_exposure']
            self.assertEqual(actual['sample_count'], 12)
            self.assertEqual(actual['zero_supervision_sample_count'], 4)
            self.assertEqual(actual['point_exposure']['valid_position_queries']['counts'],
                             dict(zero=2, one=18, two_or_more=0))
            self.assertAlmostEqual(actual['point_exposure']['valid_position_queries']['fractions']['zero'], .1)
            coefficients = actual['sample_equal_loss_coefficients']
            self.assertAlmostEqual(coefficients['shares']['zero'], 1/6)
            self.assertAlmostEqual(coefficients['shares']['one'], .5)
            self.assertAlmostEqual(coefficients['coefficient_mass'], 2/3)
            scene = actual['scene_equal']
            self.assertAlmostEqual(scene['sample_equal_loss_coefficients']['shares']['zero'], .125)
            self.assertAlmostEqual(scene['sample_equal_loss_coefficients']['shares']['one'], .375)
            self.assertAlmostEqual(scene['sample_equal_loss_coefficients']['coefficient_mass'], .5)
            self.assertEqual(scene['scene_point_fractions']['valid_position_queries']['zero_point_scenes'], 1)
            self.assertEqual(result['unique_scene_anchor_mode']['sample_count'], 6)
            self.assertEqual(result['repeated_scene_anchor_mode_exposures'], 6)
            # Same anchor has different support in different corruption modes;
            # those are distinct cases, not a consistency error or one sample.
            self.assertEqual(result['unique_scene_anchor_mode']['modes']['clean']['sample_count'], 2)
            self.assertEqual(actual['modes']['clean_plus_wrong']['zero_supervision_sample_count'], 2)

    def test_rejects_incomplete_failed_or_counter_inconsistent_sources(self):
        changes = [
            ('failed source', lambda r, rows: r.update(status='FAILED'), 'completed'),
            ('missing successful update', lambda r, rows: rows.pop(), 'every successful'),
            ('extra update', lambda r, rows: rows.append(deepcopy(rows[-1])), 'step/mode/count'),
            ('mode order', lambda r, rows: rows[1].update(mode='clean'), 'step/mode/count'),
            ('objective denominator', lambda r, rows: rows[0]['objectives'].update(valid_position_points=4), 'objective valid'),
            ('saved exposure', lambda r, rows: r['exposure'].update(valid_position_points=21), 'saved exposure'),
            ('saved helper count', lambda r, rows: r['helper_support_exposure']['all_queries'].update(zero=0), 'saved helper exposure'),
            ('gradient updates', lambda r, rows: r['helper_gradients'].update(phi_nonzero_updates=5), 'gradient counters'),
        ]
        for name, change, message in changes:
            with self.subTest(name=name), TemporaryDirectory() as temp:
                fit, _, record, rows = _fixture(Path(temp))
                change(record, rows)
                _save(fit, record, rows)
                with self.assertRaisesRegex(ValueError, message):
                    audit_fit(fit, allowed_training_scene_ids=['0', '3'])

    def test_rejects_bad_population_mapping_and_repeated_support(self):
        def helper_counts(rows):
            rows[0]['helper_support_batch'][0]['valid_position_queries']['zero'] = 2
        def different_support(rows):
            rows[3]['helper_support_batch'][0]['all_queries'].update(zero=2, one=62)
        def outside_scene(rows):
            rows[0]['batch'][0].update(scene_id='1', node_id='1:0')
        def anchor_mapping(rows):
            rows[3]['batch'][0]['node_id'] = '0:unseen'
        def denominator_change(rows):
            rows[1]['batch'][0]['valid_position_points'] = 2
        def impossible_helper(rows):
            rows[0]['helper_support_batch'][0]['all_queries'].update(zero=1, one=62, multiple=1)
        for change, message in [(helper_counts, 'bucket sum'), (different_support, 'repeated scene-anchor-mode'),
                (outside_scene, 'legal T_core'), (anchor_mapping, 'anchor-node'),
                (denominator_change, 'target valid-point'), (impossible_helper, 'impossible')]:
            with self.subTest(change=change.__name__), TemporaryDirectory() as temp:
                fit, _, record, rows = _fixture(Path(temp))
                change(rows)
                _save(fit, record, rows)
                with self.assertRaisesRegex(ValueError, message):
                    audit_fit(fit, allowed_training_scene_ids=['0', '3'])

    def test_validation_description_is_not_training_loss_weight_estimate(self):
        with TemporaryDirectory() as temp:
            path, value = _validation(Path(temp))
            result = validation_description(path, allowed_validation_scene_ids=['1'])
            self.assertTrue(result['descriptive_only'])
            self.assertIn('V per-anchor training-style loss coefficients', result['unavailable'])
            self.assertNotIn('sample_equal_loss_coefficients', result)
            value['helper_population']['unique_selected_valid_position_queries']['counts']['zero'] = 2
            with gzip.open(path, 'wt') as stream:
                json.dump(value, stream)
            with self.assertRaisesRegex(ValueError, 'sum/subset'):
                validation_description(path, allowed_validation_scene_ids=['1'])

    def test_cli_two_seeds_no_overwrite_and_failed_partial_evidence(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first, c1, _, _ = _fixture(root)
            second, c2, _, _ = _fixture(root, 20261021)
            validation, _ = _validation(root)
            args = ['--fit', str(first), '--fit', str(second), '--validation-summary', str(validation), '--output', str(root/'out')]
            protocol = (['0', '3'], ['1'], {20261023: c1, 20261021: c2})
            before = {str(p): p.read_bytes() for p in [first/'run.json', first/'training.jsonl',
                                                     second/'run.json', second/'training.jsonl', validation]}
            with patch.object(entry, '_protocol', return_value=protocol):
                self.assertEqual(entry.main(args), 0)
                with self.assertRaises(FileExistsError):
                    entry.main(args)
                # A complete first source remains recorded if a later source
                # is corrupt; no source log or first output is rewritten.
                (second/'training.jsonl').write_text('{}\n')
                args[-1] = str(root/'failed')
                with self.assertRaisesRegex(ValueError, 'step/mode/count'):
                    entry.main(args)
            state = json.loads((root/'failed/run.json').read_text())
            partial = json.loads((root/'failed/summary.json').read_text())
            self.assertEqual(state['status'], 'FAILED')
            self.assertEqual(state['completed_fits'], 1)
            self.assertEqual(len(partial['fits']), 1)
            self.assertEqual(partial['status'], 'FAILED')
            for name, content in before.items():
                if name != str(second/'training.jsonl'):
                    self.assertEqual(Path(name).read_bytes(), content)
            report = (root/'out/report.md').read_text()
            self.assertIn('zero-supervision samples retained', report)
            self.assertIn('not recoverable per-anchor training-loss coefficients', report)
            result = json.loads((root/'out/summary.json').read_text())
            self.assertEqual([x['seed'] for x in result['fits']], [20261023, 20261021])

    def test_real_tiny_training_logs_and_stdlib_only_audit_import(self):
        import torch
        from association.training.sequence_position import train_sequence_position
        from association.training.helper_sequence import train_helper_sequence
        from tests.test_sequence_position import tiny_training, tiny_configuration
        from tests.test_helper_sequence_training import helper_config
        torch.set_num_threads(1)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            train = tiny_training()
            config = tiny_configuration('transformer')
            config['training']['updates'] = 6
            train_sequence_position(train, config, root/'control')
            helper = helper_config('pre_pool', updates=6)
            train_helper_sequence(train, helper, root/'fit', paired_reference_fit=root/'control')
            result = audit_fit(root/'fit', allowed_training_scene_ids=[str(s['scene_id']) for s in train],
                               expected_configuration=helper)
            self.assertEqual(result['successful_updates'], 6)
            self.assertEqual(result['actual_exposure']['sample_count'], 12)
            self.assertTrue(result['checks']['counters_exact'])
        process = subprocess.run([sys.executable, '-c',
            "import sys; from tools.analysis import helper_exposure_audit; assert 'torch' not in sys.modules"],
            cwd=entry.ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_cli_resource_source_run_protection_and_duplicate_seed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fit, config, _, _ = _fixture(root)
            validation, _ = _validation(root)
            args = Namespace(fit=[str(fit)], validation_summary=str(validation), output=str(fit.parent/'audit'), cpu_threads=4)
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                entry.run(args)
            self.assertFalse((fit.parent/'audit').exists())
            args.output = str(root/'out')
            args.cpu_threads = 5
            with self.assertRaisesRegex(ValueError, 'CPU4'):
                entry.run(args)
            self.assertFalse((root/'out').exists())
            # Distinct source paths with the same seed still duplicate the same
            # intended population, and may not be counted as a second seed.
            other = root/'other'
            other.mkdir()
            duplicated, _, _, _ = _fixture(other)
            args.fit.append(str(duplicated))
            args.cpu_threads = 4
            with patch.object(entry, '_protocol', return_value=(['0', '3'], ['1'], {20261023: config})):
                with self.assertRaisesRegex(ValueError, 'one frozen'):
                    entry.run(args)
            self.assertEqual(json.loads((root/'out/run.json').read_text())['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
