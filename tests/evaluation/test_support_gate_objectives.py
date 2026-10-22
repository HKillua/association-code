"""069 cache reconstruction and explicitly unequal aggregation populations."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.support_gate import evaluate_support_gate
from association.evaluation.support_gate_objectives import (
    METHODS, REGIONS, load_scene, summarize, verify_source_summaries,
)
from association.models.support_gate import SupportGate
from tests.evaluation.test_support_gate import support_gate_cache_fixture


def fixture(root):
    source, metadata, raw = support_gate_cache_fixture(root)
    model = SupportGate()
    with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda scene, *_: scene):
        initial = evaluate_support_gate(model, raw, source, metadata, root/'initial')
        with torch.no_grad():
            model.head[-1].bias.fill_(0.)
        final = evaluate_support_gate(model, raw, source, metadata, root/'final')
    return initial, final


def mutate(path, change):
    with np.load(path, allow_pickle=False) as cache:
        data = dict(cache)
    change(data)
    np.savez_compressed(path, **data)


class SupportGateObjectiveTests(unittest.TestCase):
    def test_full_cache_reconstruction_preserves_duplicates_missing_targets_and_truth(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial, final = fixture(root)
            with patch('torch.load', side_effect=AssertionError('checkpoint forbidden')), \
                    patch.object(SupportGate, 'forward', side_effect=AssertionError('forward forbidden')):
                row = load_scene(root/'initial', root/'final', '3035')
            self.assertEqual(row['errors']['initial'].shape, (5, 64))
            self.assertEqual(np.isfinite(row['errors']['initial']).sum(), 127)
            self.assertEqual(row['population']['duplicate_prediction_group_count'], 1)
            self.assertEqual(row['population']['target_identity_count'], 5)
            self.assertEqual(row['model_state_comparison']['all_output_groups']['query_count'], 256)
            self.assertEqual(row['model_state_comparison']['unique_selected_groups']['query_count'], 192)
            np.testing.assert_array_equal(row['selected_group_by_target'], [0, 1, -1, -1, 3])
            self.assertTrue(np.isnan(row['errors']['initial'][2:]).all())
            self.assertEqual(row['truth'].shape, (5, 64, 4))
            self.assertEqual(row['truth_valid'].dtype, bool)
            verified = verify_source_summaries([row], initial, final)
            self.assertTrue(verified['source_position_metrics_verified'])
            report = summarize([row], {'3035': 'shared'})
            self.assertEqual(report['regions']['all']['point_count'], 127)
            self.assertEqual(report['regions']['all']['scene_target_record_count'], 2)
            self.assertEqual(report['regions']['no_predicted_group']['point_count'], 0)
            self.assertIsNone(report['regions']['no_predicted_group']['methods']['final']['point_equal']['log1p'])
            self.assertEqual(report['regions']['all']['methods']['initial']['point_equal']['p99'],
                             np.quantile(row['errors']['initial'][np.isfinite(row['errors']['initial'])], .99))
            tampered = deepcopy(final)
            tampered['methods']['learned_gate']['common_target_strata']['all']['position']['mae'] += 1.
            with self.assertRaisesRegex(ValueError, 'metric differs'):
                verify_source_summaries([row], initial, tampered)

    def test_cache_error_mask_group_duplicate_and_ordinary_corruption_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            fragment_path = root/'final/predictions/states-3035.npz'
            system_path = root/'final/system-predictions/states-3035.npz'
            original_fragment, original_system = fragment_path.read_bytes(), system_path.read_bytes()
            cases = (
                ('saved_error', fragment_path, lambda d: d['learned_gate_position_error'].__setitem__((0, 0), 123.)),
                ('truth_mask', system_path, lambda d: d['anchor_aligned_truth_valid'].__setitem__((0, 0, 0), False)),
                ('group_first', fragment_path, lambda d: d['group_anchor_node_indices'].__setitem__(0, 1)),
                ('duplicate_state', system_path, lambda d: d['learned_gate_group_states'].__setitem__((2, 0, 0), 20.)),
                ('ordinary', system_path, lambda d: d['support_switch_group_states'].__setitem__((0, 0, 0), 20.)),
                ('saved_error_missing', fragment_path, lambda d: d['learned_gate_position_error'].__setitem__((0, 0), np.nan)),
                ('gate_nonfinite', fragment_path, lambda d: d['group_gate'].__setitem__((0, 0), np.nan)),
            )
            for name, path, change in cases:
                with self.subTest(case=name):
                    mutate(path, change)
                    with self.assertRaises(ValueError):
                        load_scene(root/'initial', root/'final', '3035')
                    fragment_path.write_bytes(original_fragment)
                    system_path.write_bytes(original_system)

    def test_correlated_group_corruption_does_not_pass_cross_endpoint_equality(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            for endpoint in ('initial', 'final'):
                mutate(root/endpoint/'predictions/states-3035.npz',
                       lambda d: d['group_anchor_node_indices'].__setitem__(0, 1))
            with self.assertRaisesRegex(ValueError, 'representatives'):
                load_scene(root/'initial', root/'final', '3035')

    def test_unequal_points_targets_scenes_have_explicitly_different_weights(self):
        def row(sid, targets, values):
            regions = {name: np.zeros(values.shape, bool) for name in REGIONS}
            regions['all'][:] = True
            regions['anchor_inside_support'][:] = True
            return dict(scene_id=sid, target_ids=np.array(targets),
                errors={method: values*(.5 if method == 'final' else 1.) for method in METHODS},
                regions=regions, population=dict(target_identity_count=len(targets)), model_state_comparison={})
        values = np.zeros((2, 9))
        values[0] = np.nan
        values[0, 0] = 10000.
        rows = [row('a', ['shared', 'other'], values), row('b', ['shared'], np.array([[100.]]))]
        result = summarize(rows, {'a': 'same_component', 'b': 'same_component'})
        region = result['regions']['all']
        self.assertEqual(result['scene_target_record_count'], 3)
        self.assertEqual(result['distinct_identity_string_count'], 2)
        self.assertEqual(result['component_count'], 1)
        self.assertEqual(region['point_count'], 11)
        self.assertEqual(region['scene_target_record_count'], 3)
        self.assertEqual(region['nonempty_scene_count'], 2)
        metrics = region['methods']['initial']
        self.assertAlmostEqual(metrics['point_equal']['mae'], 10100./11)
        self.assertAlmostEqual(metrics['scene_target_equal']['mae'], 10100./3)
        self.assertEqual(metrics['scene_equal_point_mean']['mae'], 550.)
        self.assertEqual(metrics['scene_equal_target_mean']['mae'], 2550.)
        self.assertAlmostEqual(metrics['point_equal']['log1p'], (np.log1p(10.)+np.log1p(.1))/11)
        self.assertAlmostEqual(metrics['scene_equal_point_mean']['rmse'], np.sqrt((10000.**2/10+100.**2)/2))
        paired = region['paired']['final__minus__initial']['point_equal']
        self.assertEqual(paired['unit_count'], 11)
        self.assertEqual(paired['mae']['negative_unit_count'], 2)
        self.assertEqual(paired['mae']['exact_tie_unit_count'], 9)
        self.assertAlmostEqual(paired['mae']['mean_difference'], -5050./11)
        wrong = deepcopy(rows)
        wrong[0]['errors']['final'][1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'paired masks'):
            summarize(wrong, {'a': 'shared', 'b': 'shared'})

    def test_import_does_not_load_torch_models_or_archive_readers(self):
        script = '''import sys
import association.evaluation.support_gate_objectives
assert 'torch' not in sys.modules
assert not any(x.startswith('association.models') for x in sys.modules)
assert 'association.data.archive' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
