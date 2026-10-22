"""070 immutable initial-cache binding and matched endpoint comparisons."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.support_gate import evaluate_support_gate
from association.evaluation.support_gate_linear import (
    _component_split, bind_linear_source, compare_linear_run,
)
from association.evaluation.support_gate_objectives import METHODS, REGIONS, summarize
from association.models.support_gate import SupportGate
from association.utils.runtime import read_json, write_json
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.evaluation.test_support_gate_objectives import mutate
from tests.training.test_support_gate import tiny_config


def fixture(root):
    source068 = root/'old068'
    source068.mkdir()
    source056, metadata, raw = support_gate_cache_fixture(root)
    model = SupportGate()
    with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda scene, *_: scene):
        initial = evaluate_support_gate(model, raw, source056, metadata, source068/'initial')
        with torch.no_grad():
            model.head[-1].bias.fill_(0.)
        final = evaluate_support_gate(model, raw, source056, metadata, source068/'final')
        with torch.no_grad():
            model.head[-1].bias.fill_(1.)
        evaluate_support_gate(model, raw, source056, metadata, root/'new070/final')
    old = tiny_config()
    config = dict(old, position_loss='linear')
    manifest = dict(splits=dict(T_core=['train0'], V_select=['3035']), components={'3035': 'shared'})
    metadata_binding = dict(scene_ids=['3035'], components={'3035': 'shared'},
        endpoints=dict(initial=initial, final=final),
        source_run=dict(configuration=old, source_system=str(source056)))
    fit = dict(status='COMPLETED', configuration=old, optimizer_updates=3, test_opened=False,
        loss_semantics='mean_valid_position_log1p_haversine_m_div_1000_all_queries_no_auxiliary',
        training_scene_ids=['train0'], exposure=dict(successful_updates=3), final_checkpoint='final.pt')
    (source068/'fit').mkdir()
    write_json(source068/'fit/run.json', fit)
    for filename in ('step-0.pt', 'final.pt'):
        (source068/'fit'/filename).touch()  # Binding must not deserialize these files.
    (root/'new070/fit').mkdir()
    write_json(root/'new070/fit/run.json', dict(fit, configuration=config,
        loss_semantics='mean_valid_position_linear_haversine_m_div_1000_all_queries_no_auxiliary',
        pairing=dict(reference_fit=str(source068/'fit'), initial_parameters_exact=True,
                     ordered_training_scene_ids_exact=True, source_successful_updates=3,
                     matched_updates=3, matched_anchors=6)))
    checkpoint = dict(path=str(root/'new070/fit/final.pt'), step=3, actual_updates=3,
        configured_final_updates=3, routing_semantics=config['routing_semantics'],
        loss_semantics='mean_valid_position_linear_haversine_m_div_1000_all_queries_no_auxiliary')
    for filename in ('summary.json', 'run.json'):
        path = root/'new070/final'/filename
        record = read_json(path)
        record.update(configuration=config, position_loss='linear', checkpoint=checkpoint)
        write_json(path, record)
    return source068, source056, config, manifest, metadata_binding


class SupportGateLinearTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cache_binding_and_two_comparisons_need_no_models_and_keep_missing_duplicates(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, original, config, manifest, metadata = fixture(root)
            with patch('association.evaluation.support_gate_linear.bind_source', return_value=metadata), \
                    patch('torch.load', side_effect=AssertionError('checkpoint forbidden')), \
                    patch.object(SupportGate, 'forward', side_effect=AssertionError('forward forbidden')):
                binding = bind_linear_source(source, original, config, manifest)
                result = compare_linear_run(source, root/'new070/final', binding, root/'reports')
            self.assertTrue(binding['initial_route_and_mix_exact'])
            self.assertTrue(binding['source_position_metrics_verified'])
            self.assertTrue((root/'reports/objective-summary.json.gz').is_file())
            self.assertTrue((root/'reports/scene-summary.json.gz').is_file())
            self.assertEqual(set(result['comparisons']), {'vs_initial', 'vs_log_final'})
            for comparison in result['comparisons'].values():
                objective = comparison['objectives']
                self.assertEqual(objective['population']['target_identity_count'], 5)
                self.assertEqual(objective['population']['duplicate_prediction_group_count'], 1)
                self.assertEqual(objective['regions']['all']['point_count'], 127)
                self.assertEqual(len(objective['regions']['all']['methods']['final']), 4)
                self.assertIsNone(comparison['largest_component_vs_remaining']['remaining'])
                physical = comparison['physical']['paired']['all']
                self.assertEqual(physical['speed']['signed_error_change_sum'], 0.)
                self.assertEqual(physical['heading']['signed_error_change_sum'], 0.)
            self.assertNotEqual(result['comparisons']['vs_initial']['labels']['initial'],
                                result['comparisons']['vs_log_final']['labels']['initial'])

    def test_binding_rejects_other_training_change_failed_fit_wrong_source_and_initial_route(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, original, config, manifest, metadata = fixture(root)
            with patch('association.evaluation.support_gate_linear.bind_source', return_value=metadata):
                changed = deepcopy(config)
                changed['training']['learning_rate'] *= 2
                with self.assertRaisesRegex(ValueError, 'only position_loss'):
                    bind_linear_source(source, original, changed, manifest)
                with self.assertRaisesRegex(ValueError, 'immutable056'):
                    bind_linear_source(source, root/'other056', config, manifest)
                fit_path = source/'fit/run.json'
                fit = read_json(fit_path)
                write_json(fit_path, dict(fit, status='FAILED'))
                with self.assertRaisesRegex(ValueError, 'completed original068'):
                    bind_linear_source(source, original, config, manifest)
                write_json(fit_path, fit)
                path = source/'initial/predictions/states-3035.npz'
                mutate(path, lambda d: d['group_logits'].__setitem__((2, 0), 0.))
                with self.assertRaisesRegex(ValueError, 'initial logits'):
                    bind_linear_source(source, original, config, manifest)

    def test_comparison_rejects_stale_checkpoint_and_changed_duplicate_route(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, original, config, manifest, metadata = fixture(root)
            with patch('association.evaluation.support_gate_linear.bind_source', return_value=metadata):
                binding = bind_linear_source(source, original, config, manifest)
            path = root/'new070/final/summary.json'
            report = read_json(path)
            tampered = deepcopy(report)
            tampered['checkpoint']['step'] = 0
            write_json(path, tampered)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                compare_linear_run(source, root/'new070/final', binding)
            write_json(path, report)
            fit_path = root/'new070/fit/run.json'
            fit = read_json(fit_path)
            changed = deepcopy(fit)
            changed['pairing']['matched_updates'] = 2
            write_json(fit_path, changed)
            with self.assertRaisesRegex(ValueError, 'matched linear training'):
                compare_linear_run(source, root/'new070/final', binding)
            write_json(fit_path, fit)
            mutate(root/'new070/final/predictions/states-3035.npz',
                   lambda d: d['group_gate'].__setitem__((2, 0), .1))
            with self.assertRaisesRegex(ValueError, 'sigmoid route'):
                compare_linear_run(source, root/'new070/final', binding)

    def test_largest_component_and_complement_selected_by_fixed_points_not_gain(self):
        def row(sid, count, before, after):
            values = np.full((1, count), before, dtype=float)
            regions = {name: np.zeros(values.shape, bool) for name in REGIONS}
            regions['all'][:] = regions['anchor_inside_support'][:] = True
            errors = {method: values.copy() for method in METHODS}
            errors['final'][:] = after
            return dict(scene_id=sid, target_ids=np.array(['shared']), errors=errors,
                regions=regions, population=dict(target_identity_count=1), model_state_comparison={})
        rows = [row('a', 3, 100., 50.), row('b', 1, 100., 500.)]
        components = {'a': 'large', 'b': 'small'}
        result = _component_split(rows, summarize(rows, components), components)
        self.assertEqual(result['largest_component_id'], 'large')
        self.assertEqual(result['largest']['regions']['all']['point_count'], 3)
        self.assertEqual(result['remaining']['regions']['all']['point_count'], 1)
        self.assertEqual(result['remaining']['regions']['all']['paired']['final__minus__initial']
                         ['point_equal']['mae']['mean_difference'], 400.)


if __name__ == '__main__':
    unittest.main()
