"""098 falsifiable gates, actual source/budget pairing and full cache invariants."""
from copy import deepcopy
from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.frozen_position_adapter import evaluate_frozen_position_adapter
from association.evaluation.query_relation import evaluate_query_relation
from association.evaluation.sequence_position import WEIGHTINGS, evaluate_sequence_position
from association.models.frozen_position_adapter import FrozenPositionAdapter, initialize_from_teacher
from association.models.query_relation import SCOPE, ROUTING_SEMANTICS, QueryRelationPosition, initialize_from_reference
from association.training.frozen_position_adapter import train_frozen_position_adapter
from association.training.query_relation import train_query_relation
from association.training.sequence_position import load_checkpoint, train_sequence_position
from association.utils.runtime import read_json, write_json, write_json_gzip
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_train
from tools.analysis import compare_query_relation as compare


def gate_fixture():
    values = {'relation': (94., 150., 220., 300.), 'generic': (98., 160., 230., 310.),
              'finetune': (100., 170., 240., 320.), 'relation_off': (95., 155., 225., 305.)}
    table = {name: {weight: dict(zip(('mae', 'rmse', 'p95', 'p99'), numbers), unit_count=100)
                   for weight in WEIGHTINGS} for name, numbers in values.items()}
    item = dict(position_weightings={'all': table})
    summary = dict(item, components={str(i): deepcopy(item) for i in range(3)},
        remaining_components=deepcopy(item), no_multi_prediction_exact=True,
        group_cohorts={'any_multi_query': {'relation': {'point_equal': {'unit_count': 60}}}})
    return summary, [deepcopy(item) for _ in range(3)]


def tiny_configs():
    base = read_json(compare.ROOT/'configs/sequence-position-transformer-extended-level1.json')
    base['training'].update(updates=3, batch_size=2)
    adapter = deepcopy(base)
    # Read the canonical historical strings, preserving the paired tiny budget.
    original_adapter = read_json(compare.ROOT/'configs/frozen-position-adapter-full-finetune-level1.json')
    adapter.update({key: deepcopy(original_adapter[key]) for key in ('scope', 'routing_semantics', 'model')})
    candidates = {}
    for mode in ('generic', 'relation'):
        config = deepcopy(base)
        config.update(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS)
        config['model'] = dict(mode=mode, hidden_size=64, num_layers=2, initial_gate=.95)
        candidates[mode] = config
    return base, adapter, candidates


class QueryRelationComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_gate_accepts_only_both_strong_controls_and_reports_no_promotion(self):
        summary, scenes = gate_fixture()
        decision = compare.scientific_screen(summary, scenes)
        self.assertTrue(decision['seed_screen_passed'])
        self.assertEqual(decision['versus_budget']['required_reduction'], .03)
        self.assertEqual(decision['versus_generic']['required_reduction'], .01)
        self.assertFalse(decision['default_promoted'])
        self.assertEqual(decision['required_seeds'], [20261023, 20261021])
        self.assertEqual(decision['outcome'], 'PASS_THIS_SEED_ONLY')

    def test_gate_rejects_small_effect_tail_regression_population_concentration_and_unused_bias(self):
        for failure in ('budget', 'generic', 'rmse', 'p99', 'weight', 'outside', 'scenes',
                        'components', 'bias_no_effect', 'no_multi_population', 'invariance'):
            with self.subTest(failure=failure):
                summary, scenes = gate_fixture()
                table = summary['position_weightings']['all']
                if failure == 'budget': table['finetune']['point_equal']['mae'] = 96.
                elif failure == 'generic': table['generic']['point_equal']['mae'] = 94.5
                elif failure in ('rmse', 'p99'): table['relation']['point_equal'][failure] = 500.
                elif failure == 'weight': table['relation']['scene_equal_target_mean']['mae'] = 101.
                elif failure == 'outside':
                    summary['remaining_components']['position_weightings']['all']['relation']['point_equal']['mae'] = 101.
                elif failure == 'scenes':
                    for item in scenes[:2]: item['position_weightings']['all']['relation']['point_equal']['mae'] = 101.
                elif failure == 'components':
                    for key in ('0', '1'):
                        summary['components'][key]['position_weightings']['all']['relation']['point_equal']['mae'] = 101.
                elif failure == 'bias_no_effect': table['relation_off']['point_equal']['mae'] = 94.
                elif failure == 'no_multi_population':
                    summary['group_cohorts']['any_multi_query']['relation']['point_equal']['unit_count'] = 0
                else: summary['no_multi_prediction_exact'] = False
                self.assertFalse(compare.scientific_screen(summary, scenes)['seed_screen_passed'])
        summary, scenes = gate_fixture()
        summary['position_weightings']['all']['relation']['point_equal']['mae'] = 99.
        self.assertEqual(compare.scientific_screen(summary, scenes)['outcome'], 'GENERIC_BASELINE_SIGNAL_ONLY')

    def test_actual_three_added_fits_pair_to_teacher_final_and_reject_changed_batch_or_initial_tensor(self):
        base, adapter, configs = tiny_configs()
        train = tiny_train()
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            train_sequence_position(train, base, root/'linear/fit')
            train_frozen_position_adapter(train, adapter, root/'budget/fit', paired_reference_fit=root/'linear/fit')
            arms = dict(linear=dict(path=root/'linear'),
                finetune=dict(path=root/'budget', configuration=adapter, paired_control=root/'linear'))
            for mode, config in configs.items():
                train_query_relation(train, config, root/mode/'fit', paired_reference_fit=root/'linear/fit')
                arms[mode] = dict(path=root/mode, configuration=config,
                    paired_control=root/'linear', budget_control=root/'budget')
            paired = compare.verify_pairing(arms)
            self.assertTrue(paired['generic_relation_initial_parameters_exact'])
            self.assertTrue(paired['generic_relation_parameter_count_equal'])
            self.assertEqual(paired['matched_updates'], 3)
            log = root/'generic/fit/training.jsonl'
            original = log.read_text()
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0]['batch'][0]['eligible_query_points'] += 1
            log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(ValueError): compare.verify_pairing(arms)
            log.write_text(original)
            checkpoint = root/'generic/fit/step-0.pt'
            payload = torch.load(checkpoint, weights_only=False)
            payload['model']['phi.0.weight'][0, 0] += .1
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, 'initialization'):
                compare.verify_pairing(arms)

    def test_real_cache_population_initial_source_and_fixed_weight_off_invariance(self):
        base, adapter, configs = tiny_configs()
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, metadata, validation = support_gate_cache_fixture(root)
            train_sequence_position(tiny_train(), base, root/'linear/fit')
            original, saved = load_checkpoint(root/'linear/fit/final.pt')
            arms = dict(linear=dict(path=root/'linear'),
                finetune=dict(path=root/'budget', configuration=adapter))
            budget = FrozenPositionAdapter(mode='full_finetune')
            initialize_from_teacher(budget, saved['model'], branch_seed=base['training']['seed']+90000)
            with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                evaluate_sequence_position(original, validation, source, metadata, root/'linear/final')
                evaluate_frozen_position_adapter(budget, validation, source, metadata, root/'budget/initial',
                    source_teacher_checkpoint=root/'linear/fit/final.pt', initial=True)
                with torch.no_grad(): budget.teacher.output.bias[0] += .01
                evaluate_frozen_position_adapter(budget, validation, source, metadata, root/'budget/final',
                    source_teacher_checkpoint=root/'linear/fit/final.pt')
                for mode in ('generic', 'relation'):
                    model = QueryRelationPosition(mode=mode)
                    initialize_from_reference(model, saved['model'], branch_seed=base['training']['seed']+98000)
                    arms[mode] = dict(path=root/mode, configuration=configs[mode])
                    evaluate_query_relation(model, validation, source, metadata, root/mode/'initial',
                        source_teacher_checkpoint=root/'linear/fit/final.pt', initial=True)
                    with torch.no_grad():
                        model.output.bias[0] += .02
                        model.branch_projection.weight.fill_(.01)
                    evaluate_query_relation(model, validation, source, metadata, root/mode/'final',
                        source_teacher_checkpoint=root/'linear/fit/final.pt')
                    if mode == 'relation':
                        evaluate_query_relation(model, validation, source, metadata, root/mode/'relation_off',
                            source_teacher_checkpoint=root/'linear/fit/final.pt', relation_enabled=False)
            row, arrays = compare.scene_pair(arms, '3035')
            result = compare.summarize([row], [arrays])
            self.assertEqual(result['population']['predicted_group_count'], 4)
            self.assertEqual(result['population']['duplicate_prediction_group_count'], 1)
            self.assertEqual(result['position_population']['covered_target_points'], 127)
            self.assertTrue(result['no_multi_prediction_exact'])
            self.assertEqual(result['group_relation_population']['groups_with_no_multi_query'], 4)
            self.assertEqual(result['group_cohorts']['any_multi_query']['relation']['point_equal']['unit_count'], 0)
            values = arrays['relation_position_error'][np.isfinite(arrays['relation_position_error'])]
            self.assertEqual(result['position_weightings']['all']['relation']['point_equal']['p99'], np.quantile(values, .99))
            path = root/'relation/relation_off/system-predictions/states-3035.npz'
            with np.load(path, allow_pickle=False) as cache:
                corrupted = {key: cache[key].copy() for key in cache.files}
            corrupted['learned_sequence_aligned_prediction_valid'][0, 0, 0] = False
            np.savez_compressed(path, **corrupted)
            with self.assertRaisesRegex(ValueError, 'target/population/masks'):
                compare.scene_pair(arms, '3035')

    def test_two_seed_closeout_recomputes_gates_and_rejects_mixed_missing_or_false_pass(self):
        from tools.analysis import summarize_query_relation as closeout
        manifest = dict(splits={'V_select': ['a', 'b', 'c']}, components={'a': '0', 'b': '1', 'c': '2'})
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def table(seed, passed=True):
                path = root/str(seed)
                path.mkdir(exist_ok=True)
                summary, scenes = gate_fixture()
                if not passed:
                    summary['position_weightings']['all']['relation']['point_equal']['mae'] = 99.
                summary['position_weightings']['all']['linear'] = deepcopy(summary['position_weightings']['all']['finetune'])
                summary.update(scope=compare.SCOPE, status='COMPLETED', seed=seed,
                    model_forwards=0, optimizer_updates=0, test_opened=False, default_promoted=False,
                    source_system='/fixture/056/system', scene_ids=['a', 'b', 'c'], scene_count=3,
                    pairing=dict(status='PAIR_VERIFIED', seed=seed), initial_output_source_final_exact=True,
                    population={'predicted_group_count': 8}, position_population={'covered_target_points': 300},
                    helper_population={'unique_selected_valid_position_queries': {'total': 300}})
                for sid, scene in zip(summary['scene_ids'], scenes):
                    scene.update(scene_ids=[sid], component_id=manifest['components'][sid])
                summary['scientific_screen'] = compare.scientific_screen(summary, scenes)
                record = {key: summary[key] for key in ('scope', 'status', 'seed', 'model_forwards',
                    'optimizer_updates', 'test_opened', 'default_promoted')}
                record.update(completed_scenes=3, seed_screen_passed=summary['scientific_screen']['seed_screen_passed'])
                write_json(path/'run.json', record)
                write_json_gzip(path/'summary.json.gz', summary)
                write_json_gzip(path/'scene-summary.json.gz', scenes)
                return path, summary

            first, _ = table(20261023)
            second, _ = table(20261021)
            with patch.object(closeout, 'declared_manifest', return_value=manifest):
                params = Namespace(seed23_table=str(first), seed21_table=str(second), output=str(root/'both'))
                result = closeout.run(params)
                self.assertTrue(result['eligible_for_independent_confirmation'])
                self.assertFalse(result['default_promoted'])
                self.assertFalse(result['test_opened'])
                with self.assertRaises(FileExistsError): closeout.run(params)
                table(20261021, passed=False)
                params.output = str(root/'one')
                result = closeout.run(params)
                self.assertFalse(result['eligible_for_independent_confirmation'])
                params.seed23_table, params.seed21_table, params.output = str(second), str(first), str(root/'mixed')
                with self.assertRaisesRegex(ValueError, 'correctly bound seed'):
                    closeout.run(params)
                params.seed23_table, params.seed21_table, params.output = str(first), str(root/'missing'), str(root/'missing-out')
                with self.assertRaises(FileNotFoundError): closeout.run(params)
                self.assertEqual(read_json(root/'missing-out/run.json')['status'], 'FAILED')
                _, changed = table(20261021, passed=False)
                changed['scientific_screen']['seed_screen_passed'] = True
                write_json_gzip(second/'summary.json.gz', changed)
                params.seed21_table, params.output = str(second), str(root/'fake-pass')
                with self.assertRaisesRegex(ValueError, 'scientific gate'):
                    closeout.run(params)


if __name__ == '__main__':
    unittest.main()
