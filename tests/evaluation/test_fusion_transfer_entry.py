"""062 tiny integration: real frozen inference, cache replay and unique-target output.

Only the native archive reader and production provenance envelope are replaced.
The envelope itself has separate strict-source tests; these tests use real small
checkpoints and the existing system entry point to produce all source arrays.
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.models.evidence_fusion import EvidenceFusion
from association.evaluation.fusion_controls import attribute_errors
from association.training.evidence_fusion import bind_ols_working_point, save_checkpoint
from association.utils.runtime import read_json, write_json
from tests.evaluation import test_level1_system_contracts as system_fixture
from tools.analysis.compare_association_duration import _load_metadata
from tools.experiments.level1_system_evaluate import DATA, POLICY, run as system_run


class FusionTransferEntryTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        scene = system_fixture.Level1SystemContractTests.scene()
        # A genuine support opportunity exists without changing labels or clocks.
        scene['motion']['support'][0, 16:32] = False
        scene['motion']['nearest_seconds'][0, 16:32] = 700.
        scene['motion']['exact_observation'][0, 16:32] = False
        manifest = dict(dataset_level='Level1', split_unit='scene',
            splits=dict(T_core=['t'], V_select=['v'], T_test=['sealed']))
        wp = dict(scope='ols-working-point-v1', synthetic=False, dataset_level='Level1',
            motion_observation_policy=POLICY,
            data={key: DATA[key] for key in ('length', 'max_observations', 'time_unit_seconds')},
            training_scene_ids=['t'], validation_scene_ids=['v'],
            score_semantics='-log1p(rms_haversine_m/1000)', selection_split='V_select', threshold=-1.)
        for name, value in (('manifest', manifest), ('working-point', wp), ('alignment', {})):
            write_json(root/(name + '.json'), value)
        system_fixture.Level1SystemContractTests.association_fixture(root, scene, wp)
        loaded = dict(validation=[scene], report=dict(validation_scene_ids=['v'],
            training_scene_ids=[], test_opened=False))
        common = dict(device='cpu', cpu_threads=1, archive=['synthetic-not-read.zip'],
            manifest=root/'manifest.json', alignment=root/'alignment.json',
            ols_working_point=root/'working-point.json', association_endpoint='final',
            ordinary_controls=['anchor', 'mean_supported', 'attribute_hybrid'])
        source = root/'056'/'system'
        args = SimpleNamespace(**common, output=source,
            association_scores=root/'scores.npz', association_summary=root/'association-summary.json',
            association_data_audit=root/'association-audit.json',
            association_run=root/'association-run.json', association_checkpoint=root/'association-final.pt',
            fusion_checkpoint=None, fusion_run=None)
        models, fusion_metadata = {}, {}
        fusion_root = root/'057'
        with patch('tools.experiments.level1_system_evaluate.load_timed_dataset', return_value=loaded):
            system_run(args)
            for objective, bias in (('helper_quality', .2), ('direct_only', 0.), ('mixed_action', .1)):
                arm = fusion_root/objective.replace('_', '-')
                fit = arm/'fit'
                fit.mkdir(parents=True)
                config = bind_ols_working_point(dict(data=DATA, motion_observation_policy=POLICY,
                    model=dict(exact_observation_lock=False, utility_output_bias=.01),
                    training=dict(updates=1, utility_objective=objective, mixed_action_alpha=.5)), manifest, wp)
                torch.manual_seed(101)
                model = EvidenceFusion(**config['model'])
                with torch.no_grad():
                    model.utility[-1].bias.fill_(bias)
                model.motion_observation_policy = POLICY
                model.ols_grouping = config['grouping']
                model.eval()
                save_checkpoint(fit/'final.pt', model, torch.optim.Adam(model.parameters()),
                    1, config, np.random.default_rng(101))
                write_json(fit/'run.json', dict(status='COMPLETED', configuration=config,
                    optimizer_updates=1, final_evaluation='evaluation-step-1.json',
                    training_scene_ids=['t'], validation_scene_ids=['v']))
                fusion_args = SimpleNamespace(**common, output=arm/'system', association_scores=None,
                    association_summary=None, association_data_audit=None, association_run=None,
                    association_checkpoint=None, fusion_checkpoint=fit/'final.pt', fusion_run=fit/'run.json')
                result = system_run(fusion_args)
                models[objective] = model
                fusion_metadata[objective] = dict(system=str(arm/'system'), summary=result,
                    checkpoint=str(fit/'final.pt'), run=str(fit/'run.json'),
                    expected={'v': result['per_scene'][0]['unique_target']['ols_learned']['full_target_metrics']})
        metadata = _load_metadata(source, manifest, 1)
        metadata['component_ids'] = {'v': 'v'}
        # An independent literal construction for this synthetic fixture; the
        # production061 expected value remains frozen and is not weakened.
        with np.load(source/'predictions'/'states-v.npz', allow_pickle=False) as saved:
            control = saved['learned_assoc_mean_supported'].copy()
            control[..., :2] = saved['learned_assoc_anchor'][..., :2]
            control[0, 16:32, :2] = saved['learned_assoc_mean_supported'][0, 16:32, :2]
        errors = attribute_errors(control[0], scene['fusion_targets']['ship-a']['states'])['position']
        expected_support = dict(intersection_count=64, mae=float(errors.mean()),
                                rmse=float(np.sqrt(np.square(errors).mean())))
        entry = SimpleNamespace(source_system=source, source_fusion=fusion_root,
            archive=['synthetic-not-read.zip'], manifest=root/'manifest.json',
            alignment=root/'alignment.json', output=root/'062-output', cpu_threads=1)
        return entry, (metadata, models, fusion_metadata), loaded, expected_support

    def test_real_frozen_inference_replay_alignment_and_saved_predictions(self):
        from tools.analysis.frozen_fusion_transfer import run
        with TemporaryDirectory() as temporary:
            args, binding, loaded, expected_support = self.fixture(Path(temporary))
            before = {name: deepcopy(model.state_dict()) for name, model in binding[1].items()}
            with patch('tools.analysis.frozen_fusion_transfer.bind_sources', return_value=binding), \
                    patch('tools.analysis.frozen_fusion_transfer.load_timed_dataset', return_value=loaded) as loader, \
                    patch('tools.analysis.frozen_fusion_transfer.EXPECTED_SUPPORT_POSITION', expected_support):
                result = run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertTrue(loader.call_args.kwargs['validation_only'])
            self.assertEqual(read_json(args.output/'run.json')['status'], 'COMPLETED')
            self.assertFalse(result['test_opened'])
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertEqual(set(result['methods']), {'anchor', 'mean_supported', 'attribute_hybrid',
                'support_switch', 'helper_quality', 'direct_only', 'mixed_action'})
            for name, model in binding[1].items():
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name][key], atol=0, rtol=0)
            self.assertEqual(result['methods']['anchor']['population']['target_identity_count'], 3)
            self.assertEqual(result['methods']['anchor']['population']['predicted_group_count'], 1)
            self.assertEqual(result['methods']['anchor']['population']['anchor_matched_target_count'], 1)
            self.assertTrue(result['support_control_analytical_verified'])
            self.assertEqual(result['routing']['helper_quality']['all_output_groups']['position']['output_query_count'], 64)
            self.assertEqual(result['routing']['direct_only']['all_output_groups']['position']['positive_helper_query_count'], 0)
            with np.load(args.output/'predictions'/'states-v.npz', allow_pickle=False) as saved:
                self.assertEqual(int(saved['support_switch_position_mask'].sum()), 16)
                np.testing.assert_array_equal(saved['output_support_switch'][0, 16:32, :2],
                                              saved['output_mean_supported'][0, 16:32, :2])
                np.testing.assert_array_equal(saved['output_support_switch'][0, :16, :2],
                                              saved['output_anchor'][0, :16, :2])
                np.testing.assert_array_equal(saved['output_support_switch'][..., 2:],
                                              saved['output_mean_supported'][..., 2:])
                np.testing.assert_array_equal(saved['output_direct_only'], saved['native_states'])
                self.assertEqual(saved['helper_quality_routing_helper_weights'].shape, (6, 64, 3))
            with np.load(args.output/'system-predictions'/'states-v.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['support_switch_aligned_group_indices'], [0, -1, -1])
                np.testing.assert_array_equal(saved['support_switch_aligned_prediction_valid'],
                                              saved['anchor_aligned_prediction_valid'])
                self.assertEqual(saved['support_switch_group_states'].shape, (1, 64, 4))
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual(read_json(args.output/'run.json')['status'], 'COMPLETED')

    def test_tampered_ordinary_cache_and_native_inputs_fail_without_success_report(self):
        from tools.analysis.frozen_fusion_transfer import run
        for case in ('ordinary_prediction', 'native_input', 'frozen_prediction', 'reference_registry'):
            with self.subTest(case=case), TemporaryDirectory() as temporary:
                args, binding, loaded, expected_support = self.fixture(Path(temporary))
                if case == 'ordinary_prediction':
                    path = args.source_system/'predictions'/'states-v.npz'
                    with np.load(path, allow_pickle=False) as saved:
                        arrays = {key: saved[key].copy() for key in saved.files}
                    arrays['learned_assoc_mean_supported'][0, 0, 0] += .001
                    np.savez_compressed(path, **arrays)
                elif case in ('native_input', 'reference_registry'):
                    loaded = deepcopy(loaded)
                    if case == 'native_input':
                        loaded['validation'][0]['motion']['states'][0, 0, 0] += .001
                    else:
                        loaded['validation'][0]['fusion_targets']['ship-c']['states'][0, 0] += .001
                else:
                    path = Path(binding[2]['helper_quality']['system'])/'predictions'/'states-v.npz'
                    with np.load(path, allow_pickle=False) as saved:
                        arrays = {key: saved[key].copy() for key in saved.files}
                    arrays['ols_learned'][0, 0, 0] += .001
                    np.savez_compressed(path, **arrays)
                with patch('tools.analysis.frozen_fusion_transfer.bind_sources', return_value=binding), \
                        patch('tools.analysis.frozen_fusion_transfer.load_timed_dataset', return_value=loaded), \
                        patch('tools.analysis.frozen_fusion_transfer.EXPECTED_SUPPORT_POSITION', expected_support):
                    with self.assertRaises(ValueError):
                        run(args)
                self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')
                self.assertFalse((args.output/'summary.json').exists())

    def test_diagnostic_preserves_rejection_without_continuing_system_evaluation(self):
        from tools.analysis.frozen_fusion_transfer import run
        for corrupt_utility in (False, True):
            with self.subTest(corrupt_utility=corrupt_utility), TemporaryDirectory() as temporary:
                args, binding, loaded, _ = self.fixture(Path(temporary))
                args.diagnose_first_scene = True
                path = Path(binding[2]['helper_quality']['system'])/'predictions'/'states-v.npz'
                if corrupt_utility:
                    with np.load(path, allow_pickle=False) as saved:
                        arrays = {key: saved[key].copy() for key in saved.files}
                    arrays['ols_routing_utility'][0, 0, 0] += 1.
                    np.savez_compressed(path, **arrays)
                model = binding[1]['helper_quality']
                with patch('tools.analysis.frozen_fusion_transfer.bind_sources', return_value=binding), \
                        patch('tools.analysis.frozen_fusion_transfer.load_timed_dataset', return_value=loaded) as loader, \
                        patch.object(model, 'forward', wraps=model.forward) as forward, \
                        patch.object(binding[1]['direct_only'], 'forward', side_effect=AssertionError('second arm')), \
                        patch.object(binding[1]['mixed_action'], 'forward', side_effect=AssertionError('third arm')), \
                        patch('tools.analysis.frozen_fusion_transfer.evaluate_unique_population',
                              side_effect=AssertionError('full system evaluation')):
                    result = run(args)
                self.assertTrue(loader.call_args.kwargs['validation_only'])
                self.assertEqual(forward.call_count, 2 * len(loaded['validation'][0]['node_ids']))
                self.assertEqual(result['source_replay_passed'], not corrupt_utility)
                self.assertEqual(result['first_vs_old']['utility']['passed'], not corrupt_utility)
                self.assertTrue(result['repeat_vs_first']['utility']['passed'])
                self.assertFalse(result['full_system_evaluation'])
                self.assertEqual(result['learned_partition_forwards'], 0)
                self.assertEqual(result['diagnostic_scene_count'], 1)
                state = read_json(args.output/'run.json')
                self.assertEqual(state['status'], 'COMPLETED')
                self.assertEqual(state['completed_scenes'], 0)
                self.assertFalse((args.output/'summary.json').exists())
                self.assertFalse((args.output/'system-predictions').exists())
                self.assertTrue((args.output/'first-forward.npz').is_file())
                self.assertTrue((args.output/'trace-second-forward.npz').is_file())
                self.assertTrue((args.output/'diagnostic-summary.json').is_file())
                if corrupt_utility:
                    # Diagnostic completion must not weaken the normal062 gate.
                    args.diagnose_first_scene = False
                    args.output = Path(temporary)/'normal-062'
                    with patch('tools.analysis.frozen_fusion_transfer.bind_sources', return_value=binding), \
                            patch('tools.analysis.frozen_fusion_transfer.load_timed_dataset', return_value=loaded):
                        with self.assertRaisesRegex(ValueError, 'replay tolerance'):
                            run(args)
                    self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')

    def test_diagnostic_safe_probe_is_compared_and_script_is_only_archived(self):
        from tools.analysis.frozen_fusion_transfer import run
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, binding, loaded, _ = self.fixture(root)
            args.diagnose_first_scene = True
            args.probe_motion, args.probe_script = root/'probe.npz', root/'probe.py'
            probe = deepcopy(loaded['validation'][0]['motion'])
            probe['native_count'][0] += 1
            np.savez_compressed(args.probe_motion, **probe)
            args.probe_script.write_text('raise RuntimeError("must never execute")\n')
            with patch('tools.analysis.frozen_fusion_transfer.bind_sources', return_value=binding), \
                    patch('tools.analysis.frozen_fusion_transfer.load_timed_dataset', return_value=loaded):
                result = run(args)
            self.assertEqual(result['probe_input_comparison']['status'], 'compared')
            self.assertFalse(result['probe_input_comparison']['fields']['native_count']['passed'])
            self.assertTrue(result['probe_input_comparison']['fields']['states']['passed'])
            self.assertFalse(result['probe_source']['script_executed'])
            self.assertEqual((args.output/'supplied-probe-script.txt').read_text(), args.probe_script.read_text())

    def test_diagnostic_does_not_accept_corrupt_source_artifacts(self):
        from tools.analysis.frozen_fusion_transfer import run
        with TemporaryDirectory() as temporary:
            args, binding, loaded, _ = self.fixture(Path(temporary))
            args.diagnose_first_scene = True
            path = Path(binding[2]['helper_quality']['system'])/'system-predictions'/'states-v.npz'
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            arrays['ols_learned_aligned_group_indices'][0] = -1
            np.savez_compressed(path, **arrays)
            with patch('tools.analysis.frozen_fusion_transfer.bind_sources', return_value=binding), \
                    patch('tools.analysis.frozen_fusion_transfer.load_timed_dataset', return_value=loaded):
                with self.assertRaisesRegex(ValueError, 'complete learned system'):
                    run(args)
            self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')
            self.assertFalse((args.output/'diagnostic-summary.json').exists())
            self.assertTrue((args.output/'first-forward.npz').is_file())


if __name__ == '__main__':
    unittest.main()
