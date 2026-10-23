"""077 all-group physical shrink and source pairing, without model execution."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.sequence_position import evaluate_sequence_position
from association.evaluation.system_attribution import pool_metrics
from association.models.sequence_position import EARTH_RADIUS_M, SequencePosition
from association.utils.runtime import read_json, write_json, write_json_gzip
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tools.analysis.compare_position_harm import scene_pair, summarize as original_summary
from tools.analysis import compare_residual_shrink as shrink


def fixture(root, *, sparse=True, two_scenes=False):
    """Production-generated endpoint arrays plus explicit toy metadata attestations.

    No claim that a 4000-update real job was trained here: production budget
    metadata is deliberately synthetic, to test the cache-only binding path.
    """
    root = root.resolve()
    source, metadata, validation = support_gate_cache_fixture(root, sparse=sparse)
    if two_scenes:
        for directory in ('predictions', 'system-predictions'):
            (source/directory/'states-3036.npz').write_bytes((source/directory/'states-3035.npz').read_bytes())
        validation.append(dict(deepcopy(validation[0]), scene_id='3036', component_id='other'))
        metadata.update(scene_ids=['3035', '3036'], component_ids={'3035': 'shared', '3036': 'other'},
            fragment_count=2*metadata['fragment_count'], expected=dict(metadata['expected'], **{'3036': metadata['expected']['3035']}),
            pooled={method: pool_metrics([row['full_target_metrics']]*2) for method, row in metadata['expected']['3035'].items()})
    configs = root/'repository/configs'
    (configs/'data').mkdir(parents=True)
    manifest = dict(splits=dict(T_core=['toy-training'], V_select=metadata['scene_ids']), components=metadata['component_ids'])
    write_json(configs/'data/mtad-level1-scene3000.json', manifest)
    arguments = dict(output=str(root/'output'), cpu_threads=4)
    for seed, suffixes in shrink.CONFIGS.items():
        arms = {}
        for name, suffix in zip(('linear', 'harm'), suffixes):
            config_name = f'sequence-position-transformer-{suffix}.json'
            config = read_json(shrink.ROOT/'configs'/config_name)
            write_json(configs/config_name, config)
            path = root/(name+str(seed)[-2:])
            model = SequencePosition(backbone='transformer')
            with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                evaluate_sequence_position(model, validation, source, metadata, path/'initial', initial=True)
                with torch.no_grad():
                    model.output.bias.copy_(torch.tensor([.1, -.2])*(1. if name == 'linear' else .4))
                evaluate_sequence_position(model, validation, source, metadata, path/'final')
            record = dict(status='COMPLETED', configuration=config, optimizer_updates=4000, test_opened=False,
                source_system=str(source), completed_endpoints=['initial', 'final'], validation_selection=False,
                final_checkpoint='final.pt', training_scene_ids=['toy-training'])
            write_json(path/'run.json', record)
            write_json(path/'summary.json', record)
            (path/'fit').mkdir()
            write_json(path/'fit/run.json', record)
            for label, step, filename in (('initial', 0, 'step-0.pt'), ('final', 4000, 'final.pt')):
                cp = dict(step=step, actual_updates=step, configured_final_updates=4000, path=str(path/'fit'/filename))
                for file in ('run.json', 'summary.json'):
                    values = read_json(path/label/file)
                    values.update(checkpoint=cp, configuration=config)
                    write_json(path/label/file, values)
            arms[name] = dict(path=path)
            arguments[name+str(seed)[-2:]] = str(path)
        pairs = [scene_pair(arms, sid) for sid in metadata['scene_ids']]
        summary = original_summary([row for row, _ in pairs], [arrays for _, arrays in pairs])
        summary.update(status='COMPLETED', seed=seed, inputs={key: str(value['path']) for key, value in arms.items()},
            model_forwards=0, optimizer_updates=0, test_opened=False, pairing=dict(status='PAIR_VERIFIED',
                seed=seed, matched_updates=4000, matched_anchors=32000, initial_parameters_exact=True,
                initial_sampler_rng_exact=True, actual_successful_batches_exact=True))
        paired = root/('paired'+str(seed)[-2:])
        paired.mkdir()
        write_json_gzip(paired/'summary.json.gz', summary)
        write_json(paired/'run.json', dict(status='COMPLETED', model_forwards=0, optimizer_updates=0,
            test_opened=False, completed_scenes=len(metadata['scene_ids']), population=summary['population']))
        arguments['paired'+str(seed)[-2:]] = str(paired)
    return Namespace(**arguments), root/'repository'


class ResidualShrinkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_full_two_seed_cache_entry_without_weights_or_forwards(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args, repository = fixture(root)
            with ExitStack() as stack:
                stack.enter_context(patch.object(shrink, 'ROOT', repository))
                stack.enter_context(patch('torch.load', side_effect=AssertionError('checkpoint forbidden')))
                stack.enter_context(patch.object(SequencePosition, 'forward', side_effect=AssertionError('forward forbidden')))
                result = shrink.run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(set(result['seeds']), {'20261023', '20261021'})
            self.assertEqual(result['alpha'], .5)
            state = read_json(root/'output/run.json')
            self.assertEqual(state['completed_scenes'], 2)
            self.assertEqual(state['checkpoints_loaded'], 0)
            for seed, summary in result['seeds'].items():
                self.assertEqual(summary['population']['predicted_group_count'], 4)
                self.assertEqual(summary['population']['duplicate_prediction_group_count'], 1)
                self.assertEqual(summary['position_population']['covered_target_points'], 1)
                self.assertEqual(summary['relative_harm']['all']['half_linear']['common_point_count'], 1)
                geo = summary['residual_geometry']
                self.assertEqual(geo['all_output_groups']['metrics']['norm_ratio']['defined_count'], 256)
                self.assertEqual(geo['duplicate_output_groups']['metrics']['norm_ratio']['defined_count'], 64)
                self.assertEqual(geo['unique_selected_groups']['metrics']['norm_ratio']['defined_count'], 192)
                self.assertEqual(geo['unique_valid_points']['metrics']['norm_ratio']['defined_count'], 1)
                self.assertAlmostEqual(geo['all_output_groups']['metrics']['norm_ratio']['mean'], .4, places=7)
                self.assertEqual(sum(geo[key]['metrics']['norm_ratio']['total_count']
                    for key in ('linear_improved', 'linear_harmed', 'linear_tied')), 1)
                with np.load(root/f'output/{seed}/predictions/states-3035.npz', allow_pickle=False) as half:
                    self.assertEqual(half['half_linear_group_states'].shape, (4, 64, 4))
                    self.assertEqual(half['aligned_group_indices'].shape, (6,))
                    self.assertEqual(half['alpha'], .5)
                    self.assertNotIn('aligned_truth', half.files)
            with self.assertRaises(FileExistsError):
                shrink.run(args)

    def test_geometry_zeros_reverse_orthogonal_and_no_pool_mean_of_quantiles(self):
        values = shrink.geometry(np.array([[0., 0.], [2., 0.], [2., 0.], [2., 0.]]),
                                 np.array([[1., 1.], [0., 0.], [-1., 0.], [0., 2.]]))
        self.assertTrue(np.isnan(values['norm_ratio'][0]))
        self.assertTrue(np.isnan(values['projection_scale'][0]))
        self.assertTrue(np.isnan(values['cosine'][:2]).all())
        np.testing.assert_array_equal(values['cosine'][2:], [-1., 0.])
        np.testing.assert_array_equal(values['projection_scale'][1:], [0., -.5, 0.])
        np.testing.assert_array_equal(values['orthogonal_norm_m'][1:], [0., 0., 2.])
        self.assertEqual(shrink.distribution(values['cosine'])['undefined_count'], 2)
        self.assertIsNone(shrink.distribution([np.nan])['p99'])
        with self.assertRaises(ValueError):
            shrink.geometry([[np.inf, 0.]], [[1., 1.]])

    def test_remaining_components_retain_real_point_and_scene_denominators(self):
        with TemporaryDirectory() as directory:
            args, repository = fixture(Path(directory), two_scenes=True)
            with patch.object(shrink, 'ROOT', repository):
                result = shrink.run(args)
            for summary in result['seeds'].values():
                self.assertEqual(summary['largest_component_id'], 'other')  # Fixed id tie-break.
                self.assertEqual(summary['remaining_components']['scene_ids'], ['3035'])
                self.assertEqual(summary['remaining_components']['position_population']['covered_target_points'], 1)
                self.assertEqual(summary['position_population']['covered_target_points'], 2)

    def test_half_uses_sphere_across_date_line_exact_zero_and_native_other_attributes(self):
        # Half of a tangent displacement is not an average of geographic longitudes.
        prior = np.array([[[0., 179., 3., 350.], [90., 33., 4., 10.]]], np.float64)
        residual = np.array([[[0., EARTH_RADIUS_M], [0., 0.]]], np.float64)
        before = prior.copy(), residual.copy()
        output = shrink.half_residual_states(prior, residual)
        self.assertAlmostEqual(output[0, 0, 1], (179.+np.rad2deg(np.arctan(.5))+180.) % 360.-180., places=10)
        np.testing.assert_array_equal(output[0, 1], prior[0, 1])
        np.testing.assert_array_equal(output[..., 2:], prior[..., 2:])
        np.testing.assert_array_equal(prior, before[0])
        np.testing.assert_array_equal(residual, before[1])
        with self.assertRaises(ValueError):
            shrink.half_residual_states(prior, residual[:, :, :1])

    def test_duplicate_only_residual_tampering_rejected_before_unique_evaluation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args, repository = fixture(root)
            arms = {name: dict(path=Path(getattr(args, name+'23'))) for name in ('linear', 'harm')}
            path = Path(args.harm23)/'final/predictions/states-3035.npz'
            with np.load(path, allow_pickle=False) as saved:
                values = dict(saved)
            values['group_tangent_residual_m'][1, 0, 0] += 1.
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, 'all-group tangent replay'):
                shrink.scene_analysis(arms, '3035')

    def test_source_binding_rejects_provenance_missing_population_and_unverified_pair(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args, repository = fixture(root)
            with patch.object(shrink, 'ROOT', repository):
                invoke = lambda: shrink.bind_pair(args.linear23, args.harm23, args.paired23, 20261023)
                arms, binding = invoke()
                self.assertFalse(binding['checkpoint_loaded_this_run'])
                run_path = Path(args.harm23)/'fit/run.json'
                original = read_json(run_path)
                for changed in (dict(original, test_opened=True), dict(original, optimizer_updates=3999),
                                dict(original, training_scene_ids=[])):
                    write_json(run_path, changed)
                    with self.assertRaises(ValueError):
                        invoke()
                write_json(run_path, original)
                source = Path(args.linear23)/'final/predictions/states-extra.npz'
                np.savez_compressed(source, dummy=np.array([0]))
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    invoke()
                source.unlink()
                paired = Path(args.paired23)/'summary.json.gz'
                saved = shrink.read_json(paired)
                for field, change in (('inputs', {'linear': args.harm23, 'harm': args.linear23}),
                                      ('seed', 20261021), ('scene_ids', []),
                                      ('pairing', dict(saved['pairing'], initial_parameters_exact=False))):
                    write_json_gzip(paired, dict(saved, **{field: change}))
                    with self.assertRaises(ValueError):
                        invoke()
                write_json_gzip(paired, saved)
                invoke()

    def test_rejects_changed_original_paired_metrics_and_preserves_failed_attempt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args, repository = fixture(root)
            path = Path(args.paired23)/'summary.json.gz'
            changed = shrink.read_json(path)
            changed['position_weightings']['all']['linear']['point_equal']['mae'] += 1.
            write_json_gzip(path, changed)
            with patch.object(shrink, 'ROOT', repository), self.assertRaisesRegex(ValueError, 'paired table metrics'):
                shrink.run(args)
            state = read_json(root/'output/run.json')
            self.assertEqual(state['status'], 'FAILED')
            self.assertEqual(state['retries'], 0)
            self.assertTrue((root/'output/20261023/predictions/states-3035.npz').exists())
            self.assertFalse((root/'output/summary.json.gz').exists())

    def test_parser_disallows_alpha_and_execution_cannot_overwrite_sources(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            shrink.main(['--alpha', '.7'])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args, repository = fixture(root)
            args.output = str(Path(args.harm23)/'new-output')
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                shrink.run(args)
            args.output = str(root/'untouched')
            args.cpu_threads = 2
            with self.assertRaisesRegex(ValueError, 'CPU4'):
                shrink.run(args)
            self.assertFalse(Path(args.output).exists())


if __name__ == '__main__':
    unittest.main()
