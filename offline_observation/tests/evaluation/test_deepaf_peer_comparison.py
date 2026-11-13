"""099 actual-cache scoring, fixed LR choice and immutable-population boundaries."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.sequence_position import _prior, _scene
from association.evaluation.support_control import support_switch_prediction
from association.evaluation.support_gate_objectives import _artifact
from association.utils.runtime import read_json, write_json
from tests.fixtures.cache import six_arm_cache_fixture
from tools.analysis import compare_deepaf_peer as compare
from tools.experiments.level1_system_evaluate import evaluate_unique_population


def cached_arm(root, name, *, latitude_offset=.0001, lr=.001):
    """Create real evaluator artifacts from prescribed coordinates, no NN calls."""
    fragment, original_system, _ = six_arm_cache_fixture()
    base = _artifact(original_system, 'learned_assoc_anchor')
    groups = [[0, 1], [2, 3], [4], [5]]
    predictions = {method: fragment['learned_assoc_'+method].copy()
        for method in ('anchor', 'mean_supported', 'attribute_hybrid')}
    predictions['support_switch'], switch = support_switch_prediction(
        predictions['anchor'], predictions['mean_supported'], fragment['support'], groups)
    predictions['fixed_prior095'] = _prior(predictions['anchor'], predictions['mean_supported'], switch)
    learned = predictions['mean_supported'].copy()
    for group, target in zip(groups, base['group_anchor_target_indices']):
        valid = base['aligned_truth_valid'][target, :, :2].all(-1)
        point = base['aligned_truth'][target, valid, :2][0] if valid.any() else np.array([30.04, 120.])
        learned[group[0], :, :2] = point
        learned[group[0], :, 0] += latitude_offset
    predictions['learned_sequence'] = learned
    scene = dict(identities=fragment['identities'], target_identity_ids=fragment['target_identity_ids'].tolist(),
        target=fragment['target'], valid=fragment['target_valid'],
        motion=dict(states=fragment['native_states'], support=fragment['support'],
            nearest_seconds=fragment['nearest_seconds'], query_seconds=fragment['query_seconds']),
        fusion_targets={identity: dict(states=base['aligned_truth'][i], valid=base['aligned_truth_valid'][i])
            for i, identity in enumerate(fragment['target_identity_ids'])})
    reports, artifacts = evaluate_unique_population(scene, {'learned_assoc': groups}, {'learned_assoc': predictions})
    artifacts = {key.removeprefix('learned_assoc_'): value for key, value in artifacts.items()}
    row, arrays = _scene(fragment, artifacts, reports, np.zeros((4, 64, 2)), '3035', 'shared')
    payload = {key: fragment[key] for key in compare.INPUT_KEYS if key != 'group_anchor_node_indices'}
    payload.update(group_anchor_node_indices=np.array([0, 2, 4, 5], np.int64), **arrays)
    payload.update({method+'_group_states': item['group_states'] for method, item in artifacts.items()})
    path = (root/name).resolve()
    for directory, values in (('predictions', payload), ('system-predictions', {
            method+'_'+key: value for method, item in artifacts.items() for key, value in item.items()})):
        output = path/'final'/directory
        output.mkdir(parents=True)
        np.savez_compressed(output/'states-3035.npz', **values)
    config = read_json(compare.ROOT/'configs'/compare.CONFIG_NAMES[(20261023, lr)])
    write_json(path/'run.json', dict(source_system=str(root/'source056/system')))
    return dict(path=path, configuration=config, config_name=compare.CONFIG_NAMES[(20261023, lr)],
        method='learned_sequence', method_name='090C', source_system=str(root/'source056/system'),
        scene_ids=['3035'], components={'3035': 'shared'}, paired_control=root/'reference',
        summary={'methods': row['methods']}, initial_state={'p': torch.tensor([1.]), 'v': torch.tensor([2.])},
        initial_rng={'state': 'same'}, pairing={'source_fit': str(root/'reference/fit'),
            'exposure': {'successful_updates': 8000}})


class DeepAFPeerComparisonTests(unittest.TestCase):
    def test_actual_coordinates_recompute_known_mae_and_complete_population_without_forward(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            ours = cached_arm(root, 'ours', latitude_offset=.0001)
            peer = cached_arm(root, 'peer', latitude_offset=.0002)
            with patch('association.models.deepaf_peer.DeepAFPeer.forward_batch',
                       side_effect=AssertionError('cached comparison cannot forward')):
                summary, scenes = compare.cached_comparison(dict(ours=ours, deepaf=peer))
            expected = 6371000.*np.deg2rad(.0001)
            metrics = summary['position_weightings']['all']
            self.assertAlmostEqual(metrics['ours']['point_equal']['mae'], expected, places=7)
            self.assertAlmostEqual(metrics['deepaf']['point_equal']['mae'], 2*expected, places=7)
            self.assertAlmostEqual(metrics['ours']['point_equal']['p99'], expected, places=7)
            self.assertEqual(metrics['ours']['point_equal']['unit_count'], 127)
            self.assertEqual(summary['population']['predicted_group_count'], 4)
            self.assertEqual(summary['population']['duplicate_prediction_group_count'], 1)
            decision = compare.numerical_comparison(summary, scenes)
            self.assertAlmostEqual(decision['relative_mae_reduction'], .5, places=8)
            self.assertFalse(decision['peer_quality_certified'])
            self.assertFalse(decision['stable_peer_superiority_established'])

    def test_changed_mask_truth_group_input_or_cached_error_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            arms = dict(ours=cached_arm(root, 'ours'), deepaf=cached_arm(root, 'peer', latitude_offset=.0002))
            cases = (
                ('system-predictions', 'learned_sequence_aligned_truth_valid', lambda value: value.__setitem__((0, 0, 0), False)),
                ('system-predictions', 'learned_sequence_aligned_truth', lambda value: value.__setitem__((0, 0, 0), 31.)),
                ('system-predictions', 'learned_sequence_aligned_group_indices', lambda value: value.__setitem__(0, 1)),
                ('predictions', 'support', lambda value: value.__setitem__((0, 0), False)),
                ('predictions', 'learned_assoc_group_node_indices', lambda value: value.__setitem__(0, 1)),
                ('predictions', 'learned_sequence_position_error', lambda value: value.__setitem__((0, 0), 0.)),
            )
            for directory, key, mutate in cases:
                with self.subTest(key=key):
                    path = root/'peer/final'/directory/'states-3035.npz'
                    original = path.read_bytes()
                    with np.load(path, allow_pickle=False) as saved:
                        arrays = {name: saved[name].copy() for name in saved.files}
                    mutate(arrays[key]); np.savez_compressed(path, **arrays)
                    with self.assertRaises(ValueError):
                        compare.scene_pair(arms, '3035')
                    path.write_bytes(original)

    def test_learning_rate_chooses_stronger_peer_from_cache_and_fixed_tie_ignores_ours(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = cached_arm(root, 'a', latitude_offset=.0003, lr=.001)
            b = cached_arm(root, 'b', latitude_offset=.0001, lr=.0003)
            arms = {a['path']: a, b['path']: b}
            with patch.object(compare, 'peer_metadata', side_effect=lambda path: arms[Path(path)]):
                result = compare.select_learning_rate(a['path'], b['path'])
                self.assertEqual(result['selected_learning_rate'], .0003)
                self.assertEqual(result['selected']['peer_path'], str(b['path']))
                self.assertEqual(result['selected']['step'], 8000)
                self.assertEqual(result['model_forwards'], 0)
                self.assertFalse(result['peer_quality_certified'])
                for directory in ('predictions', 'system-predictions'):
                    source = a['path']/'final'/directory/'states-3035.npz'
                    destination = b['path']/'final'/directory/'states-3035.npz'
                    destination.write_bytes(source.read_bytes())
                b['summary'] = deepcopy(a['summary'])
                result = compare.select_learning_rate(a['path'], b['path'])
                self.assertEqual(result['selected_learning_rate'], .001)
                b['initial_state']['p'] += 1
                with self.assertRaisesRegex(ValueError, 'initialization/sampling'):
                    compare.select_learning_rate(a['path'], b['path'])

    def test_chart_excursions_are_recomputed_and_prevent_automatic_quality_claim(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            ours = cached_arm(root, 'ours')
            peer = cached_arm(root, 'peer', latitude_offset=.0002)
            peer['published_method'] = 'DeepAF'
            peer['summary']['chart_excursion_count'] = 1
            path = peer['path']/'final/predictions/states-3035.npz'
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            positions = np.zeros((4, 64, 2), dtype=np.float64)
            positions[0, 0, 0] = 1.01*np.pi*6371000.
            arrays.update(group_deepaf_positions_m=positions,
                group_deepaf_chart_excursion_count=np.array([1, 0, 0, 0], np.int64))
            np.savez_compressed(path, **arrays)
            summary, scenes = compare.cached_comparison(dict(ours=ours, deepaf=peer))
            self.assertEqual(summary['chart_excursions']['deepaf'], 1)
            self.assertEqual(compare.numerical_comparison(summary, scenes)['scientific_decision'],
                'UNRESOLVED_PEER_GEOMETRY')
            arrays['group_deepaf_chart_excursion_count'][0] = 0
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'finite-chart excursions'):
                compare.scene_pair(dict(ours=ours, deepaf=peer), '3035')

    def test_cli_writes_reviewable_selection_and_preserves_existing_results(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = cached_arm(root, 'a', latitude_offset=.0002, lr=.001)
            b = cached_arm(root, 'b', latitude_offset=.0001, lr=.0003)
            arms = {a['path']: a, b['path']: b}
            args = Namespace(select_lr=True, peer_a=str(a['path']), peer_b=str(b['path']),
                ours_run=None, peer_run=None, output=str(root/'selection'), cpu_threads=4)
            with patch.object(compare, 'peer_metadata', side_effect=lambda path: arms[Path(path)]):
                compare.run(args)
                saved = read_json(root/'selection/selection.json')
                self.assertEqual(saved['selected_learning_rate'], .0003)
                self.assertEqual(read_json(root/'selection/run.json')['status'], 'COMPLETED')
                with self.assertRaises(FileExistsError):
                    compare.run(args)
                self.assertEqual(read_json(root/'selection/selection.json'), saved)
                args.select_lr, args.ours_run, args.peer_run = False, str(a['path']), str(b['path'])
                args.peer_a = args.peer_b = None
                args.output = str(root/'paired')
                with patch.object(compare, 'ours_metadata', return_value=a):
                    result = compare.run(args)
                    self.assertFalse(result['stable_peer_superiority_established'])
                    self.assertEqual(result['seed'], 20261023)
                    b['configuration']['training']['seed'] = 20261021
                    args.output = str(root/'wrong-seed')
                    with self.assertRaisesRegex(ValueError, 'same declared seed'):
                        compare.run(args)
                    self.assertEqual(read_json(root/'wrong-seed/run.json')['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
