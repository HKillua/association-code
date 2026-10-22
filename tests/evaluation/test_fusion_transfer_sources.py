"""062 actual endpoint bindings and complete OLS replay, including duplicates."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from association.evaluation.scene import evaluate_full_scene
from association.models.evidence_association import EvidenceAssociation, FEATURE_NAMES
from association.models.evidence_fusion import EvidenceFusion
from association.training.evidence_association import _save_checkpoint
from association.training.evidence_fusion import bind_ols_working_point, save_checkpoint
from association.utils.runtime import write_json
from tools.analysis.fusion_transfer_sources import (
    CONFIG_ROOT, _bind_association_checkpoint, _bind_fusion_arm, replay_fusion_source,
)
from association.utils.runtime import read_json_or_gzip as read
from tests.evaluation import test_learned_group_oracle as oracle_fixture


class FusionTransferSourceTests(unittest.TestCase):
    @staticmethod
    def replay_fixture():
        fragment, saved, _ = oracle_fixture.LearnedGroupOracleTests.fixture()
        offsets, flat = fragment['ols_group_offsets'], fragment['ols_group_node_indices']
        groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
        learned = fragment['native_states'].copy()
        own = np.ones((*learned.shape[:2], 3))
        anchors, helpers = [], []
        for group in groups:
            for anchor in group:
                for helper in group:
                    if anchor != helper:
                        anchors.append(anchor)
                        helpers.append(helper)
                        learned[anchor] = .75 * fragment['native_states'][anchor] + .25 * fragment['native_states'][helper]
                        own[anchor] = .75
        routing = dict(helper_anchor=np.asarray(anchors), helper_node=np.asarray(helpers),
            membership=fragment['ols_membership'].copy(), anchor_weights=own,
            helper_weights=np.full((len(anchors), 64, 3), .25),
            utility=np.full((len(anchors), 64, 3), 1. / 3.))
        old = {key: value.copy() for key, value in fragment.items() if not key.startswith('learned_assoc_')}
        old.update(ols_learned=learned)
        old.update({'ols_routing_' + key: value.copy() for key, value in routing.items()})
        old_saved = {key: deepcopy(value) for key, value in saved.items() if key.startswith('ols_')}
        targets = fragment['target_identity_ids']
        report = evaluate_full_scene(
            [dict(node_indices=group, states=learned[group[0]]) for group in groups],
            node_identities=fragment['identities'], target_identities=targets,
            target_states=saved['ols_anchor_aligned_truth'][:len(targets)],
            target_valid=saved['ols_anchor_aligned_truth_valid'][:len(targets)],
            node_observed=fragment['support'], split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        old_saved.update({'ols_learned_' + key: value for key, value in report.pop('artifact').items()})
        predictions = dict(mean_supported=fragment['ols_mean_supported'].copy(), learned=learned.copy())
        return fragment, saved, old, old_saved, predictions, routing, report['full_target_metrics']

    def test_full_replay_and_small_cross_device_change(self):
        *args, expected = self.replay_fixture()
        report = replay_fusion_source(*args, expected=expected)
        self.assertTrue(report['passed'])
        self.assertTrue(report['complete_old_learned_artifact_exact'])
        args[4]['learned'][0, 0, 0] += 1e-9
        self.assertTrue(replay_fusion_source(*args, expected=expected)['passed'])

    def test_utility_gate_accepts_cross_device_rounding_at_array_scale(self):
        # 064 measured this on Level1 scene 1 (direct_only): one element sat
        # 1.63e-06 from the saved GPU value against a 1.35e-06 fixed bound, while
        # the accept set was identical and the array scale was 152.8. The gate must
        # follow the array's own float32 scale, not a fixed absolute floor.
        *args, expected = self.replay_fixture()
        args[5]['utility'][0, 0, 0] = 150.
        args[2]['ols_routing_utility'][0, 0, 0] = 150.
        previous_small = float(args[2]['ols_routing_utility'][1, 0, 0])
        args[5]['utility'][1, 0, 0] += 1.6e-5
        report = replay_fusion_source(*args, expected=expected)
        self.assertTrue(report['passed'])
        self.assertTrue(report['checks']['utility_decision_equal'])
        self.assertTrue(report['checks']['utility_scale_close'])
        self.assertEqual(report['utility_sign_flip_count'], 0)
        self.assertEqual(report['utility_numeric_exceed_count'], 0)
        self.assertEqual(report['utility_accepted_count'][0], report['utility_accepted_count'][1])
        self.assertAlmostEqual(report['utility_scale'], 150.)
        self.assertAlmostEqual(report['utility_atol_ulp_floor'],
                               8 * float(np.spacing(np.float32(150.))))
        # The perturbation still exceeds the previous fixed floor, which is the point.
        self.assertGreater(1.6e-5, 1e-6 + 1e-5 * abs(previous_small))
        self.assertGreater(report['max_utility_difference'], 1e-5)

    def test_rejects_utility_decision_change_corruption_and_weight_drift(self):
        for case in ('sign_flip', 'utility_corrupted', 'helper_weights'):
            with self.subTest(case=case):
                *args, expected = self.replay_fixture()
                args[5]['utility'][0, 0, 0] = 150.
                args[2]['ols_routing_utility'][0, 0, 0] = 150.
                if case == 'sign_flip':
                    args[5]['utility'][1, 0, 0] = -abs(args[5]['utility'][1, 0, 0])
                elif case == 'utility_corrupted':
                    args[2]['ols_routing_utility'][1, 0, 0] += 1.
                else:
                    args[5]['helper_weights'][0, 0, 0] += 1e-4
                with self.assertRaises(ValueError):
                    replay_fusion_source(*args, expected=expected)

    def test_rejects_unretained_output_mask_input_and_routing_changes(self):
        for case in ('duplicate', 'mask', 'node_order', 'group', 'weights', 'state'):
            with self.subTest(case=case):
                *args, expected = self.replay_fixture()
                if case == 'duplicate':
                    # Group2 repeats an already-covered identity; unique-target
                    # metrics alone would not detect its altered state.
                    args[3]['ols_learned_group_states'][2, 0, 0] += .1
                elif case == 'mask':
                    args[3]['ols_learned_aligned_prediction_valid'][0, 0, 0] = False
                elif case == 'node_order':
                    args[2]['node_ids'][[0, 1]] = args[2]['node_ids'][[1, 0]]
                elif case == 'group':
                    args[2]['ols_group_node_indices'][[0, 1]] = [1, 0]
                elif case == 'weights':
                    args[5]['helper_weights'][0, 0, 0] += .1
                else:
                    args[4]['learned'][0, 0, 0] += 1.
                with self.assertRaises(ValueError):
                    replay_fusion_source(*args, expected=expected)

    @staticmethod
    def binding_fixture(root):
        manifest = dict(dataset_level='Level1', split_unit='scene',
                        splits=dict(T_core=['t'], V_select=['01'], T_test=['sealed']))
        wp = dict(scope='ols-working-point-v1', synthetic=False, dataset_level='Level1',
            motion_observation_policy='canonical_all', data=dict(length=64, max_observations=128, time_unit_seconds=20.),
            training_scene_ids=['t'], validation_scene_ids=['01'], threshold=-1.,
            score_semantics='-log1p(rms_haversine_m/1000)', selection_split='V_select')
        base = dict(ols_working_point=wp, fragment_count=6)
        config = bind_ols_working_point(read(CONFIG_ROOT/'evidence-fusion-level1.json'), manifest, wp)
        fit, system = root/'helper-quality'/'fit', root/'helper-quality'/'system'
        fit.mkdir(parents=True)
        for directory in ('predictions', 'system-predictions'):
            (system/directory).mkdir(parents=True)
            np.savez(system/directory/'states-01.npz', fixture=np.ones(1))
        model = EvidenceFusion(**config['model'])
        save_checkpoint(fit/'final.pt', model, torch.optim.Adam(model.parameters()), 1000,
                        config, np.random.default_rng(0))
        write_json(fit/'run.json', dict(status='COMPLETED', optimizer_updates=1000,
            configuration=config, training_scene_ids=['t'], validation_scene_ids=['01'],
            final_evaluation='evaluation-step-1000.json', test_opened=False, split_unit='scene'))
        methods = ['ols_anchor', 'ols_attribute_hybrid', 'ols_learned', 'ols_mean_supported']
        common = dict(status='COMPLETED', dataset_level='Level1', split='V_select', test_opened=False,
            optimizer_updates=0, association_endpoint=None,
            ordinary_controls=['anchor', 'mean_supported', 'attribute_hybrid'], method_names=methods,
            fusion_checkpoint_step=1000,
            inputs=dict(fusion_checkpoint=str(fit/'final.pt'), fusion_run=str(fit/'run.json')))
        write_json(system/'run.json', dict(common, motion_observation_policy='canonical_all',
            threshold_reselected=False, normalizer_refitted=False, completed_scenes=1, total_scenes=1))
        write_json(system/'summary.json', dict(common, scene_ids=['01'], scene_count=1, fragment_count=6,
            ols_working_point=wp, per_scene=[dict(scene_id='01', component_id='c1',
                unique_target={method: dict(full_target_metrics={}) for method in methods})]))
        return manifest, base, config

    def test_real_fusion_checkpoint_is_bound_to_arm_and_endpoint(self):
        for failure in (None, 'step', 'objective', 'source_path', 'missing_scene'):
            with self.subTest(failure=failure), TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, base, config = self.binding_fixture(root)
                path = root/'helper-quality'/'fit'/'final.pt'
                if failure in ('step', 'objective'):
                    cfg = deepcopy(config)
                    if failure == 'objective':
                        cfg['training']['utility_objective'] = 'direct_only'
                    model = EvidenceFusion(**cfg['model'])
                    save_checkpoint(path, model, torch.optim.Adam(model.parameters()),
                                    0 if failure == 'step' else 1000, cfg, np.random.default_rng(0))
                elif failure == 'source_path':
                    for name in ('run', 'summary'):
                        p = root/'helper-quality'/'system'/f'{name}.json'
                        record = read(p)
                        record['inputs']['fusion_checkpoint'] = str(root/'another-final.pt')
                        write_json(p, record)
                elif failure == 'missing_scene':
                    (root/'helper-quality'/'system'/'predictions'/'states-01.npz').unlink()
                if failure:
                    with self.assertRaises(ValueError):
                        _bind_fusion_arm(root, 'helper-quality', manifest, base, 'cpu')
                else:
                    model, metadata = _bind_fusion_arm(root, 'helper-quality', manifest, base, 'cpu')
                    self.assertEqual(metadata['step'], 1000)
                    self.assertEqual(metadata['objective'], 'helper_quality')
                    self.assertFalse(model.training)
                    self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_association_checkpoint_normalizer_and_full_pair_population(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, training = root/'system', root/'association'
            (source/'predictions').mkdir(parents=True)
            training.mkdir()
            cfg = read(CONFIG_ROOT/'evidence-association-level1.json')
            normalizer = dict(feature_names=list(FEATURE_NAMES), fitted_split='T_core',
                              mean=[0.] * len(FEATURE_NAMES), std=[1.] * len(FEATURE_NAMES))
            model = EvidenceAssociation(cfg['variant'])
            selection = dict(threshold=-1.)
            _save_checkpoint(training/'final.pt', model, torch.optim.Adam(model.parameters()),
                             cfg, normalizer, 1000, selection)
            metrics = dict(tp=1, fp=0, fn=0, tn=2, macro_f1=1.)
            population = dict(scenes=1, nodes=3, pairs=3)
            summary = dict(status='COMPLETED', optimizer_updates=1000, scope=cfg['scope'], data=cfg['data'],
                variant=cfg['variant'], selected={}, final_selection=selection,
                final_result=dict(metrics=metrics), population=population,
                motion_observation_policy='canonical_all', test_opened=False, reference_read=False)
            audit = dict(normalizer=normalizer, motion_observation_policy='canonical_all', data=cfg['data'],
                training_scene_ids=['t'], validation_scene_ids=['01'], split_unit='scene',
                test_opened=False, reference_read=False, validation_population=population)
            write_json(training/'summary.json', summary)
            write_json(training/'data_audit.json', audit)
            nodes, sources = np.array(['n0', 'n1', 'n2']), np.array(['a', 'a', 'b'])
            np.savez(source/'predictions'/'states-01.npz', node_ids=nodes, sources=sources)
            np.savez(training/'final-scores.npz', scene_ids=np.array(['01']),
                **{'01_node_ids': nodes, '01_sources': sources,
                   '01_pairs': np.column_stack(np.triu_indices(3, 1)), '01_scores': np.zeros(3)})
            manifest = dict(splits=dict(T_core=['t'], V_select=['01']))
            base = dict(scene_ids=['01'], fragment_count=3, association_threshold=-1., association=metrics)
            record = dict(config=cfg, selected={})
            checked = _bind_association_checkpoint(source, manifest, base, record)
            self.assertEqual(checked['pairs'], 3)
            audit['normalizer']['mean'][0] = 1.
            write_json(training/'data_audit.json', audit)
            with self.assertRaisesRegex(ValueError, 'normalizer'):
                _bind_association_checkpoint(source, manifest, base, record)


if __name__ == '__main__':
    unittest.main()
