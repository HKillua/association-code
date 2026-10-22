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
        self.assertEqual(report['utility_replay_domain'], 'raw')
        self.assertTrue(report['checks']['utility_close'])
        self.assertTrue(report['complete_old_learned_artifact_exact'])
        args[4]['learned'][0, 0, 0] += 1e-9
        self.assertTrue(replay_fusion_source(*args, expected=expected)['passed'])

    def test_relu_contract_preserves_raw_failure_for_inactive_utility(self):
        for current, previous in ((-0.035231105983257294, -0.03522947430610657),
                                  (-100., -0.01)):
            with self.subTest(current=current, previous=previous):
                *args, expected = self.replay_fixture()
                # ReLU discards the magnitude in both cases. The second case
                # explicitly documents inference equivalence, not head replay.
                # This isolates the utility gate; unchanged fixture weights are
                # not recomputed. The entry test covers real zero-weight inference.
                args[5]['utility'][0, 0, 0] = current
                args[2]['ols_routing_utility'][0, 0, 0] = previous
                with self.assertRaisesRegex(ValueError, 'replay tolerance'):
                    replay_fusion_source(*args, expected=expected)
                report = replay_fusion_source(*args, expected=expected, utility_replay_domain='relu')
                self.assertTrue(report['passed'])
                self.assertEqual(report['utility_replay_domain'], 'relu')
                self.assertNotIn('utility_close', report['checks'])
                self.assertTrue(report['checks']['utility_activation_equal'])
                self.assertTrue(report['checks']['utility_relu_close'])
                self.assertFalse(report['utility_diagnostics']['raw']['passed'])
                self.assertEqual(report['utility_diagnostics']['raw']['exceed_count'], 1)
                self.assertTrue(report['utility_diagnostics']['relu']['passed'])
                self.assertEqual(report['utility_diagnostics']['activation_flip_count'], 0)

    def test_relu_contract_rejects_activation_changes_and_positive_drift(self):
        for previous, current in ((-1e-8, 1e-8), (0., 1e-8), (1e-8, 0.),
                                  (1. / 3., 1. / 3. + 1.6e-5)):
            with self.subTest(previous=previous, current=current):
                *args, expected = self.replay_fixture()
                # An unrelated large value must not enlarge the scalar bound.
                args[5]['utility'][1, 0, 0] = 150.
                args[2]['ols_routing_utility'][1, 0, 0] = 150.
                args[5]['utility'][0, 0, 0] = current
                args[2]['ols_routing_utility'][0, 0, 0] = previous
                with self.assertRaisesRegex(ValueError, 'replay tolerance'):
                    replay_fusion_source(*args, expected=expected, utility_replay_domain='relu')

    def test_relu_contract_rejects_nonfinite_and_shape_mismatch(self):
        for side in ('current', 'previous'):
            for bad in (np.nan, np.inf, -np.inf, 'shape'):
                with self.subTest(side=side, bad=bad):
                    *args, expected = self.replay_fixture()
                    container, key = ((args[5], 'utility') if side == 'current'
                                      else (args[2], 'ols_routing_utility'))
                    if bad == 'shape':
                        container[key] = container[key][:-1]
                    else:
                        container[key][0, 0, 0] = bad
                    with self.assertRaisesRegex(ValueError, 'replay tolerance'):
                        replay_fusion_source(*args, expected=expected, utility_replay_domain='relu')

    def test_relu_contract_keeps_weight_and_physical_gates(self):
        for case in ('helper_weights', 'anchor_weights', 'physical_state'):
            with self.subTest(case=case):
                *args, expected = self.replay_fixture()
                if case == 'physical_state':
                    args[4]['learned'][0, 0, 0] += 1.
                else:
                    args[5][case][0, 0, 0] += 1e-4
                with self.assertRaisesRegex(ValueError, 'replay tolerance'):
                    replay_fusion_source(*args, expected=expected, utility_replay_domain='relu')

    def test_output_contract_accepts_internal_drift_and_sign_flip_with_real_weights(self):
        for previous, current in ((.12807676196098328, .12807908654212952), (-1e-8, 1e-8)):
            with self.subTest(previous=previous, current=current):
                *args, _ = self.replay_fixture()
                fragment, saved, old, old_saved, predictions, routing = args
                # Recompute actual normalized weights and speed predictions on
                # both sides; utility drift is not paired with invented routes.
                anchor, helper = routing['helper_anchor'][0], routing['helper_node'][0]
                for value, utility, weights, own, states in (
                        (previous, old['ols_routing_utility'], old['ols_routing_helper_weights'],
                         old['ols_routing_anchor_weights'], old['ols_learned']),
                        (current, routing['utility'], routing['helper_weights'],
                         routing['anchor_weights'], predictions['learned'])):
                    weight = max(value, 0.) / (1. + max(value, 0.))
                    utility[0, 0, 1] = value
                    weights[0, 0, 1], own[anchor, 0, 1] = weight, 1. - weight
                    a, h = fragment['native_states'][[anchor, helper], 0, 2]
                    states[anchor, 0, 2] = a + weight * (h - a)
                offsets, flat = fragment['ols_group_offsets'], fragment['ols_group_node_indices']
                groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
                targets = fragment['target_identity_ids']
                original = evaluate_full_scene(
                    [dict(node_indices=group, states=old['ols_learned'][group[0]]) for group in groups],
                    node_identities=fragment['identities'], target_identities=targets,
                    target_states=saved['ols_anchor_aligned_truth'][:len(targets)],
                    target_valid=saved['ols_anchor_aligned_truth_valid'][:len(targets)],
                    node_observed=fragment['support'], split='V_select', grouping_kind='predicted',
                    correspondence_policy='anchor_identity_first', return_artifact=True,
                    node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
                old_saved.update({'ols_learned_' + key: value for key, value in original.pop('artifact').items()})
                with self.assertRaisesRegex(ValueError, 'replay tolerance'):
                    replay_fusion_source(*args, utility_replay_domain='relu')
                result = replay_fusion_source(*args, expected=original['full_target_metrics'],
                                              utility_replay_domain='output')
                self.assertTrue(result['passed'])
                self.assertTrue(result['complete_old_learned_artifact_exact'])
                self.assertFalse(any(key.startswith('utility') for key in result['checks']))
                if previous > 0:
                    self.assertFalse(result['utility_diagnostics']['relu']['passed'])
                else:
                    self.assertEqual(result['utility_diagnostics']['activation_flip_count'], 1)

    def test_output_contract_collects_numerical_failures_but_still_checks_old_artifact(self):
        for case in ('weights', 'physical'):
            with self.subTest(case=case):
                *args, expected = self.replay_fixture()
                if case == 'weights':
                    args[5]['helper_weights'][0, 0, 0] += 1e-4
                    anchor = args[5]['helper_anchor'][0]
                    args[5]['anchor_weights'][anchor, 0, 0] -= 1e-4
                else:
                    args[4]['learned'][0, 0, 0] += 1.
                result = replay_fusion_source(*args, expected=expected, utility_replay_domain='output')
                self.assertFalse(result['passed'])
                self.assertFalse(result['checks']['helper_weights_close' if case == 'weights' else 'physical_close'])
                self.assertTrue(result['complete_old_learned_artifact_exact'])
                # A numerical mismatch never skips the old complete-output audit.
                args[3]['ols_learned_group_states'][2, 0, 0] += .1
                with self.assertRaisesRegex(ValueError, '057 complete learned system'):
                    replay_fusion_source(*args, expected=expected, utility_replay_domain='output')

    def test_output_contract_rejects_malformed_nonfinite_and_invalid_weights(self):
        for side in ('current', 'previous'):
            for name in ('learned', 'mean_supported', 'utility', 'helper_weights', 'anchor_weights'):
                for bad in ('shape', np.nan, np.inf, -np.inf):
                    with self.subTest(side=side, name=name, bad=bad):
                        *args, expected = self.replay_fixture()
                        prediction = name in ('learned', 'mean_supported')
                        container, key = ((args[4] if prediction else args[5], name)
                            if side == 'current' else (args[2], ('ols_' if prediction else 'ols_routing_') + name))
                        if bad == 'shape':
                            container[key] = container[key][:-1]
                        else:
                            container[key][0, 0, 0] = bad
                        with self.assertRaises(ValueError):
                            replay_fusion_source(*args, expected=expected, utility_replay_domain='output')
            for value in (-1e-12, 1.000000000001, .8):
                with self.subTest(side=side, invalid_weight=value):
                    *args, expected = self.replay_fixture()
                    container, key = ((args[5], 'helper_weights') if side == 'current'
                                      else (args[2], 'ols_routing_helper_weights'))
                    container[key][0, 0, 0] = value
                    with self.assertRaisesRegex(ValueError, 'weights must'):
                        replay_fusion_source(*args, expected=expected, utility_replay_domain='output')
        *args, expected = self.replay_fixture()
        args[4]['mean_supported'][0, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'structural/ordinary'):
            replay_fusion_source(*args, expected=expected, utility_replay_domain='output')

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
