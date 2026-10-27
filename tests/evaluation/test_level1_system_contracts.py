"""Contract tests for Level1 frozen comparison and complete population."""
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from association.data.motion_evidence import build_motion_evidence
from association.evaluation.association import evaluate_scenes
from association.evaluation.fusion_controls import ordinary_fusion
from association.models.evidence_association import EvidenceAssociation
from association.models.evidence_fusion import EvidenceFusion
from association.training.evidence_fusion import bind_ols_working_point, save_checkpoint
from association.training.evidence_association import (
    DATA as ASSOCIATION_DATA, prepare_features, fit_normalizer, score_scenes,
    _save_checkpoint, _save_scores, fit_evidence_association,
)
from association.utils.runtime import read_json, write_json

from tools.experiments.level1_system_evaluate import (
    DATA, POLICY, run,
    _pool_fragment, exact_score_population, verify_fusion_snapshot,
    evaluate_unique_population, common_target_metrics,
    ordinary_predictions, validate_association_endpoint,
)


class Level1SystemContractTests(unittest.TestCase):
    @staticmethod
    def association_fixture(root, scene, wp):
        config = dict(scope='native-ols-evidence-association-v1', variant='full_evidence',
            data=ASSOCIATION_DATA, seed=0, motion_observation_policy=POLICY,
            training=dict(updates=1, batch_size=2, validate_every=1, inference_batch_size=16,
                          learning_rate=.001, gradient_clip=1., advisory_hours=1.))
        features = prepare_features([scene], 'V_select')
        normalizer = fit_normalizer([{**features[0], 'split': 'T_core'}])
        model = EvidenceAssociation()
        model.motion_observation_policy = POLICY
        baseline_records = score_scenes(model, features, normalizer, 'cpu', 16)
        base_selection = dict(threshold=wp['threshold'])
        baseline = dict(motion_observation_policy=POLICY, selection=base_selection,
            selected_threshold=dict(metrics=evaluate_scenes(baseline_records, wp['threshold'])))
        with torch.no_grad():
            model.network[-1].bias.fill_(1.)
        records = score_scenes(model, features, normalizer, 'cpu', 16)
        selection = dict(threshold=-10.)
        result = dict(metrics=evaluate_scenes(records, selection['threshold']))
        population = dict(scenes=1, components=1, nodes=len(scene['node_ids']),
                          pairs=len(scene['pairs']), positives=int(scene['labels'].sum()))
        summary = dict(status='COMPLETED', scope=config['scope'], variant=config['variant'],
            motion_observation_policy=POLICY, optimizer_updates=1, data=ASSOCIATION_DATA,
            selected=dict(update=1, selection=selection), selected_result=result,
            final_selection=selection, final_result=result, population=population,
            reference_read=False, test_opened=False)
        audit = dict(motion_observation_policy=POLICY, split_unit='scene', data=ASSOCIATION_DATA,
            training_scene_ids=['t'], validation_scene_ids=['v'], normalizer=normalizer,
            validation_population=population, reference_read=False, test_opened=False)
        run_record = dict(status='COMPLETED', optimizer_updates=1, config=config,
            selected=summary['selected'], reference_read=False, test_opened=False)
        for filename, value in (('baseline', baseline), ('association-summary', summary),
                                ('association-audit', audit), ('association-run', run_record)):
            write_json(root/f'{filename}.json', value)
        _save_scores(root/'baseline-scores.npz', baseline_records)
        _save_scores(root/'scores.npz', records)
        _save_checkpoint(root/'association-final.pt', model, torch.optim.Adam(model.parameters()),
                         config, normalizer, 1, selection)
        scores = {r['scene_id']: r for r in records}
        kwargs = dict(endpoint='final', working_point=wp, training_run=run_record,
            baseline_scores={r['scene_id']: r for r in baseline_records}, baseline=baseline,
            checkpoint_path=root/'association-final.pt', device='cpu')
        return summary, audit, scores, kwargs

    @staticmethod
    def scene(q=64):
        """Three visible fragments, including a duplicate identity, plus a missing target."""
        query = np.linspace(0., 1260., q)
        states = np.tile([30., 120., 6., 90.], (3, 2, 1))
        states[1, :, 0] = 30.001
        states[2, :, 0] = 31.
        sources = np.array(['9001', '9002', '9001'])
        motion = build_motion_evidence(states, np.tile([0., 1260.], (3, 1)),
            np.ones((3, 2), bool), query, sources)
        identities = np.array(['ship-a', 'ship-a', 'ship-b'])
        truth = np.tile([30., 120., 6., 90.], (3, q, 1))
        truth[1, :, 0], truth[2, :, 0] = 31., 32.
        valid = np.ones_like(truth, bool)
        pairs = np.column_stack(np.triu_indices(3, 1))
        return dict(scene_id='v', component_id='v', split='V_select', split_unit='scene',
            grid=query, node_ids=np.array(['a0', 'a1', 'b0']), identities=identities,
            sources=sources, pairs=pairs, labels=identities[pairs[:, 0]] == identities[pairs[:, 1]],
            motion=motion, motion_observation_policy=POLICY,
            target_identity_ids=['ship-a', 'ship-b', 'ship-c'],
            target=truth[[0, 0, 1]], valid=valid[[0, 0, 1]],
            fusion_targets={identity: dict(states=truth[i], valid=valid[i])
                for i, identity in enumerate(['ship-a', 'ship-b', 'ship-c'])})

    def test_exact_scene_and_pair_order(self):
        scene = dict(scene_id='a', node_ids=np.array(['x', 'y']),
                     sources=np.array(['9001', '9002']), pairs=np.array([[0, 1]]))
        good = {'a': dict(node_ids=scene['node_ids'].copy(), sources=scene['sources'].copy(),
                          pairs=scene['pairs'].copy(), scores=np.array([-.5]))}
        exact_score_population([scene], good)
        with self.assertRaises(ValueError):
            exact_score_population([scene], {})
        wrong = {'a': {**good['a'], 'node_ids': np.array(['y', 'x'])}}
        with self.assertRaises(ValueError):
            exact_score_population([scene], wrong)

    def test_endpoint_rejects_mismatched_weights_steps_threshold_scores_and_population(self):
        scene = self.scene()
        manifest = dict(split_unit='scene', splits=dict(T_core=['t'], V_select=['v']))
        with TemporaryDirectory() as temp:
            summary, audit, scores, kwargs = self.association_fixture(Path(temp), scene, dict(threshold=-1.))
            answer = validate_association_endpoint(summary, audit, manifest, scores, [scene], **kwargs)
            self.assertEqual(answer['update'], 1)
            self.assertEqual(answer['score_replay']['max_absolute_difference'], 0.)
            baseline_kwargs = {**kwargs, 'endpoint': 'baseline', 'checkpoint_path': None}
            baseline = validate_association_endpoint(summary, audit, manifest,
                kwargs['baseline_scores'], [scene], **baseline_kwargs)
            self.assertEqual(baseline['update'], 0)
            selected = validate_association_endpoint(summary, audit, manifest, scores, [scene],
                                                      **{**kwargs, 'endpoint': 'selected'})
            self.assertEqual(selected['update'], 1)
            mutations = [
                ('step', lambda s, a, c, k: s.update(optimizer_updates=2)),
                ('run', lambda s, a, c, k: k['training_run'].update(status='FAILED')),
                ('threshold', lambda s, a, c, k: s['final_selection'].update(threshold=-9.)),
                ('normalizer', lambda s, a, c, k: a['normalizer']['mean'].__setitem__(0, 1.)),
                ('partial V', lambda s, a, c, k: c.clear()),
                ('population', lambda s, a, c, k: s['population'].update(nodes=2)),
                ('score shape', lambda s, a, c, k: c['v'].update(scores=np.zeros((3, 1)))),
                ('score values', lambda s, a, c, k: c['v']['scores'].__setitem__(0, 2.)),
                ('step zero cache', lambda s, a, c, k: k['baseline_scores']['v']['scores'].__setitem__(0, 2.)),
                ('step zero threshold', lambda s, a, c, k: k['baseline']['selection'].update(threshold=-2.)),
            ]
            for label, mutate in mutations:
                with self.subTest(label=label):
                    s, a, c, k = deepcopy((summary, audit, scores, kwargs))
                    mutate(s, a, c, k)
                    with self.assertRaises(ValueError):
                        validate_association_endpoint(s, a, manifest, c, [scene], **k)
            snapshot = torch.load(kwargs['checkpoint_path'], weights_only=False)
            wrong_step = {**snapshot, 'update': 0}
            torch.save(wrong_step, kwargs['checkpoint_path'])
            with self.assertRaisesRegex(ValueError, 'checkpoint/config/update'):
                validate_association_endpoint(summary, audit, manifest, scores, [scene], **kwargs)
            # Even an allowed tiny CPU/GPU score difference cannot change an edge.
            s, c = deepcopy((summary, scores))
            near_threshold = float(c['v']['scores'][0]+1e-7)
            snapshot['selection'] = {'threshold': near_threshold}
            s['final_selection'] = snapshot['selection']
            c['v']['scores'][0] += 2e-7
            torch.save(snapshot, kwargs['checkpoint_path'])
            with self.assertRaisesRegex(ValueError, 'threshold decisions'):
                validate_association_endpoint(s, audit, manifest, c, [scene], **kwargs)
            snapshot['selection'] = kwargs['training_run']['selected']['selection']
            torch.save(snapshot, kwargs['checkpoint_path'])
            # A valid endpoint record cannot make a different state_dict its final weights.
            snapshot = torch.load(kwargs['checkpoint_path'], weights_only=False)
            snapshot['state_dict']['network.4.bias'].add_(1.)
            torch.save(snapshot, kwargs['checkpoint_path'])
            with self.assertRaisesRegex(ValueError, 'replay checkpoint'):
                validate_association_endpoint(summary, audit, manifest, scores, [scene], **kwargs)

    def test_attribute_hybrid_uses_circular_mean_and_keeps_target_masks_external(self):
        scene = self.scene(q=2)
        states = scene['motion']['states'].copy()
        states[0, :, 1], states[1, :, 1] = 179.9, -179.9
        states[0, :, 3], states[1, :, 3] = 359., 1.
        states[0, :, 2], states[1, :, 2] = 4., 8.
        scene['motion']['states'] = states
        groups = [[0, 1], [2]]
        controls = ordinary_fusion(states, scene['motion']['nearest_seconds'],
                                   scene['motion']['support'], groups)
        arms = ordinary_predictions(controls, ['anchor', 'mean_supported', 'attribute_hybrid'])
        np.testing.assert_array_equal(arms['attribute_hybrid'][..., :2], states[..., :2])
        np.testing.assert_array_equal(arms['attribute_hybrid'][..., 2:], arms['mean_supported'][..., 2:])
        self.assertAlmostEqual(arms['attribute_hybrid'][0, 0, 2], 6.)
        self.assertAlmostEqual((arms['attribute_hybrid'][0, 0, 3]+180.) % 360.-180., 0.)
        scene['fusion_targets']['ship-a']['valid'][0, 2] = False
        scene['valid'][[0, 1], 0, 2] = False
        reports, artifacts = evaluate_unique_population(scene, {'ols': groups}, {'ols': arms})
        base = reports['ols_anchor']['full_target_metrics']['all']
        hybrid = reports['ols_attribute_hybrid']['full_target_metrics']['all']
        mean = reports['ols_mean_supported']['full_target_metrics']['all']
        self.assertEqual(hybrid['position'], base['position'])
        self.assertEqual(hybrid['speed'], mean['speed'])
        self.assertEqual(hybrid['heading'], mean['heading'])
        np.testing.assert_array_equal(artifacts['ols_attribute_hybrid']['aligned_truth_valid'],
                                     artifacts['ols_anchor']['aligned_truth_valid'])

    def test_pooled_errors_use_intersection_count_not_scene_average(self):
        def row(n, mae):
            return dict(intersection_count=n, available_target_count=10,
                        mae=mae if n else None, mse=mae*mae if n else None,
                        unit='metres')
        regions = ('all', 'inside_support', 'outside_support', 'far_gt_1800s')
        scenes = [dict(fragment={'ols_mean_supported': {k: {'position': row(n, v),
                'speed': row(0, 0), 'heading': row(0, 0)} for k in regions}})
                  for n, v in [(1, 10), (9, 2)]]
        answer = _pool_fragment(scenes, ['ols_mean_supported'])
        summary = answer['ols_mean_supported']['all']['position']
        self.assertAlmostEqual(summary['mae'], 2.8)
        self.assertAlmostEqual(summary['scene_macro_mae'], 6.0)
        self.assertAlmostEqual(summary['coverage'], 0.5)
        self.assertIsNone(answer['ols_mean_supported']['all']['speed']['mae'])

    def test_completed_run_cannot_be_paired_with_step_zero_weights(self):
        group = dict(ols_score_threshold=-2., working_point=dict(threshold=-2.))
        cfg = dict(training=dict(updates=1000), motion_observation_policy='canonical_all', grouping=group)
        manifest = dict(splits=dict(T_core=['0'], V_select=['1']))
        run = dict(status='COMPLETED', optimizer_updates=1000, configuration=cfg,
                   final_evaluation='evaluation-step-1000.json',
                   training_scene_ids=['0'], validation_scene_ids=['1'])
        model = SimpleNamespace(motion_observation_policy='canonical_all', ols_grouping=group)
        snapshot = dict(configuration=cfg, step=1000)
        self.assertEqual(verify_fusion_snapshot(model, snapshot, run, manifest, group), 1000)
        with self.assertRaises(ValueError):
            verify_fusion_snapshot(model, {**snapshot, 'step':0}, run, manifest, group)
        with self.assertRaises(ValueError):
            verify_fusion_snapshot(model, snapshot, {**run, 'configuration':{}}, manifest, group)

    def test_no_input_identity_stays_in_full_target_registry(self):
        scene = self.scene(q=2)
        partitions = dict(separate=[[0], [1], [2]], merged=[[0, 1, 2]])
        predictions = {name: dict(mean_supported=scene['target'].copy()) for name in partitions}
        reports, artifacts = evaluate_unique_population(scene, partitions, predictions)
        separate = reports['separate_mean_supported']
        self.assertEqual(separate['target_identity_count'], 3)
        self.assertEqual(separate['duplicate_prediction_group_count'], 1)
        self.assertEqual(separate['unmatched_target_ids'], ['ship-c'])
        self.assertEqual(separate['full_target_metrics']['all']['position']['coverage'], 2/3)
        self.assertEqual(reports['merged_mean_supported']['unmatched_target_ids'], ['ship-b', 'ship-c'])
        paired, common = common_target_metrics(artifacts)
        self.assertEqual(int(common[..., 0].sum()), 2)
        for report in paired.values():
            self.assertEqual(report['full_target_metrics']['all']['position']['coverage'], 1/3)

        empty = {**scene, 'identities': np.array([], str),
            'target': np.empty((0, 2, 4)), 'valid': np.empty((0, 2, 4), bool),
            'motion': dict(query_seconds=scene['grid'], support=np.empty((0, 2), bool),
                           nearest_seconds=np.empty((0, 2)))}
        reports, _ = evaluate_unique_population(empty, dict(ols=[]),
            dict(ols=dict(mean_supported=np.empty((0, 2, 4)))))
        missing = reports['ols_mean_supported']['full_target_metrics']['all']['position']
        self.assertEqual(missing['available_target_count'], 6)
        self.assertEqual(missing['missing_prediction_count'], 6)
        self.assertEqual(missing['coverage'], 0.)

    def test_four_arm_entrypoint_persists_recomputable_metrics_and_rejects_step_zero(self):
        """Only archive loading is mocked; no optimizer update or real dataset is used."""
        scene = self.scene()
        manifest = dict(dataset_level='Level1', split_unit='scene',
                        splits=dict(T_core=['t'], V_select=['v'], T_test=['sealed']))
        wp = dict(scope='ols-working-point-v1', synthetic=False, dataset_level='Level1',
            motion_observation_policy=POLICY,
            data={key: DATA[key] for key in ('length', 'max_observations', 'time_unit_seconds')},
            training_scene_ids=['t'], validation_scene_ids=['v'],
            score_semantics='-log1p(rms_haversine_m/1000)', selection_split='V_select', threshold=-1.)
        config = bind_ols_working_point(dict(data=DATA, motion_observation_policy=POLICY,
            model=dict(exact_observation_lock=False), training=dict(updates=1)), manifest, wp)
        fusion = EvidenceFusion(**config['model'])
        optimizer = torch.optim.Adam(fusion.parameters())
        fusion_run = dict(status='COMPLETED', configuration=config, optimizer_updates=1,
            final_evaluation='evaluation-step-1.json', training_scene_ids=['t'], validation_scene_ids=['v'])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.association_fixture(root, scene, wp)
            for filename, value in (('manifest', manifest), ('working-point', wp), ('alignment', {}),
                                    ('fusion-run', fusion_run)):
                write_json(root/f'{filename}.json', value)
            save_checkpoint(root/'final.pt', fusion, optimizer, 1, config, np.random.default_rng(0))
            save_checkpoint(root/'step-0.pt', fusion, optimizer, 0, config, np.random.default_rng(0))
            args = SimpleNamespace(device='cpu', cpu_threads=1, archive=['synthetic-not-read.zip'],
                manifest=root/'manifest.json', alignment=root/'alignment.json',
                ols_working_point=root/'working-point.json', association_scores=root/'scores.npz',
                association_summary=root/'association-summary.json',
                association_data_audit=root/'association-audit.json', fusion_checkpoint=root/'final.pt',
                association_endpoint='final', association_run=root/'association-run.json',
                association_checkpoint=root/'association-final.pt',
                fusion_run=root/'fusion-run.json', output=root/'evaluation')
            with patch('tools.experiments.level1_system_evaluate.load_timed_dataset',
                       return_value=dict(validation=[scene], report=dict(
                           validation_scene_ids=['v'], training_scene_ids=[], test_opened=False))) as loader:
                report = run(args)
                self.assertTrue(loader.call_args.kwargs['validation_only'])
                self.assertEqual(report['status'], 'COMPLETED')
                self.assertEqual(report['fusion_checkpoint_step'], 1)
                self.assertEqual(report['association_endpoint_step'], 1)
                self.assertTrue(report['association_score_replay']['decisions_equal'])
                self.assertEqual(len(report['method_names']), 4)
                with np.load(args.output/'predictions/states-v.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['ols_group_offsets'], [0, 2, 3])
                    np.testing.assert_array_equal(saved['learned_assoc_group_offsets'], [0, 3])
                    self.assertEqual(saved['learned_assoc_routing_helper_weights'].shape, (6, 64, 3))
                with np.load(args.output/'system-predictions/states-v.npz', allow_pickle=False) as saved:
                    artifacts = {name: {key[len(name)+1:]: saved[key].copy()
                        for key in saved.files if key.startswith(name+'_')} for name in report['method_names']}
                    paired, mask = common_target_metrics(artifacts)
                    np.testing.assert_array_equal(mask, saved['common_prediction_valid_on_target_rows'])
                for name in report['method_names']:
                    expected = report['common_valid_target_points'][name]['all']['position']
                    actual = paired[name]['full_target_metrics']['all']['position']
                    self.assertEqual(actual['intersection_count'], expected['intersection_count'])
                    self.assertAlmostEqual(actual['mae'], expected['mae'])
                    self.assertEqual(actual['coverage'], 1/3)
                args.fusion_checkpoint, args.output = root/'step-0.pt', root/'rejected'
                with self.assertRaisesRegex(ValueError, 'same final Level1 job'):
                    run(args)
                self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')

    def test_real_short_fit_artifacts_feed_six_arm_final_system_evaluation(self):
        """Only native loading is replaced: train, save, bind, replay and score are real."""
        validation = self.scene()
        training = {**deepcopy(validation), 'scene_id': 't', 'component_id': 't', 'split': 'T_core'}
        manifest = dict(dataset_level='Level1', split_unit='scene',
                        splits=dict(T_core=['t'], V_select=['v'], T_test=['sealed']))
        config = dict(scope='native-ols-evidence-association-v1', variant='full_evidence',
            data=ASSOCIATION_DATA, seed=7, motion_observation_policy=POLICY,
            training=dict(updates=2, batch_size=256, validate_every=2, inference_batch_size=16,
                          learning_rate=.001, gradient_clip=1., advisory_hours=1.))
        ordinary_weights = None
        for mode in (None, 'uniform', 'hard_negative'):
            with self.subTest(sampling=mode):
                if mode is not None:
                    config['sampling'] = dict(mode=mode, intervention_seed=20261023,
                                              hard_slots=64, hard_fraction=.1)
                with TemporaryDirectory() as temp:
                    root = Path(temp)
                    association = root/'association'
                    with patch('association.training.evidence_association.prepare_motion_scenes',
                               side_effect=[[training], [validation]]) as motion_loader:
                        fitted = fit_evidence_association(config, association, 'cpu',
                            archive_path=['synthetic-not-read.zip'], manifest=manifest)
                    self.assertEqual([call.args[2] for call in motion_loader.call_args_list], ['T_core', 'V_select'])
                    self.assertEqual(fitted['optimizer_updates'], 2)
                    self.assertEqual(read_json(association/'run.json')['status'], 'COMPLETED')
                    final = torch.load(association/'final.pt', weights_only=False)
                    self.assertEqual(final['update'], 2)
                    self.assertEqual(final['config'], config)
                    if mode is None:
                        ordinary_weights = final['state_dict']
                    elif mode == 'uniform':
                        for name, values in ordinary_weights.items():
                            torch.testing.assert_close(final['state_dict'][name], values, rtol=0, atol=0)
                    if mode is not None:
                        audit = read_json(association/'paired_sampling_audit.json')
                        self.assertEqual(audit['completed_batch_count'], 2)
                        self.assertEqual(audit['actual_optimizer_updates'], 2)
                        self.assertEqual(audit['hard_source_slots'], 128 if mode == 'hard_negative' else 0)
                        self.assertEqual(fitted['paired_sampling']['hard_source_slots'], audit['hard_source_slots'])
                    else:
                        self.assertFalse((association/'paired_sampling_audit.json').exists())
                    self.assertGreater(float(final['state_dict']['network.4.weight'].abs().sum()), 0.)
                    write_json(root/'manifest.json', manifest)
                    write_json(root/'alignment.json', {})
                    args = SimpleNamespace(device='cpu', cpu_threads=1, archive=['synthetic-not-read.zip'],
                        manifest=root/'manifest.json', alignment=root/'alignment.json',
                        ols_working_point=association/'ols-working-point.json',
                        association_scores=association/'final-scores.npz',
                        association_summary=association/'summary.json',
                        association_data_audit=association/'data_audit.json',
                        association_endpoint='final', association_run=association/'run.json',
                        association_checkpoint=association/'final.pt',
                        ordinary_controls=['anchor', 'mean_supported', 'attribute_hybrid'],
                        fusion_checkpoint=None, fusion_run=None, output=root/'system')
                    with patch('tools.experiments.level1_system_evaluate.load_timed_dataset',
                               return_value=dict(validation=[validation], report=dict(
                                   validation_scene_ids=['v'], training_scene_ids=[], test_opened=False))) as loader:
                        report = run(args)
                    self.assertTrue(loader.call_args.kwargs['validation_only'])
                    self.assertEqual(report['status'], 'COMPLETED')
                    self.assertEqual(report['association_endpoint_step'], 2)
                    self.assertEqual(report['association_threshold'], fitted['final_selection']['threshold'])
                    self.assertEqual(report['association_score_replay']['max_absolute_difference'], 0.)
                    self.assertEqual(report['association'], fitted['final_result']['metrics'])
                    self.assertEqual(set(report['method_names']), {group+'_'+arm
                        for group in ('ols', 'learned_assoc')
                        for arm in ('anchor', 'mean_supported', 'attribute_hybrid')})
                    with np.load(args.output/'predictions/states-v.npz', allow_pickle=False) as saved:
                        for group in ('ols', 'learned_assoc'):
                            np.testing.assert_array_equal(saved[group+'_attribute_hybrid'][..., :2],
                                                          saved[group+'_anchor'][..., :2])
                            np.testing.assert_array_equal(saved[group+'_attribute_hybrid'][..., 2:],
                                                          saved[group+'_mean_supported'][..., 2:])
                    self.assertFalse(report['test_opened'])
                    self.assertEqual(report['optimizer_updates'], 0)

    def test_common_target_intersection_prevents_coverage_artifact(self):
        target = np.array([[[30.,120.,6.,90.]],[[31.,121.,6.,90.]]])
        valid = np.ones_like(target,bool)
        observed = np.ones((2,1),bool)
        a = dict(target_ids=np.array(['a','b']), aligned_truth=target.copy(),
                 aligned_truth_valid=valid.copy(),aligned_observed=observed.copy(),
                 aligned_prediction=target.copy(), aligned_prediction_valid=valid.copy())
        b = {**a,'aligned_prediction':target.copy(), 'aligned_prediction_valid':valid.copy()}
        a['aligned_prediction_valid'][1]=False
        reports, common = common_target_metrics({'a':a,'b':b})
        self.assertEqual(int(common.sum()),4)
        for row in reports.values():
            metric=row['full_target_metrics']['all']['position']
            self.assertEqual(metric['available_target_count'],2)
            self.assertEqual(metric['intersection_count'],1)
            self.assertEqual(metric['coverage'],.5)

    def test_common_target_preserves_explicit_speed_units_and_rejects_mixed_units(self):
        from association.data.physical_units import SPEED_UNIT_POLICY
        truth = np.array([[[30., 120., 6., 90.]]])
        prediction = truth.copy(); prediction[..., 2] += 2.
        artifact = dict(target_ids=np.array(['a']), aligned_truth=truth,
            aligned_truth_valid=np.ones_like(truth, bool), aligned_observed=np.ones((1, 1), bool),
            aligned_prediction=prediction, aligned_prediction_valid=np.ones_like(truth, bool))
        legacy, _ = common_target_metrics({'a': artifact, 'b': deepcopy(artifact)})
        self.assertEqual(legacy['a']['full_target_metrics']['all']['speed']['unit'], 'native_speed')
        current = dict(artifact, state_speed_unit='mps', speed_unit_policy=SPEED_UNIT_POLICY)
        reports, _ = common_target_metrics({'a': current, 'b': deepcopy(current)})
        speed = reports['a']['full_target_metrics']['all']['speed']
        self.assertEqual(speed['unit'], 'mps')
        self.assertEqual(speed['mae'], 2.)  # Only the label is preserved; no second conversion.
        for changed in (artifact, dict(current, state_speed_unit='kn'),
                        dict(current, speed_unit_policy='different')):
            with self.assertRaisesRegex(ValueError, 'speed units or policies'):
                common_target_metrics({'a': current, 'b': changed})



if __name__ == '__main__':
    unittest.main()
