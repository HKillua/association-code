"""Contract tests for Level1 frozen comparison and 049 activation audit."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from association.data.motion_evidence import build_motion_evidence
from association.evaluation.association import evaluate_scenes
from association.models.evidence_fusion import EvidenceFusion
from association.training.evidence_fusion import bind_ols_working_point, save_checkpoint
from association.utils.runtime import read_json, write_json

from tools.experiments.level1_system_evaluate import (
    DATA, POLICY, run,
    _pool_fragment, exact_score_population, verify_fusion_snapshot,
    evaluate_unique_population, common_target_metrics,
)
from tools.analysis.tfm_information_probe import (
    _dispersion, _physical_output_dispersion, _heading_dispersion,
)


class Level1SystemContractTests(unittest.TestCase):
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
        # Deliberately merge different identities to exercise different arm coverage.
        score_record = {**{key: scene[key] for key in ('scene_id', 'component_id', 'pairs', 'labels', 'sources')},
                        'scores': np.ones(3)}
        summary = dict(status='COMPLETED', scope='native-ols-evidence-association-v1',
            motion_observation_policy=POLICY, optimizer_updates=1,
            selected=dict(update=1, selection=dict(threshold=0.)),
            selected_result=dict(metrics=evaluate_scenes([score_record], 0.)))
        audit = dict(motion_observation_policy=POLICY, split_unit='scene',
                     training_scene_ids=['t'], validation_scene_ids=['v'])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for filename, value in (('manifest', manifest), ('working-point', wp), ('alignment', {}),
                                    ('fusion-run', fusion_run), ('association-summary', summary),
                                    ('association-audit', audit)):
                write_json(root/f'{filename}.json', value)
            np.savez_compressed(root/'scores.npz', scene_ids=np.array(['v']),
                v_pairs=scene['pairs'], v_scores=score_record['scores'],
                v_node_ids=scene['node_ids'], v_sources=scene['sources'])
            save_checkpoint(root/'final.pt', fusion, optimizer, 1, config, np.random.default_rng(0))
            save_checkpoint(root/'step-0.pt', fusion, optimizer, 0, config, np.random.default_rng(0))
            args = SimpleNamespace(device='cpu', cpu_threads=1, archive=['synthetic-not-read.zip'],
                manifest=root/'manifest.json', alignment=root/'alignment.json',
                ols_working_point=root/'working-point.json', association_scores=root/'scores.npz',
                association_summary=root/'association-summary.json',
                association_data_audit=root/'association-audit.json', fusion_checkpoint=root/'final.pt',
                fusion_run=root/'fusion-run.json', output=root/'evaluation')
            with patch('tools.experiments.level1_system_evaluate.load_timed_dataset',
                       return_value=dict(validation=[scene], report=dict(
                           validation_scene_ids=['v'], training_scene_ids=[], test_opened=False))) as loader:
                report = run(args)
                self.assertTrue(loader.call_args.kwargs['validation_only'])
                self.assertEqual(report['status'], 'COMPLETED')
                self.assertEqual(report['fusion_checkpoint_step'], 1)
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

    def test_physical_attributes_have_separate_diagnostics(self):
        states=np.zeros((2,4,4))
        states[...,0]=30.; states[...,1]=120.
        states[0,:,2]=4.; states[1,:,2]=12.
        states[0,:,3]=359.; states[1,:,3]=1.
        stat=_physical_output_dispersion(states)
        self.assertEqual(stat['position_metres']['max_pair_0_1_separation_metres'],0.)
        self.assertGreater(stat['speed_native']['cross_sample_centered_rms'],0.)
        self.assertAlmostEqual(stat['heading_circular']['pair_0_1_mean_absolute_short_arc_deg'],2.)
        self.assertLess(_heading_dispersion(states[...,3])['cross_sample_mean_circular_std_deg'],5.)

    def test_activation_dispersion_not_called_information_loss(self):
        a = np.array([[[0., 1.], [1., 0.]], [[2., 3.], [3., 2.]]])
        result = _dispersion(a)
        self.assertGreater(result['cross_sample_centered_rms'], 0)
        self.assertGreater(result['within_sample_time_centered_rms'], 0)
        self.assertEqual(result['shape'], [2, 2, 2])


if __name__ == '__main__':
    unittest.main()
