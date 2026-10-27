import copy
from dataclasses import asdict
import unittest

import numpy as np

from association.data.physical_units import KNOT_TO_MPS, SPEED_UNIT_POLICY, unit_metadata
from association.evaluation.fixed056_controls import controls_from_visible
from tests.fixtures.native_zip import NativeZipFixture


class Fixed056ControlTests(unittest.TestCase):
    def setUp(self):
        self.groups = [[0, 1], [2]]
        q = np.arange(4, dtype=float)
        native = np.zeros((3, 2, 4))
        native[..., 0] = 10
        native[..., 1] = 20
        native[..., 2] = 10*KNOT_TO_MPS
        native[..., 3] = 90
        self.visible = dict(scene_id='v', node_ids=np.array(['0', '1', '2']),
            sources=np.array(['a', 'b', 'a']), grid=q, native_values=native,
            native_times=np.tile([0., 3.], (3, 1)), native_observed=np.ones((3, 2), bool),
            physical_units=unit_metadata(SPEED_UNIT_POLICY))
        states = np.repeat(native[:, :1], 4, axis=1)
        states[..., 2] = 10.  # frozen056 source is knots
        states[0, 1, 0] = 11.  # A canonical observation missed by the capped input.
        self.fragment = dict(node_ids=self.visible['node_ids'].copy(), sources=self.visible['sources'].copy(),
            query_seconds=q.copy(), learned_assoc_group_offsets=np.array([0, 2, 3]),
            learned_assoc_group_node_indices=np.array([0, 1, 2]), learned_assoc_membership=np.array([0, 0, 1]),
            native_states=states, support=np.ones((3, 4), bool), nearest_seconds=np.zeros((3, 4)))

    def test_units_and_input_budgets_are_distinct_without_mutation(self):
        before = copy.deepcopy(self.fragment)
        result = controls_from_visible(self.visible, self.groups, self.fragment)
        full = result['canonical_all_anchor_ols']['generated'][0]['states']
        capped = result['selected_cap128_anchor_ols']['generated'][0]['states']
        self.assertEqual(full[1, 0], 11.)
        self.assertEqual(capped[1, 0], 10.)
        np.testing.assert_allclose(full[:, 2], 10*KNOT_TO_MPS)
        np.testing.assert_allclose(capped[:, 2], 10*KNOT_TO_MPS)
        np.testing.assert_array_equal(self.fragment['native_states'], before['native_states'])

    def test_reordered_group_node_or_wrong_units_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'groups'):
            controls_from_visible(self.visible, [[1, 0], [2]], self.fragment)
        changed = copy.deepcopy(self.visible)
        changed['node_ids'] = changed['node_ids'][::-1]
        with self.assertRaisesRegex(ValueError, 'population'):
            controls_from_visible(changed, self.groups, self.fragment)
        changed = copy.deepcopy(self.visible)
        changed['physical_units'] = None
        with self.assertRaisesRegex(ValueError, 'm/s'):
            controls_from_visible(changed, self.groups, self.fragment)


class Fixed056IntegrationTests(NativeZipFixture, unittest.TestCase):
    def test_paper_association_uses_full_same_and_cross_source_population(self):
        from association.evaluation.association import evaluate_scenes
        from tools.experiments.tmtaff_fixed056_evaluate import verify_056_association
        records = [dict(scene_id='s', component_id='c', sources=np.array(['a', 'a', 'b']),
            pairs=np.array([[0, 1], [0, 2], [1, 2]]), scores=np.array([1., 1., 0.]),
            labels=np.array([1, 0, 1]), intervals=np.array([[0., 1.], [0., 1.], [0., 1.]]))]
        original = evaluate_scenes(records, .5)
        actual = verify_056_association(records, .5, original)
        self.assertEqual(actual['itsa']['pair_count'], 1)
        self.assertEqual(actual['mstsa']['pair_count'], 2)
        self.assertEqual(actual['itsa']['paper_metrics']['m_pca'], 1.)
        # 0.4 yields the same edge decisions as 0.5, but is not the saved threshold.
        with self.assertRaisesRegex(ValueError, 'threshold'):
            verify_056_association(records, .4, original)
        changed = copy.deepcopy(original)
        changed['mstsa']['per_scene'][0]['fp'] = 0
        with self.assertRaisesRegex(ValueError, 'per-scene counts'):
            verify_056_association(records, .5, changed)

    def test_native_verification_receives_only_validation_raw_inputs(self):
        seen = []
        def verify(native):
            seen.append((native['scene_id'], native['split'],
                         native['tracks'][0]['canonical_values'][0, 2] if native['tracks'] else None))
        self.load(data_config={**self.data, 'speed_unit_policy': SPEED_UNIT_POLICY},
                  validation_native_verifier=verify)
        self.assertEqual([row[:2] for row in seen], [('2', 'V_select'), ('4', 'V_select')])
        self.assertEqual(seen[0][2], 5.)  # callback sees original knots before conversion

    def test_actual_tcm_and_tfm_exposure_binding_rejects_changed_completion(self):
        import torch
        from association.utils import runtime
        from tools.experiments.tmtaff_fixed056_evaluate import source_exposure
        source = self.root/'tcm'; source.mkdir()
        norm = dict(adapter_config=dict(speed_unit_policy=SPEED_UNIT_POLICY),
                    physical_units=unit_metadata(SPEED_UNIT_POLICY))
        runtime.write_json(source/'run.json', dict(status='COMPLETED', phase='tcm', test_opened=False,
            cursor=dict(completed_updates=2), elapsed_seconds=3.))
        runtime.write_json(source/'summary.json', dict(normalizer=norm, coverage={'TCM': {'track_presentations': 4}}))
        tcm = dict(phase='tcm', data_contract=['contract'], states=[{'positions': torch.tensor([1.]),
                   'input_embedding.weight': torch.tensor([2.])}])
        torch.save(tcm, source/'checkpoint.pt')
        payload = dict(initialization=dict(source_run=str(source), source_phase='tcm', source_tcm_updates=2),
            normalizer=norm, data_contract=['contract'], states=[copy.deepcopy(tcm['states'][0])],
            cursor=dict(completed_updates=1), coverage={'TFM_FROZEN': {'pair_presentations': 4}},
            history=[dict(stage='TFM_FROZEN', fusion_valid_counts=[1, 2, 3, 4])], pairing_audit=True)
        exposure = source_exposure(dict(elapsed_seconds=4.), payload)
        self.assertEqual(exposure['tfm']['valid_attribute_presentations'], [1, 2, 3, 4])
        self.assertEqual(exposure['tcm']['exposure']['track_presentations'], 4)
        payload['states'][0]['positions'] = torch.tensor([0.])
        with self.assertRaisesRegex(ValueError, 'frozen completion'):
            source_exposure({}, payload)

    def test_fixed_groups_and_complete_reference_registry_are_bound(self):
        from tests.fixtures.cache import support_gate_cache_fixture
        from tools.experiments.tmtaff_fixed056_evaluate import bind_scenes
        source, metadata, raw = support_gate_cache_fixture(self.root)
        with np.load(source/'predictions'/'states-3035.npz') as z:
            fragment = {key: z[key].copy() for key in z.files}
        scene = dict(raw[0], node_ids=fragment['node_ids'], sources=fragment['sources'],
                     grid=fragment['query_seconds'])
        n = len(scene['node_ids'])
        pairs = np.column_stack(np.triu_indices(n, 1))
        scores = np.zeros(len(pairs)); scores[(pairs == [0, 1]).all(1) | (pairs == [2, 3]).all(1)] = 1.
        record = dict(pairs=pairs, scores=scores)
        metadata['association_threshold'] = .5
        result = bind_scenes([scene], [record], source, metadata)
        self.assertTrue(result['fixed_groups_exact'])
        altered = dict(record, scores=np.zeros_like(scores))
        with self.assertRaisesRegex(ValueError, 'groups'):
            bind_scenes([scene], [altered], source, metadata)
        changed = copy.deepcopy(scene)
        changed['fusion_targets'][list(changed['fusion_targets'])[0]]['states'][0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'reference'):
            bind_scenes([changed], [record], source, metadata)

    def test_checkpoint_requires_completed_fixed_final_new_units(self):
        import torch
        from association.models.tmtaff import ModelConfig
        from association.models.fusion import PairFusionConfig
        from association.training.tmtaff import _configuration
        from association.utils import runtime
        from tools.experiments.tmtaff_fixed056_evaluate import bind_endpoint
        cfg = copy.deepcopy(self.config)
        cfg['model'] = asdict(ModelConfig(length=64))
        cfg['fusion'] = asdict(PairFusionConfig(fc_hidden=8, length=64))
        cfg['data'].update(max_observations=128, time_unit_seconds=20.,
                           reference_longitude_policy='shortest_arc', speed_unit_policy=SPEED_UNIT_POLICY)
        cfg['training']['updates'] = dict(TCM=0, TAM_FROZEN=0, TFM_FROZEN=2, JOINT=0)
        cfg['training']['fusion_monitor_updates'] = [0, 2]
        cfg['training'] = _configuration(cfg['training'], phase='tfm')
        cursor = dict(stage=2, update=2, completed_updates=2)
        run = dict(status='COMPLETED', phase='tfm', final_evaluation_status='COMPLETED', test_opened=False,
                   config=cfg['training'], cursor=cursor, checkpoint_cursor=cursor, selection_performed=False)
        payload = dict(format_version=1, phase='tfm', input_adapter='native_time_v1', config=cfg['training'],
                       cursor=cursor, model_config=cfg['model'], fusion_config=cfg['fusion'],
                       normalizer=dict(adapter_config=dict(speed_unit_policy=SPEED_UNIT_POLICY),
                                       physical_units=unit_metadata(SPEED_UNIT_POLICY)))
        path = self.root/'endpoint'; path.mkdir()
        runtime.write_json(path/'run.json', run)
        runtime.write_json(path/'preparation.json', dict(configuration=cfg))
        torch.save(payload, path/'checkpoint.pt')
        self.assertEqual(bind_endpoint(path)[2]['cursor']['completed_updates'], 2)
        payload['cursor'] = dict(stage=2, update=1, completed_updates=1)
        torch.save(payload, path/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError, 'completed fixed final'):
            bind_endpoint(path)

    def test_real_evaluator_callback_receives_no_truth_and_preserves_missing_population(self):
        import torch
        from association.data.motion_evidence import build_motion_evidence
        from association.data.timed import fit_timed_normalizer
        from association.evaluation.grouping import group_predicted_tracks
        from association.evaluation.tmtaff import evaluate_complete
        from association.models.timed import TimedAssociationModel
        from association.models.tmtaff import ModelConfig
        from association.models.fusion import PairFusion, PairFusionConfig
        from tools.experiments.tmtaff_fixed056_evaluate import ComparisonWriter

        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        data = self.load(data_config={**self.data, 'speed_unit_policy': SPEED_UNIT_POLICY})
        normalizer = fit_timed_normalizer(data['train'])
        model = TimedAssociationModel(ModelConfig(**self.config['model']), 'ce',
                                      max_observations=4, normalizer=normalizer)
        fusion = PairFusion(PairFusionConfig(**self.config['fusion']), normalizer)
        records, fragments = [], {}
        for scene in data['validation']:
            row = {key: scene[key] for key in ('scene_id', 'component_id', 'node_ids', 'identities',
                   'sources', 'intervals', 'pairs', 'labels')}
            row['scores'] = np.zeros(len(scene['pairs']))
            records.append(row)
            groups = group_predicted_tracks(len(scene['node_ids']), row['pairs'], row['scores'],
                threshold=1., grouping_policy='anchor_direct', node_order='input')
            motion = build_motion_evidence(scene['native_values'], scene['native_times'],
                scene['native_observed'], scene['grid'], scene['sources'])
            raw = motion['states'].copy(); raw[..., 2] /= KNOT_TO_MPS
            fragments[scene['scene_id']] = dict(node_ids=scene['node_ids'], sources=scene['sources'],
                query_seconds=scene['grid'], learned_assoc_group_offsets=np.r_[0, np.cumsum([len(g) for g in groups])],
                learned_assoc_group_node_indices=np.array([n for g in groups for n in g]),
                learned_assoc_membership=np.arange(len(groups)), native_states=raw,
                support=motion['support'], nearest_seconds=motion['nearest_seconds'])

        def callback(visible, groups):
            self.assertFalse({'identities', 'labels', 'fusion_targets', 'target'} & set(visible))
            return controls_from_visible(visible, groups, fragments[visible['scene_id']])

        writer = ComparisonWriter(self.root/'fair-predictions')
        result = evaluate_complete(model, fusion, normalizer, data['validation'], device='cpu',
            batch_tracks=4, pair_batch_size=4, select_on_validation=False, threshold=1.,
            singleton_policy='observed', association_records=records, predicted_controls=True,
            extra_predicted_controls=callback, prediction_writer=writer)
        self.assertEqual(result['tam_scoring_forwards'], 0)
        self.assertEqual(len(writer.rows), 2)
        self.assertIn('canonical_all_anchor_ols', result['fusion'])
        self.assertEqual(writer.pooled()['canonical_all_anchor_ols']['tmtaff']['speed']['unit'], 'mps')
        self.assertEqual(writer.rows[0]['pairs']['canonical_all_anchor_ols']['methods']
                         ['tmtaff']['full_target_metrics']['all']['speed']['unit'], 'mps')
        first = result['per_scene'][0]['methods']['canonical_all_anchor_ols']
        self.assertGreater(len(first['unmatched_target_ids']), 0)
        self.assertTrue((self.root/'fair-predictions'/'000000.json').exists())

    def test_cli_orchestrates_fixed_scores_controls_and_090c_on_small_artifacts(self):
        from unittest.mock import patch
        import torch
        from association.data.motion_evidence import build_motion_evidence
        from association.data.timed import fit_timed_normalizer
        from association.evaluation.grouping import group_predicted_tracks
        from association.evaluation.scene import evaluate_full_scene
        from association.models.timed import TimedAssociationModel
        from association.models.tmtaff import ModelConfig
        from association.models.fusion import PairFusion, PairFusionConfig
        from association.utils import runtime
        from tools.experiments import tmtaff_fixed056_evaluate as cli

        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        data = self.load(data_config={**self.data, 'speed_unit_policy': SPEED_UNIT_POLICY})
        norm = fit_timed_normalizer(data['train'])
        model = TimedAssociationModel(ModelConfig(**self.config['model']), 'ce', max_observations=4, normalizer=norm)
        fusion = PairFusion(PairFusionConfig(**self.config['fusion']), norm)
        source = self.root/'056'/'system'; (source/'predictions').mkdir(parents=True)
        old = self.root/'090c'; (old/'final'/'system-predictions').mkdir(parents=True)
        runtime.write_json(old/'run.json', {'source_system': str(source)})
        records = []
        for scene in data['validation']:
            record = {k: scene[k] for k in ('scene_id', 'component_id', 'node_ids', 'identities',
                      'sources', 'intervals', 'pairs', 'labels')}
            record['scores'] = np.zeros(len(scene['pairs'])); records.append(record)
            groups = group_predicted_tracks(len(scene['node_ids']), record['pairs'], record['scores'],
                threshold=1., grouping_policy='anchor_direct', node_order='input')
            motion = build_motion_evidence(scene['native_values'], scene['native_times'],
                scene['native_observed'], scene['grid'], scene['sources'])
            raw = motion['states'].copy(); raw[..., 2] /= KNOT_TO_MPS
            np.savez(source/'predictions'/f"states-{scene['scene_id']}.npz", node_ids=scene['node_ids'],
                sources=scene['sources'], query_seconds=scene['grid'],
                learned_assoc_group_offsets=np.r_[0, np.cumsum([len(g) for g in groups])],
                learned_assoc_group_node_indices=np.array([n for g in groups for n in g]),
                learned_assoc_membership=np.arange(len(groups)), native_states=raw,
                support=motion['support'], nearest_seconds=motion['nearest_seconds'])
            targets = scene['fusion_targets']
            generated = [dict(node_indices=g, states=scene['values'][g[0]],
                prediction_valid=np.repeat(scene['observed'][g[0], :, None], 4, axis=1)) for g in groups]
            report = evaluate_full_scene(generated, node_identities=scene['identities'], target_identities=list(targets),
                target_states=np.stack([x['states'] for x in targets.values()]),
                target_valid=np.stack([x['valid'] for x in targets.values()]), node_observed=scene['observed'],
                split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first',
                return_artifact=True)
            np.savez(old/'final'/'system-predictions'/f"states-{scene['scene_id']}.npz",
                **{'learned_sequence_'+k: v for k, v in report['artifact'].items()
                   if k not in ('split', 'grouping_kind', 'correspondence_policy')})
        manifest = self.root/'manifest.json'; alignment = self.root/'alignment.json'
        runtime.write_json(manifest, self.manifest); runtime.write_json(alignment, self.alignment)
        cfg = dict(data=self.config['data'], train_splits=['T_core'])
        run = dict(config=dict(validation_batches=dict(tracks=4, pairs=4), singleton_policy='observed'))
        payload = dict(data_contract='stub-contract', cursor={'completed_updates': 1})
        bound = {'model': model, 'fusion': fusion, 'normalizer': norm}
        from association.evaluation.association import evaluate_scenes
        metadata = dict(scene_ids=['2', '4'], association_threshold=1., association=evaluate_scenes(records, 1.))
        with patch.object(cli, 'bind_endpoint', return_value=(run, cfg, payload)), \
             patch.object(cli, 'source_exposure', return_value={'fixture': True}), \
             patch.object(cli, '_load_metadata', return_value=metadata), \
             patch.object(cli, 'reference090c_metadata', return_value={'status': 'fixture'}), \
             patch.object(cli, 'load_timed_dataset', return_value=data), \
             patch.object(cli, '_scenes', return_value='stub-contract'), \
             patch.object(cli, 'read_scores', return_value={}), \
             patch.object(cli, '_records_for', return_value=records), \
             patch.object(cli, 'bind_scenes', return_value={'fixture': True}), \
             patch.object(cli, 'load_complete_snapshot', return_value=bound):
            cli.main(['--tfm-run', str(self.root/'tfm'), '--source-system', str(source),
                '--reference090c-run', str(old), '--manifest', str(manifest), '--alignment', str(alignment),
                '--archive', str(self.path), '--output', str(self.root/'out'), '--device', 'cpu', '--cpu-threads', '1'])
        summary = runtime.read_json(self.root/'out'/'summary.json')
        self.assertEqual(runtime.read_json(self.root/'out'/'run.json')['status'], 'COMPLETED')
        self.assertEqual(summary['evaluation']['tam_scoring_forwards'], 0)
        self.assertIn('internal_090c', summary['pairwise_common_pooled'])
        self.assertIn('itsa', summary['inherited_056_association'])


if __name__ == '__main__':
    unittest.main()
