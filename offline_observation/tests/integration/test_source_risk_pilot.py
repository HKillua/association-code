"""Small cache fixtures exercise 133 wiring only, never scientific effects."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.scene import evaluate_full_scene
from association.utils import runtime
from tools.analysis import source_risk_pilot as p
from tools.analysis import audit_source_risk_arrays as auditor


OLD_METHODS = ('WS_MEAN', 'WS_NEAREST_SUPPORTED', 'WS_LOW_SOURCE_SUPPORTED',
               'WS_FIXED_LOW_SOURCE_SUPPORTED', 'WS_QUADRATIC_VALUE_GATE')


def scene_arrays(sid):
    """Four tiny nodes include one duplicate group and an unmatched target."""
    q = 64
    states = np.zeros((4, q, 4), np.float64)
    states[..., 0] = 30.+int(sid)*1e-4+np.arange(4)[:, None]*1e-4+np.arange(q)*1e-5
    states[..., 1] = 120.+np.arange(4)[:, None]*2e-4+np.arange(q)*1e-5
    states[..., 2] = 10.+np.arange(4)[:, None]
    states[..., 3] = np.asarray([359., 1., 20., 30.])[:, None]
    support = np.ones((4, q), bool)
    support[:, 0] = False
    support[1, 1] = False
    nodes = np.asarray([f'{sid}/node-{i}' for i in range(4)])
    query = np.arange(q, dtype=float)*10.
    membership = np.asarray([0, 0, 1, 2], np.int64)
    groups = [[0, 1], [2], [3]]
    nearest = np.broadcast_to(np.asarray([2., 8., 20., 30.])[:, None], (4, q)).copy()
    motion = dict(states=states, support=support, nearest_seconds=nearest,
        sources=np.asarray(['source-z', 'source-a', 'source-z', 'source-a']),
        native_count=np.asarray([10, 20, 15, 25]), span_seconds=np.asarray([300., 400., 350., 450.]),
        fit_rmse_m=np.asarray([10., 30., 20., 40.]),
        ols_velocity_mps=np.asarray([[1., 2.], [1.1, 2.2], [1.2, 2.1], [2., 1.]]),
        endpoint_velocity_mps=np.asarray([[1.2, 2.1], [1.3, 2.4], [1.2, 2.2], [2.2, 1.3]]),
        ols_slope_degrees_per_second=np.ones((4, 2))*1e-5,
        query_seconds=query, exact_observation=np.zeros((4, q), bool), node_ids=nodes,
        pairs=np.column_stack(np.triu_indices(4, 1)))
    mean = ordinary_fusion(states, nearest, support, groups)['predictions']['mean_supported'][[0, 2, 3]]
    truth = np.stack([states[0].copy(), states[3].copy(), states[0].copy()])
    truth[0, :, 0] += (1 if int(sid) % 2 else -1)*2e-5
    truth[2, :, 0] += .1
    valid = np.ones_like(truth, bool)
    valid[0, 2, :2] = False
    evaluation = dict(node_ids=nodes, identities=np.asarray(['t0', 't0', 't0', 't1']),
        target_ids=np.asarray(['t0', 't1', 't2']), query_seconds=query,
        target_states=truth, target_valid=valid)
    artifact = evaluate_full_scene(emit_group_outputs(mean, groups, node_count=4),
        node_identities=evaluation['identities'], target_identities=evaluation['target_ids'],
        target_states=truth, target_valid=valid, node_observed=support, split='V_select',
        grouping_kind='predicted', correspondence_policy='anchor_identity_first',
        return_artifact=True, node_input_support=dict(support=support, nearest_seconds=nearest))['artifact']
    return motion, evaluation, membership, artifact


def cached_input(root):
    native, value, partition = (root/name for name in ('native127', 'value125', 'partition114'))
    for path in (native, value, partition):
        (path/'private').mkdir(parents=True)
    train = [str(i) for i in range(32)]
    validation = [str(i) for i in range(32, 64)]
    manifest = dict(splits=dict(T_core=train, V_select=validation, T_test=['sealed']),
        components={**{sid: f'T{i % 12}' for i, sid in enumerate(train)},
                    **{sid: f'V{i % 9}' for i, sid in enumerate(validation)}})
    manifest_path = root/'manifest.json'
    runtime.write_json(manifest_path, manifest)
    source = root/'unused056'/'system'
    runtime.write_json(native/'run.json', dict(status='COMPLETED', exit_code=0,
        idea_id='IDEA-NATIVE-CV-STATE-CONTROL-20261029', plan_version='1.0',
        source_system=str(source), partition_run=str(partition)))
    runtime.write_json(native/'native-cache.json', dict(status='CACHE_COMPLETED',
        schema_version='native-events-ragged-v1', scene_ids=dict(T_core=train, V_select=validation),
        identity_reference_stored_separately=True, native_motion_policy='canonical_all', test_opened=False))
    runtime.write_json(value/'run.json', dict(status='COMPLETED', exit_code=0,
        idea_id='IDEA-CONDITIONAL-SOURCE-VALUE-20261029', plan_version='1.0',
        native_run=str(native), source_system=str(source), partition_run=str(partition),
        training_scene_ids=train, scene_ids=validation, completed_T_scenes=32, completed_V_scenes=32,
        training_component_ids=sorted({manifest['components'][sid] for sid in train})))
    runtime.write_json(partition/'run.json', dict(status='COMPLETED', exit_code=0,
        idea_id='IDEA-STATIC-PARTITION-PILOT-20261029', plan_version='1.0', pilot_scene_ids=validation))
    for split, ids in (('T_core', train), ('V_select', validation)):
        for category in ('native-motion', 'native-evaluation'):
            (native/'private'/category/split).mkdir(parents=True)
        for sid in ids:
            motion, evaluation, membership, artifact = scene_arrays(sid)
            np.savez_compressed(native/'private/native-motion'/split/f'scene-{sid}.npz', **motion)
            np.savez_compressed(native/'private/native-evaluation'/split/f'scene-{sid}.npz', **evaluation)
            if split == 'T_core':
                np.savez_compressed(value/'private'/f'T-features-{sid}.npz', membership=membership,
                    scores=np.ones(len(motion['pairs']))*2.)
            else:
                p.old.frozen.write(partition/'private'/f'scene-{sid}.json.gz', dict(memberships=dict(WS=membership.tolist())))
                np.savez_compressed(value/'private'/f'states-{sid}.npz',
                    **{name+'_'+key: array for name in OLD_METHODS for key, array in artifact.items()})
    return SimpleNamespace(native_run=native, value_run=value, partition_run=partition,
        manifest=manifest_path, output=root/'new133', cpu_threads=1, time_policy='advisory')


class SourceRiskPilotTests(unittest.TestCase):
    def test_complete_fixed_endpoint_and_preserve_binding_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = cached_input(Path(temporary).resolve())
            result = p.run(args)
            state = runtime.read_json(args.output/'run.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertEqual(state['exit_code'], 0)
            self.assertEqual(result['status'], 'COMPLETED')
            aggregate = result['all_components']
            self.assertEqual(len(aggregate['methods']), 10)
            self.assertEqual(len(aggregate['pairwise']), 45)
            self.assertEqual(len(result['by_component']), 9)
            self.assertEqual(aggregate['scene_count'], 32)
            self.assertEqual(result['primary_bootstrap']['component_count'], 9)
            self.assertEqual(result['primary_bootstrap']['resamples'], 2000)
            self.assertEqual(state['completed_training_scene_count'], 32)
            self.assertEqual(state['completed_scene_count'], 32)
            self.assertEqual(state['analytic_head_fits'], 39)
            self.assertEqual(state['ridge_system_solves'], 39)
            self.assertEqual(state['no_label_fallback_heads'], 0)
            for key in ('teacher_forward_calls', 'neural_models_constructed', 'optimizer_updates'):
                self.assertEqual(state[key], 0)
            for key in ('archive_read', 'csv_read', 'test_opened', 'gpu_used'):
                self.assertFalse(state[key])
            self.assertTrue(state['source_features_frozen_before_labels'])
            self.assertTrue(state['full_T_models_frozen_before_V'])
            folds = runtime.read_json(args.output/'T-fold-index.json')
            self.assertEqual(len(folds), 13)
            for fold in folds:
                if fold['heldout_component'] is not None:
                    self.assertNotIn(fold['heldout_component'], fold['training_components'])
            with np.load(args.output/'private/states-32.npz', allow_pickle=False) as saved:
                baseline_valid = saved['WS_MEAN_aligned_prediction_valid']
                baseline_group = saved['WS_MEAN_node_group_indices']
                for name in aggregate['methods']:
                    np.testing.assert_array_equal(saved[name+'_aligned_prediction_valid'], baseline_valid)
                    np.testing.assert_array_equal(saved[name+'_node_group_indices'], baseline_group)
                    np.testing.assert_array_equal(saved[name+'_group_states'][..., 2:],
                                                  saved['WS_MEAN_group_states'][..., 2:])
            audit = auditor.audit(args.output, args.value_run, args.native_run, args.manifest,
                                  args.output/'independent-audit.json')
            self.assertEqual(audit['status'], 'COMPLETED', audit.get('error'))
            self.assertEqual(audit['audited_model_count'], 39)
            self.assertEqual(audit['exact_imported_scene_arms'], 160)
            self.assertEqual(audit['pair_count_per_scene'], 45)
            self.assertTrue(audit['public_oof_compared'])
            self.assertTrue(audit['public_bootstrap_compared'])
            # Audit must detect corruption independently of production scoring.
            pack = auditor.load(args.output/'private/training-0.npz')
            motion = auditor.load(args.native_run/'private/native-motion/T_core/scene-0.npz')
            ev = auditor.load(args.native_run/'private/native-evaluation/T_core/scene-0.npz')
            checks = auditor.core.Checks()
            groups = auditor.inspect_pack(pack, motion, pack['membership'], checks, 'mutation fixture')
            pack['sample_weight'][pack['labeled']] *= 2.
            with self.assertRaisesRegex(ValueError, 'label_weight/source'):
                auditor.inspect_labels(pack, groups, ev, auditor.core.Checks())
            # New run, altered reference/query binding: preserve both old result
            # and the second run's actual partial failure record.
            path = args.native_run/'private/native-evaluation/T_core/scene-0.npz'
            with np.load(path, allow_pickle=False) as saved:
                evaluation = {key: saved[key] for key in saved.files}
            evaluation['query_seconds'] = evaluation['query_seconds']+1.
            np.savez_compressed(path, **evaluation)
            broken = SimpleNamespace(**vars(args)); broken.output = args.output.parent/'broken133'
            with self.assertRaises((ValueError, AssertionError)):
                p.run(broken)
            self.assertEqual(runtime.read_json(broken.output/'run.json')['status'], 'FAILED')
            self.assertEqual(runtime.read_json(args.output/'run.json')['status'], 'COMPLETED')
            self.assertFalse((broken.output/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
