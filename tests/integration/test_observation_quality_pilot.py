"""Tiny immutable-cache integration fixture; not a scientific dataset/run."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from association.evaluation import source_risk as risk
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.scene import evaluate_full_scene
from association.utils import runtime
from tests.integration.test_source_risk_pilot import cached_input
from tools.analysis import observation_common as common
from tools.analysis import observation_quality_pilot as pilot
from tools.analysis import source_risk_pilot as source


def fixture(root):
    """Build original-sized lists of tiny records to exercise binding counts."""
    args = cached_input(root)
    manifest = runtime.read_json(args.manifest)
    sr = root/'fixture133'
    (sr/'private').mkdir(parents=True)
    tids, vids = manifest['splits']['T_core'], manifest['splits']['V_select']
    runtime.write_json(sr/'run.json', dict(status='COMPLETED', exit_code=0, fixture_only=True,
        training_scene_ids=tids, scene_ids=vids, completed_training_scene_count=32, completed_scene_count=32,
        native_run=str(args.native_run), value_run=str(args.value_run), partition_run=str(args.partition_run)))
    for split, ids in (('T_core', tids), ('V_select', vids)):
        (args.native_run/'private/native-events'/split).mkdir(parents=True)
        for sid in ids:
            member = common.membership(args.value_run, args.partition_run, split, sid)
            motion, groups, mean, pack, _ = common.physical(args.native_run, split, sid, member)
            # Uneven event counts and one two-event fragment make unknown-proxy
            # fallback observable without deleting that source or its outputs.
            sizes = [5, 2, 4, 3]
            times, values, event_nodes, canonical = [], [], [], []
            for node, size in enumerate(sizes):
                t = np.linspace(0., 630., size)
                value = np.tile(motion['states'][node, 0], (size, 1))
                value[:, 0] += t*1e-6+np.sin(np.arange(size))*(node+1)*1e-5
                value[:, 1] += t*1e-6
                times.extend(t)
                values.extend(value)
                event_nodes.extend([motion['node_ids'][node]]*size)
                canonical.extend(range(size))
            native = dict(node_ids=motion['node_ids'], sources=motion['sources'], query_seconds=motion['query_seconds'],
                event_offsets=np.r_[0, np.cumsum(sizes)].astype(np.int64), event_times_seconds=np.asarray(times),
                event_values=np.asarray(values), event_valid=np.ones((sum(sizes), 4), bool),
                event_node_ids=np.asarray(event_nodes), event_canonical_indices=np.asarray(canonical, np.int64))
            np.savez_compressed(args.native_run/'private/native-events'/split/f'scene-{sid}.npz', **native)
            if split == 'V_select':
                ev = common.evaluation(args.native_run, split, sid, motion)
                equal = risk.recover_source_policy(states=motion['states'], support=motion['support'], groups=groups,
                    pack=pack, scores=np.zeros(pack['available'].shape), kind='equal', mean_states=mean)['group_states']
                retained = common.load_artifacts(args.value_run/'private'/f'states-{sid}.npz', source.IMPORTED)
                artifacts = dict(retained)
                for name in source.POLICIES:
                    # The fixture assigns all learned policies the same fixed
                    # prediction; it does not fit or claim a scientific model.
                    artifacts[name] = evaluate_full_scene(emit_group_outputs(equal, groups, node_count=len(motion['states'])),
                        node_identities=ev['identities'], target_identities=ev['target_ids'],
                        target_states=ev['target_states'], target_valid=ev['target_valid'], node_observed=motion['support'],
                        split=split, grouping_kind='predicted', correspondence_policy='anchor_identity_first',
                        return_artifact=True, node_input_support=dict(support=motion['support'],
                            nearest_seconds=motion['nearest_seconds']))['artifact']
                common.save_artifacts(sr/'private'/f'states-{sid}.npz', artifacts)
    return SimpleNamespace(native_run=args.native_run, value_run=args.value_run, partition_run=args.partition_run,
        source_risk_run=sr, manifest=args.manifest, output=root/'fixture136', cpu_threads=1, time_policy='advisory')


class ObservationQualityPilotTests(unittest.TestCase):
    def test_fixed_endpoint_old_binding_and_failure_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary).resolve())
            result = pilot.run(args)
            state = runtime.read_json(args.output/'run.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertEqual(state['exit_code'], 0)
            self.assertEqual(state['completed_training_scene_count'], 32)
            self.assertEqual(state['completed_scene_count'], 32)
            self.assertEqual(state['target_reference_fit_count'], 0)
            for key, expected_n, expected_pairs in (('all_components', 15, 105),):
                self.assertEqual(len(result[key]['methods']), expected_n)
                self.assertEqual(len(result[key]['pairwise']), expected_pairs)
            training = result['training_summary']['all_components']
            self.assertEqual(len(training['methods']), 6)
            self.assertEqual(len(training['pairwise']), 15)
            self.assertNotIn('WS_SHARED_Q_HARD', training['methods'])
            self.assertEqual(len(result['primary_bootstrap']), 4)
            for split, sid, methods in (('T_core', '0', pilot.T_METHODS), ('V_select', '32', pilot.V_METHODS)):
                artifact = common.load_artifacts(args.output/'private'/f'states-{split}-{sid}.npz', methods,
                                                 template=methods[0])
                with np.load(args.native_run/'private/native-motion'/split/f'scene-{sid}.npz', allow_pickle=False) as stored:
                    motion = {k: stored[k] for k in stored.files}
                for name in methods:
                    np.testing.assert_array_equal(artifact[name]['group_states'][..., 2:],
                                                  artifact[methods[0]]['group_states'][..., 2:])
                    np.testing.assert_array_equal(artifact[name]['aligned_prediction_valid'],
                                                  artifact[methods[0]]['aligned_prediction_valid'])
                policy = common.load_npz(args.output/'private'/f'policy-{split}-{sid}.npz')
                self.assertFalse(policy['native_loo_known'][1])
                self.assertTrue(policy['WS_OBS_LOO_F1__uninformative_fallback'][0, 2:].all())
                for name in pilot.T_METHODS:
                    np.testing.assert_array_equal(policy[name+'__group_states'], artifact[name]['group_states'])
            # Predict-only execution does not take an evaluation object, and
            # a wrong native lineage fails with the prior completed run intact.
            path = args.native_run/'private/native-events/T_core/scene-0.npz'
            native = common.load_npz(path)
            native['event_node_ids'][0] = 'wrong-node'
            np.savez_compressed(path, **native)
            broken = SimpleNamespace(**vars(args))
            broken.output = args.output.parent/'broken136'
            with self.assertRaisesRegex(ValueError, 'lineage'):
                pilot.run(broken)
            self.assertEqual(runtime.read_json(broken.output/'run.json')['status'], 'FAILED')
            self.assertFalse((broken.output/'summary.json').exists())
            self.assertEqual(runtime.read_json(args.output/'run.json')['status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
