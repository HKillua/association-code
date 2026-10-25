"""Small shared learning fixtures; no experiment-specific acceptance tests."""
import numpy as np

from association.data.motion_evidence import scene_motion_evidence
from association.models.support_gate import ROUTING_SEMANTICS, SCOPE
from tools.experiments.evidence_fusion_fit import synthetic_scenes


def fixture_motion():
    states = np.array([
        [[20., 179., 2., 350.], [20., 179., 2., 350.], [20., 179., 2., 350.]],
        [[20.01, -179., 8., 10.], [20.01, -179., 8., 10.], [20.01, -179., 8., 10.]],
        [[20.02, -177., 6., 20.], [20.02, -177., 6., 20.], [20.02, -177., 6., 20.]]])
    return dict(states=states, support=np.array([[True, False, False], [True, True, False], [False, True, False]]),
        nearest_seconds=np.array([[0., 20., 40.], [0., 0., 20.], [20., 0., 20.]]),
        exact_observation=np.zeros((3, 3), bool), native_count=np.array([2, 4, 3]),
        span_seconds=np.array([10., 40., 20.]), fit_rmse_m=np.array([0., 10., 20.]),
        ols_velocity_mps=np.array([[2., 3.], [2., 3.], [3., 2.]]),
        endpoint_velocity_mps=np.array([[2., 3.], [3., 2.], [2., 3.]]))


def tiny_config():
    return dict(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS,
        data=dict(length=64, max_observations=128, time_unit_seconds=20., reference_longitude_policy='shortest_arc'),
        train_splits=['T_core'], model=dict(initial_gate=.95), motion_observation_policy='canonical_all',
        training=dict(seed=20261021, updates=3, batch_size=2, learning_rate=.001,
                      gradient_clip=1., advisory_seconds=3600.))


def tiny_train():
    train, _ = synthetic_scenes(length=64, max_observations=4)
    for scene in train:
        scene['tracks'] = [dict(node_id=node_id, source=scene['sources'][node],
            canonical_values=scene['native_values'][node, scene['native_observed'][node]],
            canonical_times=scene['native_times'][node, scene['native_observed'][node]])
            for node, node_id in enumerate(scene['node_ids'])]
        scene['motion'] = scene_motion_evidence(scene, motion_observation_policy='canonical_all')
        scene['motion_observation_policy'] = 'canonical_all'
        del scene['tracks']
        # A definite, finite position learning signal independent of other attrs.
        for target in scene['fusion_targets'].values():
            target['states'][:, 0] += .02
    return train
