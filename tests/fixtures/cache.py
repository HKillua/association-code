"""Shared complete-population cache fixtures for current learned models."""
import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.oracle_diagnostics import POPULATION_KEYS
from association.evaluation.system_attribution import pool_metrics
from tools.analysis.compare_association_duration import _replay
from tools.experiments.level1_system_evaluate import evaluate_unique_population, ordinary_predictions
from tests.fixtures.scenes import duplicate_missing_target_fixture


def six_arm_cache_fixture(sparse_reference=False):
    """Reuse058's swallowed target/duplicate layout, expanded to the Q64 cache."""
    fragment, _, _ = duplicate_missing_target_fixture()
    native = np.repeat(fragment['native_states'], 16, axis=1)
    truth = np.broadcast_to(np.array([[30., 120., 5., 1.], [30.01, 120., 6., 1.],
        [30.02, 120., 7., 2.], [30.03, 120., 8., 180.], [30.04, 120., 9., 90.]])[:, None],
        (5, 64, 4)).copy()
    valid = np.ones(truth.shape, bool)
    valid[1, 9, :2] = False
    valid[1, 10, 2] = False
    valid[1, 11, 3] = False
    valid[4] = False
    native[:, :, 1] = 120.
    native[2, :, 3], native[3, :, 3] = 359., 1.
    native[3, :, 0] = 30.  # Mixing helps A, but a later duplicate is better still.
    native[4] = truth[1]
    native[2, 3, 3], native[3, 3, 3] = 0., 180.  # Heading fallback.
    native[2, 4, 1], native[3, 4, 1] = 0., 180.  # Longitude fallback, latitude still mixes.
    native[3, 8] = native[2, 8]  # A supported helper can leave every candidate unchanged.
    support = np.ones((6, 64), bool)
    support[2, 5:7] = False
    support[3, 6:8] = False
    nearest = np.full((6, 64), 10.)
    nearest[2, 5:7] = 2000.
    nearest[3, 6:8] = 2000.
    if sparse_reference:
        truth[0, :, 0] += .2
        valid[:] = False
        valid[0, 0] = True
    truth[~valid] = np.nan
    nodes, targets = fragment['identities'], fragment['target_identity_ids']
    target_rows = [0, 1, 1, 2, 1, 4]
    fragment.update(native_states=native, target=truth[target_rows], target_valid=valid[target_rows],
        query_seconds=np.linspace(0., 2000., 64), support=support, nearest_seconds=nearest,
        exact_observation=np.zeros((6, 64), bool))
    groups = [[0, 1], [2, 3], [4], [5]]
    controls = ordinary_fusion(native, nearest, support, groups)
    arms = ordinary_predictions(controls, ('anchor', 'mean_supported', 'attribute_hybrid'))
    scene = dict(identities=nodes, target_identity_ids=targets.tolist(), target=fragment['target'],
        valid=fragment['target_valid'], motion=dict(states=native, support=support, nearest_seconds=nearest,
                                                   query_seconds=fragment['query_seconds']),
        fusion_targets={identity: dict(states=truth[i], valid=valid[i]) for i, identity in enumerate(targets)})
    reports, artifacts = evaluate_unique_population(scene, dict(ols=groups, learned_assoc=groups),
                                                   dict(ols=arms, learned_assoc=arms))
    for grouping in ('ols', 'learned_assoc'):
        fragment.update({grouping + '_' + arm: value for arm, value in arms.items()})
        fragment[grouping + '_membership'] = controls['membership']
        fragment[grouping + '_group_offsets'] = np.array([0, 2, 4, 5, 6])
        fragment[grouping + '_group_node_indices'] = np.arange(6)
    saved = {name + '_' + key: value for name, item in artifacts.items() for key, value in item.items()}
    expected = {name: dict(population={key: item[key] for key in POPULATION_KEYS},
                           full_target_metrics=item['full_target_metrics']) for name, item in reports.items()}
    return fragment, saved, expected


def support_gate_cache_fixture(root, *, sparse=False, scene_id='3035'):
    """Actual six-arm Q64 cache with swallowed/missing targets and a duplicate.

    Motion reconstruction alone is substituted by callers: the existing fixture
    has cached physical evidence but no source observations. All ordinary replay,
    model forward, group alignment and metrics use production implementations.
    """
    fragment, saved, expected = six_arm_cache_fixture(sparse)
    source = root/'source056'/'system'
    source.mkdir(parents=True)
    for directory, values in (('predictions', fragment), ('system-predictions', saved)):
        (source/directory).mkdir()
        np.savez_compressed(source/directory/f'states-{scene_id}.npz', **values)
    replay = _replay(fragment, saved, expected)
    artifact = replay[1]['learned_assoc_anchor']
    n = len(fragment['node_ids'])
    motion = dict(states=fragment['native_states'], support=fragment['support'],
        nearest_seconds=fragment['nearest_seconds'], query_seconds=fragment['query_seconds'],
        exact_observation=fragment['exact_observation'], native_count=np.full(n, 8),
        span_seconds=np.full(n, 100.), fit_rmse_m=np.ones(n),
        ols_velocity_mps=np.ones((n, 2)), endpoint_velocity_mps=np.ones((n, 2)))
    raw = dict(scene_id=scene_id, split='V_select', component_id='shared', motion=motion,
        identities=fragment['identities'], target_identity_ids=fragment['target_identity_ids'],
        target=fragment['target'], valid=fragment['target_valid'],
        fusion_targets={identity: dict(states=artifact['aligned_truth'][i], valid=artifact['aligned_truth_valid'][i])
                        for i, identity in enumerate(fragment['target_identity_ids'])})
    metadata = dict(scene_ids=[scene_id], component_ids={scene_id: 'shared'},
        expected={scene_id: expected}, fragment_count=n,
        pooled={method: pool_metrics([row['full_target_metrics']]) for method, row in expected.items()})
    return source, metadata, [raw]
