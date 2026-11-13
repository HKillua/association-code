"""Small synthetic populations with explicit clocks, labels and missing targets."""
import numpy as np
from association.data.timed import prepare_timed_scene

def native_scene(name, split, *, shift=0.0):
    times = ([12., 13.], [0., 15., 30.], [4., 15., 24.], [7., 16., 28.])
    identities = np.array([name + '-A', name + '-A', name + '-B', name + '-B'])
    sources = ['9001', '9002', '9001', '9002']
    tracks = []
    for node, stamps in enumerate(times):
        stamps = np.asarray(stamps)
        values = np.column_stack((30 + stamps * .0001 + (node // 2) * .01,
                                  120 + stamps * .0002, 5 + stamps * .01 + node * .1,
                                  40 + stamps * .1 + node))
        tracks.append(dict(node_id=name + ':' + str(node), identity_id=identities[node],
                           source=sources[node], canonical_times=stamps + shift,
                           canonical_values=values))
    pairs = np.column_stack(np.triu_indices(4, 1)).astype(np.int64)
    return dict(scene_id=name, component_id=name + '-component', split=split,
                tracks=tracks, node_ids=[x['node_id'] for x in tracks], identities=identities,
                sources=sources, intervals=np.array([[x['canonical_times'][0], x['canonical_times'][-1]]
                                                    for x in tracks]),
                pairs=pairs, labels=(identities[pairs[:, 0]] == identities[pairs[:, 1]]).astype(np.float32))

def timed_scene(name, split, *, shift=0.0):
    result = prepare_timed_scene(native_scene(name, split, shift=shift), length=4,
                                max_observations=3, time_unit_seconds=10.)
    grid = result['grid'] - shift
    targets = {}
    for index, identity in enumerate(dict.fromkeys(result['identities'])):
        values = np.column_stack((30 + grid * .0001 + index * .01, 120 + grid * .0002,
                                  5 + grid * .01, 40 + grid * .1))
        targets[str(identity)] = dict(states=values, valid=np.ones_like(values, bool),
                                      query_times=result['grid'].copy())
    result['fusion_targets'] = targets
    return result


def registered_scene(name, split, *, singleton=False):
    grid = np.arange(4, dtype=np.float64) * 10
    values = np.empty((4, 4, 4), dtype=np.float32)
    for node in range(4):
        values[node] = np.array([30 + .01 * (node // 2), 120, 5 + node / 10, 45])
        values[node, :, 0] += np.arange(4) * .0001
        values[node, :, 1] += np.arange(4) * .0002 + (node % 2) * .00001
    observed = np.array([[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=bool)
    if singleton:
        observed[:] = False
        observed[:, 1] = True
    targets = {identity: dict(states=values[node].copy(), valid=np.ones((4, 4), dtype=bool), query_times=grid.copy())
               for identity, node in (('a', 0), ('b', 2))}
    values[~observed] = np.nan
    pairs = np.array([(a, b) for a in range(4) for b in range(a + 1, 4)], dtype=np.int64)
    identities = np.array(['a', 'a', 'b', 'b'])
    return dict(scene_id=name, component_id=name + '-component', split=split,
                grid=grid, observation_times=np.broadcast_to(grid, observed.shape).copy(),
                values=values, observed=observed, fusion_targets=targets,
                node_ids=np.array(['n0', 'n1', 'n2', 'n3']), identities=identities,
                sources=np.array(['A', 'B', 'A', 'B']), pairs=pairs,
                labels=(identities[pairs[:, 0]] == identities[pairs[:, 1]]).astype(np.float32),
                intervals=np.array([[grid[np.flatnonzero(mask)[0]], grid[np.flatnonzero(mask)[-1]]]
                                    for mask in observed]))


def scene(scene_id='one', include_missing_target=False):
    grid = np.arange(4, dtype=float)
    observed = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]], bool)
    values = np.zeros((4, 4, 4), dtype=np.float32)
    for node in range(4):
        values[node, :, 0] = .001 * np.arange(4) + .01 * (node // 2)
        values[node, :, 1] = .002 * np.arange(4)
        values[node, :, 2] = 1 + node // 2
        values[node, :, 3] = 15 + node
    values[~observed] = np.nan
    identities = ['A', 'A', 'B', 'B']
    pairs = np.array([[i, j] for i in range(4) for j in range(i+1, 4)], dtype=np.int64)
    targets = {}
    for index, identity in enumerate(['A', 'B'] + (['C'] if include_missing_target else [])):
        states = np.zeros((4, 4), dtype=float)
        states[:, 0] = .001 * grid + .01 * index
        states[:, 1] = .002 * grid
        states[:, 2] = index + 1
        states[:, 3] = 15
        targets[identity] = {'states': states, 'valid': np.ones((4, 4), bool), 'query_times': grid.copy()}
    return dict(scene_id=scene_id, component_id='component-' + scene_id, split='V_select',
                grid=grid, observation_times=np.where(observed, grid[None], np.nan),
                values=values, observed=observed, node_ids=[scene_id + ':' + str(i) for i in range(4)],
                identities=identities, sources=['9001', '9002', '9001', '9002'],
                intervals=np.array([[0, 1], [1, 2], [0, 3], [0, 1]], float), pairs=pairs,
                labels=np.array([identities[i] == identities[j] for i, j in pairs], dtype=np.int64),
                fusion_targets=targets)

def empty_scene(scene_id='empty', with_target=False):
    result = scene(scene_id)
    for key in ('values', 'observed', 'observation_times', 'intervals'):
        result[key] = result[key][:0]
    result.update(node_ids=[], identities=[], sources=[], pairs=np.empty((0, 2), dtype=np.int64),
                  labels=np.empty(0, dtype=np.int64), fusion_targets={})
    if with_target:
        result['fusion_targets']['Z'] = {'states': np.zeros((4, 4)), 'valid': np.ones((4, 4), bool),
                                         'query_times': result['grid'].copy()}
    return result


def empty_timed_scene(name='empty', with_target=True):
    scene = native_scene(name, 'V_select')
    scene.update(tracks=[], node_ids=[], identities=np.array([], dtype=str), sources=[],
                 intervals=np.empty((0, 2)), pairs=np.empty((0, 2), dtype=np.int64),
                 labels=np.empty(0, dtype=np.float32))
    result = prepare_timed_scene(scene, length=4, max_observations=3,
                                 time_unit_seconds=10., query_times=np.array([0., 10., 20., 30.]))
    result['fusion_targets'] = ({name + '-unobserved': dict(states=np.zeros((4, 4)),
        valid=np.ones((4, 4), dtype=bool), query_times=result['grid'].copy())} if with_target else {})
    return result


def artifact_for(example):
    identities = list(example['fusion_targets'])
    targets = example['fusion_targets']
    # Split identity A and leave identity C without any input/prediction. This
    # exercises duplicated predictions and missing targets simultaneously.
    outputs = [dict(node_indices=[0], states=example['values'][0],
                    prediction_valid=np.repeat(example['observed'][0, :, None], 4, 1)),
               dict(node_indices=[1], states=None),
               dict(node_indices=[2, 3], states=targets['B']['states'])]
    kwargs = dict(node_identities=example['identities'], target_identities=identities,
        target_states=np.stack([targets[i]['states'] for i in identities]),
        target_valid=np.stack([targets[i]['valid'] for i in identities]),
        node_observed=example['observed'], split=example['split'], grouping_kind='predicted',
        correspondence_policy='anchor_identity_first')
    return outputs, kwargs


def duplicate_missing_target_fixture():
    """A's first fragment is swallowed by X; A has later duplicate anchors.

    B has input but no predicted representative; C has no input. D has an
    output without valid reference. Truth grouping switches A's anchor.
    """
    from association.evaluation.fusion_controls import ordinary_fusion
    from association.evaluation.scene import evaluate_full_scene
    from association.evaluation.system_attribution import BASE

    q = 4
    nodes = np.asarray(['x', 'a', 'a', 'b', 'a', 'd'])
    targets = np.asarray(['x', 'a', 'b', 'c', 'd'])
    truth = np.broadcast_to(np.array([
        [30., 120., 5., 1.], [30.01, 120.01, 6., 359.],
        [30.02, 120.02, 7., 2.], [30.03, 120.03, 8., 180.],
        [30.04, 120.04, 9., 90.],
    ])[:, None], (5, q, 4)).copy()
    valid = np.ones_like(truth, bool)
    indices = np.array([0, 1, 1, 2, 1, 4])
    native = truth[indices].copy()
    native[0, :, 0] += .003
    native[1, :, 0] += .001
    native[2, :, 0] += .006
    native[4, :, 0] -= .001
    native[1, :, 3] = 1.
    native[2, :, 3] = 355.
    native[4, :, 3] = 359.
    native[2, :, 2] += 4.
    valid[1, 1, :2] = False
    valid[2, 2, 3] = False
    valid[4] = False
    truth[~valid] = np.nan
    support = np.ones((6, q), bool)
    support[:, -1] = False
    nearest = np.broadcast_to(np.array([0., 60., 600., 2000.]), (6, q)).copy()
    groups = [[0, 1], [2, 3], [4], [5]]
    ordinary = ordinary_fusion(native, nearest, support, groups)
    mean = ordinary['predictions']['mean_supported']
    fragment = dict(node_ids=np.asarray([f'node-{i}' for i in range(6)]),
        identities=nodes, sources=np.asarray(['1', '2', '1', '2', '3', '1']),
        target_identity_ids=targets, query_seconds=np.array([0., 60., 600., 2000.]),
        target=truth[indices], target_valid=valid[indices], native_states=native,
        nearest_seconds=nearest, support=support, exact_observation=nearest == 0,
        ols_mean_supported=mean, ols_membership=ordinary['membership'],
        ols_group_offsets=np.asarray([0, 2, 4, 5, 6]),
        ols_group_node_indices=np.asarray([0, 1, 2, 3, 4, 5]))
    system = evaluate_full_scene([dict(node_indices=g, states=mean[g[0]]) for g in groups],
        node_identities=nodes, target_identities=targets, target_states=truth, target_valid=valid,
        node_observed=support, split='V_select', grouping_kind='predicted',
        correspondence_policy='anchor_identity_first', return_artifact=True,
        node_input_support=dict(support=support, nearest_seconds=nearest))
    saved = system.pop('artifact')
    expected = dict(scene_id='1', component_id='1', unique_target={BASE: system})
    return fragment, saved, expected
