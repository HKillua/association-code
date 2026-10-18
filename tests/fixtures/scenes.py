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
