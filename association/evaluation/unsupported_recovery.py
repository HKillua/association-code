"""145 physical-only endpoint/velocity interventions outside group support."""
from __future__ import annotations

import numpy as np

from association.evaluation.query_contribution import recover_group_states

NEW = ('WS_OBS_ZERO_CVP_OLSV', 'WS_OBS_ZERO_NATIVEP_CVV',
       'WS_OBS_ZERO_NATIVEP_OLSV', 'WS_OBS_ZERO_HOLD',
       'WS_Q_ZERO_CVP_OLSV', 'WS_Q_ZERO_ALLSOURCE_CVP_OLSV')
OBS = 'WS_OBS_SCALE_RTS_EQUAL'
QHARD = 'WS_SHARED_Q_HARD'


def outside_positions(times, query, trace, ols_velocity):
    """Use the exact original meter chart; no labels, identity or tunable cap."""
    times, query, velocity = (np.asarray(x, float) for x in (times, query, ols_velocity))
    if (len(times) < 2 or np.any(np.diff(times) <= 0) or velocity.shape != (2,)
            or not all(np.isfinite(x).all() for x in (times, query, velocity))):
        raise ValueError('finite native times/query and north/east velocity required')
    endpoints = np.asarray(trace['RTS']['endpoints'], float)
    native = np.asarray(trace['native_local'], float)[[0, -1]]
    origin, scale = np.asarray(trace['chart_origin_latlon']), np.asarray(trace['meters_per_degree'])
    result = {name: np.asarray(trace['RTS']['positions']).copy()
              for name in ('CVP_OLSV', 'NATIVEP_CVV', 'NATIVEP_OLSV', 'HOLD')}
    for side, mask in ((0, query < times[0]), (1, query > times[-1])):
        dt = query[mask]-times[[0, -1]][side]
        specs = {'CVP_OLSV': (endpoints[side, :2], velocity),
                 'NATIVEP_CVV': (native[side], endpoints[side, 2:]),
                 'NATIVEP_OLSV': (native[side], velocity),
                 'HOLD': (endpoints[side, :2], np.zeros(2))}
        for name, (position, rate) in specs.items():
            local = position+dt[:, None]*rate
            physical = origin+local/scale
            physical[:, 1] = (physical[:, 1]+180.) % 360.-180.
            result[name][mask] = physical
    if any(not np.isfinite(x).all() or (np.abs(x[:, 0]) > 90).any() for x in result.values()):
        raise FloatingPointError('illegal extrapolated latitude or nonfinite position; no clipping')
    return result


def assemble(motion, groups, observation_states, qhard_states, node_positions):
    """Apply declared interventions to all zero-support group queries only."""
    support, sources = motion['support'], motion['sources']
    zero = np.asarray([~support[g].any(0) for g in groups])
    result = {name: (qhard_states if name in NEW[4:] else observation_states).copy() for name in NEW}
    factor_keys = ('CVP_OLSV', 'NATIVEP_CVV', 'NATIVEP_OLSV', 'HOLD', 'CVP_OLSV')
    for gid, group in enumerate(groups):
        if not zero[gid].any():
            continue
        anchor = int(group[0])
        for name, key in zip(NEW[:5], factor_keys):
            result[name][gid, zero[gid], :2] = node_positions[anchor][key][zero[gid]]
    nodes = motion['states'].copy()
    mass = np.zeros(support.shape, float)
    for gid, group in enumerate(groups):
        if not zero[gid].any():
            continue
        identities = list(dict.fromkeys(sources[group].tolist()))
        for source in identities:
            members = group[sources[group] == source]
            mass[np.ix_(members, np.flatnonzero(zero[gid]))] = 1./(len(identities)*len(members))
        for node in group:
            nodes[node, zero[gid], :2] = node_positions[int(node)]['CVP_OLSV'][zero[gid]]
    fused = recover_group_states(nodes, mass, groups)['group_states']
    result[NEW[5]][zero, :2] = fused[zero, :2]
    for name, value in result.items():
        baseline = qhard_states if name in NEW[4:] else observation_states
        if not np.array_equal(value[~zero], baseline[~zero]) or not np.array_equal(value[..., 2:], baseline[..., 2:]):
            raise ValueError('support-in or non-position output changed')
    return result, zero
