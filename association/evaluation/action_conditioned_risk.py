"""Frozen-risk transfer to motion-conditioned candidates; no labels or source priority."""
from __future__ import annotations
import numpy as np
from association.evaluation.history_candidates import source_balanced
from association.evaluation.native_temporal import spherical_position
from association.evaluation.native_motion_interventions import mean_velocity_control
from association.evaluation.source_risk import _position_distance


def candidates(m, events):
    n, q = m['support'].shape
    groups = [np.array([i]) for i in range(n)]
    mc, _, diagnostic = mean_velocity_control(m, events, groups, m['support'].astype(int), np.ones((n, q)), m['states'])
    velocity = np.asarray(m['ols_velocity_mps']).copy()
    bounds = np.empty((n, 2))
    known = np.zeros((n, q), bool)
    for node in range(n):
        a, b = events['event_offsets'][node:node + 2]
        t = events['event_times_seconds'][a:b]
        value = events['event_values'][a:b]
        valid = events['event_valid'][a:b, 2:4].all(1)
        bounds[node] = t[[0, -1]]
        if valid.sum() < 2:
            continue
        speed = value[valid, 2] * 1852 / 3600
        angle = np.deg2rad(value[valid, 3])
        velocity[node] = np.array([np.mean(speed * np.cos(angle)), np.mean(speed * np.sin(angle))])
        known[node] = (((m['query_seconds'] < t[0]) & valid[0]) | ((m['query_seconds'] > t[-1]) & valid[-1])) & ~m['support'][node]
    return mc, velocity, bounds, known, diagnostic['native_mean_computations']


def features(m, groups, count, original, mc, velocity, bounds, known):
    """Update only action-dependent coordinates against the ORIGINAL context."""
    x = original.copy()
    eligible = np.zeros(count.shape, bool)
    for gid, group in enumerate(groups):
        nodes = np.asarray(group, int)
        eligible[gid] = (count[gid] == 0) & ((m['query_seconds'] < bounds[nodes, 0].min()) | (m['query_seconds'] > bounds[nodes, 1].max()))
        for j in np.flatnonzero(eligible[gid]):
            mass = source_balanced(nodes, m['sources'])
            center = spherical_position(m['states'][nodes, j, :2], mass)
            center_v = mass @ m['ols_velocity_mps'][nodes]
            for node in nodes[known[nodes, j]]:
                own = nodes[m['sources'][nodes] == m['sources'][node]]
                own_center = spherical_position(m['states'][own, j, :2], np.ones(len(own)))
                x[node, j, 6] = np.log1p(_position_distance(mc[node, j], center) / 1000)
                x[node, j, 7] = np.log1p(_position_distance(mc[node, j], own_center) / 1000)
                x[node, j, 8] = np.log1p(np.linalg.norm(velocity[node] - center_v))
    if not np.isfinite(x).all():
        raise FloatingPointError('finite action features required')
    return x, eligible


def tie_mass(nodes, modes, sources):
    """Equal source, then node, then surviving mode mass; no numeric ID priority."""
    mass = np.zeros(len(nodes), float)
    classes = set(sources[nodes].tolist())
    for source in classes:
        ids = np.flatnonzero(sources[nodes] == source)
        unique = np.unique(nodes[ids])
        for node in unique:
            ix = ids[nodes[ids] == node]
            mass[ix] = 1 / len(classes) / len(unique) / len(ix)
    return mass


def readouts(m, groups, old_score, mc_score, mc, eligible, base):
    if not np.isfinite(mc_score).all() or mc_score.shape != old_score.shape:
        raise ValueError('finite same-shape action scores required')
    outputs = {key: base.copy() for key in ('CF', 'RM', 'JM')}
    weights = {key: np.zeros((*old_score.shape, 2)) for key in outputs}
    values = np.stack((m['states'][..., :2], mc[..., :2]), axis=2)
    for gid, group in enumerate(groups):
        nodes = np.asarray(group, int)
        for j in np.flatnonzero(eligible[gid]):
            original = nodes[old_score[nodes, j] == old_score[nodes, j].min()]
            for key in outputs:
                pool = original if key == 'CF' else nodes
                choices = np.array([1]) if key == 'RM' else np.array([0, 1])
                nn, mm = np.meshgrid(pool, choices, indexing='ij')
                nn, mm = nn.ravel(), mm.ravel()
                scores = np.where(mm == 0, old_score[nn, j], mc_score[nn, j])
                keep = scores == scores.min()
                nn, mm = nn[keep], mm[keep]
                mass = tie_mass(nn, mm, m['sources'])
                weights[key][nn, j, mm] = mass
                outputs[key][gid, j, :2] = spherical_position(values[nn, j, mm], mass)
    return outputs, weights
