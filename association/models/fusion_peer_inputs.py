"""Declared measurement-only interfaces for new-unit peer comparisons.

PTFM keeps two source branches on contiguous common support. DeepAF retains its
P/V core, with visible Anchor-OLS bootstrap instead of the old fixed095 mixture.
Neither interface accepts identity, reference or fixed source-priority weights.
"""
from __future__ import annotations

import numpy as np
import torch

from association.evaluation.fusion_controls import ordinary_fusion
from association.models.ptfm import local_states
from association.models import deepaf_peer as deep


def group_mean(motion, members):
    members = np.asarray(members, np.int64)
    if len(members) == 1:
        return np.asarray(motion['states'][members[0]], np.float64).copy()
    return ordinary_fusion(motion['states'][members], motion['nearest_seconds'][members],
        motion['support'][members], [list(range(len(members)))])['predictions']['mean_supported'][0]


def sensor_segments(motion, members):
    """Two anonymous source equality classes, ordered by first input occurrence.

    Returns every contiguous query segment with both sources supported. A
    same-source multi-fragment pool is an explicit MTAD adaptation; original
    two-track inputs reduce exactly to their two independent streams.
    """
    members = np.asarray(members, np.int64)
    if members.ndim != 1 or not len(members) or len(set(members.tolist())) != len(members):
        raise ValueError('distinct nonempty members required')
    sources = motion['sources'][members]
    names = list(dict.fromkeys(str(s) for s in sources))
    if len(names) != 2:
        return []
    local = [members[sources == s] for s in names]
    supported = [motion['support'][g].any(0) for g in local]
    common = supported[0] & supported[1]
    if not common.any():
        return []
    states = [group_mean(motion, g) for g in local]
    edges = np.diff(np.r_[False, common, False].astype(np.int8))
    return [dict(query_indices=np.arange(a, b), left=states[0][a:b], right=states[1][a:b])
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def ptfm_segment_inputs(segment, *, device='cpu', swapped=False):
    left, right = segment['left'], segment['right']
    if swapped:
        left, right = right, left
    # One common visible origin separates branch order from output coordinates.
    origin = np.asarray(segment['left'][0], np.float64).copy()
    context = torch.ones((len(left), 2), dtype=torch.float32, device=device)
    return (torch.as_tensor(local_states(left, origin), dtype=torch.float64, device=device),
            torch.as_tensor(local_states(right, origin), dtype=torch.float64, device=device), context,
            torch.as_tensor(origin, dtype=torch.float64, device=device))


def deepaf_visible_inputs(motion, members, *, device='cpu'):
    """No fixed095 initializer; full Q64 offline OLS anchor remains available.

    Always-available anchor and target finite-difference V are task adaptations,
    not claims about the original online association/lifecycle implementation.
    """
    members = np.asarray(members, np.int64)
    mean = group_mean(motion, members)
    base = np.asarray(motion['states'][members[0]], np.float64).copy()
    base[:, 2:] = mean[:, 2:]
    origin = torch.as_tensor(base[0, :2], dtype=torch.float64, device=device)
    positions = torch.as_tensor(motion['states'][members, :, :2], dtype=torch.float64, device=device)
    support = torch.as_tensor(motion['support'][members], dtype=torch.bool, device=device).clone()
    count = support[1:].sum(0)
    support[0] = True
    angular = torch.as_tensor(motion['ols_slope_degrees_per_second'][members[0]], dtype=torch.float64, device=device)
    velocity = deep.EARTH_RADIUS_M*torch.deg2rad(angular)*torch.stack((origin.new_tensor(1.), torch.deg2rad(origin[0]).cos()))
    base = torch.as_tensor(base, dtype=torch.float64, device=device)
    return dict(base_states=base, mean_states=torch.as_tensor(mean, dtype=torch.float64, device=device),
        measurements_m=deep.positions_to_local(positions, origin[None, None]), candidate_valid=support,
        query_seconds=torch.as_tensor(motion['query_seconds'], dtype=torch.float64, device=device),
        initial_velocity_mps=velocity, origin_latlon=origin,
        base_positions_m=deep.positions_to_local(base[:, :2], origin[None]), helper_support_count=count)
