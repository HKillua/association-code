"""Reference-only mixture opportunity diagnostics, never a prediction policy.

The grid oracle consumes target positions explicitly. Its weights must not be
used as legal predictions or described as learned contributions. The sufficient
statistics preserve node-level longitude resultants rather than averaging the
already normalized longitude angles of the two sources.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.grouping import _partition


EARTH_RADIUS_M = 6371000.
RESULTANT_EPSILON = 1e-12


def _finite(value, name):
    result = np.asarray(value)
    if result.dtype.kind not in 'iuf' or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite numeric values')
    return result.astype(np.float64, copy=False)


def source_position_moments(*, states, support, groups, pack):
    """Return [S,Q,3] latitude, mean sine longitude, mean cosine longitude.

    Only supported members enter each source mean. An unavailable source has
    three finite zeros and must be excluded by the caller's availability mask.
    The supplied source mapping and counts are bound to the physical partition.
    """
    states = _finite(states, 'states')
    support = np.asarray(support)
    if (states.ndim != 3 or states.shape[-1] != 4 or not states.shape[1]
            or support.shape != states.shape[:2] or support.dtype != bool):
        raise ValueError('states[N,Q,4] and boolean support[N,Q], Q>0 required')
    n, q = support.shape
    groups = _partition(groups, n)
    mapping = np.asarray(pack['node_to_source_row'])
    counts = np.asarray(pack['supported_counts'])
    offsets = np.asarray(pack['group_offsets'])
    source_groups = np.asarray(pack['source_group_indices'])
    available = np.asarray(pack['available'])
    s = len(counts)
    if (counts.shape != (s, q) or counts.dtype.kind not in 'iu' or (counts < 0).any()
            or mapping.shape != (n,) or mapping.dtype.kind not in 'iu'
            or (mapping < 0).any() or (mapping >= s).any()
            or offsets.shape != (len(groups)+1,) or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or offsets[-1] != s or (np.diff(offsets) <= 0).any()
            or source_groups.shape != (s,) or source_groups.dtype.kind not in 'iu'
            or available.shape != (s, q) or available.dtype != bool):
        raise ValueError('source pack shapes or mapping incompatible with physical inputs')
    actual_counts = np.zeros_like(counts)
    for gid, group in enumerate(groups):
        rows = mapping[group]
        if (not np.all((rows >= offsets[gid]) & (rows < offsets[gid+1]))
                or not np.all(source_groups[offsets[gid]:offsets[gid+1]] == gid)):
            raise ValueError('source rows must belong to their supplied group')
        np.add.at(actual_counts, rows, support[group].astype(np.int64))
    if (not np.array_equal(actual_counts, counts)
            or not np.array_equal(available, counts > 0)):
        raise ValueError('source support count/availability binding differs')
    longitude = np.deg2rad(states[..., 1])
    values = np.stack((states[..., 0], np.sin(longitude), np.cos(longitude)), axis=-1)
    moments = np.zeros((s, q, 3), np.float64)
    np.add.at(moments, mapping, values*support[..., None])
    moments /= np.maximum(counts[..., None], 1)
    if not np.isfinite(moments).all():
        raise FloatingPointError('nonfinite source position moments')
    return moments


def _pair_inputs(moment_pairs, anchors):
    moments, anchors = _finite(moment_pairs, 'moment_pairs'), _finite(anchors, 'anchors')
    if moments.ndim != 3 or moments.shape[1:] != (2, 3) or anchors.shape != (len(moments), 2):
        raise ValueError('moment_pairs[M,2,3] and anchors[M,2] required')
    return moments, anchors


def _positions(moments, alpha, anchors):
    """Broadcasting implementation shared by point and bounded grid calls."""
    first, second = moments[..., 0, :], moments[..., 1, :]
    mixed = second+alpha[..., None]*(first-second)
    # Preserve the explicit endpoint candidates without subtraction roundoff.
    mixed = np.where((alpha == 0)[..., None], second, mixed)
    mixed = np.where((alpha == 1)[..., None], first, mixed)
    resultant = np.hypot(mixed[..., 1], mixed[..., 2])
    longitude = (np.rad2deg(np.arctan2(mixed[..., 1], mixed[..., 2]))+180.) % 360.-180.
    longitude = np.where(resultant <= RESULTANT_EPSILON, anchors[..., 1], longitude)
    position = np.stack((mixed[..., 0], longitude), axis=-1)
    if not np.isfinite(position).all() or not np.isfinite(resultant).all():
        raise FloatingPointError('nonfinite mixture position or resultant')
    return position, resultant


def mixture_positions(moment_pairs, alpha, anchors):
    """Recover [M,2] positions; alpha is the first source's contribution.

    This physical helper accepts no reference. It uses the original group's
    longitude at an undefined circular resultant, matching node recovery.
    """
    moments, anchors = _pair_inputs(moment_pairs, anchors)
    alpha = _finite(alpha, 'alpha')
    if alpha.ndim == 0:
        alpha = np.full(len(moments), float(alpha))
    if alpha.shape != (len(moments),) or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError('alpha must be a scalar or [M] contributions in [0,1]')
    return _positions(moments, alpha, anchors)[0]


def _distance(position, truth):
    lat, target_lat = np.deg2rad(position[..., 0]), np.deg2rad(truth[..., 0])
    delta_lon = np.deg2rad((position[..., 1]-truth[..., 1]+180.) % 360.-180.)
    hav = np.clip(np.sin((lat-target_lat)/2.)**2
        + np.cos(lat)*np.cos(target_lat)*np.sin(delta_lon/2.)**2, 0., 1.)
    return 2.*EARTH_RADIUS_M*np.arctan2(np.sqrt(hav), np.sqrt(1.-hav))


def _continuous_gap(moments, grid_size):
    """Lipschitz grid-error gap; undefined across circular fallback regions."""
    first, second = moments[:, 0], moments[:, 1]
    base, delta = second[:, 1:], first[:, 1:]-second[:, 1:]
    length_squared = np.sum(delta*delta, axis=1)
    closest_alpha = np.clip(np.divide(-np.sum(base*delta, axis=1), length_squared,
        out=np.zeros(len(moments)), where=length_squared > 0), 0., 1.)
    closest = base+closest_alpha[:, None]*delta
    minimum = np.hypot(closest[:, 0], closest[:, 1])
    cross = base[:, 0]*delta[:, 1]-base[:, 1]*delta[:, 0]
    gap = np.full(len(moments), np.nan)
    certified = minimum > RESULTANT_EPSILON
    angular_rate = cross[certified]/minimum[certified]**2
    latitude_rate = np.deg2rad(first[certified, 0]-second[certified, 0])
    bound = EARTH_RADIUS_M*np.hypot(latitude_rate, angular_rate)
    if not np.isfinite(minimum).all() or not np.isfinite(bound).all():
        raise FloatingPointError('nonfinite continuous mixture gap calculation')
    gap[certified] = bound/(2.*(grid_size-1))+1e-6
    return minimum, gap


def grid_mixture_oracle(moment_pairs, anchors, truth, *, grid_size=4097, chunk_size=128):
    """Reference-only best fixed-grid mixture, with a continuous-error bound.

    All grid points, including both single-source endpoints, are evaluated once.
    Exact numerical error ties choose the grid point nearest 0.5, then its
    original increasing-alpha grid index. No optimizer or adaptive retry runs.
    The endpoint error columns follow source order: alpha=1, then alpha=0.
    ``gap_bound_m`` bounds grid error minus continuous optimum when certified;
    NaN explicitly preserves queries whose circular fallback prevents a bound.
    """
    moments, anchors = _pair_inputs(moment_pairs, anchors)
    truth = _finite(truth, 'truth')
    if truth.shape != (len(moments), 2):
        raise ValueError('truth must be [M,2] reference positions')
    for name, value, minimum, maximum in (('grid_size', grid_size, 2, 4097),
                                         ('chunk_size', chunk_size, 1, 128)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not minimum <= value <= maximum):
            raise ValueError(f'{name} must be an integer in [{minimum},{maximum}]')
    grid = np.linspace(0., 1., grid_size)
    # Integer distances avoid a rounding-dependent preference between tied
    # grid locations equidistant from the midpoint (including even grid sizes).
    order = np.lexsort((np.arange(grid_size), np.abs(2*np.arange(grid_size)-(grid_size-1))))
    m = len(moments)
    alpha, error, position = np.empty(m), np.empty(m), np.empty((m, 2))
    endpoints = np.empty((m, 2))
    selected_index, tie_count = np.empty(m, np.int64), np.empty(m, np.int64)
    degenerate_candidates = selected_degenerate = 0
    for start in range(0, m, chunk_size):
        stop = min(start+chunk_size, m)
        coordinates, resultant = _positions(moments[start:stop, None], grid[None, :], anchors[start:stop, None])
        distances = _distance(coordinates, truth[start:stop, None])
        if not np.isfinite(distances).all():
            raise FloatingPointError('nonfinite oracle grid error')
        chosen = order[np.argmin(distances[:, order], axis=1)]
        row = np.arange(stop-start)
        selected_index[start:stop] = chosen
        alpha[start:stop] = grid[chosen]
        error[start:stop] = distances[row, chosen]
        position[start:stop] = coordinates[row, chosen]
        endpoints[start:stop] = distances[:, [-1, 0]]
        tie_count[start:stop] = (distances == error[start:stop, None]).sum(axis=1)
        degenerate_candidates += int((resultant <= RESULTANT_EPSILON).sum())
        selected_degenerate += int((resultant[row, chosen] <= RESULTANT_EPSILON).sum())
    single = endpoints.min(axis=1)
    gain = single-error
    if (gain < -1e-7).any():
        raise FloatingPointError('oracle grid omitted a better single-source endpoint')
    minimum, gap = _continuous_gap(moments, grid_size)
    return dict(endpoint_errors_m=endpoints, best_single_error_m=single, grid_error_m=error,
        alpha=alpha, gain_m=gain, gap_bound_m=gap, min_resultant=minimum, position=position,
        grid_index=selected_index, tied_grid_count=tie_count,
        audit=dict(reference_only=True, grid_size=int(grid_size), chunk_size=int(chunk_size),
            query_count=m, grid_candidate_count=m*int(grid_size),
            grid_tie_query_count=int((tie_count > 1).sum()),
            max_tied_grid_count=int(tie_count.max()) if m else 0,
            endpoint_selected_count=int(((selected_index == 0) | (selected_index == grid_size-1)).sum()),
            interior_selected_count=int(((selected_index > 0) & (selected_index < grid_size-1)).sum()),
            degenerate_grid_candidate_count=degenerate_candidates,
            degenerate_selected_count=selected_degenerate,
            certified_gap_count=int(np.isfinite(gap).sum()), uncertified_gap_count=int(np.isnan(gap).sum()),
            tie_rule='exact error; nearest midpoint, then increasing original grid index'))
