"""Independent, identity-blind C74 readout audit on the original V32.

Only the population and raw scene loaders are shared with science. Augmented
least squares reconstructs both fits, and every output and branch mask is
checked. Identity/reference fields and physical evaluation results are unused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np


EXPECTED_SCENES = 32
EXPECTED_COMPONENTS = 9
MEMORY_BYTES = 8 * 1024**3
STATE_ATOL = 1e-6
STATE_RTOL = 1e-8
SCALE = 6371000. * np.pi / 180.
MASKS = ('active', 'internal_gap', 'near_endpoint', 'B_low_speed', 'C_low_speed')
FIELDS = {'A', 'B', 'C', 'membership', *MASKS}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def peak_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))


def groups_of(membership):
    member = np.asarray(membership)
    require(member.ndim == 1 and member.dtype.kind in 'iu' and (member >= 0).all(), 'complete integer membership')
    return sorted((np.flatnonzero(member == value) for value in np.unique(member)), key=lambda nodes: int(nodes[0]))


def basis(fraction):
    t = np.asarray(fraction, np.float64).clip(0., 1.)
    return np.maximum(1. - 4. * np.abs(t[:, None] - np.linspace(0., 1., 5)), 0.)


def fit(design, position, weights):
    """Direct weighted observations plus explicit penalty rows; no Gram cache."""
    root = np.sqrt(weights)[:, None]
    penalty = np.vstack((.1 * np.diff(np.eye(5), n=2, axis=0), np.sqrt(1e-9)*np.eye(5)))
    matrix = np.vstack((design * root, penalty))
    target = np.vstack((position * root, np.zeros((len(penalty), 2))))
    beta, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    require(rank == 5 and np.isfinite(beta).all(), 'finite full-rank independent ridge fit')
    return beta


def reconstruct(scene):
    nodes, sources, support, nearest, membership, native, baseline = scene
    nodes, sources, support, nearest, membership, baseline = map(
        np.asarray, (nodes, sources, support, nearest, membership, baseline))
    query, times, latlon, offsets = map(np.asarray, native)
    n = len(nodes)
    require(nodes.shape == sources.shape == membership.shape == (n,) and len(np.unique(nodes)) == n,
            'unique original nodes and aligned source/membership vectors')
    require(support.shape == nearest.shape == (n, 64) and support.dtype == bool
            and np.isfinite(nearest).all() and (nearest >= 0).all(), 'original support and nearest seconds')
    require(query.shape == (64,) and np.isfinite(query).all() and (np.diff(query) > 0).all(), 'fixed increasing Q64')
    require(offsets.shape == (n+1,) and offsets.dtype.kind in 'iu' and offsets[0] == 0
            and offsets[-1] == len(times) and (np.diff(offsets) >= 2).all(), 'original native event offsets')
    require(latlon.shape == (len(times), 2) and np.isfinite(times).all() and np.isfinite(latlon).all(),
            'finite native positions and times')
    require((np.abs(latlon) <= [90., 180.]).all(), 'legal native geographic coordinates')
    require(times.min() >= query[0]-1e-8 and times.max() <= query[-1]+1e-8, 'native events inside query window')
    for start, stop in zip(offsets[:-1], offsets[1:]):
        require((np.diff(times[start:stop]) > 0).all(), 'strict per-fragment native time order')
    groups = groups_of(membership)
    require(baseline.shape == (len(groups), 64, 4) and np.isfinite(baseline).all(), 'complete mps baseline states')
    duration = float(query[-1]-query[0])
    query_fraction = (query-query[0])/duration
    event_fraction = (times-query[0])/duration
    query_design, event_design = basis(query_fraction), basis(event_fraction)
    origin = latlon[int(np.argmin(times))]
    scale = SCALE * np.array([1., np.cos(np.deg2rad(origin[0]))])
    require(scale[1] > 1e-9, 'defined east coordinate scale')

    def project(position):
        delta = np.asarray(position)-origin
        delta[:, 1] = (delta[:, 1]+180.) % 360.-180.
        return delta*scale

    native_position = project(latlon)
    native_node = np.repeat(np.arange(n), np.diff(offsets))
    native_weight = 1./np.diff(offsets)[native_node]
    intervals = np.minimum(np.floor(query_fraction.clip(0., 1.)*4.).astype(int), 3)
    output = dict(A=baseline.copy(), B=baseline.copy(), C=baseline.copy(), membership=membership.copy())
    output.update({name: np.zeros((len(groups), 64), bool) for name in MASKS})
    counts = dict(groups=len(groups), active_groups=0, inactive_groups=0, eligible_q64_samples=0,
                  independent_fits=0, native_events=len(times), B_total_weight=0., C_total_weight=0.)
    for group_index, members in enumerate(groups):
        rows = np.concatenate([np.arange(offsets[node], offsets[node+1]) for node in members])
        start, stop = float(times[rows].min()), float(times[rows].max())
        inside = (query >= start) & (query <= stop)
        union = support[members].any(axis=0)
        eligible = inside & union
        counts['eligible_q64_samples'] += int(eligible.sum())
        if int(eligible.sum()) < 2:
            counts['inactive_groups'] += 1
            continue
        counts['active_groups'] += 1
        output['active'][group_index] = inside
        output['internal_gap'][group_index] = inside & ~union
        output['near_endpoint'][group_index] = inside & (
            (query-start <= query[1]-query[0]) | (stop-query <= query[1]-query[0]))
        grid_weight = np.full(int(eligible.sum()), len(members)/int(eligible.sum()), np.float64)
        np.testing.assert_allclose(grid_weight.sum(), native_weight[rows].sum(), rtol=0., atol=1e-10)
        beta_b = fit(query_design[eligible], project(baseline[group_index, eligible, :2]), grid_weight)
        beta_c = fit(event_design[rows], native_position[rows], native_weight[rows])
        counts['independent_fits'] += 2
        counts['B_total_weight'] += float(grid_weight.sum())
        counts['C_total_weight'] += float(native_weight[rows].sum())
        for arm, beta in (('B', beta_b), ('C', beta_c)):
            velocity = (beta[intervals+1]-beta[intervals]) * (4./duration)
            speed = np.sqrt(np.sum(velocity**2, axis=1))
            heading = np.rad2deg(np.arctan2(velocity[:, 1], velocity[:, 0])) % 360.
            low = inside & (speed < .5)
            output[arm+'_low_speed'][group_index] = low
            output[arm][group_index, inside, 2] = speed[inside]
            use_heading = inside & ~low
            output[arm][group_index, use_heading, 3] = heading[use_heading]
    return output, counts


def compare_payload(actual, expected):
    require(set(actual) == FIELDS, 'exact complete saved readout schema')
    np.testing.assert_array_equal(actual['membership'], expected['membership'])
    np.testing.assert_array_equal(actual['A'], expected['A'])
    for name in MASKS:
        require(actual[name].dtype == bool, 'boolean '+name)
        np.testing.assert_array_equal(actual[name], expected[name])
    maximum = np.zeros(2)
    for arm in ('B', 'C'):
        values, target = actual[arm], expected[arm]
        require(values.shape == target.shape and values.dtype.kind == 'f' and np.isfinite(values).all(), 'finite '+arm+' states')
        np.testing.assert_array_equal(values[..., :2], expected['A'][..., :2])
        inactive = ~expected['active']
        np.testing.assert_array_equal(values[inactive], expected['A'][inactive])
        low = expected[arm+'_low_speed']
        np.testing.assert_array_equal(values[..., 3][low], expected['A'][..., 3][low])
        modeled_heading = expected['active'] & ~low
        require((values[..., 2][expected['active']] >= 0.).all(), 'nonnegative modeled speed')
        require(((values[..., 3][modeled_heading] >= 0.) & (values[..., 3][modeled_heading] < 360.)).all(),
                'normalized modeled circular heading')
        error = values[..., 2:]-target[..., 2:]
        error[..., 1] = (error[..., 1]+180.) % 360.-180.
        tolerance = STATE_ATOL + STATE_RTOL*np.abs(target[..., 2:])
        require((np.abs(error) <= tolerance).all(), 'independent full '+arm+' readout reconstruction')
        maximum = np.maximum(maximum, np.max(np.abs(error), axis=(0, 1)))
    return maximum.tolist()


def audit(runs, science, output, *, source=None):
    runs, science, output = (Path(value).resolve() for value in (runs, science, output))
    require(output != runs and output != science and science not in output.parents, 'fresh separate audit output')
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = dict(status='RUNNING', exit_code=None, started_at=datetime.now(timezone.utc).isoformat(),
        runs=str(runs), science=str(science), scenes=[], max_abs_speed_heading_difference=[0., 0.],
        totals=dict(scenes=0, groups=0, active_groups=0, inactive_groups=0, native_events=0,
                    independent_fits=0, checked_state_values=0, checked_mask_values=0),
        tolerance=dict(state_atol=STATE_ATOL, state_rtol=STATE_RTOL))

    def checkpoint():
        report.update(elapsed_seconds=time.monotonic()-began, peak_rss_bytes=peak_rss_bytes())
        with (output/'audit.json').open('w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')

    try:
        checkpoint()
        if source is None:
            from tools.analysis import motion_readout_pilot as source
        record = read_json(science/'run.json')
        require(record['status'] == 'COMPLETED' and record['exit_code'] == 0, 'completed scientific output')
        ids, components = source.load_population(runs)
        require(len(ids) == len(set(ids)) == EXPECTED_SCENES and all(isinstance(sid, str) for sid in ids), 'original V32 population')
        require(len({str(components[sid]) for sid in ids}) == EXPECTED_COMPONENTS, 'original V9 components')
        require(record['scene_ids'] == ids and record['split'] == 'V_select', 'bound V population/order and split')
        for sid in ids:
            scene = source.load_scene(runs, sid)
            expected, counts = reconstruct(scene)
            actual = read_arrays(science/'private'/f'{sid}.npz')
            maximum = compare_payload(actual, expected)
            report['max_abs_speed_heading_difference'] = np.maximum(report['max_abs_speed_heading_difference'], maximum).tolist()
            low_b, low_c, active = (expected[name] for name in ('B_low_speed', 'C_low_speed', 'active'))
            branches = {f'B{b}_C{c}': int((active & (low_b == bool(b)) & (low_c == bool(c))).sum())
                        for b in (0, 1) for c in (0, 1)}
            report['scenes'].append(dict(scene_id=sid, component_id=str(components[sid]), **counts,
                active_queries=int(active.sum()), internal_gap_queries=int(expected['internal_gap'].sum()),
                near_endpoint_queries=int(expected['near_endpoint'].sum()), low_speed_branches=branches))
            report['totals']['scenes'] += 1
            for name in ('groups', 'active_groups', 'inactive_groups', 'native_events', 'independent_fits'):
                report['totals'][name] += counts[name]
            report['totals']['checked_state_values'] += sum(expected[arm].size for arm in ('A', 'B', 'C'))
            report['totals']['checked_mask_values'] += sum(expected[name].size for name in MASKS)
            checkpoint()
            require(peak_rss_bytes() <= MEMORY_BYTES, 'RSS within 8GiB')
        report.update(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat(),
            scope='shared original population/scene loaders including old127 baseline/unit conversion; independent chart, basis, augmented weighted least-squares for every active B/C group, physical-time slopes, five masks and every predicted state; A, positions, membership and prescribed fallback values exact; no identities/reference, evaluation errors, strata based on truth, or public scientific summaries read or certified; no proof of process-level historical input isolation')
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}', finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        checkpoint()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'science', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    from tools.analysis import motion_readout_pilot as source
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    audit(args.runs, args.science, args.output, source=source)
