"""142: independently bind zero-support outputs and replay their CV endpoints.

No producer imports, new model, rho selection, changed output or sealed data.
The fixed recurrence is transcribed from the published 127 working CV model.
Only anchors needed by the complete zero-support diagnostic population replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

for setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(setting, '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

RADIUS = 6371000.
IDEA = 'IDEA-STATE-RECOVERY-TAIL-CONTINUATION-20261030'
BASE = 'WS_SOURCE_EQUAL'
CV_NAMES = tuple(f'WS_{label}_SCALE_{direction}_EQUAL'
                 for label in ('REFERENCE', 'OBS') for direction in ('FORWARD', 'RTS'))
NAMES = (BASE,) + CV_NAMES


def read(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write(path, data):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=None)
        stream.write('\n')


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def require(value, label):
    if not value:
        raise ValueError(label)


def same(left, right, label, *, numeric=False):
    a, b = np.asarray(left), np.asarray(right)
    require(a.shape == b.shape, label+' shape')
    equal = (np.allclose(a, b, rtol=1e-10, atol=1e-9, equal_nan=True)
             if numeric else np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f'
             else np.array_equal(a, b))
    require(equal, label+' values')


def distance(a, b):
    a, b = np.deg2rad(np.asarray(a)), np.deg2rad(np.asarray(b))
    def xyz(p):
        return np.stack((np.cos(p[..., 0])*np.cos(p[..., 1]),
                         np.cos(p[..., 0])*np.sin(p[..., 1]), np.sin(p[..., 0])), axis=-1)
    x, y = xyz(a), xyz(b)
    return RADIUS*np.arctan2(np.linalg.norm(np.cross(x, y), axis=-1), np.sum(x*y, axis=-1))


def describe(values):
    x = np.asarray(values, float)
    require(np.isfinite(x).all(), 'finite diagnostic statistics')
    return dict(count=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                p95=float(np.quantile(x, .95)) if len(x) else None,
                p99=float(np.quantile(x, .99)) if len(x) else None,
                max=float(x.max()) if len(x) else None)


def error_stats(values):
    x = np.asarray(values, float)
    return dict(describe(x), mae=float(x.mean()) if len(x) else None,
                mse=float(np.mean(x*x)) if len(x) else None,
                rmse=float(np.sqrt(np.mean(x*x))) if len(x) else None)


def trace_cv(times, latlon, query, rho):
    """Fixed-model replay retaining endpoint states in the original meter chart."""
    times, latlon, query = map(lambda x: np.asarray(x, float), (times, latlon, query))
    require(len(times) >= 2 and np.all(np.diff(times) > 0) and np.all(np.diff(query) > 0), 'increasing trace times')
    require(latlon.shape == (len(times), 2) and np.isfinite(latlon).all(), 'trace positions')
    scale = RADIUS*np.pi/180
    east = scale*np.cos(np.deg2rad(latlon[:, 0].mean()))
    require(east > 1e-9 and np.isfinite(rho) and rho > 0, 'trace chart/rho')
    raw = np.diff(latlon[:, 1])
    delta = (raw+180.) % 360.-180.
    delta = np.where(np.abs(delta) == 180., np.copysign(180., raw), delta)
    unwrapped = latlon[0, 1]+np.r_[0., np.cumsum(delta)]
    y = np.column_stack(((latlon[:, 0]-latlon[0, 0])*scale, (unwrapped-latlon[0, 1])*east))
    inside = (query >= times[0]) & (query <= times[-1])
    timeline = np.unique(np.r_[times, query[inside]])
    indices = np.full(len(timeline), -1, np.int64)
    indices[np.searchsorted(timeline, times)] = np.arange(len(times))
    h, eye, e2 = np.eye(4)[:2], np.eye(4), np.eye(2)
    x, p = np.r_[y[0], 0., 0.], np.diag([1., 1., 1/3600., 1/3600.])
    xf, pf, xp, pp = (np.empty((len(timeline), 4)), np.empty((len(timeline), 4, 4)),
                       np.empty((len(timeline), 4)), np.empty((len(timeline), 4, 4)))
    xf[0], pf[0], xp[0], pp[0] = x, p, x, p
    def transition(dt):
        require(dt >= 0, 'no negative-time covariance')
        return (np.block([[e2, dt*e2], [np.zeros((2, 2)), e2]]),
                rho*np.block([[dt**3/3*e2, dt**2/2*e2], [dt**2/2*e2, dt*e2]]))
    def covariance(value):
        value = (value+value.T)/2
        np.linalg.cholesky(value)
        return value
    for i in range(1, len(timeline)):
        f, q = transition(timeline[i]-timeline[i-1])
        x, p = f@x, covariance(f@p@f.T+q)
        xp[i], pp[i] = x, p
        if indices[i] >= 0:
            k = np.linalg.solve(h@p@h.T+e2, h@p).T
            x = x+k@(y[indices[i]]-h@x)
            residual = eye-k@h
            p = covariance(residual@p@residual.T+k@k.T)
        xf[i], pf[i] = x, p
    xs = xf.copy()
    for i in range(len(timeline)-2, -1, -1):
        f, _ = transition(timeline[i+1]-timeline[i])
        gain = np.linalg.solve(pp[i+1], f@pf[i]).T
        xs[i] = xf[i]+gain@(xs[i+1]-xp[i+1])
    result = {}
    for name, states in (('FORWARD', xf), ('RTS', xs)):
        positions = np.empty((len(query), 2))
        positions[inside] = states[np.searchsorted(timeline, query[inside]), :2]
        for use, endpoint in ((query < times[0], 0), (query > times[-1], -1)):
            positions[use] = states[endpoint, :2]+(query[use]-times[endpoint])[:, None]*states[endpoint, 2:]
        physical = np.column_stack((latlon[0, 0]+positions[:, 0]/scale,
                                   (latlon[0, 1]+positions[:, 1]/east+180.) % 360.-180.))
        result[name] = dict(positions=physical, endpoints=states[[0, -1]].copy())
    result.update(native_local=y, chart_origin_latlon=latlon[0].copy(),
                  meters_per_degree=np.asarray([scale, east]), native_event_count=len(times), transition_steps=len(timeline)-1,
                  rts_backward_steps=len(timeline)-1, kalman_observation_updates=len(times)-1)
    return result


def inspect_scene(state_run, native_run, split, sid, expected):
    saved = load(state_run/'private'/f'native-state-{split}-{sid}.npz')
    payload = load(state_run/'private'/f'states-{split}-{sid}.npz')
    motion = load(native_run/'private/native-motion'/split/f'scene-{sid}.npz')
    native = load(native_run/'private/native-events'/split/f'scene-{sid}.npz')
    ev = load(native_run/'private/native-evaluation'/split/f'scene-{sid}.npz')
    for source in (motion, native, ev):
        for key in ('node_ids', 'query_seconds'):
            same(saved[key], source[key], 'original '+key)
    same(motion['sources'], native['sources'], 'original source registry')
    offsets = native['event_offsets']
    require(offsets.shape == (len(motion['states'])+1,) and offsets[0] == 0
            and offsets[-1] == len(native['event_times_seconds']) and np.all(np.diff(offsets) >= 2), 'complete native offsets')
    independent_support = np.zeros_like(motion['support'])
    for node, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        times = native['event_times_seconds'][a:b]
        require(np.isfinite(times).all() and np.all(np.diff(times) > 0), 'complete native times')
        require(native['event_valid'][a:b, :2].all(), 'complete visible native position events')
        same(native['event_node_ids'][a:b], np.repeat(native['node_ids'][node], b-a), 'native event node lineage')
        same(native['event_canonical_indices'][a:b], np.arange(b-a), 'native event ordering')
        independent_support[node] = (saved['query_seconds'] >= times[0]) & (saved['query_seconds'] <= times[-1])
    same(independent_support, motion['support'], 'independent all-node span support')
    art = {name: {key[len(name)+1:]: val for key, val in payload.items() if key.startswith(name+'_')}
           for name in NAMES}
    base = art[BASE]
    same(base['target_ids'], ev['target_ids'], 'target registry')
    count = len(ev['target_ids'])
    same(base['aligned_truth'][:count], ev['target_states'], 'original reference')
    same(base['aligned_truth_valid'][:count], ev['target_valid'], 'original reference mask')
    same(base['aligned_target_ids'][:count], ev['target_ids'], 'aligned target registry')
    require(not base['aligned_truth_valid'][count:].any(), 'duplicate rows have no reference')
    require(str(base['split']) == split and str(base['correspondence_policy']) == 'anchor_identity_first', 'fixed correspondence')
    groups = [base['group_node_indices'][a:b] for a, b in zip(base['group_offsets'][:-1], base['group_offsets'][1:])]
    same(np.sort(np.concatenate(groups)), np.arange(len(motion['states'])), 'node partition')
    source_count = np.zeros((len(groups), len(saved['query_seconds'])), np.int64)
    for gid, group in enumerate(groups):
        require(np.all(np.diff(group) > 0), 'original ordered group')
        same(saved['membership'][group], np.repeat(saved['membership'][group[0]], len(group)), 'saved membership')
        for source in np.unique(motion['sources'][group]):
            source_count[gid] += motion['support'][group[motion['sources'][group] == source]].any(0)
    selected = base['aligned_group_indices'][:count]
    counts = np.full((count, len(saved['query_seconds'])), -1, np.int64)
    registry = {str(target): i for i, target in enumerate(ev['target_ids'])}
    identity_support = np.zeros_like(counts, bool)
    for node, identity in enumerate(ev['identities']):
        identity_support[registry[str(identity)]] |= independent_support[node]
    first = {}
    for gid, group in enumerate(groups):
        first.setdefault(registry[str(ev['identities'][group[0]])], gid)
    same(selected, np.asarray([first.get(i, -1) for i in range(count)]), 'first group mapping')
    counts[selected >= 0] = source_count[selected[selected >= 0]]
    valid = base['aligned_truth_valid'][:count, :, :2].all(-1) & base['aligned_prediction_valid'][:count, :, :2].all(-1)
    zero = valid & (counts == 0)
    strata = dict(inside_support=valid & identity_support, outside_support=valid & ~identity_support,
                  source_support_zero=zero, source_support_one=valid & (counts == 1),
                  source_support_multi=valid & (counts >= 2))
    errors = {}
    for name, value in art.items():
        for key in base:
            if key not in ('aligned_prediction', 'group_states'):
                same(value[key], base[key], 'fixed artifact '+key)
        for aligned, gid in enumerate(value['aligned_group_indices']):
            if gid >= 0:
                same(value['aligned_prediction'][aligned], value['group_states'][gid], 'aligned prediction/group output', numeric=True)
                same(value['aligned_prediction_valid'][aligned], value['group_valid'][gid], 'aligned valid/group output')
            else:
                require(not value['aligned_prediction_valid'][aligned].any(), 'unmatched output unavailable')
        errors[name] = np.full(valid.shape, np.nan)
        errors[name][valid] = distance(value['aligned_prediction'][:count, :, :2][valid], ev['target_states'][..., :2][valid])
        for key, computed in error_stats(errors[name][valid]).items():
            if key in ('mae', 'mse', 'rmse', 'p95', 'p99'):
                require(np.isclose(computed, expected['methods'][name]['position']['all'][key], rtol=1e-9, atol=1e-6), 'original primary metric '+key)
        for subset, subset_mask in strata.items():
            original = expected['methods'][name]['fixed_target_strata'][subset]
            computed = error_stats(errors[name][subset_mask])
            require(computed['count'] == original['point_count'], 'independent stratum population '+subset)
            for key in ('mae', 'mse', 'rmse', 'p95', 'p99'):
                require(original[key] is None if computed[key] is None else
                        np.isclose(computed[key], original[key], rtol=1e-9, atol=1e-6), 'independent stratum metric '+subset+'/'+key)
    indices = np.argwhere(zero)
    anchors = sorted({int(groups[selected[target]][0]) for target, _ in indices})
    traces, private_trace = {}, {}
    work = dict(replay_calls=0, native_events=0, transition_steps=0, rts_backward_steps=0)
    for node in anchors:
        a, b = native['event_offsets'][node:node+2]
        times, values = native['event_times_seconds'][a:b], native['event_values'][a:b]
        native_support = (saved['query_seconds'] >= times[0]) & (saved['query_seconds'] <= times[-1])
        same(native_support, motion['support'][node], 'native support span')
        for label, rho_index in (('REFERENCE', int(saved['reference_rho_index'])), ('OBS', int(saved['obs_rho_index']))):
            rho = float(saved['unique_rhos'][rho_index])
            trace = trace_cv(times, values[:, :2], saved['query_seconds'], rho)
            for direction in ('FORWARD', 'RTS'):
                same(trace[direction]['positions'], saved[f'rho_{rho_index}__{direction.lower()}_positions'][node], 'independent CV replay', numeric=True)
                private_trace[f'node_{node}__{label}__{direction}__endpoint_states'] = trace[direction]['endpoints']
            for key, val in (('native_endpoint_times', times[[0, -1]]), ('native_local_endpoints', trace['native_local'][[0, -1]]),
                             ('chart_origin_latlon', trace['chart_origin_latlon']), ('meters_per_degree', trace['meters_per_degree'])):
                private_trace[f'node_{node}__{label}__{key}'] = val
            traces[node, label] = trace
            work['replay_calls'] += 1
            for dest, src in (('native_events', 'native_event_count'), ('transition_steps', 'transition_steps'), ('rts_backward_steps', 'rts_backward_steps')):
                work[dest] += trace[src]
    cases = []
    for target, qi in indices:
        gid, query = int(selected[target]), float(saved['query_seconds'][qi])
        group, node = groups[gid], int(groups[gid][0])
        a, b = native['event_offsets'][node:node+2]
        times, values = native['event_times_seconds'][a:b], native['event_values'][a:b]
        side = 0 if query < times[0] else 1
        require(query < times[0] or query > times[-1], 'zero source anchor outside native span')
        endpoint = 0 if side == 0 else -1
        horizon = query-times[endpoint]
        nearest = float(motion['nearest_seconds'][group, qi].min())
        local = traces[node, 'OBS']['native_local']
        start = 0 if side == 0 else len(times)-2
        endpoint_rate = float(np.linalg.norm((local[start+1]-local[start])/(times[start+1]-times[start])))
        case = dict(scene_id=sid, split=split, side='before' if side == 0 else 'after', horizon_seconds=horizon,
            group_nearest_seconds=nearest, identity_span_inside=bool(identity_support[target, qi]),
            pure_group=bool(np.all(ev['identities'][group] == ev['identities'][node])), group_size=len(group),
            native_event_count=len(times), native_span_seconds=float(times[-1]-times[0]),
            native_gap_max_seconds=float(np.diff(times).max()), endpoint_difference_speed_mps=endpoint_rate,
            input_ols_speed_mps=float(np.linalg.norm(motion['ols_velocity_mps'][node])),
            reported_endpoint_native_speed=float(values[endpoint, 2]) if native['event_valid'][a:b][endpoint, 2] else None,
            _target_row=int(target), _group=gid, _anchor_node=node, _query_index=int(qi),
            _native_endpoint_latlon=values[endpoint, :2].tolist(), _reference_at_query=ev['target_states'][target, qi, :2].tolist())
        same(art[BASE]['group_states'][gid, qi, :2], motion['states'][node, qi, :2], 'zero original OLS fallback', numeric=True)
        for name in NAMES:
            case[name+'__error_m'] = float(errors[name][target, qi])
        for label in ('REFERENCE', 'OBS'):
            for direction in ('FORWARD', 'RTS'):
                name = f'WS_{label}_SCALE_{direction}_EQUAL'
                trace = traces[node, label]
                state = trace[direction]['endpoints'][side]
                predicted = trace[direction]['positions'][qi]
                same(art[name]['group_states'][gid, qi, :2], predicted, 'zero CV output maps to current anchor', numeric=True)
                case[name+'__endpoint_speed_mps'] = float(np.linalg.norm(state[2:]))
                case[name+'__endpoint_shift_m'] = float(np.linalg.norm(state[:2]-trace['native_local'][endpoint]))
                case[name+'__propagation_m'] = float(abs(horizon)*np.linalg.norm(state[2:]))
                case[name+'__cached_replay_difference_m'] = float(distance(predicted, art[name]['group_states'][gid, qi, :2]))
                case['_'+name+'__endpoint_state'] = state.tolist()
        cases.append(case)
    return cases, errors, valid, strata, work, private_trace


def summarize(cases, errors):
    fields = sorted({key for case in cases for key, value in case.items()
                     if not key.startswith('_') and isinstance(value, (int, float)) and not isinstance(value, bool)})
    full = {name: error_stats(np.concatenate(parts)) for name, parts in errors.items()}
    scalar = {key: describe([c[key] for c in cases if c.get(key) is not None]) for key in fields}
    sse = {name: float(sum(c[name+'__error_m']**2 for c in cases)) for name in NAMES}
    return dict(case_count=len(cases), full_position=full, scalar_diagnostics=scalar, zero_support_sse=sse,
        zero_support_sse_share={name: sse[name]/(full[name]['mse']*full[name]['count']) if full[name]['count'] else None for name in NAMES},
        before_count=sum(c['side'] == 'before' for c in cases), after_count=sum(c['side'] == 'after' for c in cases),
        identity_span_inside_count=sum(c['identity_span_inside'] for c in cases),
        by_side={side: {name: error_stats([c[name+'__error_m'] for c in cases if c['side'] == side]) for name in NAMES} for side in ('before', 'after')},
        by_identity_span={str(inside): {name: error_stats([c[name+'__error_m'] for c in cases if c['identity_span_inside'] == inside]) for name in NAMES} for inside in (True, False)})


def run(args):
    state_run, native_run, manifest, output = map(lambda p: Path(p).resolve(),
        (args.state_run, args.native_run, args.manifest, args.output))
    require(not output.exists(), 'new output required; preserve original attempt')
    require(all(output != p and p not in output.parents and output not in p.parents for p in (state_run, native_run)), 'separate source/output')
    output.mkdir(parents=True)
    (output/'private').mkdir()
    started = time.monotonic()
    status = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', started_at=datetime.now(timezone.utc).isoformat(),
        state_run=str(state_run), native_run=str(native_run), manifest=str(manifest), command=sys.argv,
        completed_scenes=0, replay_calls=0, teacher_forward_calls=0, neural_updates=0, target_reference_fits=0,
        new_method_predictions=0, test_opened=False, time_policy='advisory', advisory_seconds=1800, gpu_used=False)
    try:
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
        original = read(state_run/'run.json')
        require(original['status'] == 'COMPLETED' and original['exit_code'] == 0, 'completed 137')
        require(Path(original['native_run']).resolve() == native_run and Path(original['manifest']).resolve() == manifest, 'original input lineage')
        meta = read(manifest)
        summaries, rows = {}, []
        work = dict(replay_calls=0, native_events=0, transition_steps=0, rts_backward_steps=0)
        for split, ids, expected_zero in (('T_core', original['training_scene_ids'], 1487), ('V_select', original['scene_ids'], 2770)):
            require(len(ids) == len(set(ids)) == 32, 'original 32 scenes')
            cached = {row['scene_id']: row for row in read(state_run/('T-scene-summary.json.gz' if split == 'T_core' else 'scene-summary.json.gz'))}
            all_cases, full_errors = [], {name: [] for name in NAMES}
            full_strata = {subset: {name: [] for name in NAMES} for subset in
                ('inside_support', 'outside_support', 'source_support_zero', 'source_support_one', 'source_support_multi')}
            for sid in ids:
                cases, errors, valid, strata, local_work, private_trace = inspect_scene(state_run, native_run, split, sid, cached[sid])
                for key in work:
                    work[key] += local_work[key]
                all_cases.extend(cases)
                for name in NAMES:
                    full_errors[name].append(errors[name][valid])
                for subset, mask in strata.items():
                    for name in NAMES:
                        full_strata[subset][name].append(errors[name][mask])
                local_errors = {name: [errors[name][valid]] for name in NAMES}
                row = dict(scene_id=sid, split=split, component_id=str(meta['components'][sid]),
                    **summarize(cases, local_errors), replay_work=local_work,
                    full_population_strata={subset: {name: error_stats(errors[name][mask]) for name in NAMES}
                                            for subset, mask in strata.items()})
                rows.append(row)
                write(output/'private'/f'cases-{split}-{sid}.json.gz', cases)
                np.savez_compressed(output/'private'/f'trace-{split}-{sid}.npz', **private_trace)
                status.update(completed_scenes=len(rows), replay_calls=work['replay_calls'], current_split=split, current_scene=sid,
                    elapsed_seconds=time.monotonic()-started)
                write(output/'run.json', status)
                require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= 2*1024**3,
                        'resident memory within fixed 2GiB')
            require(len(all_cases) == expected_zero, 'full original zero-support population')
            summaries[split] = summarize(all_cases, full_errors)
            summaries[split]['full_population_strata'] = {
                subset: {name: error_stats(np.concatenate(parts)) for name, parts in methods.items()}
                for subset, methods in full_strata.items()}
            top = sorted(all_cases, key=lambda c: c['WS_OBS_SCALE_RTS_EQUAL__error_m'], reverse=True)[:20]
            write(output/f'top-cases-{split}.json', [dict(case_id=f'{split}-top-{i:02d}', **{k: v for k, v in c.items() if not k.startswith('_')}) for i, c in enumerate(top)])
        require(status['completed_scenes'] == 64 and work['replay_calls'] <= 4426, 'fixed complete endpoint')
        summary = dict(idea_id=IDEA, plan_version='1.0', splits=summaries, replay_work=work,
            limitations=['Posthoc diagnosis of exposed T/V; no validation tuning or new method score',
                        'Endpoint speed is working CV state, not certified physical velocity',
                        'Reference/source identity only diagnoses correspondence; no source priority',
                        'No parameter selection, no new neural training, real K2 only'])
        write(output/'scene-summary.json.gz', rows)
        write(output/'summary.json', summary)
        status.update(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat(), replay_work=work)
    except BaseException:
        status.update(status='FAILED', exit_code=1, finished_at=datetime.now(timezone.utc).isoformat())
        traceback.print_exc()
        raise
    finally:
        status.update(elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024))
        write(output/'run.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('state-run', 'native-run', 'manifest', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--time-policy', choices=['advisory'], default='advisory')
    run(parser.parse_args())
