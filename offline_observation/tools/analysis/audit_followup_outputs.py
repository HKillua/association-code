"""Independent NumPy output audit for143/144/146; does not certify fitted heads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as stream: return json.load(stream)
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def distance(prediction, target):
    latitude, truth_lat = np.deg2rad(prediction[..., 0]), np.deg2rad(target[..., 0])
    longitude = np.deg2rad((prediction[..., 1]-target[..., 1]+180.) % 360.-180.)
    angle = np.clip(np.sin((latitude-truth_lat)/2)**2+np.cos(latitude)*np.cos(truth_lat)*np.sin(longitude/2)**2, 0., 1.)
    return 2*6371000.*np.arctan2(np.sqrt(angle), np.sqrt(1-angle))


def stats(e):
    return dict(intersection_count=len(e), mae=float(e.mean()) if len(e) else None,
        mse=float(np.mean(e**2)) if len(e) else None, rmse=float(np.sqrt(np.mean(e**2))) if len(e) else None,
        p95=float(np.quantile(e, .95)) if len(e) else None, p99=float(np.quantile(e, .99)) if len(e) else None)


def harm(d):
    return dict(count=len(d), improved=int((d < 0).sum()), harmed=int((d > 0).sum()), tied=int((d == 0).sum()),
        harm_sum_m=float(d[d > 0].sum()), benefit_sum_m=float(-d[d < 0].sum()),
        harm_gt1000_count=int((d > 1000).sum()), harm_gt1000_sum_m=float(d[d > 1000].sum()),
        maximum_harm_m=float(max(0., d.max())) if len(d) else None)


def coverage(artifact):
    target = artifact['aligned_truth_valid'][..., :2].all(-1)
    prediction = artifact['aligned_prediction_valid'][..., :2].all(-1)
    common = target & prediction
    return dict(available_target_count=int(target.sum()), available_prediction_count=int(prediction.sum()),
        intersection_count=int(common.sum()), missing_prediction_count=int((target & ~prediction).sum()),
        prediction_without_target_count=int((prediction & ~target).sum()),
        available_target_trajectory_count=int(target.any(1).sum()),
        covered_target_trajectory_count=int(common.any(1).sum()),
        missing_target_trajectory_count=int((target.any(1) & ~common.any(1)).sum()))


def audit(run, output):
    if output.exists(): raise FileExistsError(output)
    started = time.monotonic()
    result = dict(status='RUNNING', run=str(run), started_at=datetime.now(timezone.utc).isoformat(),
        exact_elements_checked=0, numeric_scalars_checked=0, scope='native reference/row population, all main position metrics, primary harm/bootstrap, support action invariants; not independent fitting audit')
    def exact(a, b, label):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc' and b.dtype.kind in 'fc'):
            raise ValueError('exact audit '+label)
        result['exact_elements_checked'] += a.size
    def numeric(a, b, label):
        if a is None or b is None:
            if a is not b: raise ValueError('null audit '+label)
        elif not np.isclose(a, b, rtol=1e-9, atol=1e-6):
            raise ValueError(f'numeric audit {label}: {a} vs {b}')
        result['numeric_scalars_checked'] += 1
    def metrics(actual, expected, label):
        for k, v in actual.items():
            if isinstance(v, int): exact(v, expected[k], label+'/'+k)
            else: numeric(v, expected[k], label+'/'+k)
    try:
        record = read(run/'run.json')
        exact([record['status'], str(record['exit_code'])], ['COMPLETED', '0'], 'scientific terminal')
        native = Path(record['native_run'])
        result['splits'] = {}
        for split, ids, filename, scene_file in (('T_core', record['training_scene_ids'], 'T-summary.json', 'T-scene-summary.json.gz'),
                                                ('V_select', record['scene_ids'], 'summary.json', 'scene-summary.json.gz')):
            summary, rows = read(run/filename), read(run/scene_file)
            exact([r['scene_id'] for r in rows], ids, 'ordered scenes')
            names = tuple(summary['all_components']['methods'])
            primary = tuple(summary['primary_bootstrap'])
            errors, masks, components, populations = [], [], [], []
            for sid, row in zip(ids, rows):
                payload = load(run/'private'/f'states-{split}-{sid}.npz')
                ev = load(native/'private/native-evaluation'/split/f'scene-{sid}.npz')
                motion = load(native/'private/native-motion'/split/f'scene-{sid}.npz')
                n = len(ev['target_ids'])
                art = {name: {k[len(name)+1:]: v for k, v in payload.items() if k.startswith(name+'_')} for name in names}
                base = art[names[0]]
                exact(base['target_ids'], ev['target_ids'], 'native target ids')
                exact(base['aligned_truth'][:n], ev['target_states'], 'native reference')
                exact(base['aligned_truth_valid'][:n], ev['target_valid'], 'native reference mask')
                exact(base['aligned_truth_valid'][n:], np.zeros_like(base['aligned_truth_valid'][n:]), 'duplicates no truth')
                groups = [base['group_node_indices'][a:b] for a, b in zip(base['group_offsets'][:-1], base['group_offsets'][1:])]
                exact(np.sort(np.concatenate(groups)), np.arange(len(motion['states'])), 'complete partition')
                registry = {str(s): i for i, s in enumerate(ev['target_ids'])}
                first = {}
                for gid, group in enumerate(groups): first.setdefault(registry[str(ev['identities'][group[0]])], gid)
                exact(base['aligned_group_indices'][:n], [first.get(i, -1) for i in range(n)], 'first anchor target map')
                zero = np.asarray([~motion['support'][g].any(0) for g in groups])
                one = np.asarray([sum(motion['support'][g[motion['sources'][g] == s]].any(0) for s in np.unique(motion['sources'][g])) <= 1 for g in groups])
                local, population = {}, {}
                for name in names:
                    a = art[name]
                    for key in base:
                        if key not in ('aligned_prediction', 'group_states'):
                            exact(a[key], base[key], 'fixed population '+name+'/'+key)
                    mask = a['aligned_truth_valid'][:n, :, :2].all(-1) & a['aligned_prediction_valid'][:n, :, :2].all(-1)
                    if local: exact(mask, mask_base, 'same method evaluable mask')
                    else: mask_base = mask
                    for target, gid in enumerate(a['aligned_group_indices']):
                        if gid >= 0: exact(a['aligned_prediction'][target], a['group_states'][gid], 'aligned physical output')
                    local[name] = distance(a['aligned_prediction'][:n], ev['target_states'])[mask]
                    metrics(stats(local[name]), row['methods'][name]['position']['all'], 'scene main '+name)
                    population[name] = coverage(a)
                    metrics(population[name], row['methods'][name]['position']['all'], 'scene coverage '+name)
                    if name.startswith('WS_OBS_ZERO_') or name.startswith('WS_Q_ZERO_'):
                        old_name = 'WS_OBS_SCALE_RTS_EQUAL' if name.startswith('WS_OBS_ZERO_') else 'WS_SHARED_Q_HARD'
                        exact(a['group_states'][~zero], art[old_name]['group_states'][~zero], 'support-in unchanged')
                        exact(a['group_states'][..., 2:], art[old_name]['group_states'][..., 2:], 'non-position unchanged')
                    if name.startswith(('WS_PAIR_', 'WS_REGRET_', 'WS_GLOBAL_ACTUAL_', 'WS_CONDITIONAL_ACTUAL_')):
                        exact(a['group_states'][one], art['WS_SHARED_Q_HARD']['group_states'][one], 'zero/one-source unchanged')
                errors.append(local); masks.append(mask_base); components.append(row['component_id']); populations.append(population)
            cs = sorted(set(components))
            largest = max(cs, key=lambda c: components.count(c))
            def pool(indices): return {name: np.concatenate([errors[i][name] for i in indices]) for name in names}
            def verify_pool(indices, expected, label):
                indices = list(indices)
                values = pool(indices)
                for name in names:
                    metrics(stats(values[name]), expected['methods'][name]['position'], label+'/'+name)
                    counts = {key: sum(populations[i][name][key] for i in indices) for key in populations[0][name]}
                    metrics(counts, expected['methods'][name]['position'], label+'/coverage/'+name)
                return values
            full = verify_pool(range(len(rows)), summary['all_components'], 'full')
            for c in cs: verify_pool([i for i, value in enumerate(components) if value == c], summary['by_component'][c], 'component')
            verify_pool([i for i, value in enumerate(components) if value != largest], summary['without_largest_component'], 'drop largest')
            primary_results = {}
            for p in primary:
                candidate, baseline = p.split('_vs_')
                d = full[candidate]-full[baseline]
                numeric(float(d.mean()), summary['primary_bootstrap'][p]['point_estimate_m'], 'primary mean')
                metrics(harm(d), summary['additional_actual_harm']['all_components'][p], 'primary actual harm')
                totals, counts = [], []
                for c in cs:
                    group = np.concatenate([errors[i][candidate]-errors[i][baseline] for i, value in enumerate(components) if value == c])
                    totals.append(group.sum()); counts.append(len(group))
                    metrics(harm(group), summary['additional_actual_harm']['by_component'][c][p], 'component actual harm')
                totals, counts = np.asarray(totals), np.asarray(counts)
                draws = np.random.default_rng(20).integers(0, len(cs), size=(2000, len(cs)))
                ci = np.quantile(totals[draws].sum(1)/counts[draws].sum(1), [.025, .975])
                for a, b in zip(ci, summary['primary_bootstrap'][p]['percentile95_m']): numeric(a, b, 'bootstrap')
                drop = np.concatenate([errors[i][candidate]-errors[i][baseline] for i, value in enumerate(components) if value != largest])
                metrics(harm(drop), summary['additional_actual_harm']['without_largest_component'][p], 'drop actual harm')
                primary_results[p] = dict(delta_mae_m=float(d.mean()), percentile95_m=ci.tolist(), **harm(d))
            result['splits'][split] = dict(scenes=len(ids), components=len(cs), methods=len(names),
                main_position_points=len(next(iter(full.values()))), primary=primary_results)
        result.update(status='COMPLETED', exit_code=0, elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            finished_at=datetime.now(timezone.utc).isoformat(), excluded=['independent model/label fitting or score audit', 'physical truth/noise certification', 'external sensor or K>=3 validation'])
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: result[k] for k in ('status', 'exact_elements_checked', 'numeric_scalars_checked', 'elapsed_seconds')}))
    except BaseException as error:
        result.update(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        output.with_name(output.name+'.failed.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try: audit(args.run.resolve(), args.output.resolve())
    except BaseException:
        traceback.print_exc(); raise SystemExit(1)
