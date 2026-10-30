"""Independent NumPy output audit for147, without production prediction/evaluation.

Certifies saved references, population, main position metrics, historical control
bindings and primary component bootstrap. Does not certify fitted head semantics,
speed/course metrics, raw measurement provenance or physical reference accuracy.
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
from tools.analysis import audit_shared_control_arrays as core

METHODS = ('WS_MEAN', 'WS_ANCHOR', 'WS_NEAREST_SUPPORTED', 'WS_SOURCE_EQUAL',
           'WS_SHARED_Q_SOFT', 'WS_SHARED_Q_HARD')
PRIMARY = ('WS_SHARED_Q_HARD_vs_WS_MEAN', 'WS_SHARED_Q_HARD_vs_WS_NEAREST_SUPPORTED')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def audit(run, output):
    if output.exists() or output.with_suffix('.failed.json').exists():
        raise FileExistsError(output)
    started = time.monotonic()
    checks = core.Checks()
    result = dict(status='RUNNING', run=str(run), started_at=datetime.now(timezone.utc).isoformat(),
        scope=__doc__, methods=list(METHODS))
    try:
        record, summary, rows = (core.read(run/n) for n in ('run.json', 'summary.json', 'scene-summary.json.gz'))
        checks.exact(record['status'] == 'COMPLETED' and record['exit_code'] == 0, 'terminal science')
        for key in ('analytic_head_fits', 'optimizer_updates', 'neural_models_constructed', 'teacher_forward_calls'):
            checks.value(0, record[key], key, 'no new training')
        checks.exact(not any(record[k] for k in ('test_opened', 'raw_reference_read', 'threshold_reselected', 'groups_recomputed')), 'permission flags')
        manifest = core.read(Path(record['manifest']))
        ids, original = record['scene_ids'], set(record['original32_ids'])
        checks.exact(ids == manifest['splits']['V_select'] and len(ids) == 300 and len(original) == 32,
                     'complete original development population')
        checks.exact([r['scene_id'] for r in rows] == ids, 'ordered rows')
        source, partition, pilot, control = (Path(record[k]) for k in ('source_system', 'partition_run', 'pilot_run', 'control_run'))
        results = []
        for sid, row in zip(ids, rows):
            payload = load(run/'private'/f'states-{sid}.npz')
            saved = {n: {k[len(n)+1:]: v for k, v in payload.items() if k.startswith(n+'_')} for n in METHODS}
            base = saved['WS_MEAN']
            n = len(base['target_ids'])
            raw = load(source/'predictions'/f'states-{sid}.npz')
            reference = load(source/'system-predictions'/f'states-{sid}.npz')
            prefix = 'learned_assoc_mean_supported_'
            checks.array(base['target_ids'], raw['target_identity_ids'], '056 complete targets')
            for key in ('aligned_truth', 'aligned_truth_valid'):
                checks.array(base[key][:n], reference[prefix+key][:n], 'original056 '+key)
            membership = np.asarray(core.read(partition/'private'/f'scene-{sid}.json.gz')['memberships']['WS'], np.int64)
            checks.array(base['node_group_indices'], membership, 'original114 WS')
            groups = [base['group_node_indices'][a:b] for a, b in zip(base['group_offsets'][:-1], base['group_offsets'][1:])]
            registry = {str(t): i for i, t in enumerate(base['target_ids'])}
            checks.array(base['group_anchor_target_indices'],
                np.asarray([registry[str(raw['identities'][g[0]])] for g in groups], np.int64), 'independent anchor correspondence')
            for name in METHODS:
                checks.exact(str(saved[name]['state_speed_unit']) == 'mps', 'explicit canonical output speed')
            if sid in original:
                old = load(pilot/'private'/f'states-{sid}.npz')
                names = ('WS_MEAN', 'WS_NEAREST_SUPPORTED', *METHODS[3:])
            else:
                old = load(control/'private'/f'states-{sid}.npz')
                names = METHODS[:3]
            for name in names:
                for key in ('group_states', 'aligned_prediction'):
                    expected = old[name+'_'+key].copy()
                    expected[..., 2] *= 1852/3600
                    checks.array(saved[name][key], expected, 'historical output binding '+name+'/'+key)
            results.append(core.inspect_scene(saved, row, METHODS, checks))
        components = [r['component_id'] for r in rows]
        cs = sorted(set(components))
        largest = max(cs, key=components.count)
        oi = [i for i, sid in enumerate(ids) if sid in original]
        ri = [i for i, sid in enumerate(ids) if sid not in original]
        oc = {components[i] for i in oi}
        subsets = dict(all_components=list(range(300)), original32=oi, remaining268=ri,
            without_largest_component=[i for i, c in enumerate(components) if c != largest],
            remaining268_unseen_components=[i for i in ri if components[i] not in oc])
        for label, indices in subsets.items():
            core.inspect_summary(summary[label], indices, results, rows, METHODS, checks, label)
        for c in cs:
            core.inspect_summary(summary['by_component'][c], [i for i, v in enumerate(components) if v == c],
                                 results, rows, METHODS, checks, 'component/'+c)
        result['primary'] = {}
        for label, indices in subsets.items():
            if not indices:
                continue
            expected = summary[label]
            for pair, info in expected['pairwise'].items():
                right, left = pair.split('_vs_')
                delta = np.concatenate([results[i][right]['errors']-results[i][left]['errors'] for i in indices])
                checks.value(float(delta.mean()), info['strata']['all']['candidate_minus_A0']['mean_error_change'],
                             'mean_error_change', label+'/'+pair)
                checks.value(int((delta > 1000).sum()), info['actual_harm']['new_harm_over_1km'], 'harm_count', label+'/'+pair)
                checks.value(float(max(0., delta.max())), info['actual_harm']['max_error_increase_m'], 'harm_max', label+'/'+pair)
            if label not in ('all_components', 'remaining268'):
                continue
            result['primary'][label] = {}
            boot = summary['primary_bootstrap' if label == 'all_components' else 'remaining268_bootstrap']
            for pair in PRIMARY:
                right, left = pair.split('_vs_')
                local_cs = sorted({components[i] for i in indices})
                totals, counts = [], []
                for c in local_cs:
                    d = np.concatenate([results[i][right]['errors']-results[i][left]['errors'] for i in indices if components[i] == c])
                    totals.append(d.sum()); counts.append(len(d))
                totals, counts = np.asarray(totals), np.asarray(counts)
                draws = np.random.default_rng(20).integers(0, len(local_cs), (2000, len(local_cs)))
                interval = np.quantile(totals[draws].sum(1)/counts[draws].sum(1), [.025, .975])
                mean = float(totals.sum()/counts.sum())
                checks.value(mean, boot[pair]['point_estimate_m'], 'mae_delta', label+'/'+pair)
                for a, b in zip(interval, boot[pair]['percentile95_m']):
                    checks.value(float(a), b, 'bootstrap', label+'/'+pair)
                result['primary'][label][pair] = dict(mean_error_change_m=mean, percentile95_m=interval.tolist())
        result.update(status='COMPLETED', exit_code=0, scenes=300, components=len(cs),
            position_points=summary['all_components']['methods']['WS_SHARED_Q_HARD']['position']['intersection_count'])
    except BaseException as error:
        result.update(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1024 if sys.platform.startswith('linux') else 1),
            checks=vars(checks), finished_at=datetime.now(timezone.utc).isoformat())
        destination = output if result['status'] == 'COMPLETED' else output.with_suffix('.failed.json')
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'scenes', 'position_points', 'elapsed_seconds')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.output.resolve())
