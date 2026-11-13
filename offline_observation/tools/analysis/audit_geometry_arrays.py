"""Independent saved-candidate labels/counts/rank AUC audit; no predictor calls.

Scope is candidate discrimination, not partitions, physical output or mechanism.
Only small named arrays are opened; pair-distance tensors are not materialized.
"""
import argparse
from datetime import datetime, timezone
import filecmp
import gzip
import json
from pathlib import Path
import resource
import time

import numpy as np

SCORERS = dict(SCORE_SUM=('score_sum', 1), SCORE_MEAN=('score_mean', 1),
    SCORE_MIN=('score_min', 1), SCORE_MEDIAN=('score_median', 1),
    PAIR_SHARED_MEDIAN=('pair_shared_median_m', -1), PAIR_SHARED_RMS=('pair_shared_rms_m', -1),
    PAIR_FULL_MEDIAN=('pair_full_median_m', -1), PAIR_FULL_RMS=('pair_full_rms_m', -1),
    LOO_MEDIAN=('loo_median_m', -1), LOO_RMS=('loo_rms_m', -1))


def rank_auc(y, x):
    """Mann–Whitney average-rank form, independent of production tie-block sum."""
    y, x = np.asarray(y, bool), np.asarray(x, float)
    p, n = int(y.sum()), int((~y).sum())
    if not p or not n:
        return None
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends-counts+1+ends)/2
    return float((ranks[inverse][y].sum()-p*(p+1)/2)/(p*n))


def masks(e):
    size, source = e['candidate_size'], e['source_count']
    frac = e['pair_shared_query_fraction']
    return dict(all=np.ones(len(size), bool), original=e['original'],
        size1=size == 1, size2=size == 2, size3=size == 3, size4plus=size >= 4,
        single_source=source == 1, cross_source=source > 1,
        no_shared_pair_support=e['pair_shared_query_count'] == 0,
        has_shared_pair_support=e['pair_shared_query_count'] > 0,
        shared_fraction_0_to_half=(frac > 0) & (frac <= .5), shared_fraction_over_half=frac > .5,
        common_all_scorers=np.all([np.isfinite(e[k]) for k in SCORERS], axis=0))


def audit(run_path, output):
    start = time.monotonic()
    run_path, output = Path(run_path), Path(output)
    if output.exists():
        raise FileExistsError('independent audit output must be new')
    read = lambda p: json.loads(p.read_text())
    run, summary = read(run_path/'run.json'), read(run_path/'summary.json')
    out = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='saved features, 056 identity purity, candidate population and rank AUC',
        exact_checks=0, numeric_checks=0, null_checks=0,
        not_audited=['geometry/LOO feature generation', 'candidate construction', 'label timing',
                     'physical reference', 'partition F1', 'source health or causal mechanism'])
    def eq(actual, expected, label):
        if actual != expected:
            raise AssertionError(label)
        out['exact_checks'] += 1
    def val(actual, expected, label):
        if actual is None or expected is None:
            eq(actual, expected, label); out['null_checks'] += 1
        else:
            if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
                raise AssertionError(label)
            out['numeric_checks'] += 1
    def population(e, saved):
        y, known = e['pure'], e['known']
        for label, mask in masks(e).items():
            s = saved[label]
            for k, v in dict(candidate_count=mask.sum(), known_label_count=(mask & known).sum(),
                    unknown_label_count=(mask & ~known).sum(), pure_count=(mask & known & y).sum(),
                    mixed_count=(mask & known & ~y).sum()).items():
                eq(int(v), s[k], label+'/'+k)
            for name in SCORERS:
                finite = np.isfinite(e[name]); use = mask & known & finite
                d = s['scorers'][name]
                for k, v in dict(defined_count=use.sum(), undefined_count=(mask & ~finite).sum(),
                        pure_defined_count=(use & y).sum(), mixed_defined_count=(use & ~y).sum()).items():
                    eq(int(v), d[k], label+'/'+name+'/'+k)
                val(rank_auc(y[use], e[name][use]), d['auroc'], label+'/'+name+'/auc')
    try:
        eq(run['status'], 'COMPLETED', 'run status'); eq(run['exit_code'], 0, 'run exit')
        ids = run['scene_ids']; eq(run['completed_scenes'], len(ids), 'scene completion')
        eq(len(ids) == len(set(ids)) == 32, True, '32 unique scenes')
        eq(run['idea_id'], 'IDEA-ASSOCIATION-GEOMETRY-CONTINUATION-20261029', '130 idea')
        eq(summary['idea_id'], run['idea_id'], 'summary idea')
        eq(run['plan_version'] == summary['plan_version'] == '1.0', True, 'version')
        eq(summary['status'], 'COMPLETED', 'summary status')
        eq(run['original128_imported_scene_ids'], ids[:6], 'exact original prefix')
        rows = [json.loads(gzip.decompress((run_path/'scenes'/f'scene-{sid}.json.gz').read_bytes())) for sid in ids]
        es = []
        for sid, row in zip(ids, rows):
            with np.load(run_path/'private'/f'evaluation-{sid}.npz', allow_pickle=False) as z:
                e = {k: z[k] for k in z.files}
            with np.load(run_path/'private'/f'features-{sid}.npz', allow_pickle=False) as f:
                fs, names = f['features'], f['feature_names'].tolist()
                for name, (key, sign) in SCORERS.items():
                    eq(bool(np.array_equal(e[name], fs[:, names.index(key)]*sign, equal_nan=True)), True, 'score binding')
                offsets, nodes = f['candidate_offsets'], f['candidate_node_indices']
                with np.load(Path(run['source_system'])/'predictions'/f'states-{sid}.npz', allow_pickle=False) as z:
                    identity, sources = z['identities'], z['sources']
                eq(row['scene_id'], sid, 'row scene ID'); eq(row['node_count'], len(identity), 'row nodes')
                eq(bool(offsets.ndim == 1 and offsets[0] == 0 and offsets[-1] == len(nodes)
                    and np.all(np.diff(offsets) > 0)), True, 'candidate offset bounds')
                eq(bool(np.all((nodes >= 0) & (nodes < len(identity)))), True, 'member node bounds')
                eq(bool(np.array_equal(e['candidate_size'], np.diff(offsets))), True, 'member sizes')
                pure, known = [], []
                source_counts = []
                for left, right in zip(offsets[:-1], offsets[1:]):
                    v = identity[nodes[left:right]]
                    pure.append(bool(len(np.unique(v)) == 1)); known.append(bool(np.all(v != '')))
                    source_counts.append(len(np.unique(sources[nodes[left:right]])))
                eq(bool(np.array_equal(e['source_count'], source_counts)), True, 'source counts')
                eq(bool(np.array_equal(e['pure'], pure)), True, 'purity')
                eq(bool(np.array_equal(e['known'], known)), True, 'known')
                eq(bool(np.array_equal(e['original'], f['candidate_is_original'])), True, 'original')
                for key in ['candidate_size','source_count','pair_shared_query_count','pair_shared_query_fraction','loo_member_query_count']:
                    eq(bool(np.array_equal(e[key], fs[:, names.index(key)], equal_nan=True)), True, key)
            population(e, row['candidate_population']); es.append(e)
        def combined(indices, target):
            indices = list(indices)
            eq(target['scene_ids'], [ids[i] for i in indices], 'aggregate scene IDs')
            eq(target['scene_count'], len(indices), 'aggregate scene count')
            eq(target['node_count'], sum(rows[i]['node_count'] for i in indices), 'aggregate node count')
            for key in target['counts']:
                eq(target['counts'][key], sum(rows[i]['feature_audit'][key] for i in indices), 'aggregate count '+key)
            e = {k: np.concatenate([es[i][k] for i in indices]) for k in es[0]}
            population(e, target['candidate_population'])
            for name in SCORERS:
                values = [rows[i]['candidate_population']['all']['scorers'][name]['auroc'] for i in indices]
                values = [x for x in values if x is not None]
                eq(len(values), target['scene_equal_auroc'][name]['defined_scenes'], 'defined scenes')
                val(float(np.mean(values)) if values else None, target['scene_equal_auroc'][name]['mean'], 'scene mean')
        combined(range(len(rows)), summary)
        eq(set(summary['components']), {r['component_id'] for r in rows}, 'component set')
        for comp, target in summary['components'].items():
            combined([i for i,r in enumerate(rows) if r['component_id'] == comp], target)
        without = summary['without_largest_component']
        largest = max(summary['components'], key=lambda key: (summary['components'][key]['scene_count'], key))
        kept = [i for i,r in enumerate(rows) if r['component_id'] != largest]
        eq([ids[i] for i in kept], without['scene_ids'], 'without largest population')
        combined(kept, without)
        prefix = run['original128_imported_scene_ids']
        combined([i for i,sid in enumerate(ids) if sid in prefix], summary['imported_complete128'])
        combined([i for i,sid in enumerate(ids) if sid not in prefix], summary['newly_computed130'])
        for sid in prefix:
            for rel in [f'private/features-{sid}.npz',f'private/evaluation-{sid}.npz',f'scenes/scene-{sid}.json.gz']:
                eq(filecmp.cmp(run_path/rel, Path(run['failed_run'])/rel, shallow=False), True, 'original byte import')
        out.update(status='PASSED', scene_count=len(ids), imported_prefix_count=len(prefix))
    except Exception as error:
        out.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        out.update(elapsed_seconds=time.monotonic()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            finished_at=datetime.now(timezone.utc).isoformat())
        output.write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); audit(args.run, args.output)
