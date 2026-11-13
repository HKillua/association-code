"""Read-only condition audit of predicted correspondences in returned 103.

No truth field enters extraction, component assignment, fitting or the primary
residual calculation. Oracle annotations are read only after all fits freeze.
This estimates relative displacement; it never produces corrected predictions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

SCENES = ('3035', '2819', '1043')
FIELDS = ('fixed_group_index', 'node_a', 'node_b', 'source_a', 'source_b',
          'query_index', 'query_seconds', 'both_supported', 'delta_NE_m')


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def input_projection(public):
    """The only data view available to primary fitting/evaluation functions."""
    projected, seen = [], set()
    for raw in public['query_pairs']:
        if raw['fixed_group_index'] is None or raw['source_a'] == raw['source_b']:
            continue
        row = {key: raw[key] for key in FIELDS}
        if row['source_a'] >= row['source_b'] or raw['direction'] != 'source_b minus source_a':
            raise ValueError('unexpected source orientation')
        if not 0 <= row['query_index'] < 64:
            raise ValueError('query index outside fixed Q64')
        key = row_key(row)
        if key in seen:
            raise ValueError('duplicate pair-query row')
        seen.add(key)
        delta = np.asarray(row['delta_NE_m'], float)
        if delta.shape != (2,) or not np.isfinite(delta).all():
            raise ValueError('nonfinite or malformed direct input difference')
        projected.append(row)
    return projected


def row_key(row):
    return (row['fixed_group_index'], row['node_a'], row['node_b'], row['query_index'])


def component_split(rows):
    """Split all predicted cross-source groups before the support restriction."""
    nodes = defaultdict(set)
    for row in rows:
        nodes[row['fixed_group_index']].update((row['node_a'], row['node_b']))
    remaining = set(nodes)
    components = []
    while remaining:
        component = {min(remaining)}
        reached_nodes = set(nodes[min(remaining)])
        while True:
            linked = {g for g in remaining if nodes[g] & reached_nodes}
            if linked <= component:
                break
            component.update(linked)
            for group in linked:
                reached_nodes.update(nodes[group])
        remaining -= component
        components.append(sorted(component))
    components.sort(key=min)
    heldout = {g for rank, component in enumerate(components) if rank % 3 == 0 for g in component}
    fit = set(nodes)-heldout
    return dict(components=components, fit_groups=sorted(fit), heldout_groups=sorted(heldout),
                shared_node_components=sum(len(c) > 1 for c in components))


def group_vectors(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['fixed_group_index']].append(row['delta_NE_m'])
    return {group: np.mean(values, axis=0) for group, values in sorted(groups.items())}


def fit_center(rows, estimator='coordinate_median'):
    vectors = group_vectors(rows)
    if not vectors:
        return None
    values = np.array(list(vectors.values()))
    return (np.median(values, axis=0) if estimator == 'coordinate_median' else np.mean(values, axis=0)).tolist()


def distribution(values):
    if not len(values):
        return dict(count=0, mean=None, median=None, p95=None, minimum=None, maximum=None)
    values = np.asarray(values, float)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), minimum=float(values.min()), maximum=float(values.max()))


def residual_summary(rows, center):
    if center is None:
        return dict(status='unavailable_no_fit_groups', pair_query_count=len(rows),
                    group_count=len(group_vectors(rows)), groups=[])
    groups = defaultdict(list)
    before, after = [], []
    for row in rows:
        delta = np.asarray(row['delta_NE_m'])
        b, a = float(np.linalg.norm(delta)), float(np.linalg.norm(delta-center))
        before.append(b); after.append(a)
        groups[row['fixed_group_index']].append((b, a))
    group_rows = []
    for group, values in sorted(groups.items()):
        values = np.asarray(values)
        group_rows.append(dict(group_index=group, pair_query_count=len(values),
            before_mean_m=float(values[:, 0].mean()), after_mean_m=float(values[:, 1].mean()),
            change_m=float((values[:, 1]-values[:, 0]).mean())))
    return dict(status='evaluated' if rows else 'unavailable_no_evaluation_rows',
        pair_query_count=len(rows), group_count=len(group_rows),
        group_equal_before_m=distribution([r['before_mean_m'] for r in group_rows]),
        group_equal_after_m=distribution([r['after_mean_m'] for r in group_rows]),
        group_equal_change_m=distribution([r['change_m'] for r in group_rows]),
        pooled_pair_query_before_m=distribution(before), pooled_pair_query_after_m=distribution(after),
        groups=group_rows)


def analyze_pair(rows):
    split = component_split(rows)
    supported = [r for r in rows if r['both_supported']]
    fit = [r for r in supported if r['fixed_group_index'] in split['fit_groups']]
    hold = [r for r in supported if r['fixed_group_index'] in split['heldout_groups']]
    early = [r for r in fit if r['query_index'] < 32]
    center, early_center = fit_center(fit), fit_center(early)
    mean_center, early_mean = fit_center(fit, 'mean'), fit_center(early, 'mean')
    all_groups = sorted({r['fixed_group_index'] for r in rows})
    support_groups = sorted(group_vectors(supported))
    subsets = dict(fit_all=fit, heldout_all=hold)
    time_subsets = dict(fit_early=early, fit_late=[r for r in fit if r['query_index'] >= 32],
        heldout_early=[r for r in hold if r['query_index'] < 32],
        heldout_late=[r for r in hold if r['query_index'] >= 32])
    result = dict(sources=[rows[0]['source_a'], rows[0]['source_b']], split=split,
        population=dict(cross_source_group_count=len(all_groups), supported_group_count=len(support_groups),
            no_common_support_groups=sorted(set(all_groups)-set(support_groups)),
            declared_pair_query_count=len(rows), supported_pair_query_count=len(supported),
            unsupported_pair_query_count=len(rows)-len(supported)),
        center_NE_m=center, mean_sensitivity_center_NE_m=mean_center,
        early_center_NE_m=early_center, early_mean_sensitivity_center_NE_m=early_mean,
        early_vs_full_center_distance_m=(float(np.linalg.norm(np.array(center)-early_center))
                                         if center is not None and early_center is not None else None),
        main={key: residual_summary(value, center) for key, value in subsets.items()},
        mean_sensitivity={key: residual_summary(value, mean_center) for key, value in subsets.items()},
        temporal={key: residual_summary(value, early_center) for key, value in time_subsets.items()},
        temporal_mean_sensitivity={key: residual_summary(value, early_mean) for key, value in time_subsets.items()})
    return result


def annotate_oracle(result, rows, oracle_by_key):
    """Post-fit sensitivity only; this function cannot alter any primary field."""
    supported = [r for r in rows if r['both_supported']]
    fit = [r for r in supported if r['fixed_group_index'] in result['split']['fit_groups']]
    oracle_fit = [r for r in fit if oracle_by_key[row_key(r)]]
    oracle_center = fit_center(oracle_fit)
    center = result['center_NE_m']
    sensitivity = dict(same_fit_groups_oracle_center_NE_m=oracle_center,
        oracle_fit_group_count=len(group_vectors(oracle_fit)),
        center_difference_m=(float(np.linalg.norm(np.array(center)-oracle_center))
                             if center is not None and oracle_center is not None else None), strata={})
    for split_name, groups in (('all', None), ('fit', result['split']['fit_groups']),
                               ('heldout', result['split']['heldout_groups'])):
        subset = [r for r in supported if groups is None or r['fixed_group_index'] in groups]
        bucket = {}
        for label, same in (('same_identity', True), ('different_identity', False)):
            selected = [r for r in subset if oracle_by_key[row_key(r)] == same]
            bucket[label] = residual_summary(selected, center)
            bucket[label]['distinct_node_pair_count'] = len({(r['node_a'], r['node_b']) for r in selected})
        sensitivity['strata'][split_name] = bucket
    return sensitivity


def analyze_scene(public):
    rows = input_projection(public)
    pairs = defaultdict(list)
    for row in rows:
        pairs[(row['source_a'], row['source_b'])].append(row)
    # Freeze every primary and mean-sensitivity result before exposing labels.
    results = [analyze_pair(pair_rows) for _, pair_rows in sorted(pairs.items())]
    oracle = {row_key(r): r['oracle_same_identity'] for r in public['query_pairs']
              if r['fixed_group_index'] is not None and r['source_a'] != r['source_b']}
    for result in results:
        result['oracle_postfit_sensitivity'] = annotate_oracle(result, pairs[tuple(result['sources'])], oracle)
    all_groups = {r['group_index'] for r in public['query_sources']}
    cross_groups = {r['fixed_group_index'] for r in rows}
    # These labels describe duplicates only after fitting; never filter them.
    duplicate_groups = {r['group_index'] for r in public['query_sources'] if not r['selected_unique_group']}
    return dict(population=dict(all_predicted_groups=len(all_groups),
        cross_source_groups=len(cross_groups), no_cross_source_groups=len(all_groups-cross_groups),
        duplicate_groups_postfit_annotation=len(duplicate_groups),
        duplicate_cross_source_groups_postfit_annotation=len(duplicate_groups & cross_groups)), source_pairs=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('audit output already exists; no overwrite or automatic rerun')
    start, wall = datetime.now(timezone.utc).isoformat(), time.monotonic()
    result = dict(experiment_id='EXP-PREDICTED-REGISTRATION-20261027',
        source_idea_id='IDEA-PREDICTED-REGISTRATION-20261027', plan_version='1.0',
        status='RUNNING', started_at_utc=start, command=sys.argv, interpreter=sys.executable,
        input_projection_fields=list(FIELDS), source=str(args.source.resolve()), scenes={})
    # Claim a new artifact before reading inputs; interruption preserves RUNNING.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        for scene in SCENES:
            result['scenes'][scene] = analyze_scene(read(args.source/f'points-{scene}.json.gz'))
        result['status'] = 'COMPLETED'
    except BaseException as error:
        result.update(status='FAILED', error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        result.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-wall,
            python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
            max_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_rss_unit='bytes' if sys.platform == 'darwin' else 'KiB',
            optimizer_updates=0, model_forwards=0, zip_reads=0, test_opened=False,
            limits=['postselected three development scenes; correlated groups and queries',
                    'input interpolation differences, not simultaneous native measurements',
                    'no absolute position score, no motion-direction diversity check',
                    'relative gauge remains unidentified; no corrected prediction produced'])
        # Only this invocation's exclusively created file is updated.
        with args.output.open('w', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'elapsed_seconds', 'max_rss_raw', 'max_rss_unit')}))


if __name__ == '__main__':
    main()
