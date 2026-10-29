"""Independent NumPy metric audit helpers for 136/137 saved artifacts.

Imports only the existing independent auditor, never the production evaluator,
policy, aggregation or selector. T and V populations are checked explicitly.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from tools.analysis import audit_shared_control_arrays as core


def close(actual, expected, checks, label, *, atol=1e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    checks.exact(a.shape == b.shape, label+' shape')
    if a.dtype.kind in 'fci' and b.dtype.kind in 'fci':
        checks.exact(np.all(np.isfinite(a)) == np.all(np.isfinite(b)), label+' finite pattern')
        checks.exact(np.allclose(a, b, atol=atol, rtol=1e-9, equal_nan=True), label+' numeric array')
    else:
        checks.array(a, b, label)


def bootstrap(results, rows, comparison, checks):
    left, right = None, None
    # Read explicit pair orientation from the public scene metadata; it is
    # separately checked against all combinations of the declared method list.
    pair = rows[0]['pairwise'][comparison]
    left, right = pair['baseline_method'], pair['candidate_method']
    components = sorted({r['component_id'] for r in rows})
    sums, counts = [], []
    for component in components:
        delta = np.concatenate([result[right]['errors']-result[left]['errors']
            for result, row in zip(results, rows) if row['component_id'] == component])
        sums.append(float(delta.sum()))
        counts.append(len(delta))
    sums, counts = np.asarray(sums), np.asarray(counts)
    draws = np.random.default_rng(20).integers(0, len(components), size=(2000, len(components)))
    denominator = counts[draws].sum(1)
    checks.exact(np.all(denominator > 0), 'bootstrap nonempty samples')
    values = sums[draws].sum(1)/denominator
    return dict(components=components, point_estimate_m=float(sums.sum()/counts.sum()),
                percentile95_m=np.quantile(values, [.025, .975]).tolist())


def inspect_metric_block(run_path, split, rows, summary, names, checks, *, expected_scene_ids,
                         expected_component_count, expected_primary_pairs=None,
                         native_evaluation_root=None):
    path, names = Path(run_path), tuple(names)
    checks.exact(split in ('T_core', 'V_select'), 'declared development split')
    checks.exact([r['scene_id'] for r in rows] == list(expected_scene_ids), split+' full scene order')
    checks.exact(len(rows) == 32 and len({r['scene_id'] for r in rows}) == 32, split+' 32 unique scenes')
    pairs = {candidate+'_vs_'+base: (base, candidate) for base, candidate in combinations(names, 2)}
    results = []
    for row in rows:
        sid = row['scene_id']
        checks.exact(core.read(path/'scenes'/split/f'scene-{sid}.json.gz') == row,
                     split+' per-scene record binding')
        checks.exact(set(row['methods']) == set(names) and set(row['pairwise']) == set(pairs),
                     split+' complete arms and pairs')
        for key, (base, candidate) in pairs.items():
            pair = row['pairwise'][key]
            checks.exact(pair['baseline_method'] == base and pair['candidate_method'] == candidate,
                         split+' pair orientation')
        saved = core.artifacts(path/'private'/f'states-{split}-{sid}.npz', names, template=names[0])
        if native_evaluation_root is not None:
            with np.load(Path(native_evaluation_root)/split/f'scene-{sid}.npz', allow_pickle=False) as ev:
                base = saved[names[0]]
                t = len(ev['target_ids'])
                checks.array(base['target_ids'], ev['target_ids'], split+' original target registry')
                checks.array(base['aligned_truth'][:t], ev['target_states'], split+' original reference states')
                checks.array(base['aligned_truth_valid'][:t], ev['target_valid'], split+' original reference valid')
                registry = {str(value): i for i, value in enumerate(ev['target_ids'])}
                first_nodes = base['group_node_indices'][base['group_offsets'][:-1]]
                anchors = np.asarray([registry.get(str(ev['identities'][node]), -1) for node in first_nodes])
                checks.array(base['group_anchor_target_indices'], anchors, split+' original first-node identity mapping')
        results.append(core.inspect_scene(saved, row, names, checks, expected_split=split))
    components = sorted({r['component_id'] for r in rows})
    checks.exact(len(components) == expected_component_count
                 and set(summary['by_component']) == set(components), split+' complete components')
    def inspect_summary(block, indices, label):
        core.inspect_summary(block, indices, results, rows, names, checks, label)
        if not indices:
            return
        means = {}
        for name in names:
            count = sum(len(results[i][name]['errors']) for i in indices)
            means[name] = (sum(float(results[i][name]['errors'].sum()) for i in indices)/count
                           if count else None)
        for pair in block['pairwise'].values():
            left, right = pair['baseline_method'], pair['candidate_method']
            delta = None if means[left] is None else means[right]-means[left]
            checks.value(delta, pair['strata']['all']['candidate_minus_A0']['mean_error_change'],
                         'mean_error_change', label+' independent pooled pair delta')
    inspect_summary(summary['all_components'], list(range(len(rows))), split+' all')
    for component in components:
        indices = [i for i, row in enumerate(rows) if row['component_id'] == component]
        inspect_summary(summary['by_component'][component], indices, split+' component')
    largest = max(components, key=lambda c: sum(r['component_id'] == c for r in rows))
    checks.exact(summary['largest_component_id'] == largest, split+' largest by scene count')
    indices = [i for i, row in enumerate(rows) if row['component_id'] != largest]
    inspect_summary(summary['without_largest_component'], indices, split+' without largest')
    if expected_primary_pairs is not None:
        checks.exact(set(summary['primary_bootstrap']) == set(expected_primary_pairs),
                     split+' all predeclared primary comparisons')
    for key, public in summary['primary_bootstrap'].items():
        checks.exact(key in pairs, split+' bootstrap valid comparison')
        independent = bootstrap(results, rows, key, checks)
        checks.exact(public['comparison'] == key and public['components'] == components
                     and public['component_count'] == len(components)
                     and public['resamples'] == 2000 and public['seed'] == 20
                     and public['independent_test'] is False, split+' bootstrap scope')
        close(public['point_estimate_m'], independent['point_estimate_m'], checks,
              split+' bootstrap estimate', atol=1e-5)
        close(public['percentile95_m'], independent['percentile95_m'], checks,
              split+' bootstrap interval', atol=1e-5)
    return results, dict(split=split, scenes=len(rows), components=len(components),
                         methods=len(names), pairs_per_scene=len(pairs),
                         metric_subsets=2+len(components), primary_bootstrap_count=len(summary['primary_bootstrap']))
