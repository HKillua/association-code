"""C76: exact touch bounds for multiple disjoint scopes from the saved C75 counts."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import os
from pathlib import Path
import resource
import sys
import time

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np

from tools.analysis.frozen_score_decision_diagnostic import peak_rss_bytes, read, write
from tools.analysis.offline_partition_scope_bound import AXES, FAMILIES, aggregate, axis_dict, fractions
from tools.analysis.same_source_swap_pilot import MAX_AS, MAX_NODES

BUDGETS = (1, 2, 4)
MODES = ('mixed', *FAMILIES)
ORIGINAL_TOTALS = (298, 523, 124, 385)
FAMILY_ORDER = {family: index for index, family in enumerate(FAMILIES)}
ZERO = (0,) * len(AXES)


@dataclass(frozen=True)
class Scope:
    family: str
    index: int
    groups: tuple[int, ...]
    nodes: int
    counts: tuple[int, ...]

    @property
    def canonical(self):
        return FAMILY_ORDER[self.family], self.index

    @property
    def vertices(self):
        # Give a single-group scope its own matching vertex.
        return self.groups if len(self.groups) == 2 else (self.groups[0], -1 - self.groups[0])

    @property
    def group_mask(self):
        return sum(1 << group for group in self.groups)


def score(scope, axis):
    return sum(scope.counts) if axis is None else scope.counts[axis]


def exact_matching(ranked, budget):
    """Solve up to k disjoint edges, preferring fewer scopes and then canonical sets.

    Ranked positive-weight edges form a simple graph and are ordered by decreasing
    weight, then canonical scope. An optimal solution cannot include an edge
    outside an endpoint's first 2k-1: at most 2(k-1) incident alternatives
    conflict with the other selected edges, so a better-ranked replacement exists.
    The retained graph has degree at most 2k-1. At most 2(k-1)(2k-1) edges there
    conflict with the other selections, so the same exchange argument limits the
    search to the first 2(k-1)(2k-1)+1 retained edges.
    """
    degree_cap = 2 * budget - 1
    edge_cap = 2 * (budget - 1) * degree_cap + 1
    ranks = Counter()
    pool = []
    for weight, scope in ranked:
        first, second = scope.vertices
        if ranks[first] < degree_cap and ranks[second] < degree_cap:
            pool.append((weight, scope, scope.group_mask))
            if len(pool) == edge_cap:
                break
        ranks[first] += 1
        ranks[second] += 1

    prefix = [0]
    for weight, _, _ in pool:
        prefix.append(prefix[-1] + weight)
    best = ()
    best_key = (0, 0, ())

    def visit(start, remaining, occupied, total, chosen):
        nonlocal best, best_key
        if total >= -best_key[0]:
            canonical = tuple(sorted(chosen, key=lambda scope: scope.canonical))
            key = (-total, len(canonical), tuple(scope.canonical for scope in canonical))
            if key < best_key:
                best, best_key = canonical, key
        if remaining == 0:
            return
        for index in range(start, len(pool)):
            upper = total + prefix[min(index + remaining, len(pool))] - prefix[index]
            if upper < -best_key[0]:
                break
            weight, scope, mask = pool[index]
            if not occupied & mask:
                visit(index + 1, remaining - 1, occupied | mask, total + weight, (*chosen, scope))

    visit(0, budget, 0, 0, ())
    return best, len(pool)


def brute_force(scopes, budget, axis):
    """Independent enumeration without graph reduction, bit masks or search bounds."""
    options = [scope for scope in scopes if score(scope, axis) > 0]
    best = ()
    best_key = (0, 0, ())
    for size in range(1, min(budget, len(options)) + 1):
        for group in combinations(options, size):
            vertices = [vertex for scope in group for vertex in scope.groups]
            if len(set(vertices)) != len(vertices):
                continue
            canonical = tuple(sorted(group, key=lambda scope: scope.canonical))
            key = (-sum(score(scope, axis) for scope in group), size,
                   tuple(scope.canonical for scope in canonical))
            if key < best_key:
                best, best_key = canonical, key
    return best


def totals(scopes):
    return tuple(sum(scope.counts[axis] for scope in scopes) for axis in range(len(AXES)))


def describe_scope(scope, denominator):
    return dict(scope_index=scope.index, family=scope.family, original_group_indices=list(scope.groups),
                node_count=scope.nodes, potential=axis_dict(scope.counts),
                fraction_of_original_error=fractions(scope.counts, denominator))


def analyze_mode(scopes, denominator, original=None):
    records = {str(budget): dict(independent_marginal_max={}, independent_marginal_choices={},
                                 searched_edges={}) for budget in BUDGETS}
    brute_checks = Counter()
    for axis in (None, *range(len(AXES))):
        name = 'fixed_sum' if axis is None else AXES[axis]
        ranked = [(score(scope, axis), scope) for scope in scopes]
        ranked = sorted(((weight, scope) for weight, scope in ranked if weight > 0),
                        key=lambda pair: (-pair[0], pair[1].canonical))
        last_objective = 0
        for budget in BUDGETS:
            selected, pool_size = exact_matching(ranked, budget)
            if len(ranked) <= 18:
                if selected != brute_force(scopes, budget, axis):
                    raise ValueError(f'independent exhaustive matching mismatch for {name}, k={budget}')
                brute_checks['total'] += 1
                brute_checks['nonempty'] += bool(ranked)
            if axis is None and budget == 1 and not selected and original is not None and scopes:
                # Preserve C75's first canonical row for a nonempty all-zero family.
                selected = (scopes[0],)
            values = totals(selected)
            objective = sum(values) if axis is None else values[axis]
            consumed_groups = [group for scope in selected for group in scope.groups]
            if (len(selected) > budget or len(consumed_groups) != len(set(consumed_groups))
                    or objective < last_objective or any(a > b for a, b in zip(values, denominator))):
                raise ValueError(f'inconsistent scope solution for {name}, k={budget}')
            last_objective = objective
            record = records[str(budget)]
            described = [describe_scope(scope, denominator) for scope in selected]
            record['searched_edges'][name] = pool_size
            if axis is None:
                record.update(fixed_sum_vector=axis_dict(values), fixed_sum_score=objective,
                              fixed_sum_scope_count=len(selected), fixed_sum_scopes=described)
            else:
                record['independent_marginal_max'][name] = objective
                record['independent_marginal_choices'][name] = described

    if original is not None:
        one = records['1']
        observed_choice = next(iter(one['fixed_sum_scopes']), None)
        if observed_choice is not None:
            observed_choice = {key: value for key, value in observed_choice.items() if key != 'family'}
        if (observed_choice != original['fixed_sum_choice']
                or one['independent_marginal_max'] != original['independent_marginal_max']):
            raise ValueError('k=1 scope row or marginal counts differ from the saved C75 family')
    return records, brute_checks


def bind_population(c75):
    run = read(c75 / 'run.json')
    manifest = read(c75 / 'scope-manifest.json')
    rows = read(c75 / 'scenes.json')
    summary = read(c75 / 'summary.json')
    scenes = run['scene_ids']
    components = manifest['components']
    if (run['status'] != 'COMPLETED' or run['exit_code'] != 0 or run['split'] != 'T_core'
            or run['source_partition'] != 'WS' or run['completed_scenes'] != 32
            or run['stages'] != ['all_original_scopes_frozen_before_identity_read',
                                 'all_scope_counts_independently_verified']
            or len(scenes) != 32 or len(set(scenes)) != 32
            or not all(sid.isascii() and sid.isdecimal() for sid in scenes)
            or [row['scene_id'] for row in rows] != scenes or manifest['scene_ids'] != scenes
            or set(components) != set(scenes) or set(manifest['scenes']) != set(scenes)
            or len(set(components.values())) != 12 or manifest['axes'] != list(AXES)
            or manifest['identity_fields_opened'] is not False or manifest['candidate_generation'] is not False
            or summary['schema_version'] != 'c75-scope-bound-v1' or summary['axes'] != list(AXES)):
        raise ValueError('completed C75 original T32, WS, 12-component lineage required')
    total = aggregate(rows)
    if (total != summary['all_scenes'] or total['nodes'] != 2310
            or total['original_ws_group_count'] != 911 or total['families']['recut']['scope_count'] != 40349
            or total['original_error_denominator'] != axis_dict(ORIGINAL_TOTALS)):
        raise ValueError('C75 population or error denominator differs')
    by_component = {component: aggregate([row for row in rows if row['component_id'] == component])
                    for component in sorted(set(components.values()))}
    largest = min(by_component, key=lambda component: (-by_component[component]['nodes'], component))
    if (by_component != summary['components'] or largest != summary['largest_component']
            or summary['largest_component_node_count'] != by_component[largest]['nodes']
            or aggregate([row for row in rows if row['component_id'] != largest]) != summary['without_largest']):
        raise ValueError('C75 component results or largest-component selection differ')
    return rows, manifest, largest


def load_scopes(c75, original, manifest):
    sid = original['scene_id']
    declared = manifest['scenes'][sid]
    groups, nodes = declared['groups'], declared['nodes']
    gap = abs(groups - original['observed_identity_count'])
    verified = original['independent_verification']
    if (not 0 < groups <= nodes <= MAX_NODES or groups != original['original_ws_group_count']
            or nodes != original['nodes'] or manifest['components'][sid] != original['component_id']
            or verified['passed'] is not True or verified['original_groups'] != groups
            or gap != original['absolute_group_count_gap']):
        raise ValueError(f'C75 manifest/scene mismatch: {sid}')
    filename = f'scope-counts/T_core-{sid}.npz'
    if original['scope_counts_file'] != filename:
        raise ValueError(f'wrong C75 scope file binding: {sid}')
    denominator = tuple(original['original_error_denominator'][key] for key in AXES)
    canonical = np.arange(groups, dtype=np.int64)
    single = np.column_stack((canonical, np.full(groups, -1)))
    pairs = np.column_stack(np.triu_indices(groups, 1))
    expected_files = {f'{family}_{suffix}' for family in FAMILIES
                      for suffix in ('group_indices', 'original_error_counts', 'node_count')}
    family_arrays = {}
    with np.load(c75 / filename, allow_pickle=False) as file:
        if set(file.files) != expected_files:
            raise ValueError(f'unexpected scope-array fields: {sid}')
        for family in FAMILIES:
            indices = file[f'{family}_group_indices']
            counts = file[f'{family}_original_error_counts']
            sizes = file[f'{family}_node_count']
            expected_indices = single if family == 'split' else pairs
            if (len(expected_indices) != declared['scopes'][family]
                    or len(expected_indices) != original['families'][family]['scope_count']
                    or not all(array.dtype.kind == 'i' for array in (indices, counts, sizes))
                    or not np.array_equal(indices, expected_indices)
                    or counts.shape != (len(indices), len(AXES)) or sizes.shape != (len(indices),)
                    or np.any(counts < 0) or np.any(counts > denominator)
                    or np.any(sizes <= 0) or np.any(sizes > nodes)):
                raise ValueError(f'malformed canonical C75 {family} arrays: {sid}')
            family_arrays[family] = indices, counts, sizes
    _, split_counts, split_sizes = family_arrays['split']
    _, merge_counts, merge_sizes = family_arrays['merge']
    _, recut_counts, recut_sizes = family_arrays['recut']
    left, right = pairs.T
    if (int(split_sizes.sum()) != nodes or verified['unordered_group_pairs'] != len(pairs)
            or np.any(split_counts[:, 2:]) or np.any(split_counts[split_sizes == 1])
            or np.any(merge_counts[:, :2])
            or not np.array_equal(merge_sizes, split_sizes[left] + split_sizes[right])
            or not np.array_equal(recut_sizes, merge_sizes)
            or not np.array_equal(recut_counts[:, :2], split_counts[left, :2] + split_counts[right, :2])
            or not np.array_equal(recut_counts[:, 2:], merge_counts[:, 2:])
            or tuple((split_counts.sum(axis=0) + merge_counts.sum(axis=0)).tolist()) != denominator):
        raise ValueError(f'C75 scope additivity, pair population or error denominators differ: {sid}')
    scopes = {}
    for family, (indices, counts, sizes) in family_arrays.items():
        scopes[family] = tuple(
            Scope(family, index, tuple(int(group) for group in pair if group >= 0),
                  int(size), tuple(vector.tolist()))
            for index, (pair, vector, size) in enumerate(zip(indices, counts, sizes))
        )
    # On the same pair recut includes every merge FN and adds only nonnegative FP.
    scopes['mixed'] = scopes['split'] + scopes['recut']
    return scopes, denominator


def analyze_scene(c75, original, manifest):
    scopes, denominator = load_scopes(c75, original, manifest)
    modes, checks = {}, Counter()
    for mode in MODES:
        modes[mode], tested = analyze_mode(scopes[mode], denominator, original['families'].get(mode))
        checks.update(tested)
    result = {key: original[key] for key in ('scene_id', 'component_id', 'nodes',
                                            'original_ws_group_count', 'observed_identity_count',
                                            'ws_minus_observed_identity_count', 'absolute_group_count_gap',
                                            'original_error_denominator')}
    result.update(modes=modes, small_graph_exact_checks=checks['total'],
                  nonempty_small_graph_exact_checks=checks['nonempty'],
                  group_count_gap_at_most_k={str(k): result['absolute_group_count_gap'] <= k for k in BUDGETS})
    return result


def summarize_population(rows):
    denominator = tuple(sum(row['original_error_denominator'][axis] for row in rows) for axis in AXES)
    modes = {}
    for mode in MODES:
        by_budget = {}
        for budget in BUDGETS:
            entries = [row['modes'][mode][str(budget)] for row in rows]
            fixed = tuple(sum(record['fixed_sum_vector'][axis] for record in entries) for axis in AXES)
            marginal = tuple(sum(record['independent_marginal_max'][axis] for record in entries) for axis in AXES)
            by_budget[str(budget)] = dict(
                fixed_sum_vector=axis_dict(fixed), fixed_sum_score=sum(fixed),
                fixed_sum_fraction_of_original_error=fractions(fixed, denominator),
                independent_marginal_max=axis_dict(marginal),
                independent_marginal_fraction_of_original_error=fractions(marginal, denominator),
                fixed_sum_scope_count_histogram=dict(sorted(Counter(str(entry['fixed_sum_scope_count'])
                                                                  for entry in entries).items())),
                fixed_sum_scenes_with_nonzero_touch=axis_dict(
                    sum(entry['fixed_sum_vector'][axis] > 0 for entry in entries) for axis in AXES),
                marginal_scenes_with_nonzero_touch=axis_dict(
                    sum(entry['independent_marginal_max'][axis] > 0 for entry in entries) for axis in AXES),
            )
        modes[mode] = by_budget
    return dict(scene_count=len(rows), component_count=len({row['component_id'] for row in rows}),
                nodes=sum(row['nodes'] for row in rows),
                original_error_denominator=axis_dict(denominator), modes=modes,
                small_graph_exact_checks=sum(row['small_graph_exact_checks'] for row in rows),
                nonempty_small_graph_exact_checks=sum(row['nonempty_small_graph_exact_checks'] for row in rows),
                group_count_gap_at_most_k_scenes={str(k): sum(row['group_count_gap_at_most_k'][str(k)]
                                                            for row in rows) for k in BUDGETS})


def self_check():
    random = np.random.default_rng(7600)
    checks = 0
    for count in range(1, 8):
        for repetition in range(8):
            singles = tuple(Scope('split', index, (index,), 1, tuple(vector.tolist()))
                            for index, vector in enumerate(random.integers(0, 3, (count, len(AXES)))))
            pair_groups = tuple(combinations(range(count), 2))
            pair_counts = random.integers(0, 3, (len(pair_groups), len(AXES)))
            pairs = tuple(Scope('recut', index, pair, 2, tuple(vector.tolist()))
                          for index, (pair, vector) in enumerate(zip(pair_groups, pair_counts)))
            graph = singles + pairs if repetition % 2 else pairs
            # Perturb input order so brute-force validation covers tie order as well as objectives.
            graph = tuple(graph[index] for index in random.permutation(len(graph)))
            for axis in (None, *range(len(AXES))):
                ranked = sorted(((score(scope, axis), scope) for scope in graph if score(scope, axis) > 0),
                                key=lambda pair: (-pair[0], pair[1].canonical))
                for budget in BUDGETS:
                    selected, size = exact_matching(ranked, budget)
                    if (selected != brute_force(graph, budget, axis)
                            or size > 2 * (budget - 1) * (2 * budget - 1) + 1):
                        raise AssertionError(f'synthetic exact matching mismatch for {count=}, {axis=}, {budget=}')
                    checks += 1
    # Dense equal-score graph exercises canonical ties and the incident bound.
    graph = tuple(Scope('recut', index, pair, 2, (1, 1, 1, 1))
                  for index, pair in enumerate(combinations(range(16), 2)))
    selected, size = exact_matching([(4, scope) for scope in graph], 4)
    if [scope.groups for scope in selected] != [(0, 1), (2, 3), (4, 5), (6, 7)] or size > 43:
        raise AssertionError('dense graph canonical exact matching mismatch')
    # All disjoint edges survive incident pruning, so this exercises the global cap.
    graph = tuple(Scope('split', index, (index,), 2, (1, 1, 0, 0)) for index in range(50))
    selected, size = exact_matching([(2, scope) for scope in graph], 4)
    if selected != graph[:4] or size != 43:
        raise AssertionError('global ranked edge cap or canonical matching mismatch')
    return dict(brute_force_exact_objective_and_tie_checks=checks, dense_canonical_check=True,
                global_cap_check=True)


def run(c75, output):
    if output.is_relative_to(c75):
        raise ValueError('output must be separate from the original C75 run')
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    record = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
                  input_c75=str(c75), output=str(output), command=sys.argv, time_policy='advisory',
                  raw_identity_opened=False, private_reference_opened=False, optimizer_updates=0,
                  candidates_generated=0, partition_scoring=False, gpu_used=False,
                  split='T_core', source_partition='WS', completed_scenes=0, stages=[],
                  address_space_limit_bytes=MAX_AS if sys.platform.startswith('linux') else None,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

    def report(**fields):
        record.update(fields, elapsed_seconds=time.monotonic() - start, peak_rss_bytes=peak_rss_bytes())
        write(output / 'run.json', record)

    try:
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        report(phase='binding_C75')
        originals, manifest, largest = bind_population(c75)
        record['stages'].append('C75_run_manifest_scenes_and_component_summary_bound')
        report(phase='matching_saved_counts', scene_ids=manifest['scene_ids'])
        rows = []
        for original in originals:
            rows.append(analyze_scene(c75, original, manifest))
            report(completed_scenes=len(rows), latest_scene=original['scene_id'])
        summary = dict(
            schema_version='c76-disjoint-scope-bound-v1', axes=list(AXES), budgets=list(BUDGETS),
            boundary='Optimistic counts of old complete-pair errors from T32, ignoring new errors and '
                     'partition feasibility. No actual candidates, selector, target coverage, or fusion are evaluated.',
            primary_mode='mixed',
            mixed_definition='Single-group split and two-group recut, with no repeated original group. '
                             'On a given pair recut dominates merge on all four relaxed axes.',
            fixed_sum_rule='Maximize the unweighted four-axis sum, then prefer fewer positive-scoring scopes '
                           'and the earliest canonical set (split, merge, recut; then original scope row). '
                           'For family-specific k=1, preserve the first all-zero C75 row if present.',
            marginal_rule='Optimize one axis at a time with the same constraints. The four maxima may refer '
                          'to different selections and do not form a joint scope vector.',
            group_count_gap_rule='Absolute old group/identity count gap <= k is only a necessary condition '
                                 'if allowing edits that change group count by at most one.',
            comparison_population='Same original complete upper-triangle pairs and WS partition as C75.',
            largest_component=largest, all_scenes=summarize_population(rows),
            components={component: summarize_population([row for row in rows if row['component_id'] == component])
                        for component in sorted(set(manifest['components'].values()))},
            without_largest=summarize_population([row for row in rows if row['component_id'] != largest]),
        )
        if summary['all_scenes']['nonempty_small_graph_exact_checks'] == 0:
            raise ValueError('no real nonempty small-graph exhaustive checks were exercised')
        record['stages'].extend(('all_k1_family_rows_match_C75', 'real_small_graph_matches_independent_brute_force'))
        write(output / 'scenes.json', rows)
        write(output / 'summary.json', summary)
        report(status='COMPLETED', exit_code=0, phase='done',
               small_graph_exact_checks=summary['all_scenes']['small_graph_exact_checks'],
               nonempty_small_graph_exact_checks=summary['all_scenes']['nonempty_small_graph_exact_checks'],
               completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c75-run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        if args.c75_run is not None or args.output is not None or args.time_policy is not None:
            parser.error('--self-check uses synthetic graphs and does not accept run options')
        print(self_check())
    else:
        if args.c75_run is None or args.output is None or args.time_policy is None:
            parser.error('--c75-run, --output and --time-policy advisory are required')
        run(args.c75_run.resolve(), args.output.resolve())
