"""C75: optimistic error-touch bounds for one edit of a frozen WS partition.

No partition candidates, scoring, fitting, native coordinates or reference states.
Four nonnegative axes count old errors inside an editable scope; they are NOT
jointly attainable improvements. All original groups and group pairs are kept.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path
import resource
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np

from tools.analysis.same_source_swap_pilot import (
    MAX_AS, MAX_NODES, PARENTS, canonical_membership, load_identities, load_population,
    load_scene, peak_rss_bytes, write,
)

AXES = ('same_fp_removal', 'cross_fp_removal', 'same_fn_recovery', 'cross_fn_recovery')
FAMILIES = ('split', 'merge', 'recut')
C71_RECOVERY = {'same_fn_recovery': 2, 'cross_fn_recovery': 14}
BOUNDARY = ('Counts of original errors that one scope can touch, ignoring new errors, '
            'transitivity and the required number of output groups. Even one complete '
            'scope vector is a relaxed bound, not an attainable candidate or Pareto improvement.')


def axis_dict(values):
    return {name: int(value) for name, value in zip(AXES, values)}


def fractions(values, denominator):
    return {name: int(value) / int(total) if total else None
            for name, value, total in zip(AXES, values, denominator)}


def build_scopes(membership):
    """Identity-free exhaustive scope order; singleton split scopes remain present."""
    membership = canonical_membership(membership, len(membership))
    groups = int(membership.max()) + 1
    sizes = np.bincount(membership, minlength=groups)
    singles = np.column_stack((np.arange(groups), np.full(groups, -1)))
    doubles = np.column_stack(np.triu_indices(groups, 1))
    return membership, sizes, {'split': singles, 'merge': doubles, 'recut': doubles.copy()}


def _edge_counts(membership, sources, identities, pairs):
    """Production calculation from the frozen full upper-triangle edge population."""
    groups = int(membership.max()) + 1
    left, right = pairs.T
    same = sources[left] == sources[right]
    positive = identities[left] == identities[right]
    predicted = membership[left] == membership[right]
    a = np.minimum(membership[left], membership[right])
    b = np.maximum(membership[left], membership[right])
    within_fp = np.zeros((groups, 2), dtype=np.int64)
    between_fn = np.zeros((groups, groups, 2), dtype=np.int64)
    baseline = {}
    for index, (name, mask) in enumerate((('same', same), ('cross', ~same))):
        fp = mask & predicted & ~positive
        fn = mask & ~predicted & positive
        within_fp[:, index] = np.bincount(a[fp], minlength=groups)
        between_fn[:, :, index] = np.bincount(
            a[fn] * groups + b[fn], minlength=groups * groups).reshape(groups, groups)
        baseline[name] = dict(tp=int(np.count_nonzero(mask & predicted & positive)),
                              fp=int(fp.sum()), fn=int(fn.sum()),
                              tn=int(np.count_nonzero(mask & ~predicted & ~positive)),
                              pair_count=int(mask.sum()))
    return within_fp, between_fn, baseline


def verify_scope_bounds(membership, sources, identities, within_fp, between_fn):
    """Independent exact check from group/identity/source contingencies, no pairs.

    Each identity contributes its outer product of original-group multiplicities.
    All original scope vectors are linear reads of the arrays checked here.
    """
    groups = len(within_fp)
    check_fp = np.zeros_like(within_fp)
    same_truth = np.zeros((groups, groups), dtype=np.int64)
    all_truth = np.zeros_like(same_truth)
    by_identity, by_identity_source = {}, {}
    for group, source, identity in zip(membership.tolist(), sources.tolist(), identities.tolist()):
        by_identity.setdefault(identity, Counter())[group] += 1
        by_identity_source.setdefault((identity, source), Counter())[group] += 1

    def add_products(destination, entries):
        for counts in entries.values():
            indices = np.array(list(counts), dtype=np.int64)
            values = np.array(list(counts.values()), dtype=np.int64)
            destination[np.ix_(indices, indices)] += values[:, None] * values[None, :]

    add_products(all_truth, by_identity)
    add_products(same_truth, by_identity_source)
    for group in range(groups):
        selected = membership == group
        n = int(selected.sum())
        source_counts = Counter(sources[selected].tolist())
        same_total = sum(count * (count - 1) // 2 for count in source_counts.values())
        same_tp = (int(same_truth[group, group]) - n) // 2
        all_tp = (int(all_truth[group, group]) - n) // 2
        check_fp[group] = same_total - same_tp, n * (n - 1) // 2 - same_total - (all_tp - same_tp)
    check_fn = np.stack((np.triu(same_truth, 1), np.triu(all_truth - same_truth, 1)), axis=-1)
    if not np.array_equal(check_fp, within_fp) or not np.array_equal(check_fn, between_fn):
        raise AssertionError('full-edge and independent contingency scope counts disagree')
    return dict(passed=True, method='full-edge counts versus independent identity/source contingencies',
                original_groups=groups, unordered_group_pairs=groups * (groups - 1) // 2,
                checked_scalar_counts=int(within_fp.size + groups * (groups - 1)))


def pareto_indices(vectors):
    """One representative of every nondominated distinct potential vector.

    Descending total ensures that only already accepted vectors can dominate the
    current vector. Duplicate vectors use the first, identity-free scope index.
    This is a frontier of relaxed bounds, never a candidate-performance frontier.
    """
    if not len(vectors):
        return []
    unique, first = np.unique(vectors, axis=0, return_index=True)
    order = np.lexsort((first, -unique.sum(axis=1)))
    frontier = []
    for index in order:
        if not frontier or not np.any(np.all(unique[frontier] >= unique[index], axis=1)):
            frontier.append(int(index))
    return sorted(int(first[index]) for index in frontier)


def summarize_family(scopes, vectors, sizes, denominator, family):
    def describe(index):
        scope = scopes[index]
        groups = scope[scope >= 0]
        values = vectors[index]
        return dict(scope_index=int(index), original_group_indices=groups.tolist(),
                    node_count=int(sizes[groups].sum()), potential=axis_dict(values),
                    fraction_of_original_error=fractions(values, denominator))

    maximum = vectors.max(axis=0) if len(vectors) else np.zeros(4, dtype=np.int64)
    chosen = int(np.argmax(vectors.sum(axis=1))) if len(vectors) else None
    frontier = pareto_indices(vectors)
    return dict(
        scope_count=len(scopes),
        applicable_scope_count=int(np.count_nonzero(sizes >= 2)) if family == 'split' else len(scopes),
        scopes_touching_any_error=int(np.count_nonzero(vectors.sum(axis=1))),
        scopes_touching_fp=int(np.count_nonzero(vectors[:, :2].sum(axis=1))),
        scopes_touching_fn=int(np.count_nonzero(vectors[:, 2:].sum(axis=1))),
        independent_marginal_max=axis_dict(maximum),
        independent_marginal_max_fraction=fractions(maximum, denominator),
        marginal_max_is_not_a_joint_scope=True,
        fixed_sum_choice=describe(chosen) if chosen is not None else None,
        nondominated_distinct_bound_vectors=[describe(index) for index in frontier],
        bound_interpretation=BOUNDARY,
    )


def analyze_scene(sid, component, membership, sources, identities, pairs, frozen_scopes=None):
    """Return a sanitized scene report and numeric, exhaustive per-scope arrays."""
    n = len(membership)
    if (not 0 < n <= MAX_NODES or np.asarray(sources).shape != (n,)
            or np.asarray(identities).shape != (n,)
            or not np.array_equal(pairs, np.column_stack(np.triu_indices(n, 1)))):
        raise ValueError('full original node vectors and upper-triangle pairs required')
    membership, sizes, scopes = build_scopes(membership)
    if frozen_scopes is not None and any(not np.array_equal(scopes[key], frozen_scopes[key]) for key in FAMILIES):
        raise ValueError('identity-free scope enumeration changed')
    within_fp, between_fn, baseline = _edge_counts(membership, sources, identities, pairs)
    verification = verify_scope_bounds(membership, sources, identities, within_fp, between_fn)
    denominator = np.array([baseline['same']['fp'], baseline['cross']['fp'],
                            baseline['same']['fn'], baseline['cross']['fn']], dtype=np.int64)
    arrays, families = {}, {}
    for family in FAMILIES:
        indices = scopes[family]
        values = np.zeros((len(indices), 4), dtype=np.int64)
        a, b = indices.T
        if family == 'split':
            values[:, :2] = within_fp[a]
        elif family == 'merge':
            values[:, 2:] = between_fn[a, b]
        else:
            values[:, :2] = within_fp[a] + within_fp[b]
            values[:, 2:] = between_fn[a, b]
        arrays[f'{family}_group_indices'] = indices
        arrays[f'{family}_original_error_counts'] = values
        arrays[f'{family}_node_count'] = sizes[a] if family == 'split' else sizes[a] + sizes[b]
        families[family] = summarize_family(indices, values, sizes, denominator, family)
    groups, true_groups = len(sizes), len(np.unique(identities))
    gap = abs(groups - true_groups)
    row = dict(scene_id=str(sid), component_id=str(component), nodes=n,
               original_ws_group_count=groups, observed_identity_count=true_groups,
               ws_minus_observed_identity_count=groups - true_groups,
               true_group_count_differs=groups != true_groups,
               absolute_group_count_gap=gap, group_count_gap_bin='0' if gap == 0 else '1' if gap == 1 else '>=2',
               full_truth_partition_requires_more_than_one_group_count_change=gap >= 2,
               full_truth_partition_impossible_by_recut_count=gap != 0,
               baseline=baseline, original_error_denominator=axis_dict(denominator), families=families,
               all_singletons_control={name: dict(delta_tp=-baseline[name]['tp'], delta_fp=-baseline[name]['fp'],
                                                 delta_fn=baseline[name]['tp'], delta_tn=baseline[name]['fp'])
                                       for name in ('same', 'cross')},
               independent_verification=verification)
    return row, arrays


def aggregate(rows):
    baseline = {scope: {key: sum(row['baseline'][scope][key] for row in rows)
                        for key in ('tp', 'fp', 'fn', 'tn', 'pair_count')} for scope in ('same', 'cross')}
    denominator = np.array([baseline['same']['fp'], baseline['cross']['fp'],
                            baseline['same']['fn'], baseline['cross']['fn']], dtype=np.int64)
    families = {}
    for family in FAMILIES:
        entries = [row['families'][family] for row in rows]
        marginal = np.array([sum(entry['independent_marginal_max'][key] for entry in entries) for key in AXES])
        chosen = np.array([sum(entry['fixed_sum_choice']['potential'][key]
                               for entry in entries if entry['fixed_sum_choice'] is not None) for key in AXES])
        families[family] = dict(
            scope_count=sum(entry['scope_count'] for entry in entries),
            applicable_scope_count=sum(entry['applicable_scope_count'] for entry in entries),
            scenes_touching_any_error=sum(entry['scopes_touching_any_error'] > 0 for entry in entries),
            scenes_touching_fn=sum(entry['scopes_touching_fn'] > 0 for entry in entries),
            sum_of_per_scene_independent_marginal_max=axis_dict(marginal),
            marginal_fraction_of_original_error=fractions(marginal, denominator),
            sum_of_per_scene_fixed_sum_scope_vectors=axis_dict(chosen),
            fixed_sum_scope_fraction_of_original_error=fractions(chosen, denominator),
            interpretation='One scope per scene; independent marginal maxima may use different scopes. '
                           'Fixed-sum choices retain all four axes of the same scope, but remain relaxed bounds.')
    return dict(scene_count=len(rows), component_count=len({row['component_id'] for row in rows}),
                nodes=sum(row['nodes'] for row in rows),
                original_ws_group_count=sum(row['original_ws_group_count'] for row in rows),
                scene_identity_count_sum=sum(row['observed_identity_count'] for row in rows),
                baseline=baseline, original_error_denominator=axis_dict(denominator), families=families,
                scenes_with_true_group_count_difference=sum(row['true_group_count_differs'] for row in rows),
                group_count_gap_histogram={key: sum(row['group_count_gap_bin'] == key for row in rows)
                                          for key in ('0', '1', '>=2')},
                all_singletons_control={scope: {key: sum(row['all_singletons_control'][scope][key] for row in rows)
                                                 for key in ('delta_tp', 'delta_fp', 'delta_fn', 'delta_tn')}
                                        for scope in ('same', 'cross')})


def summarize(rows):
    if not rows:
        raise ValueError('nonempty original scene population required')
    components = sorted({row['component_id'] for row in rows})
    node_counts = {component: sum(row['nodes'] for row in rows if row['component_id'] == component)
                   for component in components}
    largest = min(components, key=lambda component: (-node_counts[component], component))
    total = aggregate(rows)
    return dict(
        schema_version='c75-scope-bound-v1', axes=list(AXES), bound_interpretation=BOUNDARY,
        comparison_population='All original upper-triangle pairs, split by same/cross source; '
                              'not paper-protocol ITSA/MSTSA eligibility masks.',
        scope_definitions=dict(split='One complete original group: only its old internal FP.',
                               merge='One pair of complete original groups: only their old between-group FN.',
                               recut='One pair of complete original groups: old FP inside either group and '
                                     'old FN between them; all external relationships stay unchanged.'),
        fixed_sum_rule='Maximize the unweighted sum of four old-error counts; ties use first original '
                       'canonical group/group-pair scope. This is posthoc and not an executable selector.',
        all_scenes=total,
        components={component: aggregate([row for row in rows if row['component_id'] == component])
                    for component in components},
        largest_component=largest, largest_component_rule='Most original nodes; lexical component ID breaks ties.',
        largest_component_node_count=node_counts[largest],
        without_largest=aggregate([row for row in rows if row['component_id'] != largest]),
        c71_reference=dict(population='Original T32 WS anchor-preserving same-source swaps',
                           previously_observed_fn_recovery=C71_RECOVERY,
                           original_fn={'same': 124, 'cross': 385},
                           current_observed_fn_matches_reference=baseline_matches_c71(total),
                           scope_bound_exceeds_c71_by_axis={family: {
                               axis: total['families'][family]['sum_of_per_scene_independent_marginal_max'][axis] > value
                               for axis, value in C71_RECOVERY.items()} for family in FAMILIES},
                           interpretation='If a marginal bound cannot exceed C71 recovery on an axis, this '
                                          'one-edit scope cannot exceed that reference there. Exceeding it '
                                          'only establishes potential room, not reachable or selected gains.'),
        excluded=['candidate generation', 'old-J selection', 'motion scoring', 'training', 'V/test',
                  'reference states', 'physical fusion', 'target-output coverage certification'],
    )


def baseline_matches_c71(total):
    return total['baseline']['same']['fn'] == 124 and total['baseline']['cross']['fn'] == 385


def self_check():
    """Small independent edge-loop checks, including incompatible bound maxima."""
    cases = [([0], [0], ['a']),
             ([0, 0, 0], [0, 1, 0], ['a', 'b', 'c']),
             ([0, 1, 2], [0, 0, 1], ['a', 'a', 'a']),
             ([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], ['a', 'b', 'a', 'c', 'b', 'c'])]
    rng = np.random.default_rng(75)
    for n in range(2, 10):
        cases.append((rng.integers(0, max(1, n // 2), n), rng.integers(0, 3, n), rng.integers(0, 4, n)))
    rows = []
    for case, (membership, sources, identities) in enumerate(cases):
        membership, sources, identities = map(np.asarray, (membership, sources, identities))
        membership, _, scopes = build_scopes(membership)
        pairs = np.array(list(itertools.combinations(range(len(membership)), 2)), dtype=np.int64).reshape(-1, 2)
        row, arrays = analyze_scene(str(case), str(case % 2), membership, sources, identities, pairs, scopes)
        rows.append(row)
        for family in FAMILIES:
            expected = []
            for a, b in scopes[family]:
                counts = [0, 0, 0, 0]
                for i, j in itertools.combinations(range(len(membership)), 2):
                    gi, gj = membership[i], membership[j]
                    same = sources[i] == sources[j]
                    truth = identities[i] == identities[j]
                    if family != 'merge' and gi == gj and gi in (a, b) and not truth:
                        counts[0 if same else 1] += 1
                    if family != 'split' and gi != gj and {gi, gj} == {a, b} and truth:
                        counts[2 if same else 3] += 1
                expected.append(counts)
            np.testing.assert_array_equal(arrays[f'{family}_original_error_counts'],
                                          np.array(expected, dtype=np.int64).reshape(-1, 4))
    # Separate scopes' coordinate maxima must not become a fabricated joint choice.
    vectors = np.array([[5, 0, 0, 0], [0, 4, 3, 0], [0, 0, 0, 0], [0, 4, 3, 0]])
    assert pareto_indices(vectors) == [0, 1]
    family = summarize_family(np.array([[0, 1], [0, 2], [1, 2], [1, 3]]), vectors,
                              np.ones(4, dtype=int), np.array([5, 4, 3, 0]), 'recut')
    assert family['fixed_sum_choice']['potential'] == axis_dict([0, 4, 3, 0])
    assert family['independent_marginal_max'] == axis_dict([5, 4, 3, 0])
    assert family['independent_marginal_max_fraction']['cross_fn_recovery'] is None
    result = summarize(rows)
    assert result['all_scenes']['scene_count'] == len(cases)
    assert all(row['independent_verification']['passed'] for row in rows)
    return dict(passed=True, synthetic_cases=len(cases), checks=['all scopes versus direct edge loops',
                'edge/contingency agreement', 'Pareto duplicates', 'nonjoint marginal maxima',
                'zero-denominator nulls', 'aggregate schema'])


def run(runs, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'scope-counts').mkdir()
    start = time.monotonic()
    record = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
                  inputs={key: str(runs / name) for key, name in PARENTS.items()},
                  split='T_core', source_partition='WS', optimizer_updates=0, candidates_generated=0,
                  weights_loaded=False, gpu_used=False,
                  reference_fields_loaded=False, identity_use='posthoc error-touch upper bounds only',
                  time_policy='advisory', completed_scenes=0,
                  address_space_limit_bytes=MAX_AS if sys.platform.startswith('linux') else None,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  command=sys.argv, output=str(output), stages=[])

    def report(**fields):
        record.update(fields, elapsed_seconds=time.monotonic() - start, peak_rss_bytes=peak_rss_bytes())
        write(output / 'run.json', record)

    try:
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        report(phase='binding_inputs')
        record['synthetic_check'] = self_check()
        scenes, components = load_population(runs)
        # Freeze every scope before any identity file is opened; no label-based pruning.
        frozen = {}
        for sid in scenes:
            nodes, sources, _, pairs, _, memberships = load_scene(runs, sid)
            membership, sizes, scopes = build_scopes(memberships['WS'])
            frozen[sid] = (nodes, sources, pairs, membership, sizes, scopes)
        if sum(len(value[0]) for value in frozen.values()) != 2310:
            raise ValueError('original T32 must contain 2310 nodes')
        write(output / 'scope-manifest.json', dict(scene_ids=scenes, components=components,
              axes=list(AXES), scope_order='canonical original group index; ascending unordered group pairs',
              numeric_array_schema={'<family>_group_indices': '[scope, 2]; split uses (group, -1)',
                                    '<family>_node_count': '[scope]; size of original editable union',
                                    '<family>_original_error_counts': '[scope, 4]; columns in axes order'},
              identity_fields_opened=False, candidate_generation=False,
              scenes={sid: dict(nodes=len(values[0]), groups=len(values[4]),
                                scopes={family: len(values[5][family]) for family in FAMILIES})
                      for sid, values in frozen.items()}))
        record['stages'].append('all_original_scopes_frozen_before_identity_read')
        report(phase='posthoc_scope_counts', scene_count=len(scenes), scene_ids=scenes)
        rows = []
        for sid in scenes:
            nodes, sources, pairs, membership, _, scopes = frozen[sid]
            identities = load_identities(runs, sid, nodes)
            row, arrays = analyze_scene(sid, components[sid], membership, sources, identities, pairs, scopes)
            path = output / 'scope-counts' / f'T_core-{sid}.npz'
            np.savez_compressed(path, **arrays)
            row['scope_counts_file'] = str(path.relative_to(output))
            rows.append(row)
            report(completed_scenes=len(rows), latest_scene=sid)
        summary = summarize(rows)
        if not baseline_matches_c71(summary['all_scenes']):
            raise ValueError('original WS FN counts differ from C71 T32 reference 124/385')
        write(output / 'scenes.json', rows)
        write(output / 'summary.json', summary)
        record['stages'].append('all_scope_counts_independently_verified')
        report(status='COMPLETED', exit_code=0, phase='done', completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--output', type=Path, help='Must not already exist; failures are preserved.')
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true', help='Synthetic software checks only; no input files.')
    args = parser.parse_args()
    if args.self_check:
        if args.runs is not None or args.output is not None:
            parser.error('--self-check does not accept real input or output paths')
        print(self_check())
    else:
        if args.runs is None or args.output is None or args.time_policy is None:
            parser.error('--runs, --output and --time-policy advisory are required')
        run(args.runs.resolve(), args.output.resolve())
