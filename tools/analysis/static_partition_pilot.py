"""114: fixed056 static partition pilot, with prespecified conditional expansion.

All six outputs are legitimate partitions. Complete-pair score consistency and
partition legitimacy are different properties. No native data, model, training,
new scores, threshold selection or conditional-state mechanism is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
import math
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis import frozen_score_decision_diagnostic as frozen
import numpy as np

METHODS = ('A0', 'CC', 'CL', 'WS', 'SRC_REVERSE', 'SRC_WITHIN')
TRIGGERS = ('CC', 'CL', 'WS')
PILOT_COUNT = 32
GAIN_REQUIRED = .005
TOLERANCE = 1e-12
THRESHOLD = frozen.THRESHOLD
SEED = 20261029
DETAIL_COUNTS = ('group_count', 'singleton_count', 'raw_rejected_within_group',
    'raw_accepted_across_groups', 'raw_accepted_count', 'decoded_positive_count', 'pair_count',
    'within_group_pair_count', 'across_group_pair_count', 'raw_rejected_count')


def pilot_selection(ids):
    return sorted(random.Random(SEED).sample(sorted(ids, key=int), PILOT_COUNT), key=int)


def membership_from_clusters(clusters, n):
    result = np.full(n, -1, np.int64)
    for group, members in enumerate(clusters):
        members = np.asarray(members, np.int64)
        if not len(members) or (members < 0).any() or (members >= n).any() or (result[members] >= 0).any():
            raise ValueError('static clusters must be nonempty and cover nodes once')
        if len(np.unique(members)) != len(members):
            raise ValueError('static cluster repeats a node')
        result[members] = group
    return frozen.canonical_membership(result, n)


def agglomerate(n, pairs, scores, method, check=None):
    """Exact min/sum cross-cluster updates; positive eligibility per plan1.0.

    An active cluster ID is its minimum original node index. Exact gain ties use
    the sorted pair of these keys, then IDs (identical here). `tie_decisions`
    counts selected merges with at least two valid equal-best candidates; it
    does not pretend to count every alternative in a tied family.
    """
    if method not in ('CL', 'WS'):
        raise ValueError('agglomeration must explicitly be CL or WS')
    gain = np.full((n, n), -np.inf, dtype=np.float64)
    weights = np.asarray(scores, np.float64)-THRESHOLD
    if not np.isfinite(weights).all():
        raise FloatingPointError('nonfinite signed pair weight')
    gain[pairs[:, 0], pairs[:, 1]] = weights
    gain[pairs[:, 1], pairs[:, 0]] = weights
    eligible = (lambda value: value >= 0.) if method == 'CL' else (lambda value: value > 0.)
    active = np.ones(n, bool)
    versions = np.zeros(n, np.int64)
    clusters = {i: [i] for i in range(n)}
    heap = [(-float(weight), int(a), int(b), 0, 0)
            for (a, b), weight in zip(pairs, weights) if eligible(weight)]
    heapq.heapify(heap)
    merges, ties, trace = 0, 0, []

    def valid(entry):
        _, a, b, va, vb = entry
        return active[a] and active[b] and versions[a] == va and versions[b] == vb

    def clean_top():
        while heap and not valid(heap[0]):
            heapq.heappop(heap)

    while heap:
        clean_top()
        if not heap:
            break
        entry = heapq.heappop(heap)
        negative, a, b, _, _ = entry
        value = -negative
        if not eligible(value):
            raise ValueError('ineligible merge entered the candidate heap')
        clean_top()
        tied = bool(heap and heap[0][0] == negative)
        ties += int(tied)
        trace.append(dict(left_min_node=a, right_min_node=b, gain=value, best_gain_tied=tied))
        # a<b, so retaining a preserves the minimum-original-index cluster key.
        clusters[a].extend(clusters.pop(b))
        active[b] = False
        versions[a] += 1
        others = np.flatnonzero(active)
        others = others[others != a]
        updated = (np.minimum(gain[a, others], gain[b, others]) if method == 'CL'
                   else gain[a, others]+gain[b, others])
        if not np.isfinite(updated).all():
            raise FloatingPointError('nonfinite aggregate cluster gain')
        gain[a, others] = updated
        gain[others, a] = updated
        for c, score in zip(others, updated):
            if eligible(score):
                left, right = sorted((a, int(c)))
                heapq.heappush(heap, (-float(score), left, right, int(versions[left]), int(versions[right])))
        merges += 1
        if check and merges % 32 == 0:
            check()
    result = membership_from_clusters([sorted(clusters[key]) for key in sorted(clusters)], n)
    return result, dict(merge_count=merges, tie_decisions=ties,
                        tie_definition='merge decisions with >=2 exactly equal valid maximum-gain candidates'), trace


def source_orders(sources, scene_id):
    blocks = {}
    for node, source in enumerate(np.asarray(sources).tolist()):
        blocks.setdefault(source, []).append(node)
    ordered = list(blocks.values())
    reverse = np.asarray([node for block in reversed(ordered) for node in block], np.int64)
    rng = random.Random(SEED+int(scene_id))
    within = []
    for block in ordered:
        copied = list(block)
        rng.shuffle(copied)
        within.extend(copied)
    return dict(SRC_REVERSE=reverse, SRC_WITHIN=np.asarray(within, np.int64))


def anchor_in_order(n, pairs, scores, order):
    if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError('source control must permute every original node once')
    inverse = np.empty(n, np.int64)
    inverse[order] = np.arange(n)
    moved_pairs = np.sort(inverse[pairs], axis=1)
    if not np.array_equal(np.sort(order[moved_pairs], axis=1), pairs):
        raise ValueError('source control changed named edge endpoints')
    moved = frozen.decode(n, moved_pairs, scores)
    restored = np.empty(n, np.int64)
    restored[order] = moved
    return frozen.canonical_membership(restored, n)


def analyze_scene(record, original_membership, expected, check=None):
    n, pairs, scores = len(record['sources']), record['pairs'], record['scores']
    raw = frozen.evaluate_scenes([record], THRESHOLD)
    frozen.counts_match(raw, expected, 'static pilot original raw')
    started = time.monotonic()
    baseline = frozen.decode(n, pairs, scores)
    baseline_seconds = time.monotonic()-started
    if not np.array_equal(baseline, frozen.canonical_membership(original_membership, n)):
        raise ValueError('static pilot A0 differs from original membership relation')
    accepted = frozen.predict(scores, THRESHOLD)
    operators = dict(A0=baseline)
    operator_stats, traces = dict(A0=dict(decoder_seconds=baseline_seconds, merge_count=None, tie_decisions=None)), {}
    started = time.monotonic()
    operators['CC'] = frozen.canonical_membership(frozen._components(n, pairs, accepted), n)
    operator_stats['CC'] = dict(decoder_seconds=time.monotonic()-started, merge_count=None, tie_decisions=None)
    for method in ('CL', 'WS'):
        started = time.monotonic()
        operators[method], stats, traces[method] = agglomerate(n, pairs, scores, method, check)
        operator_stats[method] = dict(stats, decoder_seconds=time.monotonic()-started)
    orders = source_orders(record['sources'], record['scene_id'])
    for method, order in orders.items():
        started = time.monotonic()
        operators[method] = anchor_in_order(n, pairs, scores, order)
        operator_stats[method] = dict(decoder_seconds=time.monotonic()-started, merge_count=None, tie_decisions=None)
    # Labels/truth appear only after every partition/order has been constructed.
    truth = frozen.truth_membership(record)
    arms = {}
    baseline_co = baseline[pairs[:, 0]] == baseline[pairs[:, 1]]
    for method in METHODS:
        membership = operators[method]
        co, detail = frozen.partition_details(membership, pairs, accepted, truth)
        if (method == 'CL' and detail['raw_rejected_within_group']) or (
                method == 'CC' and detail['raw_accepted_across_groups']):
            raise ValueError('static decoder violated its stated pair-consistency property')
        objective = float(np.sum((scores-THRESHOLD)[co], dtype=np.float64))
        if not math.isfinite(objective):
            raise FloatingPointError('nonfinite within-group signed surrogate')
        detail.update(arm=method, decoded=frozen.evaluate_scenes([dict(record, scores=co.astype(np.float64))], .5),
            signed_surrogate_J=objective,
            changed_comembership_pairs=int(np.count_nonzero(co != baseline_co)),
            partition_changed=not np.array_equal(membership, baseline),
            **operator_stats.get(method, dict(decoder_seconds=None, merge_count=None, tie_decisions=None)))
        arms[method] = detail
    private = dict(memberships={key: value.tolist() for key, value in operators.items()}, merge_traces=traces,
                   source_orders_new_to_original={key: value.tolist() for key, value in orders.items()})
    return dict(scene_id=record['scene_id'], component_id=record['component_id'], node_count=n,
                pair_count=len(pairs), raw=raw, raw_invariance_verified=True,
                original_A0_membership_verified=True, arms=arms), private


def difference(after, before):
    return None if after is None or before is None else after-before


def aggregate(rows):
    raw = {scope: frozen._aggregate([frozen.metric_rows(row['raw'])[scope] for row in rows])
           for scope in frozen.SCOPES}
    decoded = {}
    for method in METHODS:
        result = {scope: frozen._aggregate([frozen.metric_rows(row['arms'][method]['decoded'])[scope]
                                           for row in rows]) for scope in frozen.SCOPES}
        details = [row['arms'][method] for row in rows]
        defined = [row for row in details if row['identity_diagnostics_status'] == 'VERIFIED_COMPLETE_LABEL_EQUIVALENCE']
        result['partition_diagnostics'] = dict(
            **{key: sum(row[key] for row in details) for key in DETAIL_COUNTS},
            **{key: sum(row[key] for row in defined) if len(defined) == len(details) else None
               for key in ('mixed_group_count', 'split_identity_count', 'input_identity_count')},
            identity_defined_scene_count=len(defined), identity_unavailable_scene_count=len(details)-len(defined),
            signed_surrogate_J=math.fsum(row['signed_surrogate_J'] for row in details),
            merge_count=sum(row['merge_count'] for row in details if row['merge_count'] is not None)
                if method in ('CL', 'WS') else None,
            tie_decisions=sum(row['tie_decisions'] for row in details if row['tie_decisions'] is not None)
                if method in ('CL', 'WS') else None,
            decoder_seconds=math.fsum(row['decoder_seconds'] for row in details if row['decoder_seconds'] is not None),
            changed_scene_count=sum(row['partition_changed'] for row in details),
            changed_comembership_pairs=sum(row['changed_comembership_pairs'] for row in details))
        decoded[method] = result
    paired = {}
    signs = {}
    for method in METHODS[1:]:
        paired[method], signs[method] = {}, {}
        for scope in frozen.SCOPES:
            a, b = decoded[method][scope], decoded['A0'][scope]
            paired[method][scope] = {key: difference(a[key], b[key])
                for key in ('macro_f1', 'micro_f1', 'p_ca', 'p_fa', 'tp', 'fp', 'fn', 'tn')}
            paired[method][scope]['paper_metrics'] = {key: difference(a['paper_metrics'][key], b['paper_metrics'][key])
                for key in ('m_pca', 'm_pfa', 'mw_pca', 'mw_pfa')}
            signs[method][scope] = {}
            for key in ('f1', 'p_ca', 'p_fa'):
                count = Counter()
                for row in rows:
                    before = frozen.metric_rows(row['arms']['A0']['decoded'])[scope][key]
                    after = frozen.metric_rows(row['arms'][method]['decoded'])[scope][key]
                    count['undefined' if before is None or after is None else 'zero' if after == before
                          else 'positive' if after > before else 'negative'] += 1
                signs[method][scope][key] = {sign: count[sign] for sign in ('positive', 'zero', 'negative', 'undefined')}
    return dict(scene_count=len(rows), scene_ids=[row['scene_id'] for row in rows],
                input_population=dict(nodes=sum(row['node_count'] for row in rows), pairs=sum(row['pair_count'] for row in rows)),
                raw=raw, decoded=decoded, paired_changes=paired, per_scene_delta_signs=signs)


def expansion_gate(pilot, controls_verified):
    """Prespecified decoded gate, not a score fit or method selector."""
    baseline = pilot['decoded']['A0']
    decisions = {}
    for method in TRIGGERS:
        candidate = pilot['decoded'][method]
        b = (baseline['all']['macro_f1'], baseline['mstsa']['paper_metrics']['m_pca'],
             baseline['mstsa']['paper_metrics']['m_pfa'], baseline['all']['p_fa'])
        a = (candidate['all']['macro_f1'], candidate['mstsa']['paper_metrics']['m_pca'],
             candidate['mstsa']['paper_metrics']['m_pfa'], candidate['all']['p_fa'])
        valid = [left is not None and right is not None and math.isfinite(left) and math.isfinite(right)
                 for left, right in zip(a, b)]
        defined = all(valid)
        checks = dict(all_macro_f1_gain=bool(valid[0] and a[0] >= b[0]+GAIN_REQUIRED-TOLERANCE),
                      mstsa_m_pca_nondecreasing=bool(valid[1] and a[1] >= b[1]-TOLERANCE),
                      mstsa_m_pfa_nonincreasing=bool(valid[2] and a[2] <= b[2]+TOLERANCE),
                      all_micro_pfa_nonincreasing=bool(valid[3] and a[3] <= b[3]+TOLERANCE),
                      raw_population_A0_controls=bool(controls_verified))
        decisions[method] = dict(passed=bool(defined and all(checks.values())), checks=checks,
                                 required_metrics_defined=defined,
                                 undefined_metrics=[name for name, ok in zip(
                                     ('all_macro_f1', 'mstsa_m_pca', 'mstsa_m_pfa', 'all_micro_pfa'), valid) if not ok],
                                 reason=None if defined else 'null or nonfinite required decoded metric; no defined-only fallback')
    return dict(expand=any(row['passed'] for row in decisions.values()), methods=decisions,
                eligible_methods=list(TRIGGERS), descriptive_source_controls_do_not_trigger=True,
                all_macro_f1_required_gain=GAIN_REQUIRED, tolerance=TOLERANCE)


def run(args):
    source, output = Path(args.source_association).resolve(), Path(args.output).resolve()
    if output == source.parent or source.parent in output.parents or output in source.parent.parents:
        raise ValueError('output must be separate from immutable056 source')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id='IDEA-STATIC-PARTITION-PILOT-20261029', plan_version='1.0',
        phase='preflight', seed=SEED, threshold=THRESHOLD, methods=list(METHODS),
        source_association=str(source), manifest=str(Path(args.manifest).resolve()), cpu_threads=args.cpu_threads,
        completed_scenes=0, model_forwards=0, models_constructed=0, optimizer_updates=0, gpu_used=False,
        scores_recomputed=False, threshold_reselected=False, normalizer_refitted=False,
        native_read=False, reference_read=False, test_opened=False, ram_budget_bytes=frozen.RAM_BUDGET)
    frozen.write(output/'run.json', state)
    result = None

    def resources():
        if frozen.peak_rss_bytes() > frozen.RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')

    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=frozen.peak_rss_bytes(), **extra)
        frozen.write(output/'run.json', state)
        resources()

    try:
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([frozen.MAX_AS]+[x for x in (soft, hard) if x != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard)); state['address_space_limit_bytes'] = limit
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
                os.sched_setaffinity(0, cpus); state['cpu_affinity'] = cpus
        import torch
        torch.set_num_threads(args.cpu_threads)
        manifest = frozen.read(args.manifest)
        ids, components, original, expected = frozen.metadata(source, manifest)
        index = {sid: i for i, sid in enumerate(ids)}
        pilot_ids = pilot_selection(ids)
        remaining_ids = [sid for sid in ids if sid not in set(pilot_ids)]
        checkpoint(pilot_scene_ids=pilot_ids, maximum_scene_count=len(ids), phase='pilot')
        (output/'scenes').mkdir(); (output/'private').mkdir()
        cache = source/'final-scores.npz'; before = cache.stat()
        def immutable():
            after = cache.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('original056 score cache changed during read-only pilot')
        rows, totals = [], Counter()
        with np.load(cache, allow_pickle=False) as saved:
            keys = {'scene_ids'} | {f'{sid}_{key}' for sid in ids for key in frozen.FIELDS}
            if set(saved.files) != keys or saved['scene_ids'].tolist() != ids:
                raise ValueError('original056 score schema/population/order differs')
            def consume(selected):
                for sid in selected:
                    resources()
                    record = frozen.load_record(saved, sid, components[sid])
                    for value in record.values():
                        if isinstance(value, np.ndarray):
                            value.setflags(write=False)
                    expected_scene = {scope: expected[scope][index[sid]] for scope in frozen.SCOPES}
                    row, private = analyze_scene(record, original[sid], expected_scene, resources)
                    frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                    frozen.write(output/'private'/f'scene-{sid}.json.gz', private)
                    totals.update(nodes=row['node_count'], pairs=row['pair_count'],
                                  positives=expected_scene['all']['positive_count'],
                                  **{key: row['raw'][key] for key in ('tp', 'fp', 'fn', 'tn')})
                    rows.append(row)
                    checkpoint(completed_scenes=len(rows), last_scene_id=sid)
            consume(pilot_ids)
            immutable()
            pilot = aggregate(rows)
            gate = expansion_gate(pilot, controls_verified=all(
                row['raw_invariance_verified'] and row['original_A0_membership_verified'] for row in rows))
            frozen.write(output/'pilot-summary.json', dict(pilot=pilot, expansion_gate=gate))
            checkpoint(phase='remainder' if gate['expand'] else 'pilot_complete', expansion_gate=gate)
            remainder_rows = []
            if gate['expand']:
                consume(remaining_ids)
                remainder_rows = rows[PILOT_COUNT:]
                if any(totals[key] != frozen.EXPECTED[key] for key in totals):
                    raise ValueError('expanded actual056 full population/counts differ')
            immutable()
        actual_ids = [sid for sid in ids if any(row['scene_id'] == sid for row in rows)]
        ordered_rows = sorted(rows, key=lambda row: index[row['scene_id']])
        result = dict(status='COMPLETED', idea_id=state['idea_id'], plan_version='1.0',
            pilot=pilot, expansion_gate=gate, expanded=gate['expand'],
            remainder=aggregate(remainder_rows) if remainder_rows else None,
            full_V=aggregate(ordered_rows) if gate['expand'] else None,
            analyzed=aggregate(ordered_rows), completed_scene_ids=actual_ids,
            declared_full_population=dict(frozen.EXPECTED), actual_analyzed_population=dict(totals),
            components={cid: aggregate([row for row in ordered_rows if row['component_id'] == cid])
                        for cid in sorted({row['component_id'] for row in rows})},
            pilot_components={cid: aggregate([row for row in rows[:PILOT_COUNT] if row['component_id'] == cid])
                              for cid in sorted({row['component_id'] for row in rows[:PILOT_COUNT]})},
            remainder_components={cid: aggregate([row for row in remainder_rows if row['component_id'] == cid])
                                  for cid in sorted({row['component_id'] for row in remainder_rows})},
            source_cache=dict(path=str(cache), size_bytes=before.st_size, mtime_ns=before.st_mtime_ns),
            raw_invariance_verified=True, original_A0_membership_verified=True,
            analyzed_bindings_scene_count=len(rows),
            unread_remainder_scene_count=0 if gate['expand'] else len(remaining_ids),
            no_method_selected_as_output=True, model_forwards=0, optimizer_updates=0,
            limitations=['all methods already yield legitimate partitions; score consistency is separate',
                'fixed repeatedly used development cache, not independent generalization',
                'surrogate J is not a probability or true-identity guarantee; merge-only is not global optimization',
                'source controls describe one fixed order intervention and do not select a source-quality policy',
                'raw score metrics unchanged; decoded membership metrics explicitly separate',
                'no conditional state evidence, physical output, full-target coverage or credible-peer claim'])
        frozen.write(output/'scene-summary.json.gz', ordered_rows)
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        frozen.write(output/'summary.json', result)
        checkpoint(status='COMPLETED', phase='done', exit_code=0)
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            error=f'{type(error).__name__}: {error}', exit_code=1,
            elapsed_seconds=time.monotonic()-started, peak_rss_bytes=frozen.peak_rss_bytes())
        frozen.write(output/'run.json', state)
        if result is not None and (output/'summary.json').exists():
            result.update(status=state['status'], error=state['error']); frozen.write(output/'summary.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-association', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc(); return 1
    print(f'COMPLETED: {args.output}/summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
