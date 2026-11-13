"""C53: fixed056 edge evidence, original partition plus eight fixed permutations.

Reads only the completed association run, its score cache and manifest metadata.
No native observations, reference, weights, model construction or score fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
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

# Bound numerical-library worker pools before importing NumPy/Torch. The decoder
# imports Torch's Tensor type but does not construct a model or touch CUDA.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np

from association.evaluation.association import _aggregate, _components, _records, evaluate_scenes, predict
from association.evaluation.physical import group_membership_diagnostics

SOURCE_RUN = 'LEVEL1-ASSOCIATION-20261022-056-attempt-1'
THRESHOLD = 1.7320103112387182
SEED = 20261029
PERMUTATIONS = 8
EXPECTED = dict(scenes=300, components=38, nodes=24599, pairs=4439092, positives=24675,
                tp=22666, fp=55652, fn=2009, tn=4358765)
FIELDS = ('node_ids', 'sources', 'pairs', 'scores', 'labels', 'joint_support_queries')
SCOPES = ('all', 'itsa', 'mstsa')
COUNT_KEYS = ('node_count', 'pair_count', 'positive_count', 'tp', 'fp', 'fn', 'tn')
MAX_AS = 8 * 1024**3
RAM_BUDGET = 4 * 1024**3


def read(path):
    path = Path(path)
    if not path.exists() and path.suffix == '.json':
        path = path.with_suffix('.json.gz')
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def canonical_membership(value, n):
    """Canonical labels by first original-node occurrence; never compare group IDs."""
    value = np.asarray(value)
    if value.shape == (0,) and n == 0:
        return np.empty(0, np.int64)
    if value.shape != (n,) or value.dtype.kind not in 'iu' or (value < 0).any():
        raise ValueError('membership must label every original node exactly once')
    mapping = {}
    result = np.empty(n, np.int64)
    for node, group in enumerate(value):
        result[node] = mapping.setdefault(int(group), len(mapping))
    return result


def decode(n, pairs, scores):
    from association.evaluation.grouping import group_predicted_tracks
    groups = group_predicted_tracks(n, pairs, scores, threshold=THRESHOLD,
                                    grouping_policy='anchor_direct', node_order='input')
    flat = [node for group in groups for node in group]
    if sorted(flat) != list(range(n)) or any(not group for group in groups):
        raise ValueError('decoder omitted or repeated nodes')
    membership = np.empty(n, np.int64)
    for group, nodes in enumerate(groups):
        membership[nodes] = group
    return canonical_membership(membership, n)


def load_record(saved, sid, component):
    record = {key: saved[f'{sid}_{key}'] for key in FIELDS}
    record.update(scene_id=sid, component_id=component)
    n = len(record['node_ids'])
    nodes = record['node_ids']
    if nodes.shape != (n,) or nodes.dtype.kind not in 'US' or len(set(nodes.tolist())) != n:
        raise ValueError('unique scalar original node identifiers required')
    if record['sources'].shape != (n,) or record['sources'].dtype.kind not in 'iuUS':
        raise ValueError('original source vector required')
    support = record['joint_support_queries']
    if support.shape != (len(record['scores']),) or support.dtype.kind not in 'iu' or (support < 0).any() or (support > 64).any():
        raise ValueError('nonnegative joint_support_queries required')
    record = _records([record])[0]
    if len(record['pairs']) != n * (n - 1) // 2:
        raise ValueError('complete unordered i<j candidate population required')
    return record


def counts_match(actual, expected, label):
    for scope in SCOPES:
        left = actual['per_scene'][0] if scope == 'all' else actual[scope]['per_scene'][0]
        right = expected[scope]
        if any(left[key] != right[key] for key in COUNT_KEYS):
            raise ValueError(f'{label}: raw {scope} per-scene counts differ')
        if left['scene_id'] != right['scene_id'] or left['component_id'] != right['component_id']:
            raise ValueError(f'{label}: scene/component binding differs')


def truth_membership(record):
    """Derive identity only if every complete-pair label agrees with its closure."""
    n, pairs, labels = len(record['sources']), record['pairs'], record['labels']
    truth = canonical_membership(_components(n, pairs, labels), n)
    if not np.array_equal(truth[pairs[:, 0]] == truth[pairs[:, 1]], labels):
        return None
    return truth


def partition_details(membership, pairs, accepted, truth):
    co = membership[pairs[:, 0]] == membership[pairs[:, 1]]
    sizes = np.bincount(membership) if len(membership) else np.array([], np.int64)
    item = dict(group_count=len(sizes), singleton_count=int(np.sum(sizes == 1)),
                raw_rejected_within_group=int(np.count_nonzero(co & ~accepted)),
                raw_accepted_across_groups=int(np.count_nonzero(~co & accepted)),
                pair_count=len(pairs), raw_accepted_count=int(accepted.sum()),
                decoded_positive_count=int(co.sum()),
                within_group_pair_count=int(co.sum()), across_group_pair_count=int((~co).sum()),
                raw_rejected_count=int((~accepted).sum()),
                identity_diagnostics_status='VERIFIED_COMPLETE_LABEL_EQUIVALENCE' if truth is not None
                    else 'UNAVAILABLE_INCONSISTENT_LABEL_RELATION',
                mixed_group_count=None, split_identity_count=None, input_identity_count=None)
    if truth is not None:
        diagnosis = group_membership_diagnostics(membership.tolist(), truth.tolist())
        item.update(input_identity_count=len(set(truth.tolist())), identity_partition=diagnosis,
                    mixed_group_count=sum(len(set(truth[membership == g].tolist())) > 1 for g in range(len(sizes))),
                    split_identity_count=sum(len(set(membership[truth == g].tolist())) > 1 for g in set(truth.tolist())))
    return co, item


def metric_rows(result):
    return {scope: result['per_scene'][0] if scope == 'all' else result[scope]['per_scene'][0]
            for scope in SCOPES}


def analyze_scene(record, original_membership, rng):
    """Permutation acts on node indices, with edges carried along and mapped back."""
    n, pairs = len(record['sources']), record['pairs']
    raw = evaluate_scenes([record], THRESHOLD)
    accepted = predict(record['scores'], THRESHOLD)
    baseline = decode(n, pairs, record['scores'])
    if not np.array_equal(baseline, canonical_membership(original_membership, n)):
        raise ValueError('A0 differs from original saved membership relation')
    truth = truth_membership(record)
    baseline_co = baseline[pairs[:, 0]] == baseline[pairs[:, 1]]
    arms, memberships, permutations = [], [], []
    for arm in range(PERMUTATIONS + 1):
        order = np.arange(n, dtype=np.int64) if arm == 0 else np.asarray(rng.sample(range(n), n), np.int64)
        inverse = np.empty(n, np.int64)
        inverse[order] = np.arange(n)
        mapped_pairs = np.sort(inverse[pairs], axis=1)
        # Row k remains the same named edge: scores/labels move with that edge.
        if not np.array_equal(np.sort(order[mapped_pairs], axis=1), pairs):
            raise ValueError('inverse permutation changed original edge endpoints')
        moved = dict(record, node_ids=record['node_ids'][order], sources=record['sources'][order], pairs=mapped_pairs)
        if (not np.array_equal(moved['scores'], record['scores']) or
                not np.array_equal(predict(moved['scores'], THRESHOLD), accepted)):
            raise ValueError('permutation changed raw scores/accepted edges')
        moved_raw = evaluate_scenes([moved], THRESHOLD)
        counts_match(moved_raw, metric_rows(raw), 'permutation control')
        permuted = decode(n, mapped_pairs, moved['scores'])
        restored = np.empty(n, np.int64)
        restored[order] = permuted
        restored = canonical_membership(restored, n)
        co, detail = partition_details(restored, pairs, accepted, truth)
        # Public evaluator gives exactly the existing all/ITSA/MSTSA definitions.
        decoded = evaluate_scenes([dict(record, scores=co.astype(np.float64))], .5)
        changed = int(np.count_nonzero(co != baseline_co))
        detail.update(arm=f'A{arm}', partition_changed=not np.array_equal(restored, baseline),
                      changed_comembership_pairs=changed,
                      changed_comembership_fraction=changed / len(pairs) if len(pairs) else None,
                      raw_counts_unchanged=True, decoded=decoded)
        arms.append(detail)
        memberships.append(restored.tolist())
        permutations.append(order.tolist())
    # Labels are used above for evaluation only, never to generate these orders.
    return dict(scene_id=record['scene_id'], component_id=record['component_id'], node_count=n,
                pair_count=len(pairs), raw=raw, arms=arms, memberships=memberships,
                permutations_new_to_original=permutations,
                unique_permutation_count=len({tuple(p) for p in permutations}),
                unique_partition_count=len({tuple(p) for p in memberships}),
                changed_permutation_count=sum(a['partition_changed'] for a in arms[1:]),
                any_partition_changed=any(a['partition_changed'] for a in arms[1:]))


def metadata(source, manifest):
    if source.name != 'association' or source.parent.name != SOURCE_RUN:
        raise ValueError('source must be original056 attempt-1/association')
    run, summary = read(source / 'run.json'), read(source / 'summary.json')
    config = read(Path(__file__).resolve().parents[2] / 'configs/evidence-association-level1.json')
    if (run.get('status') != 'COMPLETED' or summary.get('status') != 'COMPLETED'
            or run.get('config') != config or run.get('kind') != config['scope']
            or run.get('optimizer_updates') != 1000 or summary.get('optimizer_updates') != 1000
            or summary.get('scope') != config['scope'] or summary.get('split_unit') != 'scene'
            or any(row.get(key) is not False for row in (run, summary)
                   for key in ('synthetic', 'reference_read', 'test_opened'))):
        raise ValueError('completed original real056 config/data-use binding required')
    ids = manifest.get('splits', {}).get('V_select', [])
    if (manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene'
            or len(ids) != EXPECTED['scenes'] or len(set(ids)) != len(ids)
            or any(not isinstance(s, str) or not s.isdigit() for s in ids)):
        raise ValueError('fixed complete Level1 V300 scene manifest required')
    components = manifest.get('components', {})
    if any(s not in components for s in ids) or len({components[s] for s in ids}) != EXPECTED['components']:
        raise ValueError('fixed shared-identity components required')
    if (summary.get('final_selection', {}).get('threshold') != THRESHOLD
            or summary.get('selected', {}).get('selection', {}).get('threshold') != THRESHOLD):
        raise ValueError('actual056 final/selected thresholds do not match frozen theta')
    final = summary['final_result']
    if final.get('grouping', {}).get('policy') != 'anchor_direct/input; every fragment is an anchor in helper diagnostics':
        raise ValueError('original anchor_direct/input grouping binding required')
    groups = final['grouping']['per_scene']
    if [row['scene_id'] for row in groups] != ids:
        raise ValueError('original membership scene/order differs from fixed manifest')
    expected_rows = {scope: final['metrics']['per_scene'] if scope == 'all'
                     else final['metrics'][scope]['per_scene'] for scope in SCOPES}
    if any([r['scene_id'] for r in rows] != ids for rows in expected_rows.values()):
        raise ValueError('original raw metric population/order differs')
    population = summary.get('population', {})
    if any(population.get(key) != EXPECTED[key] for key in ('scenes', 'components', 'nodes', 'pairs', 'positives')):
        raise ValueError('original056 declared population differs from plan')
    if (final['metrics'].get('pair_count') != EXPECTED['pairs'] or
            any(final['metrics'].get(key) != EXPECTED[key] for key in ('tp', 'fp', 'fn', 'tn'))):
        raise ValueError('original056 declared confusion counts differ from plan')
    return ids, components, {row['scene_id']: row['membership'] for row in groups}, expected_rows


def aggregate(rows):
    results = {}
    for arm in range(PERMUTATIONS + 1):
        scopes = {}
        for scope in SCOPES:
            items = [metric_rows(row['arms'][arm]['decoded'])[scope] for row in rows]
            scopes[scope] = _aggregate(items)
        details = [row['arms'][arm] for row in rows]
        valid = [item for item in details if item['identity_diagnostics_status'] == 'VERIFIED_COMPLETE_LABEL_EQUIVALENCE']
        scopes['partition_diagnostics'] = dict(
            **{key: sum(item[key] for item in details) for key in (
                'group_count', 'singleton_count', 'raw_rejected_within_group',
                'raw_accepted_across_groups', 'raw_accepted_count', 'decoded_positive_count', 'pair_count',
                'within_group_pair_count', 'across_group_pair_count', 'raw_rejected_count')},
            identity_defined_scene_count=len(valid), identity_unavailable_scene_count=len(details) - len(valid),
            **{key: sum(item[key] for item in valid) if len(valid) == len(details) else None for key in (
                'mixed_group_count', 'split_identity_count', 'input_identity_count')},
            defined_identity_totals={key: sum(item[key] for item in valid) for key in (
                'mixed_group_count', 'split_identity_count', 'input_identity_count')})
        results[f'A{arm}'] = scopes
    def number_summary(values):
        return dict(count=len(values), minimum=float(min(values)), median=float(np.median(values)), maximum=float(max(values))) if values else dict(count=0, minimum=None, median=None, maximum=None)
    scene_deltas = {}
    for scope in SCOPES:
        scene_deltas[scope] = {}
        for arm in range(1, PERMUTATIONS + 1):
            scene_deltas[scope][f'A{arm}'] = {}
            for key in ('f1', 'p_ca', 'p_fa', 'tp', 'fp', 'fn', 'tn'):
                signs = Counter()
                for row in rows:
                    before = metric_rows(row['arms'][0]['decoded'])[scope][key]
                    after = metric_rows(row['arms'][arm]['decoded'])[scope][key]
                    signs['undefined' if before is None or after is None else
                          'zero' if after == before else 'positive' if after > before else 'negative'] += 1
                scene_deltas[scope][f'A{arm}'][key] = {sign: signs[sign]
                    for sign in ('positive', 'zero', 'negative', 'undefined')}
    deltas = {}
    for scope in SCOPES:
        deltas[scope] = {}
        for key in ('macro_f1', 'p_ca', 'p_fa', 'tp', 'fp', 'fn', 'tn'):
            first = results['A0'][scope][key]
            values = [None if first is None or results[f'A{k}'][scope][key] is None
                      else results[f'A{k}'][scope][key] - first for k in range(1, PERMUTATIONS + 1)]
            deltas[scope][key] = dict(permutation_deltas=values,
                **number_summary([v for v in values if v is not None]))
        for key in ('m_pca', 'm_pfa', 'mw_pca', 'mw_pfa'):
            first = results['A0'][scope]['paper_metrics'][key]
            values = [None if first is None or results[f'A{k}'][scope]['paper_metrics'][key] is None
                      else results[f'A{k}'][scope]['paper_metrics'][key] - first for k in range(1, PERMUTATIONS + 1)]
            deltas[scope][key] = dict(permutation_deltas=values,
                **number_summary([v for v in values if v is not None]))
    return dict(scene_count=len(rows), eligible_node_ge2_scenes=sum(r['node_count'] >= 2 for r in rows),
        changed_scene_count=sum(r['any_partition_changed'] for r in rows),
        changed_scene_fraction=sum(r['any_partition_changed'] for r in rows) / len(rows) if rows else None,
        eligible_changed_scene_count=sum(r['any_partition_changed'] for r in rows if r['node_count'] >= 2),
        eligible_changed_scene_fraction=(sum(r['any_partition_changed'] for r in rows if r['node_count'] >= 2) /
            sum(r['node_count'] >= 2 for r in rows)) if any(r['node_count'] >= 2 for r in rows) else None,
        decoded=results, paired_changes=deltas, per_scene_delta_signs=scene_deltas,
        changed_scene_count_by_arm={f'A{k}': sum(r['arms'][k]['partition_changed'] for r in rows) for k in range(1, PERMUTATIONS + 1)},
        scene_delta_signs={f'A{k}': {key: dict(Counter(
            'zero' if r['arms'][k]['decoded'][key] == r['arms'][0]['decoded'][key] else
            ('positive' if r['arms'][k]['decoded'][key] > r['arms'][0]['decoded'][key] else 'negative') for r in rows))
            for key in ('tp', 'fp', 'fn', 'tn')} for k in range(1, PERMUTATIONS + 1)})


def run(args):
    source, output = Path(args.source_association).resolve(), Path(args.output).resolve()
    if output == source.parent or source.parent in output.parents or output in source.parent.parents:
        raise ValueError('output must be separate from immutable original056 run')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', scope='C53-frozen056-score-decision-diagnostic',
        idea_id='IDEA-FROZEN-SCORE-DECISION-20261029', plan_version='1.0',
        source_association=str(source), manifest=str(Path(args.manifest).resolve()),
        cpu_threads=args.cpu_threads, phase='preflight', completed_scenes=0,
        threshold=THRESHOLD, permutation_count=PERMUTATIONS, seed=SEED,
        model_forwards=0, models_constructed=0, optimizer_updates=0, gpu_used=False,
        scores_recomputed=False, threshold_reselected=False, native_read=False,
        reference_read=False, test_opened=False, ram_budget_bytes=RAM_BUDGET)
    write(output / 'run.json', state)
    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic() - started, peak_rss_bytes=peak_rss_bytes(), **extra)
        write(output / 'run.json', state)
        if state['peak_rss_bytes'] > RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')
    result = None
    try:
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            selected = min([MAX_AS] + [x for x in (soft, hard) if x != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (selected, hard))
            state['address_space_limit_bytes'] = selected
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
                os.sched_setaffinity(0, cpus)
                state['cpu_affinity'] = cpus
        import torch
        torch.set_num_threads(args.cpu_threads)
        manifest = read(args.manifest)
        ids, components, memberships, expected_rows = metadata(source, manifest)
        state['total_scenes'] = len(ids)
        cache = source / 'final-scores.npz'
        before = cache.stat()
        if not cache.is_file():
            raise ValueError('original final-scores.npz required; no fallback')
        with np.load(cache, allow_pickle=False) as saved:
            expected_keys = {'scene_ids'} | {f'{sid}_{key}' for sid in ids for key in FIELDS}
            if set(saved.files) != expected_keys or saved['scene_ids'].tolist() != ids:
                raise ValueError('score-cache schema/population/order differs; extra fields forbidden')
            totals = Counter()
            for index, sid in enumerate(ids):
                record = load_record(saved, sid, components[sid])
                raw = evaluate_scenes([record], THRESHOLD)
                counts_match(raw, {scope: expected_rows[scope][index] for scope in SCOPES}, sid)
                if not np.array_equal(decode(len(record['sources']), record['pairs'], record['scores']),
                                      canonical_membership(memberships[sid], len(record['sources']))):
                    raise ValueError(f'{sid}: A0 differs from original membership relation')
                totals.update(nodes=len(record['sources']), pairs=len(record['pairs']), positives=int(record['labels'].sum()),
                              **{key: raw[key] for key in ('tp', 'fp', 'fn', 'tn')})
                checkpoint(preflight_scenes=index + 1)
            if any(totals[key] != EXPECTED[key] for key in totals):
                raise ValueError('actual frozen056 full population/counts differ')
            checkpoint(phase='permutations', preflight_verified=True, actual_population=dict(totals))
            (output / 'scenes').mkdir()
            rows, rng = [], random.Random(SEED)
            for sid in ids:
                record = load_record(saved, sid, components[sid])
                row = analyze_scene(record, memberships[sid], rng)
                write(output / 'scenes' / f'scene-{sid}.json.gz', row)
                # Retain only O(scenes x arms) summaries; no nine full edge arrays.
                row.pop('permutations_new_to_original')
                row.pop('memberships')
                rows.append(row)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        after = cache.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('original cache changed during read-only diagnostic')
        result = aggregate(rows)
        result.update(status='COMPLETED', **{key: state[key] for key in (
            'scope', 'idea_id', 'plan_version', 'threshold', 'permutation_count', 'seed')},
            scene_ids=ids, raw={**_aggregate([metric_rows(r['raw'])['all'] for r in rows]),
                **{scope: _aggregate([metric_rows(r['raw'])[scope] for r in rows]) for scope in ('itsa', 'mstsa')}},
            source_cache=dict(path=str(cache), size_bytes=before.st_size, mtime_ns=before.st_mtime_ns),
            components={cid: aggregate([r for r in rows if r['component_id'] == cid]) for cid in sorted(set(components[s] for s in ids))},
            input_population=state['actual_population'], raw_invariance_all_scenes=True,
            original_A0_membership_verified=True, elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=peak_rss_bytes(), model_forwards=0, optimizer_updates=0,
            limitations=['fixed development V population; no independent generalization or permutation-seed CI',
                'raw score metrics fixed; decoded relation is separately named',
                'input-node identity diagnostics require complete consistent label graph; no full target registry',
                'no native/reference or new conditional states; no physical-fusion/target-coverage claim'])
        checkpoint(phase='writing_results')
        write(output / 'scene-summary.json.gz', rows)
        # Check the final serialization peak before publishing a success summary.
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        write(output / 'summary.json', result)
        checkpoint(status='COMPLETED', phase='done', exit_code=0)
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                     elapsed_seconds=time.monotonic() - started, peak_rss_bytes=peak_rss_bytes())
        write(output / 'run.json', state)
        if result is not None and (output / 'summary.json').exists():
            result.update(status=state['status'], error=state['error'], exit_code=state['exit_code'])
            write(output / 'summary.json', result)
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
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        return 1
    print(f'COMPLETED: {args.output}/summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
