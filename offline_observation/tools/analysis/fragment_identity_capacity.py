"""C53-Pa: conditional hard-edge capacity on existing056 score-cache labels.

No archive, physical-state cache, reference, checkpoint, model or torch import.
Counts concern valid input nodes, never the full declared target registry.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

if __name__ == '__main__':
    for _thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_thread_setting] = '4'

import numpy as np

THRESHOLD = 1.7320103112387182
SCENE_COUNT = 300
QUERY_COUNT = 64
IDEA_ID = 'IDEA-FRAGMENT-IDENTITY-CAPACITY-20261029'
COUNT_KEYS = ('node_count', 'pair_count', 'positive_edges', 'same_source_positive_edges',
    'cross_source_positive_edges', 'identity_count', 'single_node_identities',
    'single_source_identities', 'same_source_multifragment_identities',
    'eligible_identity_source_pairs', 'zero_edge_identity_source_pairs',
    'restricted_identity_source_pairs', 'restricted_identities', 'restricted_positive_edges',
    'unit_capacity_true_edges', 'unrepresentable_direct_edges_lower_bound',
    'no_joint_support_positive_edges', 'no_joint_support_same_source_positive_edges',
    'no_joint_support_cross_source_positive_edges', 'no_joint_support_restricted_positive_edges')


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    path = Path(path)
    if not path.exists() and path.suffix != '.gz':
        path = path.with_suffix(path.suffix + '.gz')
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def validate_scene(node_ids, sources, pairs, labels, joint_support_queries, scores):
    """Complete pair eligibility and transitive label truth are qualifications."""
    ids, src, edges, truth, support, score = map(np.asarray,
        (node_ids, sources, pairs, labels, joint_support_queries, scores))
    if (ids.ndim != 1 or ids.dtype.kind not in 'US' or len(np.unique(ids)) != len(ids)
            or any(not str(x).strip() for x in ids)):
        raise ValueError('node_ids must be unique nonempty strings')
    n, e = len(ids), len(ids) * (len(ids) - 1) // 2
    if (src.shape != (n,) or src.dtype.kind not in 'US'
            or any(not str(x).strip() for x in src)):
        raise ValueError('one nonempty source string per input node required')
    expected = np.column_stack(np.triu_indices(n, 1))
    if edges.shape != (e, 2) or edges.dtype.kind not in 'iu' or not np.array_equal(edges, expected):
        raise ValueError('pairs must be the complete canonical i<j population in original order')
    if (truth.shape != (e,) or truth.dtype.kind not in 'biuf'
            or not np.isin(truth, (0, 1)).all()):
        raise ValueError('every original pair needs a binary label')
    if (support.shape != (e,) or support.dtype.kind not in 'iu'
            or (support < 0).any() or (support > QUERY_COUNT).any()):
        raise ValueError('joint_support_queries must be integer counts in [0,64] for every pair')
    if (score.shape != (e,) or score.dtype.kind not in 'iuf' or not np.isfinite(score).all()):
        raise ValueError('finite original scores required for binding, not capacity selection')
    truth = truth.astype(bool)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges[truth]:
        a, b = find(int(a)), find(int(b))
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    _, identities = np.unique(roots, return_inverse=True)
    implied = identities[edges[:, 0]] == identities[edges[:, 1]]
    if not np.array_equal(implied, truth):
        raise ValueError('complete pair labels are not an identity equivalence relation; no repair permitted')
    return src, edges, truth, support, score, identities


def analyze_scene(node_ids, sources, pairs, labels, joint_support_queries, scores, *, source_aliases=None):
    """Return anonymous conditional counts; source-pair bounds are not group recall."""
    src, edges, truth, joint, score, identities = validate_scene(
        node_ids, sources, pairs, labels, joint_support_queries, scores)
    aliases = {} if source_aliases is None else source_aliases
    for source in sorted(set(map(str, src))):
        if source not in aliases:
            aliases[source] = len(aliases)
    source_index = np.asarray([aliases[str(s)] for s in src], dtype=np.int64)
    active_sources = sorted(set(map(int, source_index)))
    source_pairs = [(a, b) for i, a in enumerate(active_sources) for b in active_sources[i + 1:]]
    no_joint = joint == 0
    same = source_index[edges[:, 0]] == source_index[edges[:, 1]]
    counts = {key: 0 for key in COUNT_KEYS}
    counts.update(node_count=len(src), pair_count=len(edges), positive_edges=int(truth.sum()),
        same_source_positive_edges=int((truth & same).sum()),
        cross_source_positive_edges=int((truth & ~same).sum()),
        no_joint_support_positive_edges=int((truth & no_joint).sum()),
        no_joint_support_same_source_positive_edges=int((truth & same & no_joint).sum()),
        no_joint_support_cross_source_positive_edges=int((truth & ~same & no_joint).sum()))
    positive_no_joint = Counter()
    for a, b in edges[truth & no_joint]:
        x, y = sorted((int(source_index[a]), int(source_index[b])))
        positive_no_joint[(int(identities[a]), x, y)] += 1
    per_identity, by_pair = [], {pair: dict(true_edges=0, unit_capacity_true_edges=0,
        no_joint_support_positive_edges=0, restricted_identity_count=0,
        zero_edge_identity_count=0) for pair in source_pairs}
    for identity in range(int(identities.max()) + 1 if len(identities) else 0):
        members = np.flatnonzero(identities == identity)
        source_counts = Counter(map(int, source_index[members]))
        rows, total_true, total_capacity, affected = [], 0, 0, False
        for a, b in source_pairs:
            na, nb = source_counts[a], source_counts[b]
            t, bound = na * nb, min(na, nb)
            nj = positive_no_joint[(identity, a, b)]
            restricted = t > bound
            rows.append(dict(source_pair=[f'source_{a}', f'source_{b}'], a=na, b=nb,
                true_edges=t, unit_capacity_true_edges=bound, direct_edge_recall_upper_bound=ratio(bound, t),
                zero_true_edge_denominator=t == 0, restricted=restricted,
                no_joint_support_positive_edges=nj))
            pair_row = by_pair[(a, b)]
            pair_row['true_edges'] += t
            pair_row['unit_capacity_true_edges'] += bound
            pair_row['no_joint_support_positive_edges'] += nj
            pair_row['restricted_identity_count'] += int(restricted)
            pair_row['zero_edge_identity_count'] += int(t == 0)
            total_true += t
            total_capacity += bound
            counts['eligible_identity_source_pairs'] += int(t > 0)
            counts['zero_edge_identity_source_pairs'] += int(t == 0)
            counts['restricted_identity_source_pairs'] += int(restricted)
            if restricted:
                counts['restricted_positive_edges'] += t
                counts['no_joint_support_restricted_positive_edges'] += nj
                affected = True
        counts['restricted_identities'] += int(affected)
        counts['single_node_identities'] += int(len(members) == 1)
        counts['single_source_identities'] += int(len(source_counts) == 1)
        counts['same_source_multifragment_identities'] += int(any(v > 1 for v in source_counts.values()))
        per_identity.append(dict(identity_index=identity, node_count=len(members),
            source_fragment_counts={f'source_{k}': int(v) for k, v in sorted(source_counts.items())},
            same_source_true_edges=sum(v * (v - 1) // 2 for v in source_counts.values()),
            cross_source_true_edges=total_true, unit_capacity_true_edges=total_capacity,
            direct_edge_recall_upper_bound=ratio(total_capacity, total_true),
            no_joint_support_positive_edges=sum(value for (cid, _, _), value in positive_no_joint.items()
                                                if cid == identity), source_pairs=rows))
    counts['identity_count'] = len(per_identity)
    counts['unit_capacity_true_edges'] = sum(x['unit_capacity_true_edges'] for x in per_identity)
    counts['unrepresentable_direct_edges_lower_bound'] = (
        counts['cross_source_positive_edges'] - counts['unit_capacity_true_edges'])
    if (sum(x['cross_source_true_edges'] for x in per_identity) != counts['cross_source_positive_edges']
            or sum(x['same_source_true_edges'] for x in per_identity) != counts['same_source_positive_edges']
            or counts['unrepresentable_direct_edges_lower_bound'] < 0):
        raise ValueError('capacity/identity pair accounting failed')
    source_pair_rows = [dict(source_pair=[f'source_{a}', f'source_{b}'], **row,
        candidate_edges=int((source_index == a).sum()) * int((source_index == b).sum()),
        direct_edge_recall_upper_bound=ratio(row['unit_capacity_true_edges'], row['true_edges']))
        for (a, b), row in by_pair.items()]
    accepted = score >= THRESHOLD
    raw = dict(tp=int((truth & accepted).sum()), fp=int((~truth & accepted).sum()),
        fn=int((truth & ~accepted).sum()), tn=int((~truth & ~accepted).sum()))
    return dict(**counts, source_count=len(active_sources),
        direct_edge_recall_upper_bound=ratio(counts['unit_capacity_true_edges'], counts['cross_source_positive_edges']),
        raw_binding_counts=raw, identities=per_identity, source_pairs=source_pair_rows,
        labels_equivalence_verified=True)


def pool_scenes(rows):
    counts = {key: sum(row[key] for row in rows) for key in COUNT_KEYS}
    bounds = [row['direct_edge_recall_upper_bound'] for row in rows if row['direct_edge_recall_upper_bound'] is not None]
    identity_bounds = [identity['direct_edge_recall_upper_bound'] for row in rows
        for identity in row['identities'] if identity['direct_edge_recall_upper_bound'] is not None]
    return dict(**counts, scene_count=len(rows),
        scenes_with_restricted_identity=sum(row['restricted_identities'] > 0 for row in rows),
        scenes_without_cross_source_true_edges=len(rows) - len(bounds),
        defined_upper_bound_scene_count=len(bounds),
        micro_direct_edge_recall_upper_bound=ratio(counts['unit_capacity_true_edges'], counts['cross_source_positive_edges']),
        defined_scene_equal_upper_bound=float(np.mean(bounds)) if bounds else None,
        strict_all_scene_equal_upper_bound=float(np.mean(bounds)) if bounds and len(bounds) == len(rows) else None,
        defined_upper_bound_identity_count=len(identity_bounds),
        defined_identity_equal_upper_bound=float(np.mean(identity_bounds)) if identity_bounds else None,
        restricted_identity_fraction=ratio(counts['restricted_identities'], counts['identity_count']),
        restricted_positive_edge_fraction=ratio(counts['restricted_positive_edges'], counts['cross_source_positive_edges']),
        no_joint_support_positive_edge_fraction=ratio(counts['no_joint_support_positive_edges'], counts['positive_edges']))


def pool_source_pairs(rows):
    """Only scenes with both source classes have a row; record that denominator."""
    groups = {}
    for scene in rows:
        for row in scene['source_pairs']:
            groups.setdefault(tuple(row['source_pair']), []).append(row)
    result = []
    for pair, values in sorted(groups.items()):
        keys = ('true_edges', 'unit_capacity_true_edges', 'candidate_edges',
                'no_joint_support_positive_edges', 'restricted_identity_count', 'zero_edge_identity_count')
        totals = {key: sum(row[key] for row in values) for key in keys}
        bounds = [row['direct_edge_recall_upper_bound'] for row in values
                  if row['direct_edge_recall_upper_bound'] is not None]
        result.append(dict(source_pair=list(pair), **totals, total_scene_count=len(rows),
            scenes_with_both_sources=len(values), defined_upper_bound_scene_count=len(bounds),
            micro_direct_edge_recall_upper_bound=ratio(totals['unit_capacity_true_edges'], totals['true_edges']),
            defined_scene_equal_upper_bound=float(np.mean(bounds)) if bounds else None))
    return result


def _input_contract(manifest, summary):
    if manifest.get('dataset_level') != 'Level1':
        raise ValueError('explicit frozen Level1 manifest required')
    scenes = manifest['splits']['V_select']
    if len(scenes) != SCENE_COUNT or len(set(scenes)) != SCENE_COUNT:
        raise ValueError('exact frozen V300 scene order required')
    if (summary.get('status') != 'COMPLETED' or summary.get('optimizer_updates') != 1000
            or summary['final_selection']['threshold'] != THRESHOLD):
        raise ValueError('completed056 final1000 summary and original threshold required')
    metrics = summary['final_result']['metrics']
    if metrics['threshold'] != THRESHOLD or [r['scene_id'] for r in metrics['per_scene']] != scenes:
        raise ValueError('original raw metrics and complete manifest scene order differ')
    expected = {row['scene_id']: row for row in metrics['per_scene']}
    if (sum(r['node_count'] for r in expected.values()) != 24599
            or metrics['pair_count'] != 4439092 or metrics['tp'] + metrics['fn'] != 24675):
        raise ValueError('original056 input population binding differs')
    if not all(sid in manifest['components'] for sid in scenes):
        raise ValueError('component registry must cover every fixed scene')
    return scenes, expected


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(status='RUNNING', idea_id=IDEA_ID, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), device='cpu', model_forwards=0,
        optimizer_updates=0, archive_read=False, reference_read=False, test_opened=False,
        scores_recomputed=False, threshold=THRESHOLD, completed_scenes=0,
        inputs={key: str(Path(getattr(args, key)).resolve()) for key in ('scores', 'summary', 'manifest')})
    write_json(output / 'run.json', record)
    try:
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([8 * 1024**3] + [x for x in (soft, hard) if x != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            record['address_space_limit_bytes'] = limit
        manifest, original = read_json(args.manifest), read_json(args.summary)
        scene_ids, expected = _input_contract(manifest, original)
        aliases, results = {}, []
        with np.load(args.scores, allow_pickle=False) as saved, (output / 'scenes.jsonl').open('x') as stream:
            if saved['scene_ids'].tolist() != scene_ids:
                raise ValueError('score cache scene order/population differs from frozen V300')
            for sid in scene_ids:
                row = analyze_scene(**{key: saved[f'{sid}_{key}'] for key in
                    ('node_ids', 'sources', 'pairs', 'labels', 'joint_support_queries', 'scores')},
                    source_aliases=aliases)
                old = expected[sid]
                if (row['node_count'] != old['node_count'] or row['pair_count'] != old['pair_count']
                        or row['positive_edges'] != old['positive_count']
                        or any(row['raw_binding_counts'][key] != old[key] for key in ('tp', 'fp', 'fn', 'tn'))):
                    raise ValueError('per-scene raw056 node/pair/label/count binding failed')
                row.update(scene_id=sid, component_id=str(manifest['components'][sid]))
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                results.append(row)
                record.update(completed_scenes=len(results), last_scene_id=sid,
                    elapsed_seconds=time.monotonic() - started,
                    advisory_exceeded=time.monotonic() - started > 600)
                write_json(output / 'run.json', record)
        summary = pool_scenes(results)
        summary.update(status='COMPLETED', idea_id=IDEA_ID, threshold=THRESHOLD,
            input_population='all valid056 V300 input nodes; not all declared targets',
            bound_scope='unmerged two-source unit-capacity hard edges, separately per source-pair; not group/backbone recall',
            no_joint_support_definition='joint_support_queries==0; does not prove absence of all native temporal overlap',
            source_count=len(aliases), source_alias_policy='first appearance in manifest order; equality only, no reliability prior',
            by_source_pair=pool_source_pairs(results),
            by_component={cid: pool_scenes([r for r in results if r['component_id'] == cid])
                          for cid in sorted({r['component_id'] for r in results})},
            author_reproduction=False, actual_method_failure_measured=False,
            limitations=['Current anchor_direct is not unit-capacity.', 'Soft nonzero edges and identity closure are not bounded by U.',
                'No-input/quarantined targets and native time intervals are absent from this cache.',
                'Fixed development population with shared identity; no independent generalization or causal claim.'])
        write_json(output / 'summary.json', summary)
        record['status'] = 'COMPLETED'
    except BaseException as error:
        record.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record.update(elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024))
        write_json(output / 'run.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('scores', 'summary', 'manifest', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('terminated; preserve this attempt without retry')
    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
