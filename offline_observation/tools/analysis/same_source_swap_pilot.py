"""C71 one-exchange diagnostic on frozen MTAD T32 partitions."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import time

import numpy as np

from association.evaluation.association import _aggregate, _counts
from tools.analysis.frozen_score_decision_diagnostic import (
    THRESHOLD, canonical_membership, peak_rss_bytes, read, write,
)

STARTS = ('WS', 'HARD', 'STRUCT')
LAYERS = ('keep', 'all')
SCOPES = ('itsa', 'mstsa')
EPSILON = 1e-10
MAX_NODES = 640
MAX_AS = 8 * 1024**3
PARENTS = {
    'value': 'LEVEL1-CONDITIONAL-SOURCE-VALUE-20261029-125-attempt-1',
    'native': 'LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1',
    'base': 'LEVEL1-TRUTH-FREE-PARTITION-20261031-165-attempt-1',
    'front': 'LEVEL1-ASSOCIATION-CONTEXT-20261031-168-attempt-1',
    'decision': 'LEVEL1-DECISION-ERROR-MINING-20261031-175-attempt-1',
}


def load_population(runs):
    roots = {key: runs / name for key, name in PARENTS.items()}
    metadata = {key: read(path / 'run.json') for key, path in roots.items()}
    if any(row['status'] != 'COMPLETED' or row['exit_code'] != 0 for row in metadata.values()):
        raise ValueError('completed frozen inputs required')
    manifest_path = Path(metadata['base']['manifest'])
    manifest = read(manifest_path)
    scenes = manifest['splits']['T_core'][:32]
    components = {sid: str(manifest['components'][sid]) for sid in scenes}
    marker = read(roots['native'] / 'native-cache.json')
    if (len(scenes) != 32 or len(components) != 32 or len(set(components.values())) != 12
            or not all(sid.isascii() and sid.isdecimal() for sid in scenes)
            or any(row['training_scene_ids'] != scenes for row in metadata.values())
            or marker['status'] != 'CACHE_COMPLETED' or marker['scene_ids']['T_core'] != scenes):
        raise ValueError('original T32 / 12 components required')
    for key in ('value', 'base', 'front', 'decision'):
        if Path(metadata[key]['native_run']).resolve() != roots['native'].resolve():
            raise ValueError(f'wrong native lineage in {key}')
    for key in ('base', 'front', 'decision'):
        if (Path(metadata[key]['value_run']).resolve() != roots['value'].resolve()
                or read(Path(metadata[key]['manifest'])) != manifest):
            raise ValueError(f'wrong value or manifest lineage in {key}')
    if (Path(metadata['decision']['front_run']).resolve() != roots['front'].resolve()
            or metadata['decision']['thresholds']['WS'] != THRESHOLD):
        raise ValueError('wrong 175 score or scene lineage')
    return scenes, components


def load_scene(runs, sid):
    filename = f'partitions-T_core-{sid}.npz'
    with np.load(runs / PARENTS['native'] / 'private/native-motion/T_core' / f'scene-{sid}.npz',
                 allow_pickle=False) as arrays:
        node_ids, sources, support, pairs = (arrays[key] for key in ('node_ids', 'sources', 'support', 'pairs'))
    n = len(node_ids)
    if (not 0 < n <= MAX_NODES or node_ids.shape != (n,) or len(np.unique(node_ids)) != n
            or sources.shape != (n,) or support.shape != (n, 64) or support.dtype != np.bool_
            or pairs.dtype.kind not in 'iu' or not np.array_equal(pairs, np.column_stack(np.triu_indices(n, 1)))):
        raise ValueError(f'invalid frozen native node order or complete pairs in {sid}')
    with np.load(runs / PARENTS['base'] / 'private' / filename, allow_pickle=False) as arrays:
        if not np.array_equal(arrays['pairs'], pairs):
            raise ValueError(f'165 pair order mismatch in {sid}')
        scores, ws = arrays['scores'], arrays['WS']
    with np.load(runs / PARENTS['value'] / 'private' / f'T-features-{sid}.npz', allow_pickle=False) as arrays:
        if not (np.array_equal(arrays['membership'], ws) and np.array_equal(arrays['scores'], scores)):
            raise ValueError(f'125 score or WS mismatch in {sid}')
    with np.load(runs / PARENTS['front'] / 'private' / f'front-cache-T_core-{sid}.npz',
                 allow_pickle=False) as arrays:
        if not all(np.array_equal(arrays[key], value)
                   for key, value in (('node_ids', node_ids), ('pairs', pairs), ('scores', scores))):
            raise ValueError(f'168 node order or score mismatch in {sid}')
    with np.load(runs / PARENTS['decision'] / 'private' / filename, allow_pickle=False) as arrays:
        if not (np.array_equal(arrays['pairs'], pairs) and np.array_equal(arrays['WS'], ws)):
            raise ValueError(f'175 pairs or WS mismatch in {sid}')
        memberships = {start: canonical_membership(arrays[start], n) for start in STARTS}
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise ValueError(f'invalid frozen 056 scores in {sid}')
    return node_ids, sources, support, pairs, scores, memberships


def load_identities(runs, sid, node_ids):
    path = runs / PARENTS['native'] / 'private/native-evaluation/T_core' / f'scene-{sid}.npz'
    with np.load(path, allow_pickle=False) as arrays:
        if not np.array_equal(arrays['node_ids'], node_ids):
            raise ValueError(f'identity node order mismatch in {sid}')
        identities = arrays['identities']
    if identities.shape != node_ids.shape or np.any(identities == ''):
        raise ValueError(f'invalid identity vector in {sid}')
    return identities


def dense_edges(pairs, values, n):
    matrix = np.zeros((n, n), dtype=values.dtype)
    left, right = pairs.T
    matrix[left, right] = values
    matrix[right, left] = values
    return matrix


def group_members(membership):
    return [np.flatnonzero(membership == group) for group in np.unique(membership)]


def delta(matrix, membership, actions):
    members = group_members(membership)
    sums = np.column_stack([matrix[:, group].sum(axis=1) for group in members])
    left, right = actions.T
    a, b = membership[left], membership[right]
    return sums[left, b] - sums[left, a] + sums[right, a] - sums[right, b] - 2 * matrix[left, right]


def build_pool(membership, sources, pairs, weights):
    members = group_members(membership)
    sizes = np.array([len(group) for group in members])
    anchors = np.array([group[0] for group in members])
    left, right = pairs.T
    a, b = membership[left], membership[right]
    valid = (a != b) & (sources[left] == sources[right]) & ((sizes[a] > 1) | (sizes[b] > 1))
    actions = pairs[valid]
    left, right = actions.T
    first_left, first_right = anchors[membership[left]], anchors[membership[right]]
    keep = (left != first_left) & (right != first_right) & (left > first_right) & (right > first_left)
    gain = delta(weights, membership, actions)
    selected = np.full(len(LAYERS), -1, dtype=np.int64)
    for index, mask in enumerate((keep, np.ones(len(actions), dtype=bool))):
        options = np.flatnonzero(mask & (gain > EPSILON))
        if len(options):
            selected[index] = options[np.argmax(gain[options])]
    return dict(actions=actions, keep=keep, gain=gain, selected=selected)


def swapped(membership, action):
    modified = membership.copy()
    left, right = action
    modified[[left, right]] = membership[[right, left]]
    return modified


def support_changes(before, after, support, sources, action):
    totals = np.zeros((3, 3), dtype=np.int64)
    for group in before[action]:
        counts = []
        for membership in (before, after):
            members = membership == group
            by_source = [support[members & (sources == src)].any(axis=0)
                         for src in np.unique(sources[members])]
            counts.append(np.minimum(2, np.sum(by_source, axis=0)))
        np.add.at(totals, tuple(counts), 1)
    return totals.tolist()


def summarize_actions(pool, truth_delta, membership, identities, sources, support, pairs):
    left, right = pairs.T
    labels = identities[left] == identities[right]
    same = sources[left] == sources[right]
    base = membership[left] == membership[right]
    good = (truth_delta >= 0).all(axis=1) & (truth_delta > 0).any(axis=1)
    members = group_members(membership)
    identity_groups = [np.flatnonzero(identities == identity) for identity in np.unique(identities)]

    def source_counts(indices):
        return tuple(sorted(Counter(sources[indices]).items()))

    anchor_truth = [identities[group[0]] for group in members]
    truth_by_identity = {identities[group[0]]: source_counts(group) for group in identity_groups}
    row = dict(
        nodes=len(membership), groups=len(members), input_identities=len(identity_groups),
        true_count_differs=len(members) != len(identity_groups),
        source_profiles_differ=sorted(source_counts(g) for g in members) != sorted(truth_by_identity.values()),
        anchored_profiles_differ=(len(set(anchor_truth)) != len(members)
                                  or {identity: source_counts(group) for identity, group
                                      in zip(anchor_truth, members, strict=True)} != truth_by_identity),
        mixed_groups=sum(len(np.unique(identities[group])) > 1 for group in members),
        split_identities=sum(len(np.unique(membership[group])) > 1 for group in identity_groups),
    )

    def counts(pred, mask):
        return dict(pair_count=int(mask.sum()), positive_count=int(labels[mask].sum()),
                    **_counts(labels[mask], pred[mask]))

    row['baseline'] = {scope: counts(base, mask) for scope, mask in zip(SCOPES, (same, ~same), strict=True)}
    for index, layer in enumerate(LAYERS):
        allowed = pool['keep'] if layer == 'keep' else np.ones(len(good), dtype=bool)
        possible = np.flatnonzero(allowed & good)
        # One diagnostic candidate per scene, with both metrics evaluated on it.
        oracle = int(possible[np.argmax(truth_delta[possible].sum(axis=1))]) if len(possible) else -1
        selected = int(pool['selected'][index])
        arm = dict(actions=int(allowed.sum()), joint_improving_actions=len(possible))
        for name, choice in (('selected', selected), ('oracle', oracle)):
            action = pool['actions'][choice] if choice >= 0 else None
            modified = swapped(membership, action) if action is not None else membership
            pred = modified[left] == modified[right]
            metrics, transitions = {}, {}
            for scope, mask in zip(SCOPES, (same, ~same), strict=True):
                metrics[scope] = counts(pred, mask)
                transitions[scope] = {
                    name: int(np.count_nonzero(mask & condition))
                    for name, condition in (
                        ('true_added', labels & pred & ~base), ('true_removed', labels & base & ~pred),
                        ('false_added', ~labels & pred & ~base), ('false_removed', ~labels & base & ~pred),
                    )
                }
            arm[name] = dict(changed=choice >= 0, joint_improvement=choice >= 0 and bool(good[choice]),
                             gain=float(pool['gain'][choice]) if choice >= 0 else 0.0,
                             metrics=metrics, transitions=transitions,
                             support_transition=support_changes(membership, modified, support, sources, action)
                             if action is not None else None)
        row[layer] = arm
    return row


def aggregate(rows):
    result = {}
    for start in STARTS:
        records = [row[start] for row in rows]
        part = dict(scenes=len(rows), baseline={scope: _aggregate([r['baseline'][scope] for r in records])
                                               for scope in SCOPES})
        for field in ('true_count_differs', 'source_profiles_differ', 'anchored_profiles_differ'):
            part[field] = sum(r[field] for r in records)
        for layer in LAYERS:
            arms = [r[layer] for r in records]
            entry = dict(actions=sum(a['actions'] for a in arms), scenes_with_actions=sum(a['actions'] > 0 for a in arms),
                         joint_improving_actions=sum(a['joint_improving_actions'] for a in arms),
                         missed_opportunity_scenes=sum(a['oracle']['joint_improvement'] and not a['selected']['joint_improvement']
                                                       for a in arms))
            for key in ('selected', 'oracle'):
                entry[key] = dict(changed_scenes=sum(a[key]['changed'] for a in arms),
                                  joint_improving_scenes=sum(a[key]['joint_improvement'] for a in arms),
                                  metrics={scope: _aggregate([a[key]['metrics'][scope] for a in arms]) for scope in SCOPES})
            part[layer] = entry
        result[start] = part
    return result


def run(runs, output):
    output.mkdir(exist_ok=False)
    (output / 'private').mkdir()
    start = time.monotonic()
    record = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
                  inputs={key: str(runs / name) for key, name in PARENTS.items()},
                  split='T_core', stages=[], optimizer_updates=0, weights_loaded=False,
                  reference_fields_loaded=False, affinity=sorted(os.sched_getaffinity(0)))

    def report(**fields):
        record.update(fields, elapsed_seconds=time.monotonic() - start, peak_rss_bytes=peak_rss_bytes())
        write(output / 'run.json', record)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        scenes, components = load_population(runs)
        report(phase='freeze_candidates', scene_count=len(scenes), scene_ids=scenes)
        for sid in scenes:
            nodes, sources, _, pairs, scores, memberships = load_scene(runs, sid)
            weights = dense_edges(pairs, scores.astype(np.float64) - THRESHOLD, len(nodes))
            payload = {f'{start_name}_{key}': value
                       for start_name, membership in memberships.items()
                       for key, value in build_pool(membership, sources, pairs, weights).items()}
            np.savez_compressed(output / 'private' / f'{sid}.npz', **payload)
        record['stages'].append('all_candidates_and_J_selections_saved')
        report(phase='score_frozen_candidates')
        rows = []
        for sid in scenes:
            nodes, sources, support, pairs, _, memberships = load_scene(runs, sid)
            identities = load_identities(runs, sid, nodes)
            left, right = pairs.T
            truth = identities[left] == identities[right]
            same = sources[left] == sources[right]
            matrices = [dense_edges(pairs, truth & scope, len(nodes)) for scope in (same, ~same)]
            row, truth_payload = dict(scene_id=sid, component_id=components[sid]), {}
            with np.load(output / 'private' / f'{sid}.npz', allow_pickle=False) as arrays:
                for start_name, membership in memberships.items():
                    pool = {key: arrays[f'{start_name}_{key}'] for key in ('actions', 'keep', 'gain', 'selected')}
                    truth_delta = np.column_stack([delta(matrix, membership, pool['actions']) for matrix in matrices])
                    truth_payload[f'{start_name}_tp_delta'] = truth_delta
                    row[start_name] = summarize_actions(pool, truth_delta, membership, identities, sources, support, pairs)
            np.savez_compressed(output / 'private' / f'{sid}-truth.npz', **truth_payload)
            rows.append(row)
        largest = Counter(components.values()).most_common(1)[0][0]
        write(output / 'scenes.json', rows)
        write(output / 'summary.json', dict(
            all_scenes=aggregate(rows), largest_component=largest,
            without_largest=aggregate([r for r in rows if r['component_id'] != largest]),
            components={c: aggregate([r for r in rows if r['component_id'] == c]) for c in dict.fromkeys(components.values())},
        ))
        record['stages'].append('T_identity_scoring_complete')
        report(status='COMPLETED', exit_code=0, completed_scenes=len(rows), phase='done')
    except BaseException as error:
        report(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs.resolve(), args.output.resolve())
