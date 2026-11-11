"""C92: frozen train score coverage and matched hard-tail association contrast."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from association.models.evidence_association import FEATURE_NAMES, features_from_motion
from association.training.evidence_association import load_checkpoint, normalized_features
from tools.analysis import association_frozen_pr_audit as audit
from tools.analysis import static_partition_pilot as ws


SCOPES = ('itsa', 'mstsa')
SEED = 20261112
CHUNK = 4096
STEPS = 1024
EACH_SCOPE = 128
CONVERSION = 1852 / 3600
PARITY = 1e-4


def require(value, description):
    if not value:
        raise ValueError(description)


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def summary_state(output, start, **state):
    write(output / 'run.json', dict(elapsed_seconds=time.monotonic() - start,
                                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                   **state))


def scene_split(ids, components):
    """Assign by shared A component before any pair scores or labels are read."""
    counts = Counter(components[sid] for sid in ids)
    ordered = sorted(counts, key=lambda component: (-counts[component], component))
    require(len(ordered) >= 8, 'insufficient train components')
    roles = {key: 'fit' for key in ordered[:2]}
    held = {'calibration': 0, 'held': 0}
    target = math.ceil(len(ids) * .08)
    for component in ordered[2:]:
        role = min(held, key=held.get)
        if held[role] >= target:
            roles[component] = 'fit'
        else:
            roles[component] = role
            held[role] += counts[component]
    return {role: [sid for sid in ids if roles[components[sid]] == role]
            for role in ('fit', 'calibration', 'held')}


def read_motion(row):
    with np.load(row['motion'], allow_pickle=False) as saved:
        motion = {key: saved[key] for key in (
            'states', 'support', 'nearest_seconds', 'native_count', 'span_seconds', 'fit_rmse_m',
            'ols_velocity_mps', 'endpoint_velocity_mps', 'sources', 'node_ids', 'query_seconds')}
        require(str(saved['state_speed_unit']) == 'mps', '151 explicit speed unit missing')
    motion['states'][..., 2] /= CONVERSION
    require(len(motion['node_ids']) == row['node_count'], '151 node count changed')
    return motion


def read_identity(row, motion):
    with np.load(row['evaluation'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['node_ids'], motion['node_ids'])
        np.testing.assert_array_equal(saved['query_seconds'], motion['query_seconds'])
        return saved['identities']


def all_pairs(n):
    return np.column_stack(np.triu_indices(n, k=1))


def features(motion, pairs, normalizer):
    raw, base = features_from_motion(motion, pairs)
    return normalized_features(raw.astype(np.float32), normalizer), base


@torch.inference_mode()
def infer(motion, pairs, teacher, normalizer, device, heads=None, saved_scores=None):
    result = {key: np.empty(len(pairs), np.float64)
              for key in (('056', *heads) if heads is not None else ('056',))}
    for start in range(0, len(pairs), CHUNK):
        block = slice(start, start + CHUNK)
        x, base = features(motion, pairs[block], normalizer)
        x = torch.from_numpy(x).to(device)
        if saved_scores is None:
            score = teacher(x, torch.from_numpy(base).to(device)).cpu().numpy()
        else:
            score = saved_scores[block]
        result['056'][block] = score
        if heads is not None:
            for name, head in heads.items():
                result[name][block] = score + head(x).squeeze(-1).cpu().numpy().astype(np.float64)
    require(all(np.isfinite(values).all() for values in result.values()), 'nonfinite pair score')
    return result


def tail_indices(scores, negatives):
    """Top 1% per scene and source relation, retaining complete boundary ties."""
    if not len(negatives):
        return negatives, negatives
    values = scores[negatives]
    cutoffs = []
    for denom in (1000, 100):
        cutoff = len(values) - math.ceil(len(values) / denom)
        cutoffs.append(np.partition(values, cutoff)[cutoff])
    return negatives[values >= cutoffs[0]], negatives[(values >= cutoffs[1]) & (values < cutoffs[0])]


def scope_masks(motion, pairs):
    same = motion['sources'][pairs[:, 0]] == motion['sources'][pairs[:, 1]]
    return same, ~same


def locations(runs, manifest):
    original = runs / audit.RUNS['056'] / 'association'
    data = runs / 'LEVEL1-FUSION-PEER-FULL-DATA-20261030-151-attempt-1'
    cache = runs / 'LEVEL1-ASSOCIATION-CONTEXT-20261031-168-attempt-1' / 'private'
    index = audit.read_json(data / 'private' / 'index.json')
    for split in ('T_core', 'V_select'):
        require(set(index[split]) == set(manifest['splits'][split]), '151 manifest population changed')
        require(all(row['component_id'] == manifest['components'][sid]
                    for sid, row in index[split].items()), '151 component assignment changed')
    return original, index, cache


def prepare(args, manifest, original, index, cache, start):
    ids = manifest['splits']['T_core']
    components = manifest['components']
    roles = scene_split(ids, components)
    by_scene = {sid: role for role, members in roles.items() for sid in members}
    write(args.output / 'frozen-split.json', roles)
    teacher, checkpoint = load_checkpoint(original / 'final.pt', args.device)
    require(checkpoint['update'] == 1000 and teacher.motion_observation_policy == 'canonical_all',
            'original 056 final checkpoint changed')
    normalizer = checkpoint['normalizer']
    tallies = defaultdict(Counter)
    identities = defaultdict(set)
    rows, reused_pairs, inferred_pairs, largest_gap = [], 0, 0, 0.
    for number, sid in enumerate(ids, 1):
        motion = read_motion(index['T_core'][sid])
        pairs = all_pairs(len(motion['node_ids']))
        path = cache / f'front-cache-T_core-{sid}.npz'
        reused = path.is_file()
        if reused:
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['node_ids'], motion['node_ids'])
                np.testing.assert_array_equal(saved['pairs'], pairs)
                scores = saved['scores']
            require(scores.shape == (len(pairs),) and np.isfinite(scores).all(), 'invalid 168 score')
            if len(pairs):
                probe = np.linspace(0, len(pairs) - 1, min(64, len(pairs)), dtype=int)
                check = infer(motion, pairs[probe], teacher, normalizer, args.device)['056']
                gap = float(np.max(np.abs(check - scores[probe])))
                require(gap <= PARITY, f'168 score provenance gap in scene {sid}: {gap}')
                largest_gap = max(largest_gap, gap)
            reused_pairs += len(pairs)
        else:
            scores = infer(motion, pairs, teacher, normalizer, args.device)['056']
            inferred_pairs += len(pairs)
        np.save(args.output / 'private' / f'scores-{sid}.npy', scores)
        truth_id = read_identity(index['T_core'][sid], motion)
        truth = truth_id[pairs[:, 0]] == truth_id[pairs[:, 1]]
        support = motion['support']
        joint = (support[pairs[:, 0]] & support[pairs[:, 1]]).any(1)
        row = dict(scene=sid, role=by_scene[sid], component=components[sid], pairs=len(pairs), reused=reused)
        for scope, mask in zip(SCOPES, scope_masks(motion, pairs), strict=True):
            pos = np.flatnonzero(mask & truth)
            neg = np.flatnonzero(mask & ~truth)
            top, rest = tail_indices(scores, neg)
            tail = np.r_[top, rest]
            selected = neg[scores[neg] >= ws.THRESHOLD]
            row[scope] = dict(positive=len(pos), negative=len(neg), tail=len(tail), top=len(top),
                              tail_joint=int(joint[tail].sum()), positive_joint=int(joint[pos].sum()),
                              above_old_threshold=len(selected), competing_scenes=int(bool(len(pos) and len(tail))))
            key = (row['role'], row['component'], scope)
            tallies[key].update(row[scope])
            identities[key].update(truth_id[pairs[tail].ravel()].tolist())
        rows.append(row)
        if number % 100 == 0:
            summary_state(args.output, start, status='RUNNING', completed_train_scenes=number,
                          reused_pairs=reused_pairs, inferred_pairs=inferred_pairs)
    require(reused_pairs + inferred_pairs == sum(row['pairs'] for row in rows), 'uncovered training pairs')
    detail = [dict(role=role, component=component, scope=scope, **counts, tail_identities=len(identities[key]))
              for key, counts in tallies.items() for role, component, scope in (key,)]
    write(args.output / 'scenes.json', rows)
    write(args.output / 'components.json', detail)
    groups = {}
    for role, members in roles.items():
        groups[role] = {}
        for scope in SCOPES:
            entries = [row for row in detail if row['role'] == role and row['scope'] == scope]
            combined = Counter()
            for entry in entries:
                combined.update({key: value for key, value in entry.items() if isinstance(value, int)})
            combined.pop('tail_identities', None)
            groups[role][scope] = dict(combined, scenes=len(members), components=len(entries),
                                       competing_components=sum(row['competing_scenes'] > 0 for row in entries),
                                       largest_tail_component=max(row['tail'] for row in entries))
    qualified = all(row['competing_components'] >= 2 and 0 < row['tail_joint'] < row['tail']
                    for group in groups.values() for row in group.values())
    write(args.output / 'summary.json', dict(qualified=qualified, train_scenes=len(ids),
          reused_scenes=sum(row['reused'] for row in rows), reused_pairs=reused_pairs, inferred_pairs=inferred_pairs,
          sampled_reuse_max_abs_gap=largest_gap, role_scopes=groups,
          components_also_in_v=len(set(components[s] for s in ids) &
                                   set(components[s] for s in manifest['splits']['V_select']))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare',), required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    (args.output / 'private').mkdir()
    start = time.monotonic()
    torch.set_num_threads(4)
    summary_state(args.output, start, status='RUNNING')
    try:
        manifest = audit.read_json(args.manifest)
        original, index, cache = locations(args.runs, manifest)
        prepare(args, manifest, original, index, cache, start)
    except Exception as exc:
        summary_state(args.output, start, status='FAILED', error=repr(exc))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, test_opened=False)


if __name__ == '__main__':
    main()
