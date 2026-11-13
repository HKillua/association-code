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

from association.evaluation.association import _aggregate, _counts
from association.models.evidence_association import EvidenceAssociation, features_from_motion
from association.training.evidence_association import load_checkpoint, normalized_features
from tools.analysis import association_frozen_pr_audit as audit
from tools.analysis import supplier_provenance_qualification as supplier
from tools.analysis import static_partition_pilot as ws


SCOPES = ('itsa', 'mstsa')
SEED = 20261112
CHUNK = 4096
STEPS = 1024
EACH_SCOPE = 128
CONVERSION = 1852 / 3600
PARITY = 1e-4
HEADS = ('BCE', 'RANK')
ARMS = ('056', '056_T', *HEADS)


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
                result[name][block] = head(x, torch.as_tensor(score, device=device)).cpu().numpy()
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


def prepare_fit(index, ids, prep, normalizer):
    pools = defaultdict(list)
    for sid in ids:
        row = index[sid]
        motion = read_motion(row)
        pairs = all_pairs(len(motion['node_ids']))
        old = np.load(prep / 'private' / f'scores-{sid}.npy', allow_pickle=False)
        ids_true = read_identity(row, motion)
        truth = ids_true[pairs[:, 0]] == ids_true[pairs[:, 1]]
        for scope, mask in zip(SCOPES, scope_masks(motion, pairs), strict=True):
            top, rest = tail_indices(old, np.flatnonzero(mask & ~truth))
            for kind, indices in (('positive', np.flatnonzero(mask & truth)), ('top', top), ('rest', rest)):
                if len(indices):
                    x, _ = features(motion, pairs[indices], normalizer)
                    pools[scope, kind].append((x, old[indices]))
    result = {}
    for scope in SCOPES:
        for kind in ('positive', 'top', 'rest'):
            blocks = pools[scope, kind]
            require(bool(blocks), f'empty fit pool {scope} {kind}')
            result[scope, kind] = tuple(np.concatenate(items) for items in zip(*blocks))
    return result


def fit_heads(pools, device):
    torch.manual_seed(SEED)
    heads = {name: EvidenceAssociation().to(device) for name in HEADS}
    heads['RANK'].load_state_dict(heads['BCE'].state_dict())
    require(all(torch.equal(value, heads['RANK'].state_dict()[key])
                for key, value in heads['BCE'].state_dict().items()), 'head initialization differs')
    opts = {name: torch.optim.Adam(head.parameters(), lr=.001) for name, head in heads.items()}
    rng = np.random.default_rng(SEED)
    seen = {key: np.zeros(len(old), bool) for key, (_, old) in pools.items()}
    population = {key: tuple(torch.from_numpy(a).to(device) for a in values) for key, values in pools.items()}
    with torch.inference_mode():
        for x, old in population.values():
            for head in heads.values():
                require(torch.equal(head(x[:32], old[:32]), old[:32]), 'u0 score differs from frozen 056')
    log = []
    for step in range(1, STEPS + 1):
        batch = {}
        for key, (x, old) in population.items():
            count = EACH_SCOPE if key[1] == 'positive' else EACH_SCOPE // 2
            drawn = rng.integers(0, len(old), size=count)
            seen[key][drawn] = True
            batch[key] = x[drawn], old[drawn]
        step_log = {}
        for name, head in heads.items():
            opts[name].zero_grad(set_to_none=True)
            logits = {key: head(x, old) for key, (x, old) in batch.items()}
            losses = []
            for scope in SCOPES:
                pos = logits[scope, 'positive']
                neg = torch.cat([logits[scope, 'top'], logits[scope, 'rest']])
                if name == 'BCE':
                    losses.append((F.softplus(-pos).mean() + F.softplus(neg).mean()) / 2)
                else:
                    losses.append(F.softplus(neg - pos).mean())
            loss = torch.stack(losses).mean()
            require(bool(torch.isfinite(loss)), 'nonfinite objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            opts[name].step()
            if step == 1 or step % 128 == 0:
                step_log[name] = dict(loss=float(loss.detach()), gradient_norm=float(norm))
        if step_log:
            log.append(dict(step=step, **step_log))
    for head in heads.values():
        head.eval().requires_grad_(False)
    counts = {scope: {kind: dict(available=len(seen[scope, kind]), unique_drawn=int(seen[scope, kind].sum()),
                                presentations=STEPS * (EACH_SCOPE if kind == 'positive' else EACH_SCOPE // 2))
                      for kind in ('positive', 'top', 'rest')} for scope in SCOPES}
    return heads, dict(seed=SEED, updates_per_head=STEPS, paired_exposure=True, u0_equal=True,
                       parameters_per_head=sum(parameter.numel() for parameter in heads['BCE'].parameters()),
                       population=counts, curve=log)


def score_scene(row, sid, normalizer, heads, device, prep, archive):
    motion = read_motion(row)
    pairs = all_pairs(len(motion['node_ids']))
    if archive is None:
        old = np.load(prep / 'private' / f'scores-{sid}.npy', allow_pickle=False)
    else:
        for key, actual in (('node_ids', motion['node_ids']), ('sources', motion['sources']), ('pairs', pairs)):
            np.testing.assert_array_equal(archive[f'{sid}_{key}'], actual)
        old = archive[f'{sid}_scores']
    require(old.shape == (len(pairs),), 'frozen pair score population changed')
    scores = infer(motion, pairs, None, normalizer, device, heads, old)
    scores['056_T'] = scores['056']
    return motion, pairs, scores


def recall_point(scores, truth, percent):
    thresholds, predicted, tp = audit.score_groups(scores, truth)
    require(int(tp[-1]) > 0, 'recall endpoint has no true edges')
    i = int(np.searchsorted(tp, (int(tp[-1]) * percent + 99) // 100))
    return dict(threshold=float(thresholds[i]), tp=int(tp[i]), fp=int(predicted[i] - tp[i]),
                positive=int(tp[-1]), predicted=int(predicted[i]), pfa=float(1 - tp[i] / predicted[i]))


def calibrate(index, ids, normalizer, heads, device, prep):
    values = {name: [] for name in ('056_T', *HEADS)}
    truth = []
    for sid in ids:
        row = index[sid]
        motion, pairs, scores = score_scene(row, sid, normalizer, heads, device, prep, None)
        ids_true = read_identity(row, motion)
        cross = scope_masks(motion, pairs)[1]
        truth.append((ids_true[pairs[:, 0]] == ids_true[pairs[:, 1]])[cross])
        for name in values:
            values[name].append(scores[name][cross])
    truth = np.concatenate(truth)
    return {name: recall_point(np.concatenate(values[name]), truth, 90) for name in values}


def decode(n, pairs, scores, thresholds, original, arms=ARMS):
    result = {'056': original}
    for name in arms[1:]:
        shifted = (scores[name] - thresholds[name]) + ws.THRESHOLD
        membership, _, _ = ws.agglomerate(n, pairs, shifted, 'WS')
        result[name] = membership
    return result


def coverage(original, changed, truth_id, support):
    def represented(group):
        _, first = np.unique(group, return_index=True)
        identities = set(truth_id[first])
        return identities, len(first) - len(identities)

    identity_equal = truth_id[:, None] == truth_id[None, :]
    original_ids, old_duplicates = represented(original)
    changed_ids, new_duplicates = represented(changed)
    old_members = original[:, None] == original[None, :]
    new_members = changed[:, None] == changed[None, :]
    old_support = supplier.supported(old_members & identity_equal, support)
    new_support = supplier.supported(new_members & identity_equal, support)
    return dict(represented=len(changed_ids), identities_lost=len(original_ids - changed_ids),
                identities_gained=len(changed_ids - original_ids), duplicates=new_duplicates,
                duplicate_increase=new_duplicates - old_duplicates,
                duplicate_increase_scenes=int(new_duplicates > old_duplicates),
                same_ship_supported=int(new_support.sum()),
                same_ship_lost=int((old_support & ~new_support).sum()),
                same_ship_recovered=int((~old_support & new_support).sum()))


def confuse(truth, chosen, mask):
    t, p = truth[mask], chosen[mask]
    return dict(pair_count=len(t), positive_count=int(t.sum()), **_counts(t, p))


def evaluate(args, split, ids, index, normalizer, heads, thresholds, archive, components):
    arms = ('056', '056_T', *heads)
    frozen_path = args.output / 'private' / split
    frozen_path.mkdir()
    original_ws = args.runs / audit.RUNS['114'] / 'private'
    for sid in ids:
        row = index[sid]
        motion, pairs, scores = score_scene(row, sid, normalizer, heads, args.device, args.prep, archive)
        n = len(motion['node_ids'])
        if archive is None:
            old_groups, _, _ = ws.agglomerate(n, pairs, scores['056'], 'WS')
        else:
            old_groups = np.asarray(audit.read_json(original_ws / f'scene-{sid}.json.gz')['memberships']['WS'])
        memberships = decode(n, pairs, scores, thresholds, old_groups, arms)
        np.savez(frozen_path / f'{sid}.npz', pairs=pairs,
                 **{f'score_{name}': values for name, values in scores.items()},
                 **{f'group_{name}': values for name, values in memberships.items()})

    rows = []
    rank = {scope: defaultdict(list) for scope in SCOPES}
    for sid in ids:
        row = index[sid]
        motion = read_motion(row)
        ids_true = read_identity(row, motion)
        with np.load(frozen_path / f'{sid}.npz', allow_pickle=False) as saved:
            pairs = saved['pairs']
            left, right = pairs.T
            truth = ids_true[left] == ids_true[right]
            if archive is not None:
                np.testing.assert_array_equal(truth, archive[f'{sid}_labels'])
            masks = dict(zip(SCOPES, scope_masks(motion, pairs), strict=True))
            before = saved['group_056']
            values = dict(scene=sid, component=components[sid], pairs=len(pairs), arms={})
            for scope, mask in masks.items():
                rank[scope]['truth'].append(truth[mask])
            for name in arms:
                scores, group = saved[f'score_{name}'], saved[f'group_{name}']
                raw = scores >= thresholds[name]
                decoded = group[left] == group[right]
                values['arms'][name] = dict(coverage=coverage(before, group, ids_true, motion['support']))
                for scope, mask in masks.items():
                    values['arms'][name][scope] = dict(
                        raw=confuse(truth, raw, mask), decoded=confuse(truth, decoded, mask),
                        removed=confuse(truth, raw & ~decoded, mask),
                        added=confuse(truth, ~raw & decoded, mask),
                        retained=confuse(truth, raw & decoded, mask))
                    rank[scope][name].append(scores[mask])
            rows.append(values)
    write(args.output / f'{split}-scenes.json', rows)
    ranking = {}
    for scope, blocks in rank.items():
        truth = np.concatenate(blocks['truth'])
        ranking[scope] = {}
        for name in arms:
            scores = np.concatenate(blocks[name])
            ranking[scope][name] = {str(p): recall_point(scores, truth, p) for p in (80, 90)}
            _, predicted, tp = audit.score_groups(scores, truth)
            fp = predicted - tp
            delta_tp, delta_fp = np.diff(tp, prepend=0), np.diff(fp, prepend=0)
            ranking[scope][name]['pooled_ap'] = float(delta_tp @ (tp / predicted) / tp[-1])
            ranking[scope][name]['pooled_auc'] = float(
                delta_tp @ (fp[-1] - fp + .5 * delta_fp) / (tp[-1] * fp[-1]))
    return summarize_results(rows, ranking, arms)


def summarize_results(rows, ranking, arms=ARMS):
    def aggregate(items):
        result = {}
        for name in arms:
            coverage_totals = Counter()
            for row in items:
                coverage_totals.update(row['arms'][name]['coverage'])
            result[name] = dict(coverage=dict(coverage_totals))
            for scope in SCOPES:
                result[name][scope] = {}
                for stage in ('raw', 'decoded', 'removed', 'added', 'retained'):
                    result[name][scope][stage] = _aggregate(
                        [row['arms'][name][scope][stage] for row in items])
        return result

    by_component = defaultdict(list)
    for row in rows:
        by_component[row['component']].append(row)
    largest = max(by_component, key=lambda c: (len(by_component[c]), c))
    result = dict(scenes=len(rows), ranking=ranking, largest_component=largest,
                  by_component={c: aggregate(items) for c, items in by_component.items()},
                  full=aggregate(rows),
                  without_largest=aggregate([r for r in rows if r['component'] != largest]))
    if arms == ARMS:
        result['gates'] = {base: qualification(result, base) for base in ('056', '056_T', 'BCE')}
    else:
        result['gates'] = {arm: qualification(result, '056', arm) for arm in arms[2:]}
    return result


def qualification(result, base, candidate='RANK'):
    def direction(arms):
        rank, old = arms[candidate], arms[base]
        return (all(rank['mstsa']['decoded'][k] < old['mstsa']['decoded'][k] for k in ('fn', 'fp'))
                and all(rank['itsa']['decoded'][k] <= old['itsa']['decoded'][k] for k in ('fn', 'fp')))

    ranking = result['ranking']
    raw = all(ranking['mstsa'][candidate][p]['fp'] < ranking['mstsa'][base][p]['fp']
              and ranking['itsa'][candidate][p]['fp'] <= ranking['itsa'][base][p]['fp'] for p in ('80', '90'))
    decoded = direction(result['full'])
    coverage_counts = result['full'][candidate]['coverage']
    retained = (all(coverage_counts[k] == 0 for k in ('identities_lost', 'same_ship_lost'))
                and coverage_counts['duplicate_increase_scenes'] == 0)
    components = sum(direction(arms) for arms in result['by_component'].values())
    sensitivity = direction(result['without_largest'])
    return dict(raw=raw, decoded=decoded, retained=retained, joint_benefit_components=components,
                without_largest=sensitivity, passed=raw and decoded and retained and components >= 2 and sensitivity)


def experiment(args, manifest, original, index, start):
    require(args.prep is not None, 'completed prep path required')
    prep = audit.read_json(args.prep / 'summary.json')
    require(prep['qualified'] and audit.read_json(args.prep / 'run.json')['status'] == 'COMPLETED',
            'prep did not qualify')
    roles = audit.read_json(args.prep / 'frozen-split.json')
    require(roles == scene_split(manifest['splits']['T_core'], manifest['components']), 'prep roles changed')
    _, checkpoint = load_checkpoint(original / 'final.pt', 'cpu')
    normalizer = checkpoint['normalizer']
    summary_state(args.output, start, status='RUNNING', phase='fit_population')
    pools = prepare_fit(index['T_core'], roles['fit'], args.prep, normalizer)
    heads, training = fit_heads(pools, args.device)
    torch.save({name: head.state_dict() for name, head in heads.items()}, args.output / 'private' / 'heads.pt')
    write(args.output / 'training.json', training)
    del pools
    summary_state(args.output, start, status='RUNNING', phase='calibration', optimizer_updates=2 * STEPS)
    calibration = calibrate(index['T_core'], roles['calibration'], normalizer, heads, args.device, args.prep)
    thresholds = {'056': ws.THRESHOLD, **{name: value['threshold'] for name, value in calibration.items()}}
    write(args.output / 'calibration.json', dict(thresholds=thresholds, points=calibration))
    components = manifest['components']
    summary_state(args.output, start, status='RUNNING', phase='held', optimizer_updates=2 * STEPS)
    held = evaluate(args, 'held', roles['held'], index['T_core'], normalizer, heads, thresholds, None, components)
    summary_state(args.output, start, status='RUNNING', phase='V_select', optimizer_updates=2 * STEPS)
    with np.load(original / 'final-scores.npz', allow_pickle=False) as archive:
        ids = manifest['splits']['V_select']
        require(archive['scene_ids'].tolist() == ids, 'V order differs from 056 score archive')
        dev = evaluate(args, 'V_select', ids, index['V_select'], normalizer, heads, thresholds, archive, components)
    expected = audit.read_json(args.runs / audit.RUNS['114'] / 'summary.json')['full_V']['decoded']['WS']
    for scope in SCOPES:
        audit.check_counts(dev['full']['056'][scope]['decoded'], expected[scope])
    write(args.output / 'summary.json', dict(held=held, V_select=dev, fixed_thresholds=thresholds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'experiment'), required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prep', type=Path)
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
        if args.mode == 'prepare':
            prepare(args, manifest, original, index, cache, start)
        else:
            experiment(args, manifest, original, index, start)
    except Exception as exc:
        summary_state(args.output, start, status='FAILED', error=repr(exc))
        raise
    updates = 0 if args.mode == 'prepare' else 2 * STEPS
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=updates, test_opened=False)


if __name__ == '__main__':
    main()
