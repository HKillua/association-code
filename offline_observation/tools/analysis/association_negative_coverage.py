"""C95 equal-budget negative allocation and anchored ranking factorial."""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from tools.analysis import offline_association_tail as previous


ARMS = {
    'T_BCE': (False, False),
    'T_ANCHORED': (False, True),
    'M_BCE': (True, False),
    'M_ANCHORED': (True, True),
}
BACKGROUND_SEED = previous.SEED + 1
RANK_WEIGHT = .5
RETAIN = previous.EACH_SCOPE // 4
NEGATIVE_HALF = previous.EACH_SCOPE // 2
FIT_KEYS = [(scope, kind) for scope in previous.SCOPES for kind in ('positive', 'top', 'rest')]
require = previous.require


def prepare_background(index, ids, prep, prep_summary, normalizer):
    rng = np.random.default_rng(BACKGROUND_SEED)
    draws, positions, inverse, population = {}, {}, {}, {}
    for scope in previous.SCOPES:
        counts = prep_summary['role_scopes']['fit'][scope]
        population[scope] = counts['negative'] - counts['tail']
        draws[scope] = rng.integers(0, population[scope], (previous.STEPS, NEGATIVE_HALF))
        positions[scope], inverse[scope] = np.unique(draws[scope], return_inverse=True)
    offsets = Counter()
    feature_blocks = {scope: [] for scope in previous.SCOPES}
    exposures = {scope: Counter() for scope in previous.SCOPES}
    for sid in ids:
        row = index[sid]
        motion = previous.read_motion(row)
        pairs = previous.all_pairs(len(motion['node_ids']))
        identities = previous.read_identity(row, motion)
        truth = identities[pairs[:, 0]] == identities[pairs[:, 1]]
        scores = np.load(prep / 'private' / f'scores-{sid}.npy', allow_pickle=False)
        require(len(scores) == len(pairs), 'background score population changed')
        for scope, mask in zip(previous.SCOPES, previous.scope_masks(motion, pairs), strict=True):
            negatives = np.flatnonzero(mask & ~truth)
            top, rest = previous.tail_indices(scores, negatives)
            background = np.setdiff1d(negatives, np.r_[top, rest], assume_unique=True)
            begin, end = offsets[scope], offsets[scope] + len(background)
            start_pos, stop_pos = np.searchsorted(positions[scope], (begin, end))
            selected = background[positions[scope][start_pos:stop_pos] - begin]
            if len(selected):
                x, _ = previous.features(motion, pairs[selected], normalizer)
                feature_blocks[scope].append((x, scores[selected]))
                exposures[scope][row['component_id']] += len(selected)
            offsets[scope] = end
    require(dict(offsets) == population, 'background complement differs from C92 census')
    result, schedule, accounting = {}, {}, {}
    for scope in previous.SCOPES:
        result[scope] = tuple(np.concatenate(blocks) for blocks in zip(*feature_blocks[scope]))
        require(len(result[scope][0]) == len(positions[scope]), 'background draw mapping incomplete')
        schedule[scope] = inverse[scope].reshape(draws[scope].shape)
        accounting[scope] = dict(available=population[scope], unique=len(positions[scope]),
                                 presentations=int(draws[scope].size),
                                 unique_by_component=dict(exposures[scope]))
    return result, schedule, accounting


def fit(pools, background, schedule, device):
    torch.manual_seed(previous.SEED)
    heads = {name: previous.EvidenceAssociation().to(device) for name in ARMS}
    first = heads['T_BCE'].state_dict()
    for head in heads.values():
        head.load_state_dict(first)
    optimizers = {name: torch.optim.Adam(head.parameters(), lr=.001) for name, head in heads.items()}
    rng = np.random.default_rng(previous.SEED)
    fit_arrays = {key: tuple(torch.from_numpy(a).to(device) for a in pools[key]) for key in FIT_KEYS}
    background_arrays = {scope: tuple(torch.from_numpy(a).to(device) for a in values)
                         for scope, values in background.items()}
    coverage = {key: np.zeros((2, len(pools[key][1])), bool) for key in FIT_KEYS}
    with torch.inference_mode():
        for head in heads.values():
            for x, old in (*fit_arrays.values(), *background_arrays.values()):
                require(torch.equal(head(x[:32], old[:32]), old[:32]), 'u0 differs from frozen 056')
    curve = []
    for step in range(previous.STEPS):
        draws = {}
        for key, (x, old) in fit_arrays.items():
            positive = key[1] == 'positive'
            count = previous.EACH_SCOPE if positive else NEGATIVE_HALF
            sampled = rng.integers(0, len(old), size=count)
            mixed_count = count if positive else RETAIN
            coverage[key][0, sampled] = True
            coverage[key][1, sampled[:mixed_count]] = True
            draws[key] = x[sampled], old[sampled]
        step_log = {}
        for name, (mixed, anchored) in ARMS.items():
            head = heads[name]
            optimizers[name].zero_grad(set_to_none=True)
            bce_terms, rank_terms = [], []
            for scope in previous.SCOPES:
                pos = head(*draws[scope, 'positive'])
                tails = []
                for kind in ('top', 'rest'):
                    x, old = draws[scope, kind]
                    tails.append(head(x[:RETAIN], old[:RETAIN]) if mixed else head(x, old))
                negatives = list(tails)
                if mixed:
                    x, old = background_arrays[scope]
                    selected = schedule[scope][step]
                    negatives.append(head(x[selected], old[selected]))
                neg = torch.cat(negatives)
                bce_terms.append((F.softplus(-pos).mean() + F.softplus(neg).mean()) / 2)
                if anchored:
                    common_tail = torch.cat([values[:RETAIN] for values in tails])
                    rank_terms.append(F.softplus(common_tail - pos[:NEGATIVE_HALF]).mean())
            bce = torch.stack(bce_terms).mean()
            rank = torch.stack(rank_terms).mean() if anchored else None
            loss = bce + RANK_WEIGHT * rank if anchored else bce
            require(bool(torch.isfinite(loss)), 'nonfinite objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizers[name].step()
            if step == 0 or (step + 1) % 128 == 0:
                step_log[name] = dict(bce=float(bce.detach()), rank=float(rank.detach()) if anchored else None,
                                      loss=float(loss.detach()), gradient_norm=float(norm))
        if step_log:
            curve.append(dict(step=step + 1, **step_log))
    for head in heads.values():
        head.eval().requires_grad_(False)
    counts = {scope: {kind: dict(available=len(pools[scope, kind][1]),
                                tail_unique=int(coverage[scope, kind][0].sum()),
                                mixed_unique=int(coverage[scope, kind][1].sum()))
                      for kind in ('positive', 'top', 'rest')} for scope in previous.SCOPES}
    return heads, dict(updates_per_arm=previous.STEPS, presentations_per_arm=512 * previous.STEPS,
                       seed=previous.SEED, background_seed=BACKGROUND_SEED, rank_weight=RANK_WEIGHT,
                       u0_equal=True, sampled_unique=counts, curve=curve)


def calibrate(args, index, ids, normalizer, heads):
    out = args.output / 'private' / 'calibration'
    out.mkdir()
    for sid in ids:
        motion, pairs, scores = previous.score_scene(
            index[sid], sid, normalizer, heads, args.device, args.prep, None)
        np.savez(out / f'{sid}.npz', pairs=pairs, nodes=motion['node_ids'],
                 sources=motion['sources'], **scores)
    names = ('056_T', *heads)
    by_scope = {scope: {name: [] for name in ('truth', *names)} for scope in previous.SCOPES}
    for sid in ids:
        row = index[sid]
        motion = previous.read_motion(row)
        identities = previous.read_identity(row, motion)
        with np.load(out / f'{sid}.npz', allow_pickle=False) as saved:
            pairs = saved['pairs']
            np.testing.assert_array_equal(saved['nodes'], motion['node_ids'])
            np.testing.assert_array_equal(saved['sources'], motion['sources'])
            truth = identities[pairs[:, 0]] == identities[pairs[:, 1]]
            for scope, mask in zip(previous.SCOPES, previous.scope_masks(motion, pairs), strict=True):
                by_scope[scope]['truth'].append(truth[mask])
                for name in names:
                    by_scope[scope][name].append(saved[name][mask])
    points = {}
    for scope, blocks in by_scope.items():
        truth = np.concatenate(blocks.pop('truth'))
        points[scope] = {name: {str(p): previous.recall_point(np.concatenate(blocks[name]), truth, p)
                               for p in (80, 90)} for name in names}
    thresholds = {'056': previous.ws.THRESHOLD,
                  **{name: points['mstsa'][name]['90']['threshold'] for name in names}}
    previous.write(args.output / 'calibration.json', dict(thresholds=thresholds, points=points))
    return thresholds


def factorial(result):
    def contrast(left, right):
        return {scope: {
            'fp_at_recall': {str(p): result['ranking'][scope][left][str(p)]['fp']
                             - result['ranking'][scope][right][str(p)]['fp'] for p in (80, 90)},
            'decoded': {key: result['full'][left][scope]['decoded'][key]
                        - result['full'][right][scope]['decoded'][key] for key in ('fn', 'fp')}}
            for scope in previous.SCOPES}

    effects = dict(background_bce=contrast('M_BCE', 'T_BCE'),
                   background_anchored=contrast('M_ANCHORED', 'T_ANCHORED'),
                   rank_tail=contrast('T_ANCHORED', 'T_BCE'),
                   rank_mixed=contrast('M_ANCHORED', 'M_BCE'))
    effects['interaction'] = {
        scope: {metric: {key: value - effects['rank_tail'][scope][metric][key]
                         for key, value in values.items()}
                for metric, values in effects['rank_mixed'][scope].items()}
        for scope in previous.SCOPES}
    return effects


def run(args, start):
    manifest = previous.audit.read_json(args.manifest)
    original, index, _ = previous.locations(args.runs, manifest)
    prep_summary = previous.audit.read_json(args.prep / 'summary.json')
    require(prep_summary['qualified'] and previous.audit.read_json(args.prep / 'run.json')['status'] == 'COMPLETED',
            'C92 prep incomplete')
    roles = previous.audit.read_json(args.prep / 'frozen-split.json')
    components = manifest['components']
    require(roles == previous.scene_split(manifest['splits']['T_core'], components), 'C92 split changed')
    _, checkpoint = previous.load_checkpoint(original / 'final.pt', 'cpu')
    normalizer = checkpoint['normalizer']
    pools = previous.prepare_fit(index['T_core'], roles['fit'], args.prep, normalizer)
    background, schedule, counts = prepare_background(index['T_core'], roles['fit'], args.prep,
                                                       prep_summary, normalizer)
    heads, training = fit(pools, background, schedule, args.device)
    training['background'] = counts
    frozen = torch.load(args.c92 / 'private' / 'heads.pt', map_location='cpu', weights_only=True)['BCE']
    delta = max(float((value.cpu() - frozen[key]).abs().max())
                for key, value in heads['T_BCE'].state_dict().items())
    training['c92_bce_weight_max_abs_delta'] = delta
    previous.write(args.output / 'training.json', training)
    torch.save({name: head.state_dict() for name, head in heads.items()}, args.output / 'private' / 'heads.pt')
    del pools, background, schedule
    previous.summary_state(args.output, start, status='RUNNING', phase='calibration')
    thresholds = calibrate(args, index['T_core'], roles['calibration'], normalizer, heads)
    previous.summary_state(args.output, start, status='RUNNING', phase='held')
    held = previous.evaluate(args, 'held', roles['held'], index['T_core'], normalizer, heads, thresholds,
                             None, components)
    previous.summary_state(args.output, start, status='RUNNING', phase='V_select')
    with np.load(original / 'final-scores.npz', allow_pickle=False) as archive:
        ids = manifest['splits']['V_select']
        require(archive['scene_ids'].tolist() == ids, 'V order differs from 056')
        dev = previous.evaluate(args, 'V_select', ids, index['V_select'], normalizer, heads, thresholds,
                                archive, components)
    old = previous.audit.read_json(args.c92 / 'summary.json')
    for label, result in (('held', held), ('V_select', dev)):
        for scope in previous.SCOPES:
            previous.audit.check_counts(result['full']['056'][scope]['decoded'],
                                        old[label]['full']['056'][scope]['decoded'])
            if delta == 0:
                previous.audit.check_counts(result['full']['T_BCE'][scope]['decoded'],
                                            old[label]['full']['BCE'][scope]['decoded'])
        result['factorial'] = factorial(result)
    previous.write(args.output / 'summary.json', dict(held=held, V_select=dev, fixed_thresholds=thresholds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'prep', 'c92', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    (args.output / 'private').mkdir()
    start = time.monotonic()
    torch.set_num_threads(4)
    previous.summary_state(args.output, start, status='RUNNING', phase='fit')
    try:
        run(args, start)
    except Exception as exc:
        previous.summary_state(args.output, start, status='FAILED', error=repr(exc))
        raise
    previous.summary_state(args.output, start, status='COMPLETED',
                           optimizer_updates=len(ARMS) * previous.STEPS, test_opened=False)


if __name__ == '__main__':
    main()
