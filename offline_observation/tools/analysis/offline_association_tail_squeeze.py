"""Read frozen C92 scores to diagnose source offsets and negative-tail transfer."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np


def describe(values):
    return dict(count=len(values), quantiles=np.quantile(values, [0, .01, .1, .5, .9, .99, 1]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    began = time.monotonic()
    summary = json.loads((args.run / 'summary.json').read_text())
    calibration = json.loads((args.run / 'calibration.json').read_text())
    index = json.loads((args.runs / 'LEVEL1-FUSION-PEER-FULL-DATA-20261030-151-attempt-1'
                       / 'private/index.json').read_text())['V_select']
    ids = json.loads(args.manifest.read_text())['splits']['V_select']
    counters, blocks, scene_rows = {}, defaultdict(list), []
    for arm in ('056', 'BCE', 'RANK'):
        counters[arm] = {scope: Counter() for scope in ('itsa', 'mstsa')}
    for sid in ids:
        with np.load(index[sid]['motion'], allow_pickle=False) as motion:
            sources, support, nodes = motion['sources'], motion['support'], motion['node_ids']
        with np.load(index[sid]['evaluation'], allow_pickle=False) as evaluation:
            np.testing.assert_array_equal(nodes, evaluation['node_ids'])
            identities = evaluation['identities']
        with np.load(args.run / 'private' / 'V_select' / f'{sid}.npz', allow_pickle=False) as saved:
            pairs = saved['pairs']
            left, right = pairs.T
            truth = identities[left] == identities[right]
            same = sources[left] == sources[right]
            joint = (support[left] & support[right]).any(axis=1)
            old = saved['score_056']
            for scope, mask in (('itsa', same), ('mstsa', ~same)):
                negative = mask & ~truth
                values = old[negative]
                cutoff = np.sort(values)[len(values) - int(np.ceil(len(values) / 100))]
                tail = negative & (old >= cutoff)
                for arm in counters:
                    score = saved[f'score_{arm}']
                    threshold = calibration['thresholds'][arm]
                    chosen = score >= threshold
                    fixed90 = score >= summary['V_select']['ranking'][scope][arm]['90']['threshold']
                    original_fp = negative & (old >= calibration['thresholds']['056'])
                    new_fp = negative & chosen & ~original_fp
                    counts = dict(
                        tp=int((chosen & truth & mask).sum()),
                        fp=int((chosen & negative).sum()),
                        new_fp=int(new_fp.sum()),
                        new_fp_outside_old_top_one_pct=int((new_fp & ~tail).sum()),
                        new_fp_with_joint_support=int((new_fp & joint).sum()),
                        fp_outside_old_top_one_pct=int((chosen & negative & ~tail).sum()),
                        fp_at_90_recall=int((fixed90 & negative).sum()),
                        fp_at_90_outside_old_top_one_pct=int((fixed90 & negative & ~tail).sum()),
                        old_tail_negative=int(tail.sum()), all_negative=int(negative.sum()))
                    counters[arm][scope].update(counts)
                    scene_rows.append(dict(scene=sid, scope=scope, arm=arm, **counts))
                    blocks[arm, scope, 'positive'].append(score[mask & truth])
                    blocks[arm, scope, 'negative'].append(score[negative])

    distributions = {}
    for arm in counters:
        distributions[arm] = {scope: {label: describe(np.concatenate(blocks[arm, scope, label]))
                                      for label in ('positive', 'negative')}
                              for scope in ('itsa', 'mstsa')}
    # Source-wise constant shifts cancel in within-source pair differences.
    # This is an algebraic diagnostic, leaving all frozen predictions untouched.
    invariance = []
    rng = np.random.default_rng(20261112)
    for scope, shift in (('itsa', -7.5), ('mstsa', 4.25)):
        positives = np.concatenate(blocks['RANK', scope, 'positive'])
        negatives = np.concatenate(blocks['RANK', scope, 'negative'])
        pos = positives[rng.integers(len(positives), size=1024)]
        neg = negatives[rng.integers(len(negatives), size=1024)]
        direct = np.logaddexp(0, neg - pos)
        moved = np.logaddexp(0, (neg + shift) - (pos + shift))
        invariance.append(dict(scope=scope, shift=shift, pair_count=1024,
                               max_loss_change=float(np.max(np.abs(direct - moved)))))
    result = dict(status='COMPLETED', seconds=time.monotonic() - began, model_forwards=0,
                  optimizer_updates=0, frozen_predictions_changed=False, scene_count=len(ids),
                  counters={arm: {scope: dict(row) for scope, row in scopes.items()}
                            for arm, scopes in counters.items()},
                  score_distributions=distributions, source_shift_invariance=invariance,
                  scene_rows=scene_rows)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
