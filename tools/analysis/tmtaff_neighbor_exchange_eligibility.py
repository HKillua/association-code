"""C122: inspect proximity and assignment risk without changing C119 predictions."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from tools.analysis.tmtaff_equal_activation_position import EPS_M, clip_positions
from tools.analysis.tmtaff_true_pair_qualification import read_npz, write_json
from tools.analysis.tmtaff_unseen_position_readout import SMALL_M, distance


C119 = 'LEVEL1-TMTAFF-PAIR-RELATIVE-HOLDOUT-20261112-attempt-1'
TOL_M = 1.e-4
CONTROLS = ('MEAN', 'WIDE', 'SUPPORT', 'OPTION', 'SUPPORT_EQUAL', 'OPTION_EQUAL')
ANCHORS = ('mean_anchor_margin', 'quality_anchor_margin', 'mean_anchor_risk', 'quality_anchor_risk')


def proximity(positions):
    return distance(positions[0], positions[1])


def margin(prediction, anchor):
    return distance(prediction[0], anchor[1]) + distance(prediction[1], anchor[0]) \
        - distance(prediction, anchor).sum(axis=0)


def matched_mask(score, opportunity, budget):
    positions = np.flatnonzero(opportunity)
    chosen = positions[np.argsort(score[positions], kind='stable')[:budget]]
    mask = np.zeros(opportunity.shape, dtype=bool)
    mask[chosen] = True
    return mask


def freeze(data, cap):
    mean, residual, eligible = (data[key] for key in ('MEAN', 'DUAL_FALLBACK', 'eligible'))
    gap = proximity(mean)
    np.testing.assert_allclose(gap, data['mean_gap'], atol=TOL_M, rtol=0.)
    limited = clip_positions(mean, residual, np.where(eligible, distance(mean, residual), 0.), cap)
    limited = np.where(eligible[..., None], limited, mean)
    movement = distance(mean, limited)
    assert (movement <= cap + TOL_M).all()
    support_gap = np.maximum(gap - cap * eligible.sum(axis=0), 0.)
    option_gap = np.minimum.reduce((gap, distance(mean[0], limited[1]),
                                    distance(limited[0], mean[1]), proximity(limited)))
    assert (support_gap <= option_gap + TOL_M).all()
    opportunity = eligible.any(axis=0)
    baseline = gap <= SMALL_M
    budget = int((baseline & opportunity).sum())
    np.testing.assert_array_equal(baseline & opportunity, matched_mask(gap, opportunity, budget))
    quality, candidate = data['QUALITY_CLIP'], data['PAIR_CLIP']
    q_action = distance(quality, mean) > EPS_M
    c_action = distance(candidate, mean) > EPS_M
    for arm, mask in ((quality, q_action), (candidate, c_action)):
        assert np.array_equal(mask & ~eligible, np.zeros_like(mask))
        np.testing.assert_allclose(arm, np.where(mask[..., None], limited, mean), atol=1.e-9, rtol=0.)
    change = distance(quality, candidate)
    return dict(MEAN=baseline, WIDE=gap <= SMALL_M + 2. * cap, SUPPORT=support_gap <= SMALL_M,
                OPTION=option_gap <= SMALL_M, SUPPORT_EQUAL=matched_mask(support_gap, opportunity, budget),
                OPTION_EQUAL=matched_mask(option_gap, opportunity, budget), gap=gap, support_gap=support_gap,
                option_gap=option_gap, opportunity=opportunity, eligible=eligible, quality_action=q_action,
                pair_action=c_action, changed=change > EPS_M, change_m=change.sum(axis=0),
                quality_gap=proximity(quality), mean_anchor_margin=margin(candidate, mean),
                quality_anchor_margin=margin(candidate, quality),
                mean_anchor_risk=gap <= distance(candidate, mean).sum(axis=0) + EPS_M,
                quality_anchor_risk=proximity(quality) <= change.sum(axis=0) + EPS_M)


def detection(data, mask, truth, predicted, baseline):
    correct = mask & truth & predicted
    false = mask & ~truth & predicted
    added = mask & predicted & ~baseline
    removed = mask & ~predicted & baseline
    return dict(candidates=int((mask & predicted).sum()), true=int(correct.sum()), false=int(false.sum()),
                true_total=int((mask & truth).sum()), added_true=int((added & truth).sum()),
                added_false=int((added & ~truth).sum()), removed_true=int((removed & truth).sum()),
                removed_false=int((removed & ~truth).sum()), true_components=len(set(data['component'][correct])),
                added_true_components=len(set(data['component'][added & truth])),
                recall=float(correct.sum() / (mask & truth).sum()) if (mask & truth).any() else None)


def classify(data, scene_names, components):
    near = data['true_gap'] <= SMALL_M
    active = data['opportunity']
    baseline = data['MEAN']
    populations = dict(common=np.ones(near.shape, dtype=bool), actionable=active)
    detection_result = {name: {arm: detection(data, mask, near, data[arm], baseline) for arm in CONTROLS}
                        for name, mask in populations.items()}
    by_scene = {scene: {arm: detection(data, active & (data['scene'] == index), near, data[arm], baseline)
                        for arm in CONTROLS} for index, scene in enumerate(scene_names)}

    q_margin, c_margin = data['quality_true_margin'], data['pair_true_margin']
    q_correct, c_correct = q_margin > EPS_M, c_margin > EPS_M
    lost, gained = q_correct & ~c_correct, ~q_correct & c_correct
    changed = data['changed_count'] > 0
    assert not ((lost | gained) & ~changed).any()
    assert np.all(np.abs(c_margin - q_margin) <= 2. * data['change_m'] + TOL_M)
    assert not (lost & (q_margin > 2. * data['change_m'] + TOL_M)).any()
    switched = []
    for index in np.flatnonzero(lost | gained):
        row = {key: data[key][index].item() for key in ('true_gap', 'gap', 'support_gap', 'option_gap',
            'quality_true_margin', 'pair_true_margin', 'changed_count', 'added_count', 'removed_count',
            'quality_action_count', 'pair_action_count', 'change_m', 'quality_gap', *ANCHORS, *CONTROLS)}
        scene = data['scene'][index]
        row.update(scene=scene_names[scene], component=components[scene], transition='lost' if lost[index] else 'gained',
                   common_index_within_scene=int((data['scene'][:index] == scene).sum()))
        switched.append(row)
    subsets = {'all': np.ones(near.shape, dtype=bool), 'near': near, 'far': ~near}
    exchange = {}
    for name, subset in subsets.items():
        eligible_lost = subset & lost
        safe_changed = subset & changed & q_correct & c_correct
        flags = {}
        for key in ANCHORS:
            hit = data[key] <= EPS_M if key.endswith('margin') else data[key]
            flags[key] = dict(lost_hit=int((hit & eligible_lost).sum()), lost_total=int(eligible_lost.sum()),
                              correct_changed_flagged=int((hit & safe_changed).sum()), correct_changed=int(safe_changed.sum()))
        exchange[name] = dict(quality_correct=int((q_correct & subset).sum()), pair_correct=int((c_correct & subset).sum()),
                              lost=int(eligible_lost.sum()), gained=int((gained & subset).sum()),
                              changed=int((changed & subset).sum()), flags=flags,
                              true_margin_certifies=int((subset & changed & q_correct
                                                         & (q_margin > 2. * data['change_m'] + TOL_M)).sum()))
    pool = detection_result['actionable']
    near_exchange = exchange['near']
    shortlist = {name: pool[name]['true'] > pool['MEAN']['true'] and pool[name]['added_false'] == 0
                 and pool[name]['added_true_components'] >= 2 for name in ('SUPPORT_EQUAL', 'OPTION_EQUAL')}
    predictive_flags = {key: bool(row['lost_total'] and row['lost_hit'] == row['lost_total']
                                  and 4 * row['correct_changed_flagged'] <= row['correct_changed'])
                        for key, row in near_exchange['flags'].items()}
    return dict(detection=detection_result, by_scene=by_scene, exchange=exchange, reversals=switched,
                qualification=dict(neighbor=shortlist, exchange=predictive_flags,
                                   new_holdout_warranted=any(shortlist.values()) and any(predictive_flags.values())))


def run(args):
    start = time.monotonic()
    source = args.runs / C119
    configuration = json.loads((source / 'freeze.json').read_text())
    assert json.loads((source / 'run.json').read_text())['status'] == 'COMPLETED'
    rows = configuration['rows']
    scenes = [row['scene'] for row in rows]
    components = [row['component'] for row in rows]
    assert len(scenes) == len(set(scenes)) == 25 and len(set(components)) == 24
    cap = configuration['calibration']['cap_m']
    private = args.output / 'private'
    private.mkdir(mode=0o700, parents=True)
    frozen = [freeze(read_npz(source / 'private' / f'{scene}.npz'), cap) for scene in scenes]
    for scene, row in zip(scenes, frozen, strict=True):
        np.savez_compressed(private / f'{scene}.npz', **row)
    write_json(args.output / 'freeze.json', dict(source=C119, scenes=scenes, components=24,
               cap_m=cap, cutoff_m=SMALL_M, private_forecasts_read=25, raw_archive_members=0,
               reference_rows_before_freeze=0, new_prediction_or_action_writes=0))

    evaluation = read_npz(source / 'public-errors.npz')
    parts = []
    for i, row in enumerate(frozen):
        pos = evaluation['scene_index'] == i
        pair = evaluation['pair_scene'] == i
        valid = evaluation['valid'][pos].all(axis=0)
        np.testing.assert_array_equal(row['eligible'], evaluation['eligible'][pos])
        for key, arm in (('quality_action', 'QUALITY_CLIP'), ('pair_action', 'PAIR_CLIP')):
            np.testing.assert_array_equal(row[key], evaluation[f'{arm}/action'][pos])
        np.testing.assert_allclose(row['gap'][valid], evaluation['mean_gap_m'][pair], atol=TOL_M, rtol=0.)
        part = {key: row[key][valid] for key in (*CONTROLS, *ANCHORS, 'gap', 'support_gap', 'option_gap',
                                                'opportunity', 'change_m', 'quality_gap')}
        part.update(scene=np.full(valid.sum(), i), component=np.repeat(components[i], valid.sum()),
                    true_gap=evaluation['gap_m'][pair], quality_true_margin=evaluation['QUALITY_CLIP/margin'][pair],
                    pair_true_margin=evaluation['PAIR_CLIP/margin'][pair],
                    changed_count=row['changed'][:, valid].sum(axis=0),
                    added_count=(row['pair_action'] & ~row['quality_action'])[:, valid].sum(axis=0),
                    removed_count=(row['quality_action'] & ~row['pair_action'])[:, valid].sum(axis=0),
                    quality_action_count=row['quality_action'][:, valid].sum(axis=0),
                    pair_action_count=row['pair_action'][:, valid].sum(axis=0))
        parts.append(part)
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    np.savez_compressed(args.output / 'public-diagnostics.npz', **data)
    near = data['true_gap'] <= SMALL_M
    assert (len(near), int(near.sum()), int((near & data['opportunity']).sum()),
            int((near & data['opportunity'] & data['MEAN']).sum())) == (729, 52, 39, 21)
    result = classify(data, scenes, components)
    result.update(queries=1600, common=int(len(near)), true_near=int(near.sum()),
                  reference_position_valid=int(evaluation['valid'].sum()),
                  full_actionable_q=sum(int(row['opportunity'].sum()) for row in frozen),
                  full_actionable_q_neighbor_counts={key: sum(int((row[key] & row['opportunity']).sum())
                                                             for row in frozen) for key in CONTROLS},
                  full_equal_budget_source=int(sum((row['MEAN'] & row['opportunity']).sum() for row in frozen)))
    write_json(args.output / 'summary.json', result)
    write_json(args.output / 'run.json', dict(status='COMPLETED', elapsed_seconds=time.monotonic() - start,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               fitted=0, forward_calls=0, test_opened=False, raw_archive_members=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
