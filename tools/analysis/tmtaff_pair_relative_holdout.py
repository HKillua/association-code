"""C119. Evaluate an equal-budget pair-relative position gate on component holdouts."""

import argparse
from pathlib import Path
import resource
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.motion_evidence import scene_motion_evidence
from association.models.physical_fusion import geographic_log
from tools.analysis import held_fusion_population as held
from tools.analysis import tmtaff_dual_position_fallback as c108
from tools.analysis import tmtaff_equal_activation_position as c115
from tools.analysis import tmtaff_true_pair_qualification as fixed
from tools.analysis import tmtaff_unseen_position_readout as c105


C107 = 'LEVEL1-OFFLINE-REGISTERED-HOLDOUT-FUSION-20261112-attempt-1'
C115 = 'LEVEL1-TMTAFF-EQUAL-ACTIVATION-POSITION-20261112-attempt-1'
ARMS = ('MEAN', 'DUAL_FALLBACK', 'MAG_CLIP', 'QUALITY_CLIP', 'PAIR_CLIP', 'FIXED_QUALITY_CLIP', 'FIXED_PAIR_CLIP')
CHOICES = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bool)
COUNTS = CHOICES.sum(axis=1)
TIE = 1e-10


def roster(runs):
    origin = runs / C107
    record = held.read(origin / 'roster.json')
    manifest = held.read(origin / 'private/supplement-manifest.json')
    history = held.read(runs / c115.C108 / 'freeze.json')
    previous = held.read(runs / C115 / 'calibration.json')
    ordered = record['isolated']
    components = manifest['components']
    used = set(fixed.SCENES) | set(history['C105_roster']) | set(previous['scenes'])
    used.update(row['scene'] for row in history['new_roster'])
    used.update(sid for sid in manifest['splits']['V_select'] if sid not in record['new'])
    used.update(manifest['splits']['T_core'])
    used.update(manifest['splits']['T_test'])  # component metadata only
    novel = {components[sid] for sid in ordered}
    fixed.require(len(ordered) == 25 and len(novel) == 24 and not novel & {components[sid] for sid in used},
                  'C119 component roster changed')
    fixed.require(all(sid in record['new'] for sid in ordered) and not record['sealed_opened'],
                  'C107 development provenance changed')
    return origin, ordered, components, previous


def prepare(archive, origin, sid, component, head):
    with np.load(origin / 'private' / f'{sid}.npz', allow_pickle=False) as data:
        registry = {key: data[key] for key in ('node_ids', 'identities', 'query_seconds')}
        metadata = {key: data[key] for key in ('node_ids', 'sources', 'query_seconds', 'native_count')}
    evaluation = origin / 'private' / f'{sid}-evaluation.npz'
    row = c105.held_scene(archive, registry, sid, evaluation, metadata)
    pairs, scene = row['pairs'], row['scene']
    fixed.require(len(pairs) == 2 and len(np.unique(row['identities'][pairs[:, 0]])) == 2,
                  f'{sid} fixed pair population changed')
    physical = c105.physical(scene, pairs)
    support = physical['support']
    physical['fraction'] = np.broadcast_to(np.linspace(-1., 1., support.shape[1]), support.shape)
    mean = physical['mean']
    residual = c105.predict(physical, head, *c105.HEADS[c108.RESIDUAL])
    motion = scene_motion_evidence(scene, motion_observation_policy='selected_cap128')
    donor = motion['states'][pairs, :, :2]
    magnitude = c105.distance(mean, residual)
    return dict(sid=sid, component=component, pairs=pairs, identities=row['identities'], evaluation=evaluation,
                nodes=registry['node_ids'], query=physical['query'], mean=mean, residual=residual, support=support,
                eligible=(support == 2) & (magnitude > c115.EPS_M), magnitude=magnitude,
                age=motion['nearest_seconds'][pairs].max(axis=1), disagreement=c105.distance(donor[:, 0], donor[:, 1]))


def option_costs(row, calibration, limited):
    scores = c115.quality(row, calibration)
    eligible = row['eligible']
    fixed.require(np.isfinite(scores[eligible]).all(), 'nonfinite eligible quality score')
    independent = np.where(eligible, scores, 0.).T @ CHOICES.T
    allowed = (eligible.T[:, None, :] | ~CHOICES).all(axis=-1)
    mean = row['mean']
    anchor = mean[0, 0]
    shift = geographic_log(limited, anchor) - geographic_log(mean, anchor)
    relative = np.linalg.norm(shift[0, :, None, :] * CHOICES[None, :, 0, None]
                              - shift[1, :, None, :] * CHOICES[None, :, 1, None], axis=-1)
    gap = c105.distance(mean[0], mean[1])
    penalty = np.where(gap[:, None] <= c105.SMALL_M, relative / np.maximum(gap[:, None], calibration['cap_m']), 0.)
    return scores, np.where(allowed, independent + penalty, np.inf), gap, penalty


def matched_options(cost, control, budget):
    """Minimize the summed option cost under the same scene budget as the control."""
    width = len(cost)
    best = np.full((width + 1, budget + 1), np.inf)
    edits = np.full((width + 1, budget + 1), 2 * width + 1, dtype=int)
    back = np.full((width + 1, budget + 1), -1, dtype=int)
    best[0, 0] = edits[0, 0] = 0
    for q, costs in enumerate(cost):
        for used in np.flatnonzero(np.isfinite(best[q])):
            for option in np.flatnonzero(np.isfinite(costs)):
                updated = used + COUNTS[option]
                if updated > budget:
                    continue
                candidate = best[q, used] + costs[option]
                changes = edits[q, used] + np.count_nonzero(CHOICES[option] != control[:, q])
                previous = best[q + 1, updated]
                if candidate < previous - TIE or (abs(candidate - previous) <= TIE and changes < edits[q + 1, updated]):
                    best[q + 1, updated] = candidate
                    edits[q + 1, updated] = changes
                    back[q + 1, updated] = option
    choice = np.empty(width, dtype=int)
    left = budget
    for i in range(width, 0, -1):
        chosen = back[i, left]
        fixed.require(chosen >= 0, 'scene budget infeasible')
        choice[i - 1] = chosen
        left -= COUNTS[chosen]
    result = CHOICES[choice].T
    fixed.require(left == 0 and result.sum() == budget, 'scene budget did not close')
    return result


def frozen_options(cost, control, threshold):
    action_cost = cost - threshold * COUNTS
    available = action_cost <= action_cost.min(axis=-1, keepdims=True) + TIE
    edits = (control.T[:, None, :] != CHOICES).sum(axis=-1)
    return CHOICES[np.where(available, edits, 3).argmin(axis=1)].T


def freeze(row, calibration, folder):
    mean, residual, eligible = (row[name] for name in ('mean', 'residual', 'eligible'))
    cap = calibration['cap_m']
    limited = c115.clip_positions(mean, residual, np.where(eligible, row['magnitude'], 0.), cap)
    scores, cost, gap, penalty = option_costs(row, calibration, limited)
    budget = np.count_nonzero(eligible) // 2
    quality = c115.low_risk(scores, eligible, budget)
    fixed_quality = eligible & (scores <= calibration['fixed_quality'])
    masks = dict(MEAN=np.zeros_like(eligible), DUAL_FALLBACK=row['support'] == 2,
                 MAG_CLIP=c115.low_risk(row['magnitude'], eligible, budget), QUALITY_CLIP=quality,
                 PAIR_CLIP=matched_options(cost, quality, budget), FIXED_QUALITY_CLIP=fixed_quality,
                 FIXED_PAIR_CLIP=frozen_options(cost, fixed_quality, calibration['fixed_quality']))
    predictions = {}
    for arm, mask in masks.items():
        prediction = mean.copy()
        position = limited if arm.endswith('_CLIP') else residual
        prediction[mask] = position[mask]
        fixed.require(np.isfinite(prediction).all() and prediction[~mask].tobytes() == mean[~mask].tobytes(),
                      f'{arm} lost coverage or changed a fallback')
        if arm.endswith('_CLIP'):
            fixed.require(not (mask & ~eligible).any() and np.all(c105.distance(prediction[mask], mean[mask]) <= cap + c115.EPS_M),
                          f'{arm} exceeded original movement bound')
        predictions[arm] = prediction
    np.savez_compressed(folder / f"{row['sid']}.npz", query=row['query'], pairs=row['pairs'],
                        support=row['support'], eligible=eligible, mean_gap=gap, penalty=penalty,
                        **predictions)
    return dict(scene=row['sid'], component=row['component'], budget=budget,
                opportunities=int(eligible.sum()), changed_slots=int((masks['PAIR_CLIP'] ^ quality).sum()),
                predicted_near_q=int((gap <= c105.SMALL_M).sum()))


def evaluate_scene(row, folder):
    frozen = fixed.read_npz(folder / f"{row['sid']}.npz")
    with np.load(row['evaluation'], allow_pickle=False) as reference:
        for key, value in (('nodes', row['nodes']), ('identities', row['identities']), ('query', frozen['query'])):
            np.testing.assert_array_equal(reference[key], value)
        by_id = {identity: i for i, identity in enumerate(reference['target_ids'])}
        indexes = [by_id[identity] for identity in row['identities'][frozen['pairs'][:, 0]]]
        truth = reference['target_states'][indexes, :, :2]
        valid = reference['target_valid'][indexes, :, :2].all(-1)
    common = valid.all(axis=0)
    y = truth[:, common]
    mean = frozen['MEAN']
    anchor = mean[0, 0]
    true_vector = c105.delta_m(y[0], y[1], anchor)
    result = dict(valid=valid, support=frozen['support'], eligible=frozen['eligible'],
                  gap_m=c105.distance(y[0], y[1]), mean_gap_m=frozen['mean_gap'][common],
                  near_opportunities=frozen['eligible'][:, common].sum(axis=0))
    for arm in ARMS:
        prediction = frozen[arm]
        err = c105.distance(prediction, truth)
        fixed.require(np.isfinite(err[valid]).all(), f'{arm} lost a valid position')
        action = c105.distance(prediction, mean) > c115.EPS_M
        first, second = prediction[:, common]
        vector = np.linalg.norm(c105.delta_m(first, second, anchor) - true_vector, axis=-1)
        margin = c105.distance(first, y[1]) + c105.distance(second, y[0]) - err[:, common].sum(axis=0)
        result.update({f'{arm}/error': err, f'{arm}/action': action, f'{arm}/pair_action': action[:, common].sum(0),
                       f'{arm}/vector': vector, f'{arm}/margin': margin})
    fixed.require(len({result[f'{arm}/action'].sum() for arm in ('MAG_CLIP', 'QUALITY_CLIP', 'PAIR_CLIP')}) == 1,
                  'actual scene action counts differ')
    return result


def summarize(rows, folder, freezes):
    scenes = [evaluate_scene(row, folder) for row in rows]
    public = {key: np.concatenate([scene[key] for scene in scenes]) for key in scenes[0]}
    public['scene_index'] = np.repeat(np.arange(len(rows)), [len(scene['valid']) for scene in scenes])
    public['pair_scene'] = np.repeat(np.arange(len(rows)), [len(scene['gap_m']) for scene in scenes])
    groups = np.asarray([row['component'] for row in rows])
    position_group, pair_group = groups[public['scene_index']], groups[public['pair_scene']]
    valid, support = public['valid'], public['support']
    small = public['gap_m'] <= c105.SMALL_M
    predicted_small = public['mean_gap_m'] <= c105.SMALL_M
    opportunity = public['near_opportunities']
    chosen, control = (public[f'{arm}/pair_action'] > 0 for arm in ('PAIR_CLIP', 'QUALITY_CLIP'))
    near_union = small & (chosen | control)
    near_intersection = small & chosen & control
    base = public['MEAN/error']
    metrics = {}
    for arm in ARMS:
        error, action, n_action = (public[f'{arm}/{name}'] for name in ('error', 'action', 'pair_action'))
        active = small & (n_action > 0)
        metrics[arm] = dict(
            all=c115.position_metrics(error, base, valid),
            dual=c115.position_metrics(error, base, valid & (support == 2)),
            active=c115.position_metrics(error, base, valid & action),
            harm_gt_250m=int(((error - base)[valid] > 250.).sum()),
            coverage=dict(output_queries=int(error.size), reference_valid=int(valid.sum()),
                          action_queries=int(action.sum()), action_valid=int((valid & action).sum())),
            small_all=c115.pair_metrics(public, arm, small),
            small_active=c115.pair_metrics(public, arm, active),
            small_active_components=len(set(pair_group[active])),
            small_active_predicted_close=int((active & predicted_small).sum()),
            small_by_n={str(n): c115.pair_metrics(public, arm, small & (n_action == n)) for n in range(3)},
            small_union=c115.pair_metrics(public, arm, near_union),
            small_intersection=c115.pair_metrics(public, arm, near_intersection),
            scenes={row['sid']: c115.position_metrics(error, base, valid & (public['scene_index'] == i)[:, None])
                    for i, row in enumerate(rows)},
            components={group: c115.position_metrics(error, base, valid & (position_group == group)[:, None])
                        for group in sorted(set(groups))},
            leave_one_component_out={group: c115.position_metrics(error, base, valid & (position_group != group)[:, None])
                                     for group in sorted(set(groups))})
    assert (int(valid.sum()), len(small), int(small.sum())) == (2169, 729, 52), 'previous reference census changed'
    return dict(metrics=metrics, components=len(set(groups)), scenes=len(rows),
                equal_budget=sum(row['budget'] for row in freezes),
                changed_slots=sum(row['changed_slots'] for row in freezes),
                all_true_near=int(small.sum()), true_near_predicted_near=int((small & predicted_small).sum()),
                predicted_near_on_common=int(predicted_small.sum()),
                true_near_with_opportunity=int((small & (opportunity > 0)).sum()),
                true_near_with_opportunity_predicted_near=int((small & (opportunity > 0) & predicted_small).sum()),
                true_near_opportunity_by_n={str(n): int((small & (opportunity == n)).sum()) for n in range(3)}), public


def gate(metrics):
    candidate = metrics['PAIR_CLIP']
    control = metrics['QUALITY_CLIP']
    base = metrics['MEAN']
    own, joint, active = (candidate[key] for key in ('small_active', 'small_union', 'active'))
    baseline_joint = control['small_union']
    return dict(
        unchanged_coverage=all(candidate['coverage'][key] == base['coverage'][key]
                               for key in ('output_queries', 'reference_valid')),
        full_mae_and_p99=candidate['all']['error']['mae'] < base['all']['error']['mae']
        and candidate['all']['error']['p99'] <= base['all']['error']['p99'],
        active_p99=bool(active['error']['count'] and active['error']['p99'] <= active['mean_same']['p99']),
        enough_near=own['vector']['count'] >= 10 and candidate['small_active_components'] >= 3,
        own_near_safe=bool(own['vector']['count'] and own['vector']['mae'] <= own['mean_same']['mae']
                           and own['correct'] >= own['mean_correct']),
        near_increment=bool(joint['vector']['count'] and joint['vector']['mae'] < baseline_joint['vector']['mae']
                            and joint['correct'] >= baseline_joint['correct']
                            and own['vector']['count'] >= control['small_active']['vector']['count']))


def run(args, start):
    origin, scene_ids, components, calibration = roster(args.runs)
    head = c108.load_head(args.runs / c108.C105)
    full = held.read(args.runs / held.PARENTS['full_data'] / 'run.json')
    folder = args.output / 'private'
    folder.mkdir(mode=0o700)
    with MTADArchive(full['archive']) as archive:
        held.validate_measurement_archive(args.runs, archive)
        rows = [prepare(archive, origin, sid, components[sid], head) for sid in scene_ids]
        members = [(entry['scene_id'], entry['kind']) for entry in archive.members_read]
    np.testing.assert_array_equal(members, [(sid, 'measurements') for sid in scene_ids])
    freezes = [freeze(row, calibration, folder) for row in rows]
    fixed.write_json(args.output / 'freeze.json', dict(rows=freezes, calibration=calibration, coefficients=44,
                     raw_measurements=len(members), raw_reference_members=0, reference_targets_before_freeze=0))
    result, public = summarize(rows, folder, freezes)
    result['qualification'] = gate(result['metrics'])
    np.savez_compressed(args.output / 'public-errors.npz', **public)
    fixed.write_json(args.output / 'summary.json', result)
    fixed.write_json(args.output / 'run.json', dict(status='COMPLETED', elapsed_seconds=time.monotonic()-start,
                     fit_operations=0, model_forward_calls=0, test_opened=False,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    start = time.monotonic()
    try:
        run(args, start)
    except Exception as error:
        fixed.write_json(args.output / 'run.json', dict(status='FAILED', error=repr(error),
                         elapsed_seconds=time.monotonic()-start, test_opened=False))
        raise


if __name__ == '__main__':
    main()
