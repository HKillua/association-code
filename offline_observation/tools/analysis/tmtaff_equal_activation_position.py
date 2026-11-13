"""C115. Compare cached residual clipping and quality at the same scene budget."""

import argparse
from collections import Counter
from pathlib import Path
import resource
import time

import numpy as np

from association.models.physical_fusion import geographic_exp, geographic_log
from tools.analysis import held_fusion_population as held
from tools.analysis import tmtaff_dual_position_fallback as c108
from tools.analysis import tmtaff_true_pair_qualification as fixed
from tools.analysis import tmtaff_unseen_position_readout as c105


C108 = 'LEVEL1-TMTAFF-DUAL-POSITION-FALLBACK-20261112-attempt-1'
ARMS = ('MEAN', 'DUAL_FALLBACK', 'MAG', 'MAG_CLIP', 'QUALITY', 'QUALITY_CLIP', 'FIXED_MAG', 'FIXED_QUALITY')
EPS_M = 1e-6


def cached_scene(origin, cohort, sid, entry, head):
    saved = fixed.read_npz(origin / 'private' / cohort / f'{sid}.npz')
    with np.load(entry['motion'], allow_pickle=False) as raw:
        motion = {key: raw[f'cap128_{key}'] for key in ('states', 'support', 'nearest_seconds')}
        motion.update({key: raw[key] for key in ('node_ids', 'sources', 'query_seconds')})
    np.testing.assert_array_equal(saved['query'], motion['query_seconds'])
    pairs = saved['pairs']
    np.testing.assert_array_equal(saved['support'], motion['support'][pairs].sum(axis=1))
    fixed.require(np.all(motion['sources'][pairs[:, 0]] != motion['sources'][pairs[:, 1]]),
                  f'{sid} contains a same-source pair')
    mean, residual = saved['MEAN'], saved[c108.RESIDUAL]
    scene = dict(motion=motion, node_ids=motion['node_ids'], grid=saved['query'], sources=motion['sources'])
    physical = c105.physical(scene, pairs)
    physical['fraction'] = np.broadcast_to(np.linspace(-1., 1., len(saved['query'])), saved['support'].shape)
    replay = c105.predict(physical, head, *c105.HEADS[c108.RESIDUAL])
    replay_distance = c105.distance(replay, residual)
    mean_distance = c105.distance(physical['mean'], mean)
    fixed.require(np.max(replay_distance) <= EPS_M and np.max(mean_distance) <= EPS_M,
                  f'{sid} source position replay changed: residual={np.max(replay_distance)} m, '
                  f'Mean={np.max(mean_distance)} m')
    magnitude = c105.distance(mean, residual)
    age = motion['nearest_seconds'][pairs].max(axis=1)
    states = motion['states'][pairs, :, :2]
    disagreement = c105.distance(states[:, 0], states[:, 1])
    eligible = (saved['support'] == 2) & (magnitude > EPS_M)
    fixed.require(all(np.isfinite(values[eligible]).all() and (values[eligible] >= 0).all()
                      for values in (age, disagreement, magnitude)), f'{sid} invalid gate inputs')
    return dict(sid=sid, component=str(entry['component_id']), evaluation=Path(entry['evaluation']),
                node_ids=motion['node_ids'], saved=saved, age=age, disagreement=disagreement, magnitude=magnitude,
                eligible=eligible, replay_max_m=float(np.max(replay_distance)), mean_replay_max_m=float(np.max(mean_distance)))


def quality(row, calibration):
    return np.maximum(row['age'] / calibration['positive_age_median_s'],
                      row['disagreement'] / calibration['positive_disagreement_median_m'])


def calibrate(rows):
    def eligible_values(name):
        return np.concatenate([row[name][row['eligible']] for row in rows])

    age, distance, magnitude = map(eligible_values, ('age', 'disagreement', 'magnitude'))
    fixed.require((age > 0).any() and (distance > 0).any() and len(magnitude), 'empty calibration')
    result = dict(opportunities=len(magnitude), positive_age_median_s=float(np.median(age[age > 0])),
                  positive_disagreement_median_m=float(np.median(distance[distance > 0])),
                  cap_m=float(np.quantile(magnitude, .25)), fixed_magnitude_m=float(np.median(magnitude)))
    scores = np.concatenate([quality(row, result)[row['eligible']] for row in rows])
    result['fixed_quality'] = float(np.median(scores))
    result['scenes'] = [row['sid'] for row in rows]
    return result


def low_risk(score, eligible, count):
    indexes = np.flatnonzero(eligible)
    ordered = indexes[np.argsort(score.flat[indexes], kind='stable')]
    chosen = np.zeros(eligible.size, bool)
    chosen[ordered[:count]] = True
    return chosen.reshape(eligible.shape)


def clip_positions(mean, residual, magnitude, cap):
    limited = residual.copy()
    for pair, q in np.argwhere(magnitude > cap):
        anchor = mean[pair, q]
        direction = geographic_log(residual[pair, q], anchor)
        limited[pair, q] = geographic_exp(direction * (cap / np.linalg.norm(direction)), anchor)
    return limited


def freeze(row, calibration, folder):
    saved = row['saved']
    mean, residual = saved['MEAN'], saved[c108.RESIDUAL]
    dual = saved['support'] == 2
    eligible, magnitude = row['eligible'], row['magnitude']
    quality_score = quality(row, calibration)
    budget = np.count_nonzero(eligible) // 2
    by_mag = low_risk(magnitude, eligible, budget)
    by_quality = low_risk(quality_score, eligible, budget)
    capped = clip_positions(mean, residual, np.where(eligible, magnitude, 0.), calibration['cap_m'])
    np.testing.assert_array_equal(c108.fallback(mean, residual, saved['support']), saved['DUAL_FALLBACK'])
    assigned = dict(MEAN=np.zeros(eligible.shape, bool), DUAL_FALLBACK=dual,
                    MAG=by_mag, MAG_CLIP=by_mag, QUALITY=by_quality, QUALITY_CLIP=by_quality,
                    FIXED_MAG=eligible & (magnitude <= calibration['fixed_magnitude_m']),
                    FIXED_QUALITY=eligible & (quality_score <= calibration['fixed_quality']))
    predictions = {}
    for arm, selected in assigned.items():
        pred = mean.copy()
        update = capped if arm.endswith('_CLIP') else residual
        pred[selected] = update[selected]
        fixed.require(pred[~selected].tobytes() == mean[~selected].tobytes(), f'{arm} fallback changed')
        if arm.endswith('_CLIP'):
            fixed.require(np.all(c105.distance(pred[selected], mean[selected]) <= calibration['cap_m'] + EPS_M),
                          f'{arm} exceeds frozen bound')
        predictions[arm] = pred
    np.savez_compressed(folder / f"{row['sid']}.npz", pairs=saved['pairs'], query=saved['query'],
                        support=saved['support'], eligible=eligible, quality=quality_score, **predictions)
    return dict(scene=row['sid'], component=row['component'], opportunities=int(eligible.sum()), budget=int(budget),
                gate_intersection=int((by_mag & by_quality).sum()), fixed_mag=int(assigned['FIXED_MAG'].sum()),
                fixed_quality=int(assigned['FIXED_QUALITY'].sum()), replay_max_m=row['replay_max_m'],
                mean_replay_max_m=row['mean_replay_max_m'])


def observe(row, frozen):
    saved = row['saved']
    registry = c105.registry(row['evaluation'])
    np.testing.assert_array_equal(registry['node_ids'], row['node_ids'])
    identities = registry['identities'][saved['pairs']]
    np.testing.assert_array_equal(identities[:, 0], identities[:, 1])
    label_row = dict(evaluation=row['evaluation'], pairs=saved['pairs'], identities=registry['identities'],
                     scene=dict(node_ids=row['node_ids']))
    truth, valid = c105.labels(label_row, saved['query'])
    fixed.require(len(truth) == 2 and len(set(identities[:, 0])) == 2, 'expected two distinct C108 truth pairs')
    common = valid.all(axis=0)
    y = truth[:, common]
    center = saved['MEAN'][0, 0]
    truth_vector = c105.delta_m(y[0], y[1], center)
    result = dict(valid=valid, support=saved['support'], eligible=frozen['eligible'], gap_m=c105.distance(y[0], y[1]),
                  age=row['age'], disagreement=row['disagreement'], magnitude=row['magnitude'], quality=frozen['quality'])
    for arm in ARMS:
        prediction = frozen[arm]
        error = c105.distance(prediction, truth)
        fixed.require(np.isfinite(error[valid]).all(), f'{arm} lost a valid position')
        moved = c105.distance(prediction, saved['MEAN']) > EPS_M
        first, second = prediction[:, common]
        result.update({f'{arm}/error': error, f'{arm}/action': moved, f'{arm}/pair_action': moved[:, common].sum(0),
                       f'{arm}/vector': np.linalg.norm(c105.delta_m(first, second, center) - truth_vector, axis=-1),
                       f'{arm}/margin': c105.distance(first, y[1]) + c105.distance(second, y[0])
                       - error[:, common].sum(0)})
    np.testing.assert_array_equal(result['MAG/action'], result['MAG_CLIP/action'])
    np.testing.assert_array_equal(result['QUALITY/action'], result['QUALITY_CLIP/action'])
    np.testing.assert_array_equal(result['MAG/action'].sum(), result['QUALITY/action'].sum())
    return result


def position_metrics(error, base, mask):
    return dict(error=fixed.statistics(error[mask]), mean_same=fixed.statistics(base[mask]),
                change=c108.changes(error[mask] - base[mask]))


def pair_metrics(public, arm, mask):
    vector, margin = (public[f'{arm}/{name}'][mask] for name in ('vector', 'margin'))
    mean_margin = public['MEAN/margin'][mask]
    return dict(vector=fixed.statistics(vector), mean_same=fixed.statistics(public['MEAN/vector'][mask]),
                correct=int((margin > EPS_M).sum()), mean_correct=int((mean_margin > EPS_M).sum()),
                ties=int((np.abs(margin) <= EPS_M).sum()),
                gained=int(((margin > EPS_M) & (mean_margin <= EPS_M)).sum()),
                lost=int(((margin <= EPS_M) & (mean_margin > EPS_M)).sum()))


def evaluate(rows, folder, prior):
    scenes = [observe(row, fixed.read_npz(folder / f"{row['sid']}.npz")) for row in rows]
    public = {key: np.concatenate([scene[key] for scene in scenes]) for key in scenes[0]}
    public['scene_index'] = np.repeat(np.arange(len(rows)), [len(scene['valid']) for scene in scenes])
    public['pair_scene'] = np.repeat(np.arange(len(rows)), [len(scene['gap_m']) for scene in scenes])
    for key in ('scene_index', 'pair_scene', 'support', 'valid', 'gap_m'):
        np.testing.assert_array_equal(public[key], prior[key])
    for arm in ('MEAN', 'DUAL_FALLBACK'):
        for name in ('error', 'vector', 'margin'):
            np.testing.assert_allclose(public[f'{arm}/{name}'], prior[f'{arm}/{name}'], atol=1e-8,
                                       rtol=1e-12, equal_nan=True)
    valid, support = public['valid'], public['support']
    dual = valid & (support == 2)
    base = public['MEAN/error']
    groups = np.asarray([row['component'] for row in rows])[public['scene_index']]
    largest = Counter(row['component'] for row in rows).most_common(1)[0][0]
    small = public['gap_m'] <= c105.SMALL_M
    mag_action = public['MAG/action']
    quality_action = public['QUALITY/action']
    pair_mag = public['MAG/pair_action'] > 0
    pair_quality = public['QUALITY/pair_action'] > 0
    metrics = {}
    for arm in ARMS:
        error, action, n_action = (public[f'{arm}/{name}'] for name in ('error', 'action', 'pair_action'))
        metrics[arm] = dict(
            all=position_metrics(error, base, valid),
            dual=position_metrics(error, base, dual),
            active=position_metrics(error, base, valid & action),
            magnitude_quality_union=position_metrics(error, base, valid & (mag_action | quality_action)),
            magnitude_quality_intersection=position_metrics(error, base, valid & mag_action & quality_action),
            coverage=dict(total_queries=int(valid.size), output_queries=int(error.size), reference_valid=int(valid.sum()),
                          action_queries=int(action.sum()), action_valid=int((valid & action).sum())),
            small_all=pair_metrics(public, arm, small),
            small_active=pair_metrics(public, arm, small & (n_action > 0)),
            small_by_n={str(n): pair_metrics(public, arm, small & (n_action == n)) for n in range(3)},
            small_union=pair_metrics(public, arm, small & (pair_mag | pair_quality)),
            small_intersection=pair_metrics(public, arm, small & pair_mag & pair_quality),
            scene={row['sid']: position_metrics(error, base, valid & (public['scene_index'] == i)[:, None])
                   for i, row in enumerate(rows)},
            component={group: position_metrics(error, base, valid & (groups == group)[:, None])
                       for group in sorted(set(groups))},
            leave_one_component_out={group: position_metrics(error, base, valid & (groups != group)[:, None])
                                     for group in sorted(set(groups))})
    return dict(metrics=metrics, largest_component=largest, scenes=len(rows), components=len(set(groups)),
                eligible_queries=int(public['eligible'].sum()), valid_queries=int(valid.sum())), public


def qualification(result):
    metrics = result['metrics']
    base = metrics['MEAN']
    checks = {}
    for arm, candidate in metrics.items():
        if arm in ('MEAN', 'DUAL_FALLBACK'):
            continue
        all_points, active, small = (candidate[key] for key in ('all', 'active', 'small_active'))
        checks[arm] = dict(
            coverage=all(candidate['coverage'][key] == base['coverage'][key]
                         for key in ('output_queries', 'reference_valid')),
            full_mae_and_p99=all_points['error']['mae'] < all_points['mean_same']['mae']
            and all_points['error']['p99'] <= all_points['mean_same']['p99'],
            active_p99=bool(active['error']['count'] and active['error']['p99'] <= active['mean_same']['p99']),
            near=bool(small['vector']['count'] and small['vector']['mae'] <= small['mean_same']['mae']
                      and small['correct'] >= small['mean_correct']),
            zero_km_harm=all_points['change']['harm_gt_1km'] == 0)
        if arm in ('QUALITY', 'QUALITY_CLIP'):
            control = metrics['MAG' + arm[len('QUALITY'):]]
            shared_near = candidate['small_union']
            checks[arm]['quality_increment'] = bool(
                all_points['change']['harm_gt_1km'] < control['all']['change']['harm_gt_1km']
                and all_points['error']['mae'] <= control['all']['error']['mae']
                and shared_near['vector']['count']
                and shared_near['vector']['mae'] <= control['small_union']['vector']['mae']
                and shared_near['correct'] >= control['small_union']['correct'])
    return checks


def run(args, start):
    origin = args.runs / C108
    source = held.read(origin / 'run.json')
    fixed.require(source['status'] == 'COMPLETED' and not source['test_opened'], 'completed C108 required')
    roster = held.read(origin / 'freeze.json')
    index = held.read(args.runs / held.PARENTS['full_data'] / 'private/index.json')['V_select']
    scene_ids = dict(replay=roster['C105_roster'], new=[row['scene'] for row in roster['new_roster']])
    groups = [{str(index[sid]['component_id']) for sid in names} for names in scene_ids.values()]
    fixed.require([len(names) for names in scene_ids.values()] == [48, 57]
                  and list(map(len, groups)) == [23, 10] and not groups[0] & groups[1],
                  'calibration and evaluation roster changed')
    head = c108.load_head(args.runs / c108.C105)
    calibration_rows = [cached_scene(origin, 'replay', sid, index[sid], head) for sid in scene_ids['replay']]
    calibration = calibrate(calibration_rows)
    fixed.write_json(args.output / 'calibration.json', calibration)
    evaluation_rows = [cached_scene(origin, 'new', sid, index[sid], head) for sid in scene_ids['new']]
    folder = args.output / 'private'
    folder.mkdir(mode=0o700)
    freeze_rows = [freeze(row, calibration, folder) for row in evaluation_rows]
    fixed.write_json(args.output / 'freeze.json', dict(rows=freeze_rows, reference_targets_read=0,
                     coefficients=44, raw_archive_members=0, calibration='C108 replay V48 observation only'))
    prior = fixed.read_npz(origin / 'new-public-errors.npz')
    result, public = evaluate(evaluation_rows, folder, prior)
    result['qualification'] = qualification(result)
    np.savez_compressed(args.output / 'public-errors.npz', **public)
    fixed.write_json(args.output / 'summary.json', result)
    fixed.write_json(args.output / 'run.json', dict(status='COMPLETED', elapsed_seconds=time.monotonic()-start,
                     fit_operations=0, optimizer_updates=0, model_forward_calls=0, test_opened=False,
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
