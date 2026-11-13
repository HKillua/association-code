"""C130: compare fixed supplier activation rankings at an equal scene budget."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from tools.analysis import fusion_tail_candidate_recovery as c127
from tools.analysis.offline_association_tail import CONVERSION, read_motion, summary_state, write


KEEP, POOL, J, IDENTITY, POSITION, JOINT = ARMS = ('KEEP', 'POOL', 'J', 'IDENTITY', 'POSITION', 'JOINT')
CONTROLLED = ARMS[2:]
C127 = 'LEVEL1-OFFLINE-TAIL-CANDIDATE-RECOVERY-20261113-attempt-1'
STATS = c127.c107.attribute_stats
DISTANCE = c127.c107._distance


def fractional_rank(values):
    values = np.asarray(values, float)
    if not len(values):
        return values
    values = np.where(np.isnan(values), -np.inf, values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    average = np.cumsum(counts) - (counts - 1) / 2
    return average[inverse] / len(values)


def visible_candidates(motion, saved, scores, blocks, choices, thresholds, original):
    groups, selected, _ = c127.frozen_scene(motion, saved, scores, blocks, choices, thresholds)
    np.testing.assert_array_equal(selected[0], original)
    group_index = np.empty(len(saved['current']), int)
    for gi, nodes in enumerate(groups):
        group_index[nodes] = gi
    support, states = motion['support'], motion['states'][..., :2]
    q_sec, age = motion['query_seconds'], motion['nearest_seconds']
    block_by_node = {node: block for block in blocks for node in block['nodes']}
    left, right = scores['pairs'].T
    strength = np.zeros((len(support), len(support)), dtype=float)
    # A weak association score in either frozen head limits identity evidence.
    if len(left):
        margin_a = scores['score_056'] - thresholds['056']
        margin_b = scores['score_M_BCE'] - thresholds['M_BCE']
        identity = np.minimum(fractional_rank(abs(margin_a))*np.sign(margin_a),
                              fractional_rank(abs(margin_b))*np.sign(margin_b))
        strength[left, right] = strength[right, left] = identity
    receiver_options = defaultdict(list)
    rows = []
    for gi, nodes in enumerate(groups):
        anchor = nodes[0]
        if anchor not in block_by_node:
            continue
        block = block_by_node[anchor]
        candidates, _ = c127.candidate_lists(anchor, block, set(), strength > 0)
        group_support = support[nodes].any(0)
        for q in np.flatnonzero(~group_support):
            for _, gain, members in candidates:
                for donor in members[support[members, q]]:
                    receiver_options[int(donor), int(q)].append((gi, float(strength[anchor, donor])))
            donor = original[gi, q]
            if donor < 0:
                continue
            assert group_index[donor] != gi and support[donor, q]
            same_moment = np.flatnonzero(group_support & support[donor])
            gap = float('nan')
            delay = float('nan')
            if len(same_moment):
                t = same_moment[np.argmin(abs(q_sec[same_moment]-q_sec[q]))]
                active = nodes[support[nodes, t]]
                gap = float(DISTANCE(states[active, t], states[donor, t]).min())
                delay = float(abs(q_sec[t]-q_sec[q]))
            gain = next(g for _, g, members in candidates if support[members, q].any())
            rows.append(dict(group=int(gi), q=int(q), donor=int(donor), donor_group=int(group_index[donor]),
                gain=float(gain), identity=float(strength[anchor, donor]),
                shared=bool(len(same_moment)), gap=gap, delay=delay, age=float(age[donor, q]),
                fit=float(motion['fit_rmse_m'][donor]), same_source=motion['sources'][anchor] == motion['sources'][donor]))
    assert len(rows) == np.count_nonzero(original >= 0)
    for row in rows:
        unique = dict(receiver_options[row['donor'], row['q']])
        best = max(unique.values())
        row['mutual'] = row['identity'] == best and list(unique.values()).count(best) == 1
    return groups, rows


def choose(rows):
    if not rows:
        return dict.fromkeys(CONTROLLED, [])
    budget = (len(rows)+1)//2
    gain = np.array([r['gain'] for r in rows])
    identity = fractional_rank([r['identity'] for r in rows])
    mutual = np.array([r['mutual'] for r in rows])
    common = np.array([r['shared'] for r in rows])
    position = np.empty(len(rows))
    for flag, fields in ((True, ('gap', 'delay', 'age')), (False, ('fit', 'age'))):
        use = np.flatnonzero(common == flag)
        if len(use):
            position[use] = sum(fractional_rank([-rows[i][key] for i in use]) for key in fields)/len(fields)
    # Stable order preserves group and query as the final tie-breaks.
    ordering = {
        J: np.argsort(-gain, kind='stable'),
        IDENTITY: np.lexsort((-identity, -mutual.astype(int))),
        POSITION: np.lexsort((-position, -common.astype(int))),
    }
    i_rank = fractional_rank(-np.argsort(ordering[IDENTITY], kind='stable'))
    p_rank = fractional_rank(-np.argsort(ordering[POSITION], kind='stable'))
    ordering[JOINT] = np.argsort(-np.minimum(i_rank, p_rank), kind='stable')
    return {arm: order[:budget].tolist() for arm, order in ordering.items()}


def score_scene(sid, component, motion, saved, truth, groups, rows, selected, threshold, focus, protect_trajectory):
    ids, target_ids = truth['identities'], truth['target_ids']
    np.testing.assert_array_equal(saved['node_ids'], truth['node_ids'])
    np.testing.assert_array_equal(saved['query'], truth['query_seconds'])
    first = {}
    for gi, group in enumerate(groups):
        first.setdefault(ids[group[0]], gi)
    baseline = saved['KEEP']
    data = np.repeat(baseline[None], len(ARMS), axis=0)
    enabled = np.zeros(data.shape[:3], bool)
    chosen = [[], list(range(len(rows))), *(selected[arm] for arm in CONTROLLED)]
    for ai, index in enumerate(chosen):
        for i in index:
            row = rows[i]
            gi, q, donor = (row[key] for key in ('group', 'q', 'donor'))
            assert not motion['support'][groups[gi], q].any() and motion['support'][donor, q]
            assert not enabled[ai, gi, q]
            data[ai, gi, q, :2] = motion['states'][donor, q, :2]
            enabled[ai, gi, q] = True
        np.testing.assert_array_equal(data[ai, ..., 2:], baseline[..., 2:])
        np.testing.assert_array_equal(data[ai, ~enabled[ai]], baseline[~enabled[ai]])
        if index:
            g, q = zip(*[(rows[i]['donor_group'], rows[i]['q']) for i in index], strict=True)
            np.testing.assert_array_equal(data[ai, g, q], baseline[g, q])
            if protect_trajectory:
                donor_groups = sorted(set(g))
                np.testing.assert_array_equal(data[ai, donor_groups], baseline[donor_groups])
    gi = np.array([first.get(identity, -1) for identity in target_ids])
    keep = gi >= 0
    evaluate = truth['target_valid'] & keep[:, None, None]
    states = np.transpose(data[:, gi[keep]], (1, 2, 0, 3))
    states[..., 2] *= CONVERSION
    y = truth['target_states'][keep]
    position_valid = evaluate[keep, :, :2].all(-1)
    errors = DISTANCE(states[..., :2], y[:, :, None, :2])[position_valid]
    speed = abs(states[..., 2] - y[:, :, None, 2])[evaluate[keep, :, 2]]
    heading = abs((states[..., 3] - y[:, :, None, 3] + 180.) % 360. - 180.)[evaluate[keep, :, 3]]
    np.testing.assert_array_equal(speed, np.repeat(speed[:, :1], len(ARMS), axis=1))
    np.testing.assert_array_equal(heading, np.repeat(heading[:, :1], len(ARMS), axis=1))
    back = {g: i for i, g in enumerate(gi) if g >= 0}
    events, arm_stats = [], {}
    for ai, (arm, index) in enumerate(zip(ARMS, chosen, strict=True)):
        stats = Counter(planned=len(index))
        stats['position_changed'] = int(np.any(data[ai, ..., :2] != baseline[..., :2], axis=-1).sum())
        stats['tail_exits'] = int(((errors[:, 0] >= threshold) & (errors[:, ai] < threshold)).sum())
        stats['tail_entries'] = int(((errors[:, 0] < threshold) & (errors[:, ai] >= threshold)).sum())
        for i in index:
            row = rows[i]
            group, q = row['group'], row['q']
            target = back.get(group)
            if target is None or not evaluate[target, q, :2].all():
                continue
            yq = truth['target_states'][target, q, :2]
            before = float(DISTANCE(baseline[group, q, :2], yq))
            after = float(DISTANCE(data[ai, group, q, :2], yq))
            true = bool(ids[row['donor']] == target_ids[target])
            tail, focal = before >= threshold, (sid, target, q) in focus
            changed = bool(np.any(data[ai, group, q, :2] != baseline[group, q, :2]))
            stats.update(evaluated=1, true=int(true), wrong=int(not true), true_tail_exit=int(true and tail and after < threshold),
                         wrong_tail_exit=int(not true and tail and after < threshold),
                         evaluated_position_changed=int(changed),
                         non_tail_harm250=int(not tail and after-before > 250),
                         true_non_tail_harm250=int(true and not tail and after-before > 250),
                         focus_true=int(focal and true), focus_wrong=int(focal and not true),
                         focus_true_exit=int(focal and true and after < threshold))
            events.append(dict(arm=arm, target=target, q=q, true=true, tail=tail, focus=focal,
                               before_m=before, after_m=after, shared=row['shared'], mutual=row['mutual'],
                               same_source=bool(row['same_source']), changed=changed))
        arm_stats[arm] = dict(stats)
    return errors, events, dict(scene=sid, component=component, candidates=len(rows),
        has_shared=sum(r['shared'] for r in rows), mutual=sum(r['mutual'] for r in rows),
        outputs=len(groups), targets=len(target_ids), assigned=int(keep.sum()),
        available=truth['target_valid'].sum(axis=(0, 1)).tolist(), evaluated=evaluate.sum(axis=(0, 1)).tolist(),
        arms=arm_stats)


def run(args):
    runs, out = args.runs, args.output
    reader, c98 = c127.read, c127.c98
    source = runs/C127
    prior_motion = runs/c127.c124.C121
    assert reader(source/'run.json')['status'] == 'COMPLETED'
    original, registration, scores = (runs/p for p in (c127.c104.C98, c127.c107.C104, c98.C95))
    manifest = reader(args.manifest)
    index = reader(runs/c98.DATA/'private/index.json')['T_core']
    scene_ids = reader(runs/c98.PREP/'frozen-split.json')['held']
    assert len(scene_ids) == 177 and set(scene_ids) <= set(manifest['splits']['T_core'])
    thresholds = reader(scores/'summary.json')['fixed_thresholds']
    private = out/'private'
    private.mkdir()
    for sid in scene_ids:
        motion = read_motion(index[sid])
        with np.load(prior_motion/'private'/f'{sid}.npz', allow_pickle=False) as saved, \
             np.load(scores/'private/held'/f'{sid}.npz', allow_pickle=False) as score:
            _, rows = visible_candidates(motion, saved, score, reader(original/'private/held'/f'{sid}.json.gz'),
                reader(registration/'private/held'/f'{sid}-choices.json'), thresholds,
                np.load(source/'private'/f'{sid}-selected.npy', allow_pickle=False)[0])
        unrestricted_count = len(rows)
        if args.protect_donor_trajectory:
            donor_groups = {r['donor_group'] for r in rows}
            rows = [r for r in rows if r['group'] not in donor_groups]
        # Serialize only the fields used in evaluation. NaN features stay out of JSON.
        frozen_rows = [{key: bool(r[key]) if key in ('shared', 'mutual', 'same_source') else r[key]
                        for key in ('group', 'q', 'donor', 'donor_group', 'shared', 'mutual', 'same_source')} for r in rows]
        write(private/f'{sid}.json', dict(rows=frozen_rows, selected=choose(rows), unrestricted_count=unrestricted_count))
    write(out/'freeze.json', dict(scenes=len(scene_ids), budget='ceil(per-scene visible opportunities / 2)',
        equal_arms=list(CONTROLLED), selection_uses_evaluation=False, external_data=False, raw_reads=0,
        protect_donor_trajectory=args.protect_donor_trajectory))
    prior = reader(source/'summary.json')
    threshold = prior['metrics'][KEEP]['p99']
    focus = {(r['scene'], r['target'], r['q']) for r in reader(source/'focus.json')}
    metrics, event_rows, scene_rows = [], [], []
    components = defaultdict(lambda: defaultdict(Counter))
    totals = defaultdict(Counter)
    for sid in scene_ids:
        frozen = reader(private/f'{sid}.json')
        comp = manifest['components'][sid]
        with np.load(prior_motion/'private'/f'{sid}.npz', allow_pickle=False) as saved, \
             np.load(index[sid]['evaluation'], allow_pickle=False) as truth:
            error, events, report = score_scene(sid, comp, read_motion(index[sid]), saved, truth,
                c127.c98.group_nodes(saved['current']), frozen['rows'], frozen['selected'], threshold, focus,
                args.protect_donor_trajectory)
        report['unrestricted_count'] = frozen['unrestricted_count']
        assert len({report['arms'][a]['planned'] for a in CONTROLLED}) == 1
        for arm, values in report['arms'].items():
            totals[arm].update(values)
            components[comp][arm].update(values)
        metrics.append(error)
        event_rows.extend(dict(scene=sid, **event) for event in events)
        scene_rows.append(report)
    all_errors = np.concatenate(metrics)
    population = {arm: STATS(all_errors[:, i]) for i, arm in enumerate(ARMS)}
    for arm in ((KEEP,) if args.protect_donor_trajectory else (KEEP, POOL)):
        for key, val in prior['metrics'][arm].items():
            np.testing.assert_allclose(population[arm][key], val, rtol=0., atol=2e-8)
    if not args.protect_donor_trajectory:
        for key, val in prior['actions'][POOL].items():
            if key in totals[POOL]:
                assert totals[POOL][key] == val
    write(out/'summary.json', dict(scenes=len(scene_ids), components=components, metrics=population,
          totals=totals, opportunity_slots=sum(r['candidates'] for r in scene_rows),
          unrestricted_slots=sum(r['unrestricted_count'] for r in scene_rows),
          nonzero_budget_scenes=sum(r['candidates'] > 0 for r in scene_rows), focus=len(focus),
          baseline_threshold_m=threshold, coverage=dict(zip(('lat', 'lon', 'speed', 'heading'),
              np.sum([r['evaluated'] for r in scene_rows], axis=0).tolist(), strict=True))))
    write(out/'scenes.json', scene_rows)
    write(out/'events.json', event_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    parser.add_argument('--protect-donor-trajectory', action='store_true')
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args)
    except Exception as error:
        summary_state(args.output, start, status='FAILED', error=repr(error))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, model_forwards=0, test_opened=False)


if __name__ == '__main__':
    main()
