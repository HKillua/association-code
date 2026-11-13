"""C127: audit frozen partition alternatives as suppliers to unsupported outputs."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from tools.analysis import association_partition_disagreement as c98
from tools.analysis import association_registered_holdout_fusion as c107
from tools.analysis import association_registered_supplier_exclusion as c104
from tools.analysis import fusion_supplier_tail_lineage as c124
from tools.analysis.offline_association_tail import read_motion, summary_state, write
from tools.analysis.same_source_swap_pilot import read


ARMS = ('KEEP', 'POOL', 'ADMITTED', 'DUAL')
STAGES = ('scene', 'block', 'pool', 'first', 'admitted', 'positive', 'dual')


def candidate_lists(anchor, block, admitted, dual):
    if block is None:
        return [], {stage: set() for stage in STAGES[2:]}
    nodes = np.asarray(block['nodes'])
    ai, = np.flatnonzero(nodes == anchor)
    rows, sets = [], {stage: set() for stage in STAGES[2:]}
    for index, item in enumerate(block['candidates']):
        membership = np.asarray(item['partition'])
        members = nodes[membership == membership[ai]]
        others = members[members != anchor]
        sets['pool'].update(others.tolist())
        if members[0] != anchor:
            continue
        sets['first'].update(others.tolist())
        if index in admitted:
            sets['admitted'].update(others.tolist())
            if item['mixed_gain'] > c98.EPS:
                sets['positive'].update(others.tolist())
                sets['dual'].update(others[dual[anchor, others]].tolist())
        rows.append((index, item['mixed_gain'], others))
    return sorted(rows, key=lambda r: (-r[1], r[0])), sets


def frozen_scene(motion, saved, scores, blocks, choices, thresholds):
    current, support = saved['current'], motion['support']
    groups = c98.group_nodes(current)
    pairs = np.column_stack(np.triu_indices(len(current), 1))
    np.testing.assert_array_equal(scores['pairs'], pairs)
    old = scores['group_056']
    admitted = choices['admitted']
    winner = choices['selections']['REG_MIX']
    if len(old):
        np.testing.assert_array_equal(c98.materialize(old, blocks, winner), current)
    assert len(blocks) == len(admitted) == len(winner)
    dual = np.zeros((len(old), len(old)), bool)
    left, right = pairs.T
    dual[left, right] = dual[right, left] = ((scores['score_056'] > thresholds['056']) &
                                              (scores['score_M_BCE'] > thresholds['M_BCE']))
    block_by_node = {node: i for i, block in enumerate(blocks) for node in block['nodes']}
    take = np.full((len(ARMS)-1, len(groups), support.shape[1]), -1, dtype=int)
    reach = []
    for gi, group in enumerate(groups):
        anchor = group[0]
        which = block_by_node.get(anchor)
        block = blocks[which] if which is not None else None
        allowed = set(admitted[which]['REG_MIX']) if block is not None else set()
        candidates, sets = candidate_lists(anchor, block, allowed, dual)
        outside = set(range(len(old))) - set(group.tolist())
        stage_sets = dict(scene=outside, block=set(block['nodes']) & outside if block else set(),
                          **{key: nodes & outside for key, nodes in sets.items()})
        reach.append({key: np.asarray(sorted(nodes), int) for key, nodes in stage_sets.items()})
        if not candidates:
            continue
        for q in np.flatnonzero(~support[group].any(axis=0)):
            for index, gain, members in candidates:
                eligible = members[support[members, q]]
                if not len(eligible):
                    continue
                eligible = eligible[np.argsort(motion['nearest_seconds'][eligible, q], kind='stable')]
                if take[0, gi, q] < 0:
                    take[0, gi, q] = eligible[0]
                if index in allowed and take[1, gi, q] < 0:
                    take[1, gi, q] = eligible[0]
                if index in allowed and gain > c98.EPS and take[2, gi, q] < 0:
                    qualified = eligible[dual[anchor, eligible]]
                    if len(qualified):
                        take[2, gi, q] = qualified[0]
                if np.all(take[:, gi, q] >= 0):
                    break
    return groups, take, reach


def scene_evaluation(sid, component, motion, saved, truth, frozen, threshold):
    groups, selected, reach = frozen
    ids = truth['identities']
    np.testing.assert_array_equal(saved['node_ids'], truth['node_ids'])
    np.testing.assert_array_equal(saved['query'], truth['query_seconds'])
    support, node_states = motion['support'], motion['states'][..., :2]
    target_states = truth['target_states'][..., :2]
    valid = truth['target_valid'][..., :2].all(-1)
    first = {}
    current_anchor = np.empty(len(ids), dtype=int)
    for gi, group in enumerate(groups):
        first.setdefault(ids[group[0]], gi)
        current_anchor[group] = group[0]
    counts, details, matrices = Counter(), [], []
    per_arm = {name: Counter() for name in ARMS[1:]}
    for ti, identity in enumerate(truth['target_ids']):
        if identity not in first:
            continue
        gi = first[identity]
        q = np.flatnonzero(valid[ti])
        y = target_states[ti, q]
        baseline = c107._distance(saved['KEEP'][gi, q, :2], y)
        counts['tail'] += int((baseline >= threshold).sum())
        values = np.repeat(baseline[None, :], len(ARMS), axis=0)
        chosen = selected[:, gi, q]
        for ai, (name, nodes) in enumerate(zip(ARMS[1:], chosen, strict=True), 1):
            edited = nodes >= 0
            values[ai, edited] = c107._distance(node_states[nodes[edited], q[edited]], y[edited])
            error = values[ai]
            wrong = edited.copy()
            wrong[edited] = ids[nodes[edited]] != identity
            per_arm[name].update(actions=int(edited.sum()), wrong=int(wrong.sum()),
                true=int((edited & ~wrong).sum()), tail_actions=int((edited & (baseline >= threshold)).sum()),
                tail_wrong=int((wrong & (baseline >= threshold)).sum()),
                tail_exits=int(((baseline >= threshold) & (error < threshold)).sum()),
                tail_entries=int(((baseline < threshold) & (error >= threshold)).sum()),
                non_tail_harm250=int(((baseline < threshold) & (error-baseline > 250)).sum()))
        matrices.append(values.T)
        group_no_support = ~support[groups[gi]][:, q].any(0)
        focus = (baseline >= threshold) & group_no_support
        counts['tail_unsupported'] += int(focus.sum())
        for offset in np.flatnonzero(focus):
            j = q[offset]
            true = {}
            wrong = {}
            for stage in STAGES:
                active = reach[gi][stage][support[reach[gi][stage], j]]
                true[stage] = active[ids[active] == identity]
                wrong[stage] = int(len(active)-len(true[stage]))
            if not len(true['scene']):
                continue
            counts['focus'] += 1
            duplicate_nodes = true['scene'][ids[current_anchor[true['scene']]] == identity]
            counts['focus_has_duplicate_group_supplier'] += int(len(duplicate_nodes) > 0)
            available = [stage for stage in STAGES if len(true[stage])]
            assert available == list(STAGES[:len(available)])
            counts[f'last_{available[-1]}'] += 1
            oracle = {stage: (float(c107._distance(node_states[nodes, j], y[offset]).min()) if len(nodes) else None)
                      for stage, nodes in true.items()}
            details.append(dict(scene=sid, target=ti, q=int(j), deepest=available[-1],
                duplicate_group_true_suppliers=len(duplicate_nodes),
                true={stage: len(nodes) for stage, nodes in true.items()}, wrong=wrong, oracle_m=oracle,
                error_m=values[:, offset].tolist(), chosen={name: ('none' if node < 0 else 'true' if ids[node] == identity else 'wrong')
                    for name, node in zip(ARMS[1:], chosen[:, offset], strict=True)}))
    matrix = np.concatenate(matrices) if matrices else np.empty((0, len(ARMS)))
    return matrix, details, dict(scene=sid, component=component, counts=dict(counts),
                                 arms={name: dict(row) for name, row in per_arm.items()})


def run(args):
    runs, output = args.runs, args.output
    data = read(runs/c98.DATA/'private/index.json')['T_core']
    manifest = read(args.manifest)
    scenes = read(runs/c98.PREP/'frozen-split.json')['held']
    assert len(scenes) == 177 and set(scenes) <= set(manifest['splits']['T_core'])
    pos, regs, raw_scores, c121 = [runs/name for name in (c104.C98, c107.C104, c98.C95, c124.C121)]
    c124_run = runs/'LEVEL1-OFFLINE-SUPPLIER-TAIL-LINEAGE-20261113-attempt-1'
    assert all(read(p/'run.json')['status'] == 'COMPLETED' for p in (pos, regs, raw_scores, c121, c124_run))
    thresholds = read(raw_scores/'summary.json')['fixed_thresholds']
    scene_data = {}
    private = output/'private'
    private.mkdir()
    for sid in scenes:
        motion = read_motion(data[sid])
        blocks = read(pos/'private/held'/f'{sid}.json.gz')
        choices = read(regs/'private/held'/f'{sid}-choices.json')
        with np.load(c121/'private'/f'{sid}.npz', allow_pickle=False) as saved, \
             np.load(raw_scores/'private/held'/f'{sid}.npz', allow_pickle=False) as scores:
            scene_data[sid] = frozen_scene(motion, saved, scores, blocks, choices, thresholds)
            np.save(private/f'{sid}-selected.npy', scene_data[sid][1], allow_pickle=False)
    write(output/'freeze.json', dict(scenes=len(scenes), scoring_changed=False, read_raw_members=0,
                                      predictions_before_evaluation=True))
    prior = read(c124_run/'summary.json')
    threshold = prior['population']['KEEP']['p99']
    matrices, focus, scene_rows = [], [], []
    for sid in scenes:
        with np.load(c121/'private'/f'{sid}.npz', allow_pickle=False) as saved, \
             np.load(data[sid]['evaluation'], allow_pickle=False) as truth:
            groups, _, reach = scene_data[sid]
            frozen = groups, np.load(private/f'{sid}-selected.npy', allow_pickle=False), reach
            errors, detail, row = scene_evaluation(sid, manifest['components'][sid], read_motion(data[sid]),
                                                     saved, truth, frozen, threshold)
        matrices.append(errors)
        focus.extend(detail)
        scene_rows.append(row)
    all_errors = np.concatenate(matrices)
    metrics = {name: c107.attribute_stats(all_errors[:, ai]) for ai, name in enumerate(ARMS)}
    for k, v in metrics['KEEP'].items():
        np.testing.assert_allclose(v, prior['population']['KEEP'][k], atol=2e-8, rtol=0.)
    expected = {(r['scene'], r['target'], r['q']) for r in read(c124_run/'tail-points.json')
                if not r['group_active'] and r['other_same_active']}
    assert len(focus) == len(expected) == 782 and expected == {(r['scene'], r['target'], r['q']) for r in focus}
    total, action = Counter(), {name: Counter() for name in ARMS[1:]}
    component = defaultdict(lambda: {name: Counter() for name in ARMS[1:]})
    for row in scene_rows:
        total.update(row['counts'])
        for name, values in row['arms'].items():
            action[name].update(values)
            component[row['component']][name].update(values)
    stage_counts = {stage: dict(points=sum(bool(r['true'][stage]) for r in focus),
        true_slots=sum(r['true'][stage] for r in focus), wrong_slots=sum(r['wrong'][stage] for r in focus),
        oracle_below_tail=sum(r['oracle_m'][stage] is not None and r['oracle_m'][stage] < threshold for r in focus))
        for stage in STAGES}
    write(output/'summary.json', dict(metrics=metrics, counts=dict(total), thresholds=thresholds,
        stage_counts=stage_counts, actions=action, components=component,
        focus_chosen={name: Counter(r['chosen'][name] for r in focus) for name in ARMS[1:]}))
    write(output/'focus.json', focus)
    write(output/'scenes.json', scene_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
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
