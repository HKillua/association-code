"""Frozen C98 partition disagreements and label-blind support preservation."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time

import numpy as np

from association.evaluation.association import _aggregate, _counts
from tools.analysis.association_frozen_pr_audit import check_counts
from tools.analysis.offline_association_tail import coverage, read_identity, read_motion, summary_state, write
from tools.analysis.static_partition_pilot import frozen


ARMS = ('WS', 'M_BCE', 'LOCAL_OLD', 'LOCAL_MIX', 'SOURCE', 'SUPPLIER', 'AGREE')
LOCAL_ARMS = ARMS[2:]
SCOPES = ('itsa', 'mstsa')
DATA = 'LEVEL1-FUSION-PEER-FULL-DATA-20261030-151-attempt-1'
C95 = 'LEVEL1-OFFLINE-NEGATIVE-COVERAGE-20261112-attempt-1'
PREP = 'LEVEL1-OFFLINE-ASSOCIATION-TAIL-PREP-20261112-attempt-1'
EPS = 1e-10
MAX_CELLS = 6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def group_nodes(values):
    groups = defaultdict(list)
    for node, label in enumerate(values):
        groups[int(label)].append(node)
    return [np.asarray(nodes) for nodes in groups.values()]


def join_blocks(old, mixed):
    parent = np.arange(len(old))

    def root(node):
        while node != parent[node]:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for partition in (old, mixed):
        for members in group_nodes(partition):
            first = root(members[0])
            for node in members[1:]:
                parent[root(node)] = first
    return group_nodes([root(i) for i in range(len(old))])


def cell_partitions(allowed):
    """Enumerate canonical set partitions respecting all allowed pairs."""
    groups = []
    labels = []

    def visit(cell):
        if cell == len(allowed):
            yield tuple(labels)
            return
        for label, members in enumerate(groups):
            if all(allowed[cell, other] for other in members):
                members.append(cell)
                labels.append(label)
                yield from visit(cell + 1)
                labels.pop()
                members.pop()
        labels.append(len(groups))
        groups.append([cell])
        yield from visit(cell + 1)
        groups.pop()
        labels.pop()

    yield from visit(0)


def options(old, mixed):
    cells = {}
    mapping = np.asarray([cells.setdefault((int(a), int(b)), len(cells))
                          for a, b in zip(old, mixed, strict=True)])
    labels = np.asarray(tuple(cells))
    first = tuple(frozen.canonical_membership(labels[:, 0], len(cells)))
    second = tuple(frozen.canonical_membership(labels[:, 1], len(cells)))
    if len(cells) <= MAX_CELLS:
        same = (labels[:, None] == labels[None, :]).any(axis=-1)
        choices = [first, *(value for value in cell_partitions(same) if value != first)]
        require(second in choices, 'mixed option missing')
    else:
        choices = [first, second]
    return [np.asarray(label)[mapping] for label in choices], len(cells)


def source_slots(membership, sources, support):
    flags = np.zeros((len(sources), len(np.unique(sources)), support.shape[1]), dtype=bool)
    for nodes in group_nodes(membership):
        for j, source in enumerate(np.unique(sources)):
            flags[nodes, j] = support[nodes[sources[nodes] == source]].any(axis=0)
    return flags


def block_catalog(nodes, old, mixed, sources, support, margins):
    labels, cells = options(old[nodes], mixed[nodes])
    left, right = np.triu_indices(len(nodes), 1)
    was = old[nodes[left]] == old[nodes[right]]
    anchor_nodes = tuple(group[0] for group in group_nodes(old[nodes]))
    local_support = support[nodes]
    local_sources = sources[nodes]
    before_slots = source_slots(old[nodes], local_sources, local_support)
    unequal_support = (local_support[left] != local_support[right]).any(axis=1)
    old_margins, mix_margins = (matrix[nodes[left], nodes[right]] for matrix in margins)
    joint = (local_support[left] & local_support[right]).any(axis=1)
    cross = local_sources[left] != local_sources[right]
    catalog = []
    for partition in labels:
        together = partition[left] == partition[right]
        added, removed = together & ~was, was & ~together
        gain = lambda margin: float(margin[added].sum() - margin[removed].sum())
        anchor_exact = tuple(group[0] for group in group_nodes(partition)) == anchor_nodes
        source_loss = int((before_slots & ~source_slots(partition, local_sources, local_support)).sum())
        catalog.append(dict(
            partition=partition.tolist(), old_gain=gain(old_margins), mixed_gain=gain(mix_margins),
            anchor_exact=bool(anchor_exact), source_lost=source_loss,
            supplier_unsafe_edges=int((removed & unequal_support).sum()),
            added=int(added.sum()), removed=int(removed.sum()),
            added_cross_cosupported=int((added & cross & joint).sum()),
            removed_cross_cosupported=int((removed & cross & joint).sum()),
            added_dual_positive=int((added & (old_margins > 0) & (mix_margins > 0)).sum()),
            removed_dual_negative=int((removed & (old_margins < 0) & (mix_margins < 0)).sum())))
    return dict(nodes=nodes.tolist(), cells=cells, candidates=catalog)


def choose(block, arm):
    def eligible(row):
        if arm in ('LOCAL_OLD', 'LOCAL_MIX'):
            return True
        if not row['anchor_exact']:
            return False
        if arm == 'SOURCE':
            return row['source_lost'] == 0
        return row['supplier_unsafe_edges'] == 0 and (arm != 'AGREE' or row['old_gain'] >= -EPS)

    field = 'old_gain' if arm == 'LOCAL_OLD' else 'mixed_gain'
    winner, gain = 0, 0.
    for i, row in enumerate(block['candidates'][1:], start=1):
        if eligible(row) and row[field] > gain + EPS:
            winner, gain = i, row[field]
    return winner


def materialize(base, blocks, choices):
    result = base.copy()
    offset = int(base.max()) + 1
    for block, selected in zip(blocks, choices, strict=True):
        if selected:
            partition = np.asarray(block['candidates'][selected]['partition'])
            result[block['nodes']] = offset + partition
            offset += int(partition.max()) + 1
    return frozen.canonical_membership(result, len(base))


def freeze_scene(index, input_path, thresholds):
    motion = read_motion(index)
    support, sources = motion['support'], motion['sources']
    count = len(sources)
    left, right = np.triu_indices(count, 1)
    with np.load(input_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['pairs'], np.column_stack((left, right)))
        old, mixed = (saved[f'group_{arm}'] for arm in ('056', 'M_BCE'))
        margins = []
        for arm in ('056', 'M_BCE'):
            matrix = np.zeros((count, count))
            matrix[left, right] = saved[f'score_{arm}'] - thresholds[arm]
            margins.append(matrix + matrix.T)
    for partition in (old, mixed):
        np.testing.assert_array_equal(partition, frozen.canonical_membership(partition, count))
    blocks = []
    for nodes in join_blocks(old, mixed):
        a = frozen.canonical_membership(old[nodes], len(nodes))
        b = frozen.canonical_membership(mixed[nodes], len(nodes))
        if not np.array_equal(a, b):
            blocks.append(block_catalog(nodes, old, mixed, sources, support, margins))
    states = {'WS': old, 'M_BCE': mixed}
    for arm in LOCAL_ARMS:
        selected = [choose(block, arm) for block in blocks]
        states[arm] = materialize(old, blocks, selected)
        for block, option in zip(blocks, selected, strict=True):
            block.setdefault('selected', {})[arm] = option
    return states, blocks


def edge_rows(truth, chosen, same):
    return {name: dict(pair_count=int(mask.sum()), positive_count=int(truth[mask].sum()),
                       **_counts(truth[mask], chosen[mask]))
            for name, mask in zip(SCOPES, (same, ~same), strict=True)}


def label_block(block, ws, identities, support, sources):
    nodes = np.asarray(block['nodes'])
    group = ws[nodes]
    truth = identities[nodes, None] == identities[None, nodes]
    before = group[:, None] == group[None, :]
    local_support = support[nodes]
    supported = ((before & truth).astype(np.int32) @ local_support.astype(np.int32)) > 0
    left, right = np.triu_indices(len(nodes), 1)
    same = sources[nodes[left]] == sources[nodes[right]]
    before_rows = edge_rows(truth[left, right], before[left, right], same)
    result = []
    for candidate in block['candidates']:
        p = np.asarray(candidate['partition'])
        after = p[:, None] == p[None, :]
        after_rows = edge_rows(truth[left, right], after[left, right], same)
        delta = {scope: {key: after_rows[scope][key] - before_rows[scope][key] for key in ('fn', 'fp')}
                 for scope in SCOPES}
        true_support = ((after & truth).astype(np.int32) @ local_support.astype(np.int32)) > 0
        lost = int((supported & ~true_support).sum())
        if candidate['supplier_unsafe_edges'] == 0:
            require(lost == 0, 'label-blind supplier guarantee failed')
        safe = bool(candidate['anchor_exact'] and lost == 0 and
                    all(value <= 0 for scope in delta.values() for value in scope.values()))
        value = -sum(delta['mstsa'].values())
        result.append(dict(delta=delta, same_ship_lost=lost, qualifying=safe and value > 0, value=value))
    return result


def evaluate_scene(row, states, blocks):
    motion = read_motion(row)
    support, sources = motion['support'], motion['sources']
    identities = read_identity(row, motion)
    ws = states['WS']
    left, right = np.triu_indices(len(sources), 1)
    truth = identities[left] == identities[right]
    same = sources[left] == sources[right]
    old_edges = ws[left] == ws[right]
    outcomes, selected_evidence, oracle = Counter(), {arm: Counter() for arm in LOCAL_ARMS}, []
    for block in blocks:
        candidates = label_block(block, ws, identities, support, sources)
        good = [i for i, candidate in enumerate(candidates) if candidate['qualifying']]
        good_strict = [i for i in good if block['candidates'][i]['supplier_unsafe_edges'] == 0]
        good_positive = [i for i in good_strict if block['candidates'][i]['mixed_gain'] > EPS]
        oracle.append(max(good, key=lambda i: candidates[i]['value']) if good else 0)
        selected = block['selected']['SUPPLIER']
        outcomes.update(dict(
            blocks=1, large_blocks=int(block['cells'] > MAX_CELLS),
            candidates=len(candidates) - 1, opportunity=int(bool(good)),
            two_cross_errors_improvable=int(any(all(x < 0 for x in candidates[i]['delta']['mstsa'].values())
                                                 for i in good)),
            opportunity_blocked_supplier=int(bool(good) and not good_strict),
            opportunity_blocked_score=int(bool(good_strict) and not good_positive),
            opportunity_misranked=int(bool(good_positive) and not candidates[selected]['qualifying']),
            qualifying_selected=int(candidates[selected]['qualifying']),
            selected_nonqualifying=int(bool(selected) and not candidates[selected]['qualifying'])))
        for name, i in block['selected'].items():
            if i:
                c = block['candidates'][i]
                keys = ('added', 'removed', 'added_cross_cosupported', 'removed_cross_cosupported',
                        'added_dual_positive', 'removed_dual_negative')
                selected_evidence[name].update({key: c[key] for key in keys})
                selected_evidence[name]['edits'] += 1
    states['ORACLE'] = materialize(ws, blocks, oracle)
    result = {}
    for name, group in states.items():
        prediction = group[left] == group[right]
        evaluation = coverage(ws, group, identities, support)
        if name in ('SUPPLIER', 'AGREE', 'ORACLE'):
            require(all(evaluation[key] == 0 for key in
                        ('identities_lost', 'same_ship_lost', 'duplicate_increase_scenes')), 'coverage certificate failed')
        result[name] = dict(coverage=evaluation, decoded=edge_rows(truth, prediction, same),
                            added=edge_rows(truth, prediction & ~old_edges, same),
                            removed=edge_rows(truth, old_edges & ~prediction, same))
    return result, dict(outcomes), {key: dict(value) for key, value in selected_evidence.items()}


def summarize(rows):
    def pool(items):
        result = {}
        for arm in (*ARMS, 'ORACLE'):
            parts = [row['arms'][arm] for row in items]
            totals = Counter()
            for arm_row in parts:
                totals.update(arm_row['coverage'])
            result[arm] = dict(coverage=dict(totals))
            for stage in ('decoded', 'added', 'removed'):
                result[arm][stage] = {scope: _aggregate([p[stage][scope] for p in parts])
                                      for scope in SCOPES}
        return result

    def direction(arms, name):
        a, b = (arms[k]['decoded'] for k in (name, 'WS'))
        return (all(a['mstsa'][k] < b['mstsa'][k] for k in ('fn', 'fp'))
                and all(a['itsa'][k] <= b['itsa'][k] for k in ('fn', 'fp')))

    groups = defaultdict(list)
    census = Counter()
    selected_evidence = {arm: Counter() for arm in LOCAL_ARMS}
    for row in rows:
        groups[row['component']].append(row)
        census.update(row['outcomes'])
        for arm, evidence in row['selected_evidence'].items():
            selected_evidence[arm].update(evidence)
    largest = max(groups, key=lambda key: (len(groups[key]), key))
    full = pool(rows)
    components = {key: pool(items) for key, items in groups.items()}
    without = pool([row for row in rows if row['component'] != largest])
    gates = {}
    for arm in (*ARMS[1:], 'ORACLE'):
        covered = all(full[arm]['coverage'][key] == 0 for key in
                      ('identities_lost', 'same_ship_lost', 'duplicate_increase_scenes'))
        count = sum(direction(group, arm) for group in components.values())
        gates[arm] = dict(decoded=direction(full, arm), retained=covered,
                          components_joint=count, without_largest=direction(without, arm),
                          passed=covered and direction(full, arm) and count >= 2 and direction(without, arm))
    return dict(full=full, without_largest=without, largest=largest, components=components,
                gates=gates, census=dict(census), selected_evidence={k: dict(v) for k, v in selected_evidence.items()})


def run(args, start):
    c95 = args.runs / C95
    require(frozen.read(c95 / 'run.json')['status'] == 'COMPLETED', 'C95 incomplete')
    expected = frozen.read(c95 / 'summary.json')
    thresholds = expected['fixed_thresholds']
    manifest = frozen.read(args.manifest)
    index = frozen.read(args.runs / DATA / 'private/index.json')
    roles = frozen.read(args.runs / PREP / 'frozen-split.json')
    splits = (('held', 'T_core', roles['held']),
              ('V_select', 'V_select', manifest['splits']['V_select']))
    summary = {}
    for split, source, ids in splits:
        private = args.output / 'private' / split
        private.mkdir(parents=True)
        summary_state(args.output, start, status='RUNNING', phase=f'freeze_{split}')
        for sid in ids:
            states, blocks = freeze_scene(index[source][sid], c95 / 'private' / split / f'{sid}.npz',
                                         thresholds)
            np.savez(private / f'{sid}.npz', **states)
            with gzip.open(private / f'{sid}.json.gz', 'wt') as stream:
                json.dump(blocks, stream, separators=(',', ':'), allow_nan=False)
        summary_state(args.output, start, status='RUNNING', phase=f'evaluate_{split}')
        rows = []
        for sid in ids:
            with np.load(private / f'{sid}.npz', allow_pickle=False) as saved:
                states = dict(saved)
            blocks = frozen.read(private / f'{sid}.json.gz')
            arms, outcomes, evidence = evaluate_scene(index[source][sid], states, blocks)
            rows.append(dict(scene=sid, component=manifest['components'][sid], arms=arms,
                             outcomes=outcomes, selected_evidence=evidence))
        result = summarize(rows)
        for here, there in (('WS', '056'), ('M_BCE', 'M_BCE')):
            for scope in SCOPES:
                check_counts(result['full'][here]['decoded'][scope],
                             expected[split]['full'][there][scope]['decoded'])
            require(result['full'][here]['coverage'] == expected[split]['full'][there]['coverage'],
                    'C95 endpoint coverage changed')
        with gzip.open(args.output / f'{split}-scenes.json.gz', 'wt') as stream:
            json.dump(rows, stream, separators=(',', ':'), allow_nan=False)
        summary[split] = result
    write(args.output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args, start)
    except Exception as error:
        summary_state(args.output, start, status='FAILED', error=repr(error))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, test_opened=False)


if __name__ == '__main__':
    main()
