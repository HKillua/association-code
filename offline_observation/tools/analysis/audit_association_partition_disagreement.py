"""Independently reconstruct and audit C98 saved local partition choices."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import resource
import time

import numpy as np

from tools.analysis import audit_offline_association_tail as audit


C95 = 'LEVEL1-OFFLINE-NEGATIVE-COVERAGE-20261112-attempt-1'
SCIENCE = 'LEVEL1-OFFLINE-PARTITION-DISAGREEMENT-20261112-attempt-1'
BLIND = ('WS', 'M_BCE', 'LOCAL_OLD', 'LOCAL_MIX', 'SOURCE', 'SUPPLIER', 'AGREE')
ARMS = (*BLIND, 'ORACLE')
LOCAL = BLIND[2:]
SCOPES = ('itsa', 'mstsa')
STAGES = ('decoded', 'added', 'removed')
EVIDENCE = ('added', 'removed', 'added_cross_cosupported', 'removed_cross_cosupported',
            'added_dual_positive', 'removed_dual_negative')
EPS = 1e-10


def canonical(labels):
    order = {}
    return np.asarray([order.setdefault(int(value), len(order)) for value in labels])


def anchors(group):
    return np.sort(np.unique(group, return_index=True)[1])


def changed_blocks(old, mixed):
    graph = (old[:, None] == old) | (mixed[:, None] == mixed)
    left = set(range(len(old)))
    blocks = []
    while left:
        seen = {min(left)}
        queue = list(seen)
        for node in queue:
            new = set(np.flatnonzero(graph[node]))-seen
            seen.update(new)
            queue.extend(new)
        left.difference_update(seen)
        idx = np.asarray(sorted(seen))
        if not np.array_equal(canonical(old[idx]), canonical(mixed[idx])):
            blocks.append(idx)
    return blocks


def legal_partitions(old, mixed):
    cells = {}
    labels = list(zip(old.tolist(), mixed.tolist(), strict=True))
    cell_of = np.asarray([cells.setdefault(label, len(cells)) for label in labels])
    unique = np.asarray(list(cells))
    count = len(unique)
    old_cells = tuple(canonical(unique[:, 0]))
    mixed_cells = tuple(canonical(unique[:, 1]))
    if count > 6:
        options = [old_cells, mixed_cells]
    else:
        left, right = np.triu_indices(count, 1)
        forbidden = (unique[left] != unique[right]).all(axis=1)
        options = [old_cells]
        for label in product(*(range(i+1) for i in range(count))):
            if label == old_cells or any(label[i] > max(label[:i])+1 for i in range(1, count)):
                continue
            values = np.asarray(label)
            if not np.any((values[left] == values[right]) & forbidden):
                options.append(label)
    return [np.asarray(option)[cell_of] for option in options], count


def source_slots(group, source, support):
    same_group = group[:, None] == group
    return np.stack([same_group.astype(np.int32) @ (support & (source == s)[:, None]).astype(np.int32) > 0
                     for s in np.unique(source)], axis=1)


def edge_counts(truth, pred, same):
    return {scope: audit.confusion(truth[mask], pred[mask])
            for scope, mask in zip(SCOPES, (same, ~same), strict=True)}


def audit_block(nodes, prior, old, mixed, data, score, checks):
    old, mixed = old[nodes], mixed[nodes]
    support, sources, identity = (data[key][nodes] for key in ('support', 'sources', 'identities'))
    left, right = np.triu_indices(len(nodes), 1)
    before = old[left] == old[right]
    old_gain, mix_gain = (arr[nodes[left], nodes[right]] for arr in score)
    joint = (support[left] & support[right]).any(axis=1)
    cross = sources[left] != sources[right]
    truth = identity[left] == identity[right]
    original_counts = edge_counts(truth, before, ~cross)
    orig_slots = source_slots(old, sources, support)
    unequal = (support[left] ^ support[right]).any(axis=1)
    partitions, cells = legal_partitions(old, mixed)
    assert len(partitions) == len(prior['candidates']) and cells == prior['cells']
    assert nodes.tolist() == prior['nodes']
    evidence, oracle = [], []
    for membership, saved in zip(partitions, prior['candidates'], strict=True):
        after = membership[left] == membership[right]
        add, remove = after & ~before, before & ~after
        row = dict(partition=membership.tolist(),
            old_gain=float(np.sum(old_gain[add])-np.sum(old_gain[remove])),
            mixed_gain=float(np.sum(mix_gain[add])-np.sum(mix_gain[remove])),
            anchor_exact=bool(np.array_equal(anchors(old), anchors(membership))),
            source_lost=int(np.count_nonzero(orig_slots & ~source_slots(membership, sources, support))),
            supplier_unsafe_edges=int(np.count_nonzero(remove & unequal)),
            added=int(add.sum()), removed=int(remove.sum()),
            added_cross_cosupported=int((add & cross & joint).sum()),
            removed_cross_cosupported=int((remove & cross & joint).sum()),
            added_dual_positive=int((add & (old_gain > 0) & (mix_gain > 0)).sum()),
            removed_dual_negative=int((remove & (old_gain < 0) & (mix_gain < 0)).sum()))
        checks.verify(row, saved, 'candidate')
        counts = edge_counts(truth, after, ~cross)
        delta = {scope: [counts[scope][field]-original_counts[scope][field] for field in ('fn', 'fp')]
                 for scope in SCOPES}
        support_lost = audit.coverage(old, membership, identity, support)['same_ship_lost']
        if row['supplier_unsafe_edges'] == 0:
            assert support_lost == 0
        value = -sum(delta['mstsa'])
        qualifies = row['anchor_exact'] and support_lost == 0 and value > 0 and all(
            error <= 0 for scope in SCOPES for error in delta[scope])
        oracle.append((qualifies, value, all(d < 0 for d in delta['mstsa'])))
        evidence.append(row)

    eligible = {
        'LOCAL_OLD': list(range(len(evidence))), 'LOCAL_MIX': list(range(len(evidence))),
        'SOURCE': [i for i, r in enumerate(evidence) if r['anchor_exact'] and r['source_lost'] == 0],
        'SUPPLIER': [i for i, r in enumerate(evidence) if r['anchor_exact'] and r['supplier_unsafe_edges'] == 0],
    }
    eligible['AGREE'] = [i for i in eligible['SUPPLIER'] if evidence[i]['old_gain'] >= -EPS]
    selections = {}
    for arm, candidates in eligible.items():
        field = 'old_gain' if arm == 'LOCAL_OLD' else 'mixed_gain'
        best = 0
        for i in candidates:
            if evidence[i][field]-evidence[best][field] > EPS:
                best = i
        selections[arm] = best
    checks.verify(selections, prior['selected'], 'selected block')
    good = [i for i, (qualified, _, _) in enumerate(oracle) if qualified]
    certified = [i for i in good if evidence[i]['supplier_unsafe_edges'] == 0]
    positive = [i for i in certified if evidence[i]['mixed_gain'] > EPS]
    chosen = selections['SUPPLIER']
    outcomes = dict(blocks=1, large_blocks=int(cells > 6), candidates=len(evidence)-1,
        opportunity=int(bool(good)), two_cross_errors_improvable=int(any(oracle[i][2] for i in good)),
        opportunity_blocked_supplier=int(bool(good) and not certified),
        opportunity_blocked_score=int(bool(certified) and not positive),
        opportunity_misranked=int(bool(positive) and not oracle[chosen][0]),
        qualifying_selected=int(oracle[chosen][0]), selected_nonqualifying=int(bool(chosen) and not oracle[chosen][0]))
    selections['ORACLE'] = max(good, key=lambda i: oracle[i][1]) if good else 0
    return partitions, selections, outcomes, evidence


def audit_scene(root, c95, split, sid, source, thresholds, checks):
    data = audit.scene_data(source)
    n = len(data['node_ids'])
    left, right = np.triu_indices(n, 1)
    with np.load(c95/'private'/split/f'{sid}.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['pairs'], np.column_stack((left, right)))
        old, mixed = z['group_056'], z['group_M_BCE']
        score = np.zeros((2, n, n))
        for i, arm in enumerate(('056', 'M_BCE')):
            score[i, left, right] = z[f'score_{arm}']-thresholds[arm]
    private = root/'private'/split
    blocks = audit.read(private/f'{sid}.json.gz')
    expected = changed_blocks(old, mixed)
    assert len(blocks) == len(expected)
    states = {arm: old.copy() for arm in (*LOCAL, 'ORACLE')}
    offsets = dict.fromkeys(states, n)
    outcomes = Counter()
    selected_evidence = {arm: Counter() for arm in LOCAL}
    for nodes, block in zip(expected, blocks, strict=True):
        partitions, choices, census, evidence = audit_block(nodes, block, old, mixed, data, score, checks)
        outcomes.update(census)
        for arm, i in choices.items():
            if i:
                states[arm][nodes] = offsets[arm]+partitions[i]
                offsets[arm] += int(partitions[i].max())+1
                if arm in LOCAL:
                    selected_evidence[arm].update({key: evidence[i][key] for key in EVIDENCE})
                    selected_evidence[arm]['edits'] += 1
    states = dict(WS=old, M_BCE=mixed, **{arm: canonical(state) for arm, state in states.items()})
    with np.load(private/f'{sid}.npz', allow_pickle=False) as z:
        assert set(z.files) == set(BLIND)
        for arm in BLIND:
            np.testing.assert_array_equal(states[arm], z[arm], err_msg=f'{split}/{sid}/{arm}')
    truth = data['identities'][left] == data['identities'][right]
    same = data['sources'][left] == data['sources'][right]
    before = old[left] == old[right]
    result = {}
    for arm, membership in states.items():
        after = membership[left] == membership[right]
        support = audit.coverage(old, membership, data['identities'], data['support'])
        if arm in ('SOURCE', 'SUPPLIER', 'AGREE', 'ORACLE'):
            assert np.array_equal(anchors(old), anchors(membership))
        if arm in ('SUPPLIER', 'AGREE', 'ORACLE'):
            assert not support['same_ship_lost']
        result[arm] = dict(coverage=support, decoded=edge_counts(truth, after, same),
                           added=edge_counts(truth, after & ~before, same),
                           removed=edge_counts(truth, before & ~after, same))
    return result, dict(outcomes), {arm: dict(c) for arm, c in selected_evidence.items()}, len(left)


def summarize(rows):
    def pool(items):
        result = {}
        for arm in ARMS:
            coverage = Counter()
            for item in items:
                coverage.update(item['arms'][arm]['coverage'])
            result[arm] = dict(coverage=dict(coverage))
            for stage in STAGES:
                result[arm][stage] = {scope: audit.aggregate([r['arms'][arm][stage][scope] for r in items])
                                      for scope in SCOPES}
        return result

    def improves(groups, name):
        original, candidate = (groups[arm]['decoded'] for arm in ('WS', name))
        for key in ('fn', 'fp'):
            if candidate['mstsa'][key] >= original['mstsa'][key] or candidate['itsa'][key] > original['itsa'][key]:
                return False
        return True

    groups, census = defaultdict(list), Counter()
    selected = {arm: Counter() for arm in LOCAL}
    for row in rows:
        groups[row['component']].append(row)
        census.update(row['outcomes'])
        for arm, counts in row['selected_evidence'].items():
            selected[arm].update(counts)
    largest = max(groups, key=lambda c: (len(groups[c]), c))
    full = pool(rows)
    components = {key: pool(items) for key, items in groups.items()}
    without = pool([row for row in rows if row['component'] != largest])
    gates = {}
    for arm in ARMS[1:]:
        retained = all(full[arm]['coverage'][key] == 0 for key in
                       ('identities_lost', 'same_ship_lost', 'duplicate_increase_scenes'))
        decoded, leave = improves(full, arm), improves(without, arm)
        n = sum(improves(result, arm) for result in components.values())
        gates[arm] = dict(decoded=decoded, retained=retained, without_largest=leave, components_joint=n,
                          passed=decoded and retained and leave and n >= 2)
    return dict(full=full, largest=largest, without_largest=without, components=components, gates=gates,
                census=dict(census), selected_evidence={arm: dict(c) for arm, c in selected.items()})


def run(runs, output, manifest_path):
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    root, c95 = runs/SCIENCE, runs/C95
    state = dict(status='RUNNING', new_predictions=0, optimizer_updates=0, raw_archive_reads=0,
                 started_at=datetime.now(timezone.utc).isoformat())
    checks = audit.Comparisons()

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-start,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        audit.write(output/'run.json', state)

    try:
        science = audit.read(root/'run.json')
        assert science['status'] == 'COMPLETED' and science['optimizer_updates'] == 0 and science['test_opened'] is False
        assert audit.read(runs/SCIENCE.replace('-attempt-', '-launch-attempt-')/'status.json')['returncode'] == 0
        manifest = audit.read(manifest_path)
        index = audit.read(runs/audit.DATA/'private/index.json')
        roles = audit.read(runs/audit.PREP/'frozen-split.json')
        expected = audit.read(root/'summary.json')
        control = audit.read(c95/'summary.json')
        thresholds = control['fixed_thresholds']
        sections = (('held', 'T_core', roles['held']), ('V_select', 'V_select', manifest['splits']['V_select']))
        assert tuple(len(ids) for _, _, ids in sections) == (177, 300)
        results = {}
        for split, source, ids in sections:
            checkpoint(phase=split)
            public = root/f'{split}-scenes.json.gz'
            published = audit.read(public)
            assert [r['scene'] for r in published] == ids
            private = root/'private'/split
            expected_files = {f'{sid}{suffix}' for sid in ids for suffix in ('.npz', '.json.gz')}
            assert {p.name for p in private.iterdir()} == expected_files
            assert max(p.stat().st_mtime_ns for p in private.iterdir()) < public.stat().st_mtime_ns
            rows, pairs = [], 0
            for sid, prior in zip(ids, published, strict=True):
                states, outcomes, evidence, n = audit_scene(root, c95, split, sid, index[source][sid], thresholds, checks)
                row = dict(scene=sid, component=manifest['components'][sid], arms=states,
                           outcomes=outcomes, selected_evidence=evidence)
                checks.verify(row, prior, f'{split}/{sid}')
                rows.append(row)
                pairs += n
            aggregate = summarize(rows)
            checks.verify(aggregate, expected[split], split)
            for arm, baseline in (('WS', '056'), ('M_BCE', 'M_BCE')):
                reference = control[split]['full'][baseline]
                checks.verify(aggregate['full'][arm]['coverage'], reference['coverage'], f'{split}/{arm}/C95coverage')
                for scope in SCOPES:
                    counts = aggregate['full'][arm]['decoded'][scope]
                    checks.verify({key: counts[key] for key in ('tp', 'fn', 'fp', 'tn')},
                                  reference[scope]['decoded'], f'{split}/{arm}/C95/{scope}')
            results[split] = dict(scenes=len(ids), components=len(aggregate['components']), pairs=pairs,
                replayed_blind_states=len(ids)*len(BLIND), reconstructed_oracles=len(ids),
                census=aggregate['census'], selected_evidence=aggregate['selected_evidence'], gates=aggregate['gates'],
                decoded={arm: {scope: {key: counts[key] for key in ('tp', 'fn', 'fp', 'tn')}
                    for scope, counts in value['decoded'].items()} for arm, value in aggregate['full'].items()},
                coverage={arm: value['coverage'] for arm, value in aggregate['full'].items()})
        audit.write(output/'audit.json', dict(status='COMPLETED', splits=results,
            exact_comparisons=checks.exact, numerical_comparisons=checks.numerical,
            independent_local_partition_enumeration=True, original_outputs_unchanged=True,
            test_opened=False, optimizer_updates=0, new_predictions=0))
        checkpoint(status='COMPLETED', phase='done', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs.resolve(), args.output.resolve(), args.manifest.resolve())
