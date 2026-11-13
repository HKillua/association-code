"""Read-only independent audit of saved C104 evidence, selections and support."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import resource
import time

import numpy as np

from tools.analysis import audit_offline_association_tail as audit
from tools.analysis.audit_association_partition_disagreement import anchors, edge_counts


ORIGIN = 'LEVEL1-OFFLINE-PARTITION-DISAGREEMENT-20261112-attempt-1'
SCORES = 'LEVEL1-OFFLINE-NEGATIVE-COVERAGE-20261112-attempt-1'
SCIENCE = 'LEVEL1-OFFLINE-REGISTERED-SUPPLIER-EXCLUSION-20261112-attempt-1'
FLOORS = dict(same=1000., motion=4., raw=4205.475, registered=420.168)
POLICIES = {
    'RAW_MIX': (('raw',), 'mixed_gain'),
    'MOTION_MIX': (('motion',), 'mixed_gain'),
    'REG_MIX': (('registered',), 'mixed_gain'),
    'REG_OLD': (('registered',), 'old_gain'),
    'REG_OR_MOTION': (('registered', 'motion'), 'mixed_gain'),
}
CONTROLS = ('WS', 'M_BCE', 'SUPPLIER')
MAX_ANCHOR_SPREAD = 840.336


def score_fields(row):
    return {key: row.get('same_distance' if key == 'same' else key) for key in FLOORS}


def independent_calibration(path, scenes, index, components):
    observed = {key: [] for key in FLOORS}
    for sid in scenes:
        rows = audit.read(path / f'{sid}.json')
        if not rows:
            continue
        ids = audit.scene_data(index[sid])['identities']
        for row in rows:
            a, b = row['pair']
            if ids[a] == ids[b]:
                for key, value in score_fields(row).items():
                    if value is not None:
                        assert np.isfinite(value) and value >= 0
                        observed[key].append((value, components[sid]))
    result = {}
    for key, rows in observed.items():
        largest = max((value for value, _ in rows), default=None)
        result[key] = dict(upper=max(FLOORS[key], largest) if rows else None, maximum=largest,
                           true_pairs=len(rows), components=len({group for _, group in rows}), floor=FLOORS[key])
    return result


def unsafe_cuts(old, nodes, partition, support):
    return {tuple(sorted((i, j))) for (i, left), (j, right) in
            combinations(zip(nodes, partition, strict=True), 2)
            if left != right and old[i] == old[j] and not np.array_equal(support[i], support[j])}


def parse_evidence(rows, data, old, scores_path, score_threshold, scopes, thresholds, checks):
    ids, support, sources = (data[key] for key in ('identities', 'support', 'sources'))
    rejections = {key: set() for key in FLOORS}
    statistics, match = Counter(), {'true': [], 'false': []}
    anchors_by_pair = {}
    rows_by_pair = {tuple(row['pair']): row for row in rows}
    assert len(rows_by_pair) == len(rows) and set(rows_by_pair) == set(scopes)
    dense = None

    def strongest_partner(node, target_source):
        nonlocal dense
        if dense is None:
            with np.load(scores_path, allow_pickle=False) as stored:
                left, right = np.triu_indices(len(ids), 1)
                np.testing.assert_array_equal(stored['pairs'], np.column_stack((left, right)))
                np.testing.assert_array_equal(stored['group_056'], old)
                dense = np.full((len(ids), len(ids)), -np.inf)
                dense[left, right] = dense[right, left] = stored['score_056']
        possible = np.flatnonzero(sources == target_source)
        candidates = dense[node, possible]
        maximum = candidates.max()
        chosen = possible[candidates == maximum]
        return int(chosen[0]) if maximum > score_threshold and len(chosen) == 1 else None

    for pair, row in rows_by_pair.items():
        a, b = pair
        assert 0 <= a < b < len(ids) and old[a] == old[b]
        same = sources[a] == sources[b]
        checks.verify(bool(same), row['same'], 'source relation')
        checks.verify(int(np.count_nonzero(support[a] ^ support[b])), row['exposure'], 'Q exposure')
        label = 'true' if ids[a] == ids[b] else 'false'
        relation = 'same' if same else 'cross'
        statistics.update((f'{relation}_{label}_pairs', f'{relation}_{label}_{row["reason"]}'))
        for key, value in score_fields(row).items():
            if value is None:
                continue
            assert np.isfinite(value) and value >= 0
            assert (key == 'same') == bool(same)
            statistics[f'{key}_{label}_eligible'] += 1
            if thresholds[key] is not None and value > thresholds[key]:
                rejections[key].add(pair)
                statistics.update({f'{key}_{label}_excluded': 1, f'{key}_{label}_exposures': row['exposure']})
        if row['reason'] != 'registered':
            assert 'anchors' not in row and 'raw' not in row and 'registered' not in row
            continue
        assert not same and 0 <= row['registered'] <= row['corrected']
        assert row['raw'] == row['unregistered'] and 0 <= row['anchor_radius'] <= MAX_ANCHOR_SPREAD
        assert row['anchor_pool'] >= row['anchor_groups'] == len(row['anchors']) >= 3
        source_order = sorted((sources[a], sources[b]))
        external_groups = set()
        for i, j in row['anchors']:
            assert 0 <= i < len(ids) and 0 <= j < len(ids)
            assert i not in scopes[pair] and j not in scopes[pair]
            assert old[i] == old[j] and old[i] not in external_groups
            assert [sources[i], sources[j]] == source_order
            checks.verify(strongest_partner(i, sources[j]), j, 'anchor unique best')
            checks.verify(strongest_partner(j, sources[i]), i, 'anchor unique best')
            members_i = [m for m in np.flatnonzero(old == old[i]) if sources[m] == sources[i]]
            eligible = []
            for member in members_i:
                peer = strongest_partner(member, sources[j])
                if peer is not None and old[peer] == old[i] and strongest_partner(peer, sources[i]) == member:
                    eligible.append((member, peer))
            checks.verify(eligible, [(i, j)], 'unique anchor within WS group and source pair')
            external_groups.add(old[i])
        statistics['anchor_uses'] += len(row['anchors'])
        statistics['anchor_false_uses'] += sum(ids[i] != ids[j] for i, j in row['anchors'])
        anchors_by_pair[pair] = row['anchors']
        match[label].append(tuple(row[key] for key in ('raw', 'corrected', 'registered', 'anchor_radius', 'shift')))
    return rejections, statistics, match, anchors_by_pair


def qualifying_options(old, nodes, support, sources, ids, candidates):
    members = np.asarray(nodes)
    baseline = old[members]
    local_ids = ids[members]
    local_support = support[members]
    local_sources = sources[members]
    li, lj = np.triu_indices(len(nodes), 1)
    truth = local_ids[li] == local_ids[lj]
    same = local_sources[li] == local_sources[lj]
    first = edge_counts(truth, baseline[li] == baseline[lj], same)
    result = []
    for row in candidates:
        new = np.asarray(row['partition'])
        kept = np.array_equal(anchors(baseline), anchors(new))
        assert bool(kept) == row['anchor_exact']
        coverage = audit.coverage(baseline, new, local_ids, local_support)
        changed = edge_counts(truth, new[li] == new[lj], same)
        improvements = {axis: [first[axis][key] - changed[axis][key] for key in ('fn', 'fp')]
                        for axis in ('itsa', 'mstsa')}
        result.append(bool(kept and not coverage['same_ship_lost'] and sum(improvements['mstsa']) > 0
                           and min(improvements['itsa'] + improvements['mstsa']) >= 0))
    return result


def absence_coverage(group, ids, support):
    """Evaluate both any suppliers and same-ship suppliers for each recipient."""
    any_slot = np.zeros_like(support)
    true_slot = np.zeros_like(support)
    for gid in np.unique(group):
        members = np.flatnonzero(group == gid)
        any_slot[members] = support[members].any(axis=0)
        for identity in np.unique(ids[members]):
            own = members[ids[members] == identity]
            true_slot[own] = support[own].any(axis=0)
    return any_slot, true_slot


def check_scene(data, origin, private, scores, sid, row, thresholds, score_threshold, checks):
    ids, support, sources = (data[key] for key in ('identities', 'support', 'sources'))
    with np.load(origin / f'{sid}.npz', allow_pickle=False) as frozen:
        old = frozen['WS']
        controls = {key: frozen[key] for key in CONTROLS}
    blocks = audit.read(origin / f'{sid}.json.gz')
    options = audit.read(private / f'{sid}-choices.json')
    evidence = audit.read(private / f'{sid}.json')
    scopes, cuts = {}, []
    for block in blocks:
        nodes = block['nodes']
        all_cuts = [unsafe_cuts(old, nodes, c['partition'], support) for c in block['candidates']]
        for c, cut in zip(block['candidates'], all_cuts, strict=True):
            checks.verify(len(cut), c['supplier_unsafe_edges'], 'unsafe cut count')
            if c['anchor_exact']:
                for pair in cut:
                    assert pair not in scopes or scopes[pair] == set(nodes)
                    scopes[pair] = set(nodes)
        cuts.append(all_cuts)
    rejections, statistics, match, external = parse_evidence(
        evidence, data, old, scores / f'{sid}.npz', score_threshold, scopes, thresholds, checks)
    checks.verify(dict(statistics), row['exclusion_counts'], 'scene exclusion census')
    assert set(statistics) == set(row['exclusion_counts'])
    with np.load(private / f'{sid}.npz', allow_pickle=False) as saved:
        predictions = {key: saved[key] for key in POLICIES}
    selection_census = {}
    pair_metrics = {}
    li, lj = np.triu_indices(len(ids), 1)
    truth = ids[li] == ids[lj]
    same = sources[li] == sources[lj]
    was = old[li] == old[lj]
    before_any, before_true = absence_coverage(old, ids, support)
    qualifies = [qualifying_options(old, b['nodes'], support, sources, ids, b['candidates']) for b in blocks]
    for arm, (kinds, field) in POLICIES.items():
        rejected = set().union(rejections['same'], *(rejections[key] for key in kinds))
        checks.verify(sorted(map(list, rejected)), options['rejected'][arm], 'all rejected pairs')
        change = old.copy()
        next_label = int(old.max()) + 1
        counters, active = Counter(), Counter()
        for b, block in enumerate(blocks):
            candidates = block['candidates']
            admitted = [j for j, candidate in enumerate(candidates)
                        if candidate['anchor_exact'] and not cuts[b][j].difference(rejected)]
            checks.verify(admitted, options['admitted'][b][arm], 'admitted candidates')
            winner = 0
            for candidate in admitted:
                if candidates[candidate][field] > candidates[winner][field] + 1.e-10:
                    winner = candidate
            checks.verify(winner, options['selections'][arm][b], 'chosen candidate')
            if winner:
                labels = np.asarray(candidates[winner]['partition'])
                change[block['nodes']] = labels + next_label
                next_label += int(labels.max())+1
            released = [j for j in admitted if cuts[b][j]]
            selected = cuts[b][winner]
            true_selected = [(i, j) for i, j in selected if ids[i] == ids[j]]
            counters.update(edited_blocks=int(bool(winner)), relaxed_edits=int(bool(selected)),
                            selected_safe=int(qualifies[b][winner]), released=len(released),
                            released_safe=sum(qualifies[b][j] for j in released),
                            released_safe_positive=sum(qualifies[b][j] and candidates[j][field] > 1.e-10
                                                       for j in released),
                            unsafe_true_cut=len(true_selected), unsafe_false_cut=len(selected)-len(true_selected),
                            unsafe_true_exposure=sum(int(np.count_nonzero(support[i] ^ support[j]))
                                                     for i, j in true_selected))
            for i, j in selected:
                if (i, j) in rejections['registered'] and 'registered' in kinds:
                    used = external[i, j]
                    active['registered_anchor_uses'] += len(used)
                    active['registered_anchor_false_uses'] += sum(ids[a] != ids[b] for a, b in used)
        registry = {}
        group = np.array([registry.setdefault(g, len(registry)) for g in change])
        np.testing.assert_array_equal(predictions[arm], group)
        checks.verify(dict(counters), row['choices'][arm], 'scene candidate census')
        after_any, after_true = absence_coverage(group, ids, support)
        now = group[li] == group[lj]
        removed = was & ~now
        lost = int(np.count_nonzero(before_true & ~after_true))
        coverage = audit.coverage(old, group, ids, support)
        checks.verify(lost, coverage['same_ship_lost'], 'enumerated true slot losses')
        checks.verify(dict(coverage=coverage, decoded=edge_counts(truth, now, same),
                           added=edge_counts(truth, now & ~was, same), removed=edge_counts(truth, removed, same)),
                      row['arms'][arm], 'whole scene result')
        active.update(all_true_cut=int(np.count_nonzero(removed & truth)), all_false_cut=int(np.count_nonzero(removed & ~truth)),
                      same_ship_lost=lost, same_ship_recovered=int(np.count_nonzero(~before_true & after_true)),
                      pure_wrong_slots_released=int(np.count_nonzero(before_any & ~before_true & ~after_any)),
                      pure_wrong_slots_introduced=int(np.count_nonzero(~before_any & after_any & ~after_true)))
        pair_metrics[arm] = active
        selection_census[arm] = counters
    for arm, group in controls.items():
        now = group[li] == group[lj]
        checks.verify(dict(coverage=audit.coverage(old, group, ids, support), decoded=edge_counts(truth, now, same),
                           added=edge_counts(truth, now & ~was, same), removed=edge_counts(truth, was & ~now, same)),
                      row['arms'][arm], 'C98 control')
    return dict(scenes=1, pairs=len(li), blocks=len(blocks),
                candidates=sum(len(block['candidates'])-1 for block in blocks),
                evidence_scenes=int(bool(evidence))), statistics, match, selection_census, pair_metrics


def check_aggregates(rows, saved, checks):
    by_component = defaultdict(list)
    for row in rows:
        by_component[row['component']].append(row)
    largest = max(by_component, key=lambda key: (len(by_component[key]), key))
    checks.verify(largest, saved['largest'], 'largest component')
    reduced = [row for row in rows if row['component'] != largest]
    assert set(by_component) == set(saved['components'])
    populations = [(rows, saved['full']), (reduced, saved['without_largest'])]
    populations += [(rs, saved['components'][key]) for key, rs in by_component.items()]

    def result(items, arm):
        return {axis: tuple(sum(r['arms'][arm]['decoded'][axis][k] for r in items) for k in ('fn', 'fp'))
                for axis in ('itsa', 'mstsa')}

    def improves(items, arm):
        old, current = result(items, 'WS'), result(items, arm)
        return (all(a < b for a, b in zip(current['mstsa'], old['mstsa'], strict=True))
                and all(a <= b for a, b in zip(current['itsa'], old['itsa'], strict=True)))

    for arm in (*CONTROLS, *POLICIES):
        for items, expected in populations:
            totals = Counter()
            for row in items:
                totals.update(row['arms'][arm]['coverage'])
            checks.verify(dict(totals), expected[arm]['coverage'], 'aggregate coverage')
            for stage in ('decoded', 'added', 'removed'):
                for axis in ('itsa', 'mstsa'):
                    merged = audit.aggregate([row['arms'][arm][stage][axis] for row in items])
                    checks.verify(merged, expected[arm][stage][axis], 'aggregate confusion')
        if arm == 'WS':
            continue
        retained = not any(row['arms'][arm]['coverage'][key] for row in rows
                           for key in ('identities_lost', 'same_ship_lost', 'duplicate_increase_scenes'))
        decoded = improves(rows, arm)
        joint = sum(improves(items, arm) for items in by_component.values())
        leave = improves(reduced, arm)
        checks.verify(dict(decoded=decoded, retained=retained, components_joint=joint,
                           without_largest=leave, passed=retained and decoded and joint >= 2 and leave),
                      saved['gates'][arm], 'aggregate qualification')


def run(runs, output, manifest_path):
    output.mkdir()
    start = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    origin, scores, science = (runs / name for name in (ORIGIN, SCORES, SCIENCE))
    state = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
                 raw_archive_reads=0, optimizer_updates=0, test_opened=False)

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-start,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        audit.write(output / 'run.json', state)

    try:
        checkpoint(phase='calibration')
        marker = audit.read(science / 'run.json')
        launcher = runs / SCIENCE.replace('-attempt-', '-launch-attempt-')
        assert marker['status'] == 'COMPLETED' and not marker['test_opened']
        assert audit.read(launcher / 'status.json')['returncode'] == 0
        assert int((launcher / 'exit_code.txt').read_text()) == 0
        freeze_path = science / 'freeze.json'
        calibration_path = science / 'calibration.json'
        freeze = audit.read(freeze_path)
        assert freeze['calibration_before_held_selection'] and freeze['all_held_and_v_selected_before_evaluation']
        assert freeze['raw_label_members'] == 0
        manifest = audit.read(manifest_path)
        roles = audit.read(runs / audit.PREP / 'frozen-split.json')
        index = audit.read(runs / audit.DATA / 'private/index.json')
        assert {manifest['components'][s] for s in roles['calibration']}.isdisjoint(
            manifest['components'][s] for s in roles['held'])
        checks = audit.Comparisons()
        all_results = audit.read(science / 'summary.json')
        score_threshold = audit.read(scores / 'summary.json')['fixed_thresholds']['056']
        cal_inputs = science / 'private/calibration'
        assert set(roles['calibration']) <= set(manifest['splits']['T_core']) and len(roles['calibration']) == 175
        assert sum(bool(audit.read(cal_inputs / f'{s}.json')) for s in roles['calibration']) == freeze['measurement_counts']['calibration']
        assert max((cal_inputs / f'{s}.json').stat().st_mtime_ns for s in roles['calibration']) <= calibration_path.stat().st_mtime_ns
        calibration = independent_calibration(cal_inputs, roles['calibration'], index['T_core'], manifest['components'])
        checks.verify(calibration, audit.read(calibration_path), 'training true-tail threshold')
        thresholds = {key: row['upper'] for key, row in calibration.items()}
        results = {}
        for split, population, scenes in (('held', 'T_core', roles['held']),
                                           ('V_select', 'V_select', manifest['splits']['V_select'])):
            checkpoint(phase=split)
            assert set(scenes) <= set(manifest['splits'][population])
            folder = science / 'private' / split
            public_path = science / f'{split}-scenes.json.gz'
            rows = audit.read(public_path)
            original = audit.read(origin / f'{split}-scenes.json.gz')
            assert [r['scene'] for r in rows] == [r['scene'] for r in original] == scenes
            assert max((folder / f'{s}.json').stat().st_mtime_ns for s in scenes) <= calibration_path.stat().st_mtime_ns
            assert min((folder / f'{s}-choices.json').stat().st_mtime_ns for s in scenes) >= calibration_path.stat().st_mtime_ns
            assert max(p.stat().st_mtime_ns for p in folder.iterdir()) <= freeze_path.stat().st_mtime_ns < public_path.stat().st_mtime_ns
            scene_counts, exclusion = Counter(), Counter()
            choices, cuts = ({arm: Counter() for arm in POLICIES} for _ in range(2))
            matches = defaultdict(list)
            for sid, row, old in zip(scenes, rows, original, strict=True):
                assert row['component'] == old['component'] == manifest['components'][sid]
                assert all(row['arms'][arm] == old['arms'][arm] for arm in CONTROLS)
                census, rejected, registration, chosen, real_cuts = check_scene(
                    audit.scene_data(index[population][sid]), origin/'private'/split, folder,
                    scores/'private'/split, sid, row, thresholds, score_threshold, checks)
                scene_counts.update(census)
                exclusion.update(rejected)
                for key, values in registration.items():
                    matches[key].extend(values)
                for arm in POLICIES:
                    choices[arm].update(chosen[arm])
                    cuts[arm].update(real_cuts[arm])
            assert scene_counts['evidence_scenes'] == freeze['measurement_counts'][split]
            expected = all_results[split]
            checks.verify(dict(exclusion), expected['exclusion_counts'], 'pooled false exclusion')
            assert set(exclusion) == set(expected['exclusion_counts'])
            checks.verify({key: dict(value) for key, value in choices.items()}, expected['choices'], 'pooled choices')
            for label, values in matches.items():
                medians = dict(zip(('raw', 'corrected', 'registered', 'anchor_radius', 'shift'),
                                   np.median(values, axis=0).tolist(), strict=True)) if values else {}
                checks.verify(dict(pairs=len(values), medians=medians), expected['registration'][label], 'matched registration')
            check_aggregates(rows, expected, checks)
            results[split] = dict(population=scene_counts, exclusion_counts=exclusion, choices=choices, actual_cuts=cuts,
                                  gates={arm: expected['gates'][arm] for arm in POLICIES})
        assert sum(row['population']['blocks'] for row in results.values()) == 1711
        assert sum(row['population']['candidates'] for row in results.values()) == 5426
        audit.write(output / 'audit.json', dict(calibration=calibration, splits=results, exact=checks.exact,
                    numerical=checks.numerical, registered_anchor_excludes_whole_candidate_block=True,
                    frozen_before_evaluation_artifact=True, raw_geometry_recomputed=False))
        checkpoint(status='COMPLETED', finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        checkpoint(status='FAILED', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'output', 'manifest'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs, args.output, args.manifest)


if __name__ == '__main__':
    main()
