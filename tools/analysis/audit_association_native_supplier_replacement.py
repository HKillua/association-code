"""Recount C101 from frozen choices, pair evidence and 151 cache only."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import resource
import time

import numpy as np

from tools.analysis import audit_offline_association_tail as audit
from tools.analysis.audit_association_partition_disagreement import anchors, edge_counts


ORIGIN = 'LEVEL1-OFFLINE-PARTITION-DISAGREEMENT-20261112-attempt-1'
SCIENCE = 'LEVEL1-OFFLINE-NATIVE-SUPPLIER-REPLACEMENT-20261112-attempt-1'
POLICIES = {
    'ANY_MIX': ('any', 'mixed_gain'),
    'GEOMETRY_MIX': ('geometry', 'mixed_gain'),
    'WITNESS_MIX': ('witness', 'mixed_gain'),
    'WITNESS_OLD': ('witness', 'old_gain'),
}
KINDS = ('any', 'geometry', 'witness')
CONTROLS = ('WS', 'M_BCE', 'LOCAL_OLD', 'SUPPLIER', 'ORACLE')
ARMS = (*CONTROLS, *POLICIES)


def membership_from_labels(labels):
    registry = {}
    return np.array([registry.setdefault(label, len(registry)) for label in labels])


def candidate_evidence(old, candidate, support, ids, partners):
    new = np.asarray(candidate['partition'])
    certificate = dict.fromkeys(('exposed', *KINDS), 0)
    provenance = {key: Counter() for key in KINDS}
    extra = Counter()
    for i in range(len(old)):
        cut = np.flatnonzero((old == old[i]) & (new != new[i]))
        if not len(cut):
            continue
        exposed = np.flatnonzero(~support[i] & np.logical_or.reduce(support[cut], axis=0))
        if not len(exposed):
            continue
        certificate['exposed'] += len(exposed)
        retained = np.flatnonzero(new == new[i])
        for q in exposed:
            cut_true = np.any(support[cut, q] & (ids[cut] == ids[i]))
            present = retained[support[retained, q]]
            available_true = present[ids[present] == ids[i]]
            extra.update(cut_true_slots=int(cut_true), cut_only_wrong_slots=int(not cut_true),
                         no_any_slots=int(not len(present)),
                         no_any_with_true_cut_slots=int(not len(present) and cut_true),
                         no_actual_same_ship_replacement=int(cut_true and not len(available_true)),
                         true_retained_slots=int(cut_true and np.any(old[available_true] == old[i])),
                         true_added_slots=int(cut_true and np.any(old[available_true] != old[i])))
            for key in KINDS:
                usable = present if key == 'any' else [j for j in present if j in partners[key][i]]
                certificate[key] += int(not len(usable))
                true = [j for j in usable if ids[j] == ids[i]]
                provenance[key].update(exposed=1, slots_with_evidence=int(len(usable) > 0),
                    slots_only_wrong_witness=int(len(usable) > 0 and not len(true)),
                    slots_with_true_witness=int(len(true) > 0),
                    slots_true_witness_old=int(any(old[j] == old[i] for j in true)),
                    slots_true_witness_new=int(any(old[j] != old[i] for j in true)),
                    slots_cut_true_supplier=int(cut_true))
    return certificate, provenance, extra


def recount_block(old, block, frozen, support, ids, sources, partners, comparisons):
    candidates = block['candidates']
    left, right = np.triu_indices(len(old), 1)
    truth = ids[left] == ids[right]
    same = sources[left] == sources[right]
    initial = edge_counts(truth, old[left] == old[right], same)
    gains, decisions, qualifications = [], [], []
    for candidate, saved in zip(candidates, frozen['candidates'], strict=True):
        result, provenance, extra = candidate_evidence(old, candidate, support, ids, partners)
        comparisons.verify(result, saved, 'candidate support')
        group = np.asarray(candidate['partition'])
        keeps_anchors = np.array_equal(anchors(group), anchors(old))
        assert keeps_anchors == candidate['anchor_exact']
        assert (result['exposed'] == 0) == (candidate['supplier_unsafe_edges'] == 0)
        changed = edge_counts(truth, group[left] == group[right], same)
        differences = [[initial[scope][key] - changed[scope][key] for key in ('fn', 'fp')]
                       for scope in ('itsa', 'mstsa')]
        qualifies = bool(keeps_anchors and not extra['no_actual_same_ship_replacement']
                         and min(min(values) for values in differences) >= 0 and sum(differences[1]) > 0)
        gains.append((result, provenance, extra))
        qualifications.append(qualifies)
        decisions.append(keeps_anchors)
    census, reasons = Counter(), Counter()
    for key in KINDS:
        good = [i for i, (decision, gain) in enumerate(zip(decisions, gains, strict=True))
                if i and decision and gain[0][key] == 0]
        release = [i for i in good if gains[i][0]['exposed']]
        census.update({f'{key}_qualified': len(good), f'{key}_released': len(release),
                       f'{key}_released_opportunity_blocks': int(any(qualifications[i] for i in release)),
                       f'{key}_opportunity_blocks': int(any(qualifications[i] for i in good)),
                       f'{key}_mixed_score_blocks': int(any(qualifications[i] and candidates[i]['mixed_gain'] > 1e-10
                                                           for i in good))})
    census['anchor_exact_unsafe'] = sum(ok and gain[0]['exposed'] > 0
                                        for ok, gain in zip(decisions, gains, strict=True))
    unsafe_opportunities = [(slots, extra) for (slots, _, extra), good in zip(gains, qualifications, strict=True)
                            if good and slots['exposed']]
    reasons.update(qualifying_blocks=int(any(qualifications)),
                   qualifying_unsafe_blocks=int(bool(unsafe_opportunities)),
                   qualifying_unsafe_candidates=len(unsafe_opportunities))
    mechanisms = set()
    for slots, extra in unsafe_opportunities:
        assert not extra['no_any_with_true_cut_slots']
        if slots['any']:
            mechanisms.add('missing_any')
            reasons['missing_any_slots'] += slots['any']
        if extra['cut_true_slots']:
            mechanisms.add('cuts_true')
        else:
            mechanisms.add('cuts_only_wrong')
        if extra['true_retained_slots']:
            mechanisms.add('retained_true')
        if extra['true_added_slots']:
            mechanisms.add('added_true')
        if slots['any'] and not extra['cut_true_slots']:
            mechanisms.add('missing_any_cuts_only_wrong')
    for mechanism in mechanisms:
        reasons[f'{mechanism}_blocks'] += 1
    selected, replacements = {}, {}
    for arm, (kind, field) in POLICIES.items():
        winner = 0
        for i in range(1, len(candidates)):
            if decisions[i] and gains[i][0][kind] == 0 and candidates[i][field] > candidates[winner][field] + 1e-10:
                winner = i
        selected[arm] = winner
        replacements[arm] = gains[winner][1][kind]
        census.update({f'{arm}_edited_blocks': int(bool(winner)),
                       f'{arm}_qualifying_blocks': int(qualifications[winner]),
                       f'{arm}_unsafe_edited_blocks': int(gains[winner][0]['exposed'] > 0)})
    comparisons.verify(selected, frozen['selected'], 'selected options')
    return selected, replacements, census, reasons


def sparse_pairs(records, kind, size):
    edges = records[kind] if kind in records else np.empty((0, 2), dtype=int)
    assert edges.shape == (len(edges), 2) and edges.dtype.kind in 'iu'
    result = set(map(tuple, edges.tolist()))
    assert len(result) == len(edges) and all(0 <= i < j < size for i, j in result)
    return result


def verify_scene(data, origin, frozen, sid, row, comparisons):
    with np.load(origin / f'{sid}.npz', allow_pickle=False) as saved:
        states = {arm: saved[arm] for arm in CONTROLS if arm != 'ORACLE'}
    old = states['WS']
    blocks = audit.read(origin / f'{sid}.json.gz')
    private = audit.read(frozen / f'{sid}.json.gz')
    assert private['pairs'] == row['evidence_pairs']
    with np.load(frozen / f'{sid}.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(POLICIES) | ({'geometry', 'witness'} if blocks else set())
        expected = {arm: saved[arm] for arm in POLICIES}
        pairs = {key: sparse_pairs(saved, key, len(old)) for key in ('geometry', 'witness')}
    assert pairs['witness'] <= pairs['geometry']
    ids, sources, support = (data[key] for key in ('identities', 'sources', 'support'))
    labels = {arm: [(0, int(g)) for g in old] for arm in POLICIES}
    partners, precision = {}, {}
    for key, edges in pairs.items():
        partners[key] = [set() for _ in old]
        for i, j in edges:
            partners[key][i].add(j)
            partners[key][j].add(i)
        for relation in ('same', 'cross'):
            eligible = [(i, j) for i, j in edges if (sources[i] == sources[j]) == (relation == 'same')]
            true = sum(int(ids[i] == ids[j]) for i, j in eligible)
            precision[f'{key}_{relation}_true'] = true
            precision[f'{key}_{relation}_false'] = len(eligible) - true
            if blocks:
                field = 'physical' if key == 'geometry' else key
                assert len(eligible) == private['pairs'][relation][field]
    comparisons.verify(precision, row['evidence_precision'], 'pair precision')
    counts, reasons = Counter(), Counter()
    replacements = {arm: Counter() for arm in POLICIES}
    for block_index, (block, saved) in enumerate(zip(blocks, private['blocks'], strict=True), start=1):
        nodes = np.asarray(block['nodes'])
        local = {key: [{j for j, b in enumerate(nodes) if b in values[a]} for a in nodes]
                 for key, values in partners.items()}
        choices, support_outcomes, census, why = recount_block(
            old[nodes], block, saved, support[nodes], ids[nodes], sources[nodes], local, comparisons)
        counts.update(census)
        reasons.update(why)
        for arm, winner in choices.items():
            replacements[arm].update(support_outcomes[arm])
            if winner:
                for i, gid in zip(nodes, block['candidates'][winner]['partition'], strict=True):
                    labels[arm][i] = (block_index, gid)
    assert dict(counts) == row['evidence_census']
    assert {key: dict(value) for key, value in replacements.items()} == row['replacement']
    assert reasons['qualifying_blocks'] == row['outcomes'].get('opportunity', 0)
    for arm, raw in labels.items():
        states[arm] = membership_from_labels(raw)
        np.testing.assert_array_equal(states[arm], expected[arm], err_msg=f'{sid} {arm}')
        np.testing.assert_array_equal(anchors(states[arm]), anchors(old))
    left, right = np.triu_indices(len(old), 1)
    truth = ids[left] == ids[right]
    same = sources[left] == sources[right]
    before = old[left] == old[right]
    for arm, group in states.items():
        chosen = group[left] == group[right]
        result = dict(coverage=audit.coverage(old, group, ids, support), decoded=edge_counts(truth, chosen, same),
                      added=edge_counts(truth, chosen & ~before, same), removed=edge_counts(truth, before & ~chosen, same))
        comparisons.verify(result, row['arms'][arm], f'{sid}/{arm}')
    counts.update(blocks=len(blocks), candidates=sum(len(block['candidates']) - 1 for block in blocks))
    return len(left), counts, reasons, precision, replacements


def verify_totals(rows, totals, comparisons):
    components = defaultdict(list)
    for row in rows:
        components[row['component']].append(row)
    largest = max(components, key=lambda key: (len(components[key]), key))
    assert largest == totals['largest'] and set(components) == set(totals['components'])
    without = [row for row in rows if row['component'] != largest]

    def errors(items, arm):
        return {scope: tuple(sum(row['arms'][arm]['decoded'][scope][key] for row in items) for key in ('fn', 'fp'))
                for scope in ('itsa', 'mstsa')}

    def improves(items, arm):
        old, candidate = errors(items, 'WS'), errors(items, arm)
        return bool(all(a < b for a, b in zip(candidate['mstsa'], old['mstsa'], strict=True))
                    and all(a <= b for a, b in zip(candidate['itsa'], old['itsa'], strict=True)))

    for arm in ARMS:
        populations = [(rows, totals['full']), (without, totals['without_largest'])]
        populations.extend((subset, totals['components'][key]) for key, subset in components.items())
        for subset, expected in populations:
            coverage = Counter()
            for row in subset:
                coverage.update(row['arms'][arm]['coverage'])
            assert dict(coverage) == expected[arm]['coverage']
            for stage in ('decoded', 'added', 'removed'):
                for scope in ('itsa', 'mstsa'):
                    summary = audit.aggregate([row['arms'][arm][stage][scope] for row in subset])
                    comparisons.verify(summary, expected[arm][stage][scope], f'aggregate/{arm}/{stage}/{scope}')
        if arm != 'WS':
            retained = not any(row['arms'][arm]['coverage'][key] for row in rows
                               for key in ('identities_lost', 'same_ship_lost', 'duplicate_increase_scenes'))
            decoded = improves(rows, arm)
            joint = sum(improves(items, arm) for items in components.values())
            leave = improves(without, arm)
            gate = dict(decoded=decoded, retained=retained, components_joint=joint,
                        without_largest=leave, passed=retained and decoded and joint >= 2 and leave)
            comparisons.verify(gate, totals['gates'][arm], 'gate')


def run(runs, output, manifest_path):
    output.mkdir()
    start = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    root, origin = runs / SCIENCE, runs / ORIGIN
    state = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
                 raw_archive_reads=0, optimizer_updates=0, test_opened=False)

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic() - start,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        audit.write(output / 'run.json', state)

    try:
        checkpoint()
        original_state = audit.read(root / 'run.json')
        assert original_state['status'] == 'COMPLETED' and not original_state['test_opened']
        launcher = runs / SCIENCE.replace('-attempt-', '-launch-attempt-')
        assert audit.read(launcher / 'status.json')['returncode'] == 0
        assert int((launcher / 'exit_code.txt').read_text()) == 0
        freeze = audit.read(root / 'freeze.json')
        assert freeze['raw_label_members'] == 0
        assert freeze['stages'] == ['all_choices_and_pair_evidence_saved_before_identity']
        expected = audit.read(root / 'summary.json')
        manifest = audit.read(manifest_path)
        roles = audit.read(runs / audit.PREP / 'frozen-split.json')
        index = audit.read(runs / audit.DATA / 'private/index.json')
        checks = audit.Comparisons()
        comparisons = (('held', 'T_core', roles['held']),
                       ('V_select', 'V_select', manifest['splits']['V_select']))
        results = {}
        measurements = 0
        for split, population, ids in comparisons:
            checkpoint(phase=split)
            assert len(ids) == freeze['populations'][split] and set(ids) <= set(manifest['splits'][population])
            public = root / f'{split}-scenes.json.gz'
            frozen = root / 'private' / split
            assert {p.name for p in frozen.iterdir()} == {f'{sid}{ext}' for sid in ids for ext in ('.npz', '.json.gz')}
            assert max(p.stat().st_mtime_ns for p in frozen.iterdir()) <= (root / 'freeze.json').stat().st_mtime_ns
            assert (root / 'freeze.json').stat().st_mtime_ns < public.stat().st_mtime_ns
            rows = audit.read(public)
            prior = audit.read(origin / f'{split}-scenes.json.gz')
            assert [r['scene'] for r in rows] == [r['scene'] for r in prior] == ids
            counts, reasons, precision = Counter(), Counter(), Counter()
            replacements = {arm: Counter() for arm in POLICIES}
            pairs = 0
            for sid, row, old in zip(ids, rows, prior, strict=True):
                assert row['component'] == manifest['components'][sid]
                assert all(row['arms'][arm] == old['arms'][arm] for arm in CONTROLS)
                scene_data = audit.scene_data(index[population][sid])
                n, census, why, pair_precision, slots = verify_scene(
                    scene_data, origin / 'private' / split, frozen, sid, row, checks)
                pairs += n
                measurements += int(census['blocks'] > 0)
                counts.update(census)
                reasons.update(why)
                precision.update(pair_precision)
                for arm in POLICIES:
                    replacements[arm].update(slots[arm])
            science = expected[split]
            assert {k: v for k, v in counts.items() if k not in ('blocks', 'candidates')} == science['evidence_census']
            assert dict(precision) == science['evidence_precision']
            assert {key: dict(value) for key, value in replacements.items()} == science['replacement']
            verify_totals(rows, science, checks)
            results[split] = dict(scenes=len(ids), complete_pairs=pairs, census=counts, mechanisms=reasons,
                                  precision=precision, replacement=replacements,
                                  gates={arm: science['gates'][arm] for arm in POLICIES})
        assert measurements == freeze['measurements']
        assert sum(row['census']['blocks'] for row in results.values()) == 1711
        assert sum(row['census']['candidates'] for row in results.values()) == 5426
        audit.write(output / 'audit.json', dict(splits=results, exact=checks.exact, numerical=checks.numerical,
                    frozen_artifact_cutoff_verified=True, complete_pair_evidence_scenes=measurements,
                    raw_pair_features_recomputed=False))
        checkpoint(status='COMPLETED', finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        checkpoint(status='FAILED', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'output', 'manifest'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--time-policy', required=True, choices=('advisory',))
    args = parser.parse_args()
    run(args.runs, args.output, args.manifest)


if __name__ == '__main__':
    main()
