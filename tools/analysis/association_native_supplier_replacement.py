"""Compare raw-observation replacement of cut C98 suppliers on identical partitions."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import association_partition_disagreement as c98
from tools.analysis.disjoint_partition_motion_evidence import competition_weights, scene_motion_evidence
from tools.analysis.held_fusion_population import _canonical_tracks, validate_measurement_archive
from tools.analysis.offline_association_tail import coverage, read_identity, read_motion, summary_state, write
from tools.analysis.same_source_swap_pilot import read


C98 = 'LEVEL1-OFFLINE-PARTITION-DISAGREEMENT-20261112-attempt-1'
NEW_ARMS = {
    'ANY_MIX': ('any', 'mixed_gain'),
    'GEOMETRY_MIX': ('geometry', 'mixed_gain'),
    'WITNESS_MIX': ('witness', 'mixed_gain'),
    'WITNESS_OLD': ('witness', 'old_gain'),
}
LIMITS = dict(cross_distance_m=300., same_distance_m=100., velocity_difference_mps=2.,
              same_gap_s=120., competition_advantage=.20)


def raw_evidence(sid, motion, archive):
    tracks = _canonical_tracks(archive._rows(sid, 'measurements'), sid, motion['node_ids'], motion['sources'])
    counts = np.array([len(track['canonical_times']) for track in tracks])
    np.testing.assert_array_equal(counts, motion['native_count'])
    native = (motion['query_seconds'], np.concatenate([t['canonical_times'] for t in tracks]),
              np.concatenate([t['canonical_values'][:, :2] for t in tracks]), np.r_[0, np.cumsum(counts)])
    sources = motion['sources']
    pairs = np.column_stack(np.triu_indices(len(sources), 1))
    fields, _ = scene_motion_evidence(pairs, sources, native)
    costs = {
        'same': np.maximum(fields['same_ols_m'] / LIMITS['same_distance_m'],
                           fields['same_slope_difference_mps'] / LIMITS['velocity_difference_mps']),
        'cross': np.maximum(fields['cross_abs_m'] / LIMITS['cross_distance_m'],
                            fields['cross_relative_mps'] / LIMITS['velocity_difference_mps']),
    }
    costs['same'][fields['same_gap_s'] > LIMITS['same_gap_s']] = np.nan
    matrices = {key: np.zeros((len(sources), len(sources)), dtype=bool) for key in ('geometry', 'witness')}
    counts = {}
    for relation, cost in costs.items():
        advantage, denominators = competition_weights(pairs, sources, native, cost, relation)
        physical = cost <= 1.
        witness = physical & (advantage >= LIMITS['competition_advantage'])
        for key, mask in (('geometry', physical), ('witness', witness)):
            left, right = pairs[mask].T
            matrices[key][left, right] = matrices[key][right, left] = True
        counts[relation] = dict(**denominators, physical=int(physical.sum()), witness=int(witness.sum()))
    return matrices, counts


def supported(edges, support):
    return (edges.astype(np.int32) @ support.astype(np.int32)) > 0


def candidate_slots(before, after, support, evidence):
    exposed = ~support & supported(before & ~after, support)
    return dict(exposed=int(exposed.sum()),
                any=int((exposed & ~supported(after, support)).sum()),
                **{key: int((exposed & ~supported(after & allowed, support)).sum())
                   for key, allowed in evidence.items()})


def choose(block, rows, requirement, score):
    winner, gain = 0, 0.
    for i, (candidate, slots) in enumerate(zip(block['candidates'], rows, strict=True)):
        if candidate['anchor_exact'] and slots[requirement] == 0 and candidate[score] > gain + c98.EPS:
            winner, gain = i, candidate[score]
    return winner


def freeze_scene(index, root, sid, out, archive):
    motion = read_motion(index)
    with np.load(root / f'{sid}.npz', allow_pickle=False) as saved:
        original = saved['WS']
    blocks = read(root / f'{sid}.json.gz')
    if blocks:
        evidence, pair_counts = raw_evidence(sid, motion, archive)
    else:
        evidence, pair_counts = {}, {}
    options = []
    for block in blocks:
        nodes = np.array(block['nodes'])
        old = original[nodes]
        before = old[:, None] == old
        support = motion['support'][nodes]
        local = {key: mask[np.ix_(nodes, nodes)] for key, mask in evidence.items()}
        candidates = []
        for candidate in block['candidates']:
            group = np.asarray(candidate['partition'])
            slots = candidate_slots(before, group[:, None] == group, support, local)
            c98.require((slots['exposed'] == 0) == (candidate['supplier_unsafe_edges'] == 0),
                        'C98 strict support certificate changed')
            c98.require(slots['exposed'] >= slots['witness'] >= slots['geometry'] >= slots['any'],
                        'nested supplier support conditions changed')
            candidates.append(slots)
        picks = {arm: choose(block, candidates, *policy) for arm, policy in NEW_ARMS.items()}
        options.append(dict(candidates=candidates, selected=picks))
    states = {arm: c98.materialize(original, blocks, [row['selected'][arm] for row in options])
              for arm in NEW_ARMS}
    np.savez_compressed(out / f'{sid}.npz', **states, **{key: np.argwhere(np.triu(mask, 1))
                                                       for key, mask in evidence.items()})
    with gzip.open(out / f'{sid}.json.gz', 'wt') as stream:
        json.dump(dict(blocks=options, pairs=pair_counts), stream, separators=(',', ':'))


def audit_replacements(before, after, support, ids, evidence, geometry, expected, requirement):
    """Rebuild selected slot coverage with explicit supplier lists."""
    counts = Counter()
    for i, times in enumerate(support):
        cut = np.flatnonzero((before[i] == before) & (after[i] != after))
        if not len(cut):
            continue
        for q in np.flatnonzero(~times & support[cut].any(axis=0)):
            if requirement == 'any':
                allowed = np.ones(len(ids), dtype=bool)
            else:
                allowed = (evidence if requirement == 'witness' else geometry)[i]
            take = np.flatnonzero((after[i] == after) & allowed & support[:, q])
            same = take[ids[take] == ids[i]]
            counts.update(exposed=1, slots_with_evidence=int(len(take) > 0),
                          slots_only_wrong_witness=int(len(take) > 0 and len(same) == 0),
                          slots_with_true_witness=int(len(same) > 0),
                          slots_true_witness_old=int(np.any(before[same] == before[i])),
                          slots_true_witness_new=int(np.any(before[same] != before[i])),
                          slots_cut_true_supplier=int(np.any(support[cut, q] & (ids[cut] == ids[i]))))
    c98.require(counts['exposed'] == counts['slots_with_evidence'] == expected, 'selected slot proof mismatch')
    return counts


def outcome_counts(block, options, labels):
    result = Counter()
    for key in ('any', 'geometry', 'witness'):
        eligible = [i for i, (candidate, slots) in enumerate(zip(block['candidates'], options['candidates'], strict=True))
                    if i and candidate['anchor_exact'] and slots[key] == 0]
        released = [i for i in eligible if options['candidates'][i]['exposed'] > 0]
        result.update({f'{key}_qualified': len(eligible), f'{key}_released': len(released),
                       f'{key}_released_opportunity_blocks': int(any(labels[i]['qualifying'] for i in released)),
                       f'{key}_opportunity_blocks': int(any(labels[i]['qualifying'] for i in eligible)),
                       f'{key}_mixed_score_blocks': int(any(labels[i]['qualifying'] and
                                                           block['candidates'][i]['mixed_gain'] > c98.EPS
                                                           for i in eligible))})
    result['anchor_exact_unsafe'] = sum(candidate['anchor_exact'] and slots['exposed'] > 0
                                        for candidate, slots in zip(block['candidates'], options['candidates'], strict=True))
    for arm, selected in options['selected'].items():
        result.update({f'{arm}_edited_blocks': int(selected > 0),
                       f'{arm}_qualifying_blocks': int(labels[selected]['qualifying']),
                       f'{arm}_unsafe_edited_blocks': int(options['candidates'][selected]['exposed'] > 0)})
    return result


def evaluate(index, root, sid, out, prior):
    motion = read_motion(index)
    ids = read_identity(index, motion)
    support, sources = motion['support'], motion['sources']
    with np.load(root / f'{sid}.npz', allow_pickle=False) as saved:
        old, mixed = (saved[key] for key in ('WS', 'M_BCE'))
    blocks = read(root / f'{sid}.json.gz')
    choices = read(out / f'{sid}.json.gz')
    with np.load(out / f'{sid}.npz', allow_pickle=False) as saved:
        states = {arm: saved[arm] for arm in NEW_ARMS}
        evidence = {}
        for key in ('geometry', 'witness'):
            mask = np.zeros((len(ids), len(ids)), dtype=bool)
            if blocks:
                left, right = saved[key].T
                mask[left, right] = mask[right, left] = True
            evidence[key] = mask
    left, right = np.triu_indices(len(ids), 1)
    truth = ids[left] == ids[right]
    same = sources[left] == sources[right]
    old_edges = old[left] == old[right]
    for key, partition in (('WS', old), ('M_BCE', mixed)):
        expected = prior['arms'][key]
        c98.require(expected['decoded'] == c98.edge_rows(truth, partition[left] == partition[right], same)
                    and expected['coverage'] == coverage(old, partition, ids, support), 'C98 endpoint changed')
    arms = dict(prior['arms'])
    census = Counter()
    replacement = {arm: Counter() for arm in NEW_ARMS}
    opportunity = 0
    for block, option in zip(blocks, choices['blocks'], strict=True):
        labels = c98.label_block(block, old, ids, support, sources)
        opportunity += any(row['qualifying'] for row in labels)
        census.update(outcome_counts(block, option, labels))
        nodes = np.asarray(block['nodes'])
        for arm, selected in option['selected'].items():
            slots = option['candidates'][selected]['exposed']
            if slots:
                replacement[arm].update(audit_replacements(
                    old[nodes], np.asarray(block['candidates'][selected]['partition']), support[nodes], ids[nodes],
                    evidence['witness'][np.ix_(nodes, nodes)], evidence['geometry'][np.ix_(nodes, nodes)],
                    slots, NEW_ARMS[arm][0]))
    c98.require(opportunity == prior['outcomes'].get('opportunity', 0), 'C98 opportunity changed')
    for name, group in states.items():
        c98.require(np.array_equal(group, c98.materialize(old, blocks,
                     [row['selected'][name] for row in choices['blocks']])), 'frozen materialization changed')
        prediction = group[left] == group[right]
        arms[name] = dict(coverage=coverage(old, group, ids, support),
                          decoded=c98.edge_rows(truth, prediction, same),
                          added=c98.edge_rows(truth, prediction & ~old_edges, same),
                          removed=c98.edge_rows(truth, old_edges & ~prediction, same))
    precision = {f'{name}_{scope}_{origin}': int((mask[left, right] & axis & label).sum())
                 for name, mask in evidence.items() for scope, axis in (('same', same), ('cross', ~same))
                 for origin, label in (('true', truth), ('false', ~truth))}
    return dict(prior, arms=arms, evidence_census=dict(census), evidence_precision=precision,
                evidence_pairs=choices['pairs'], replacement={key: dict(v) for key, v in replacement.items()})


def run(args, start):
    origin = args.runs / C98
    full_data = args.runs / c98.DATA
    c98.require(read(origin / 'run.json')['status'] == 'COMPLETED', 'C98 science incomplete')
    full = read(full_data / 'run.json')
    index = read(full_data / 'private/index.json')
    manifest = read(args.manifest)
    roles = read(args.runs / c98.PREP / 'frozen-split.json')
    splits = {'held': ('T_core', roles['held']), 'V_select': ('V_select', manifest['splits']['V_select'])}
    c98.require([len(value[1]) for value in splits.values()] == [177, 300], 'C98 populations changed')
    c98.require(set(roles['held']) <= set(manifest['splits']['T_core']), 'held outside T')
    prior = {split: {row['scene']: row for row in read(origin / f'{split}-scenes.json.gz')} for split in splits}
    measurement_members = []
    with MTADArchive(full['archive']) as archive:
        validate_measurement_archive(args.runs, archive)
        for split, (population, ids) in splits.items():
            root = origin / 'private' / split
            out = args.output / 'private' / split
            out.mkdir(parents=True)
            summary_state(args.output, start, status='RUNNING', phase=f'freeze_{split}')
            c98.require(list(prior[split]) == ids, 'frozen scene roster changed')
            for sid in ids:
                freeze_scene(index[population][sid], root, sid, out, archive)
            measurement_members.extend(sid for sid in ids if prior[split][sid]['outcomes'].get('blocks', 0))
        c98.require([(row['scene_id'], row['kind']) for row in archive.members_read] ==
                    [(sid, 'measurements') for sid in measurement_members], 'archive role changed')
    write(args.output / 'freeze.json', dict(populations={key: len(v[1]) for key, v in splits.items()},
                                           measurements=len(measurement_members), raw_label_members=0, limits=LIMITS,
                                           stages=['all_choices_and_pair_evidence_saved_before_identity']))
    result = {}
    expected = read(origin / 'summary.json')
    for split, (population, ids) in splits.items():
        summary_state(args.output, start, status='RUNNING', phase=f'evaluate_{split}')
        rows = [evaluate(index[population][sid], origin / 'private' / split, sid,
                         args.output / 'private' / split, prior[split][sid]) for sid in ids]
        totals = c98.summarize(rows, (*c98.ARMS, *NEW_ARMS))
        for arm in (*c98.ARMS, 'ORACLE'):
            for key in ('full', 'without_largest'):
                c98.require(totals[key][arm] == expected[split][key][arm], 'historical C98 aggregate changed')
        census, precision = Counter(), Counter()
        replacement = {arm: Counter() for arm in NEW_ARMS}
        pairs = {key: Counter() for key in ('same', 'cross')}
        for row in rows:
            census.update(row['evidence_census'])
            precision.update(row['evidence_precision'])
            for key, value in row['evidence_pairs'].items():
                pairs[key].update(value)
            for arm, counts in row['replacement'].items():
                replacement[arm].update(counts)
        result[split] = dict(totals, evidence_census=census, evidence_precision=precision,
                             evidence_pairs=pairs, replacement=replacement)
        with gzip.open(args.output / f'{split}-scenes.json.gz', 'wt') as stream:
            json.dump(rows, stream, separators=(',', ':'), allow_nan=False)
    write(args.output / 'summary.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args, start)
    except Exception as error:
        summary_state(args.output, start, status='FAILED', error=repr(error))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, model_forward_calls=0, test_opened=False)


if __name__ == '__main__':
    main()
