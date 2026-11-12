"""Test exclusion of old suppliers on the fixed C98 disagreement candidates."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import association_partition_disagreement as c98
from tools.analysis.association_native_supplier_replacement import C98
from tools.analysis.disjoint_partition_motion_evidence import CROSS_MIN_OVERLAP_S, CROSS_TIMES, _coordinates
from tools.analysis.held_fusion_population import _canonical_tracks, validate_measurement_archive
from tools.analysis.offline_association_tail import coverage, read_identity, read_motion, summary_state, write
from tools.analysis.same_source_swap_pilot import read
from tools.analysis.static_partition_pilot import THRESHOLD, agglomerate
from tools.analysis.supplier_external_geometry import _sample_at


FLOORS = dict(same=1000., motion=4., raw=4205.475, registered=420.168)
ANCHOR_LIMIT = 840.336
POLICIES = dict(RAW_MIX=(('raw',), 'mixed_gain'), MOTION_MIX=(('motion',), 'mixed_gain'),
                REG_MIX=(('registered',), 'mixed_gain'), REG_OLD=(('registered',), 'old_gain'),
                REG_OR_MOTION=(('registered', 'motion'), 'mixed_gain'))


def calibration_catalog(motion, path, thresholds):
    nodes, sources = motion['node_ids'], motion['sources']
    n = len(nodes)
    left, right = np.triu_indices(n, 1)
    pairs = np.column_stack((left, right))
    with np.load(path, allow_pickle=False) as saved:
        for key, value in (('pairs', pairs), ('nodes', nodes), ('sources', sources)):
            np.testing.assert_array_equal(saved[key], value)
        scores = [saved[arm] for arm in ('056', 'M_BCE')]
    membership, margins = [], []
    for arm, score in zip(('056', 'M_BCE'), scores, strict=True):
        margin = score - thresholds[arm]
        shifted = score if thresholds[arm] == THRESHOLD else margin + THRESHOLD
        membership.append(agglomerate(n, pairs, shifted, 'WS')[0])
        dense = np.zeros((n, n))
        dense[left, right] = dense[right, left] = margin
        margins.append(dense)
    old, mixed = membership
    blocks = [c98.block_catalog(rows, old, mixed, sources, motion['support'], margins)
              for rows in c98.join_blocks(old, mixed)
              if not np.array_equal(c98.frozen.canonical_membership(old[rows], len(rows)),
                                    c98.frozen.canonical_membership(mixed[rows], len(rows)))]
    return old, blocks, margins[0][left, right]


def frozen_catalog(motion, root, scores, sid, threshold):
    with np.load(root / f'{sid}.npz', allow_pickle=False) as saved:
        old = saved['WS']
    with np.load(scores / f'{sid}.npz', allow_pickle=False) as saved:
        pairs = np.column_stack(np.triu_indices(len(motion['node_ids']), 1))
        np.testing.assert_array_equal(saved['pairs'], pairs)
        np.testing.assert_array_equal(saved['group_056'], old)
        margin = saved['score_056'] - threshold
    return old, read(root / f'{sid}.json.gz'), margin


def cut_pairs(old, nodes, support, candidate):
    local = np.asarray(candidate['partition'])
    left, right = np.triu_indices(len(nodes), 1)
    a, b = nodes[left], nodes[right]
    cut = (old[a] == old[b]) & (local[left] != local[right]) & (support[a] != support[b]).any(axis=1)
    return list(zip(a[cut].tolist(), b[cut].tolist(), strict=True))


def anchor_groups(old, pairs, source, margin):
    """At most one mutually unique old link per original group and source pair."""
    n = len(old)
    left, right = pairs.T
    dense = np.full((n, n), -np.inf)
    dense[left, right] = dense[right, left] = margin
    best = np.full((n, int(source.max()) + 1), -1, dtype=np.int64)
    for label in np.unique(source):
        nodes = np.flatnonzero(source == label)
        scores = dense[:, nodes]
        j = scores.argmax(axis=1)
        top = scores[np.arange(n), j]
        unique = (top > 0.) & ((scores == top[:, None]).sum(axis=1) == 1)
        best[unique, label] = nodes[j[unique]]
    accepted = ((source[left] != source[right]) & (old[left] == old[right])
                & (best[left, source[right]] == right) & (best[right, source[left]] == left))
    groups = defaultdict(list)
    for pair in pairs[accepted]:
        a, b = sorted(map(int, pair), key=lambda i: source[i])
        groups[int(source[a]), int(source[b]), int(old[a])].append((a, b))
    result = defaultdict(list)
    for (a, b, _), matches in groups.items():
        if len(matches) == 1:
            result[a, b].append(matches[0])
    return result


def interval_sample(a, b, limits, times, xy, offsets, starts, ends):
    left = max(starts[a], starts[b], limits[0])
    right = min(ends[a], ends[b], limits[1])
    if right - left < CROSS_MIN_OVERLAP_S:
        return None
    query = np.linspace(left, right, CROSS_TIMES)
    positions, observed = _sample_at((a, b), query, times, xy, offsets)
    if not observed.all():
        return None
    return query, positions[0] - positions[1]


def cross_scores(query, displacement, anchors, args):
    motion = np.linalg.norm(np.diff(displacement, axis=0) / np.diff(query)[:, None], axis=1)
    row = dict(motion=float(np.partition(motion, 1)[1]), unregistered=float(np.median(
        np.linalg.norm(displacement, axis=1))), anchor_pool=len(anchors))
    found = [(pair, sampled) for pair in anchors
             if (sampled := interval_sample(*pair, (query[0], query[-1]), *args)) is not None]
    row['anchor_groups'] = len(found)
    if len(found) < 3:
        return dict(row, reason='anchor_count')
    windows = np.stack([sampled[0] for _, sampled in found])
    if windows[:, 0].min() > query[1] or windows[:, -1].max() < query[-2]:
        return dict(row, reason='anchor_time')
    differences = np.stack([sampled[1] for _, sampled in found])
    shifts = np.median(differences, axis=1)
    center = np.median(shifts, axis=0)
    leave_out = np.stack([np.median(np.delete(shifts, j, axis=0), axis=0) for j in range(len(shifts))])
    radius = float(np.linalg.norm(differences - leave_out[:, None, :], axis=2).max())
    if radius > ANCHOR_LIMIT:
        return dict(row, reason='anchor_spread', anchor_radius=radius)
    centers = np.vstack((center, leave_out))
    distances = np.median(np.linalg.norm(displacement[None] - centers[:, None], axis=-1), axis=1)
    return dict(row, reason='registered', raw=row['unregistered'], registered=max(0., float(distances.min())-radius),
                corrected=float(distances[0]), anchor_radius=radius, shift=float(np.linalg.norm(center)),
                anchors=[pair for pair, _ in found])


def build_evidence(sid, motion, old, blocks, margin, archive):
    support = motion['support']
    relevant = {}
    for block in blocks:
        nodes = np.asarray(block['nodes'])
        for candidate in block['candidates'][1:]:
            if candidate['anchor_exact']:
                cut = cut_pairs(old, nodes, support, candidate)
                c98.require(len(cut) == candidate['supplier_unsafe_edges'], 'fixed supplier certificate changed')
                relevant.update({pair: set(block['nodes']) for pair in cut})
    if not relevant:
        return []
    tracks = _canonical_tracks(archive._rows(sid, 'measurements'), sid, motion['node_ids'], motion['sources'])
    sizes = [len(track['canonical_times']) for track in tracks]
    np.testing.assert_array_equal(sizes, motion['native_count'])
    offsets = np.r_[0, np.cumsum(sizes)]
    times = np.concatenate([track['canonical_times'] for track in tracks])
    _, xy, _ = _coordinates(np.concatenate([track['canonical_values'][:, :2] for track in tracks]), offsets)
    args = times, xy, offsets, times[offsets[:-1]], times[offsets[1:]-1]
    _, source = np.unique(motion['sources'], return_inverse=True)
    anchors = anchor_groups(old, np.column_stack(np.triu_indices(len(old), 1)), source, margin)
    result = []
    for pair, excluded in sorted(relevant.items()):
        a, b = sorted(pair, key=lambda i: source[i])
        same = bool(source[a] == source[b])
        row = dict(pair=pair, same=same, exposure=int(np.count_nonzero(support[a] ^ support[b])))
        sampled = interval_sample(a, b, (-np.inf, np.inf), *args)
        if sampled is None:
            row['reason'] = 'no_joint_native_interval'
        elif same:
            row.update(reason='same_observed', same_distance=float(np.median(np.linalg.norm(sampled[1], axis=1))))
        else:
            outside = [pair for pair in anchors[int(source[a]), int(source[b])] if excluded.isdisjoint(pair)]
            row.update(cross_scores(*sampled, outside, args))
        result.append(row)
    return result


def value(row, key):
    return row.get('same_distance' if key == 'same' else key)


def calibrate(index, scenes, path, components):
    finite = {key: [] for key in FLOORS}
    covered = {key: set() for key in FLOORS}
    for sid in scenes:
        rows = read(path / f'{sid}.json')
        if not rows:
            continue
        data = read_motion(index[sid])
        ids = read_identity(index[sid], data)
        for row in rows:
            a, b = row['pair']
            if ids[a] != ids[b]:
                continue
            for key in FLOORS:
                score = value(row, key)
                if score is not None:
                    finite[key].append(score)
                    covered[key].add(components[sid])
    return {key: dict(upper=max(FLOORS[key], max(scores)) if scores else None,
                      true_pairs=len(scores), components=len(covered[key]),
                      maximum=max(scores) if scores else None, floor=FLOORS[key]) for key, scores in finite.items()}


def rejected(row, key, calibration):
    cap, score = calibration[key]['upper'], value(row, key)
    return cap is not None and score is not None and score > cap


def choose_scene(old, blocks, support, evidence, calibration):
    blocked = {arm: {tuple(row['pair']) for row in evidence
                     if any(rejected(row, key, calibration) for key in (('same',) if row['same'] else keys))}
               for arm, (keys, _) in POLICIES.items()}
    selections, catalog = {arm: [] for arm in POLICIES}, []
    for block in blocks:
        nodes = np.array(block['nodes'])
        good = {arm: [] for arm in POLICIES}
        for j, candidate in enumerate(block['candidates']):
            if candidate['anchor_exact']:
                cut = set(cut_pairs(old, nodes, support, candidate))
                for arm in POLICIES:
                    if cut <= blocked[arm]:
                        good[arm].append(j)
        for arm, (_, field) in POLICIES.items():
            winner, gain = 0, 0.
            for j in good[arm]:
                score = block['candidates'][j][field]
                if score > gain+c98.EPS:
                    winner, gain = j, score
            selections[arm].append(winner)
        catalog.append(good)
    states = {arm: c98.materialize(old, blocks, choices) for arm, choices in selections.items()}
    return states, dict(selections=selections, admitted=catalog,
                        rejected={arm: sorted(pairs) for arm, pairs in blocked.items()})


def summarize_evidence(rows, ids, calibration):
    counts = Counter()
    distances = []
    for row in rows:
        a, b = row['pair']
        label = 'true' if ids[a] == ids[b] else 'false'
        scope = 'same' if row['same'] else 'cross'
        counts.update((f'{scope}_{label}_pairs', f'{scope}_{label}_{row["reason"]}'))
        for key in FLOORS:
            if value(row, key) is not None:
                counts[f'{key}_{label}_eligible'] += 1
            if rejected(row, key, calibration):
                counts.update({f'{key}_{label}_excluded': 1, f'{key}_{label}_exposures': row['exposure']})
        if row['reason'] == 'registered':
            counts['anchor_uses'] += len(row['anchors'])
            counts['anchor_false_uses'] += sum(ids[i] != ids[j] for i, j in row['anchors'])
            distances.append((label, row['raw'], row['corrected'], row['registered'], row['anchor_radius'], row['shift']))
    return counts, distances


def evaluate(motion, ids, old, blocks, states, options, evidence, calibration, prior):
    sources, support = motion['sources'], motion['support']
    left, right = np.triu_indices(len(ids), 1)
    same, truth = sources[left] == sources[right], ids[left] == ids[right]
    was = old[left] == old[right]
    c98.require(prior['arms']['WS']['decoded'] == c98.edge_rows(truth, was, same), 'frozen WS does not match')
    arms = dict(prior['arms'])
    counts, distances = summarize_evidence(evidence, ids, calibration)
    census = {arm: Counter() for arm in POLICIES}
    for k, block in enumerate(blocks):
        nodes = np.asarray(block['nodes'])
        qualifying = c98.label_block(block, old, ids, support, sources)
        for arm, (_, field) in POLICIES.items():
            selected = options['selections'][arm][k]
            candidate = block['candidates'][selected]
            cuts = cut_pairs(old, nodes, support, candidate)
            false = [(i, j) for i, j in cuts if ids[i] != ids[j]]
            newly = [i for i in options['admitted'][k][arm] if block['candidates'][i]['supplier_unsafe_edges']]
            census[arm].update(edited_blocks=int(selected > 0), relaxed_edits=int(bool(cuts)),
                               selected_safe=int(qualifying[selected]['qualifying']), released=len(newly),
                               released_safe=sum(qualifying[i]['qualifying'] for i in newly),
                               released_safe_positive=sum(qualifying[i]['qualifying'] and
                                                          block['candidates'][i][field] > c98.EPS for i in newly),
                               unsafe_true_cut=len(cuts)-len(false), unsafe_false_cut=len(false),
                               unsafe_true_exposure=sum(int((support[i] ^ support[j]).sum())
                                                        for i, j in cuts if ids[i] == ids[j]))
    for arm, group in states.items():
        c98.require(np.array_equal(group, c98.materialize(old, blocks, options['selections'][arm])), 'saved state changed')
        now = group[left] == group[right]
        arms[arm] = dict(coverage=coverage(old, group, ids, support), decoded=c98.edge_rows(truth, now, same),
                         added=c98.edge_rows(truth, now & ~was, same), removed=c98.edge_rows(truth, was & ~now, same))
    return dict(prior, arms=arms, exclusion_counts=dict(counts), choices={arm: dict(c) for arm, c in census.items()}), distances


def run(args, start):
    origin, scores, data = (args.runs / name for name in (C98, c98.C95, c98.DATA))
    c98.require(read(origin / 'run.json')['status'] == 'COMPLETED', 'C98 incomplete')
    thresholds = read(scores / 'summary.json')['fixed_thresholds']
    full = read(data / 'run.json')
    index = read(data / 'private/index.json')
    manifest = read(args.manifest)
    roles = read(args.runs / c98.PREP / 'frozen-split.json')
    splits = dict(calibration=('T_core', roles['calibration']), held=('T_core', roles['held']),
                  V_select=('V_select', manifest['splits']['V_select']))
    components = manifest['components']
    c98.require([len(r[1]) for r in splits.values()] == [175, 177, 300], 'frozen populations differ')
    c98.require({components[s] for s in roles['calibration']}.isdisjoint(components[s] for s in roles['held']),
                'calibration components overlap held')
    c98.require(set(roles['calibration'] + roles['held']) <= set(manifest['splits']['T_core']), 'T population changed')
    private = args.output / 'private'
    measurement_counts = {}
    with MTADArchive(full['archive']) as archive:
        validate_measurement_archive(args.runs, archive)
        for split, (population, scenes) in splits.items():
            summary_state(args.output, start, status='RUNNING', phase=f'freeze_{split}')
            folder = private / split
            folder.mkdir(parents=True)
            first = len(archive.members_read)
            for sid in scenes:
                motion = read_motion(index[population][sid])
                old, blocks, margin = (calibration_catalog(motion, scores/'private/calibration'/f'{sid}.npz', thresholds)
                                       if split == 'calibration' else frozen_catalog(
                                           motion, origin/'private'/split, scores/'private'/split, sid, thresholds['056']))
                write(folder / f'{sid}.json', build_evidence(sid, motion, old, blocks, margin, archive))
            new = archive.members_read[first:]
            c98.require(all(row['kind'] == 'measurements' and row['scene_id'] in scenes for row in new), 'archive role changed')
            measurement_counts[split] = len(new)
    calibration = calibrate(index['T_core'], roles['calibration'], private/'calibration', components)
    write(args.output / 'calibration.json', calibration)
    for split, (population, scenes) in splits.items():
        if split == 'calibration':
            continue
        summary_state(args.output, start, status='RUNNING', phase=f'select_{split}')
        folder = private / split
        for sid in scenes:
            motion = read_motion(index[population][sid])
            old, blocks, _ = frozen_catalog(motion, origin/'private'/split, scores/'private'/split, sid, thresholds['056'])
            states, options = choose_scene(old, blocks, motion['support'], read(folder/f'{sid}.json'), calibration)
            np.savez_compressed(folder / f'{sid}.npz', **states)
            write(folder / f'{sid}-choices.json', options)
    write(args.output/'freeze.json', dict(measurement_counts=measurement_counts, calibration_before_held_selection=True,
                                         all_held_and_v_selected_before_evaluation=True, raw_label_members=0))
    result = {}
    expected = read(origin/'summary.json')
    for split, (population, scenes) in splits.items():
        if split == 'calibration':
            continue
        summary_state(args.output, start, status='RUNNING', phase=f'evaluate_{split}')
        prior = read(origin / f'{split}-scenes.json.gz')
        c98.require([row['scene'] for row in prior] == scenes, 'prior scenes differ')
        folder = private / split
        rows, distances = [], []
        for sid, previous in zip(scenes, prior, strict=True):
            record = index[population][sid]
            motion = read_motion(record)
            ids = read_identity(record, motion)
            old, blocks, _ = frozen_catalog(motion, origin/'private'/split, scores/'private'/split, sid, thresholds['056'])
            with np.load(folder/f'{sid}.npz', allow_pickle=False) as saved:
                states = dict(saved)
            row, comparison = evaluate(motion, ids, old, blocks, states, read(folder/f'{sid}-choices.json'),
                                       read(folder/f'{sid}.json'), calibration, previous)
            rows.append(row)
            distances.extend(comparison)
        pooled = c98.summarize(rows, (*c98.ARMS, *POLICIES))
        for arm in (*c98.ARMS, 'ORACLE'):
            c98.require(all(pooled[key][arm] == expected[split][key][arm] for key in ('full', 'without_largest')),
                        'C98 historical aggregation changed')
        counts, census = Counter(), {key: Counter() for key in POLICIES}
        for row in rows:
            counts.update(row['exclusion_counts'])
            for key, values in row['choices'].items():
                census[key].update(values)
        geometry = {}
        for label in ('true', 'false'):
            values = [value[1:] for value in distances if value[0] == label]
            geometry[label] = dict(pairs=len(values), medians=dict(zip(
                ('raw', 'corrected', 'registered', 'anchor_radius', 'shift'),
                np.median(values, axis=0).tolist(), strict=True)) if values else {})
        result[split] = dict(pooled, exclusion_counts=counts, choices=census, registration=geometry)
        with gzip.open(args.output/f'{split}-scenes.json.gz', 'wt') as stream:
            json.dump(rows, stream, separators=(',', ':'), allow_nan=False)
    write(args.output/'summary.json', result)


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
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, model_forward_calls=0, test_opened=False)


if __name__ == '__main__':
    main()
