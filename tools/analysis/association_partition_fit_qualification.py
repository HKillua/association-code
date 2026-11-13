"""C139: inventory protected partition preferences and native intervals on C95 fit only."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from association.data.archive import MTADArchive
from tools.analysis import association_partition_disagreement as c98
from tools.analysis import offline_association_tail as tail
from tools.analysis.association_registered_supplier_exclusion import catalog_from_scores
from tools.analysis.disjoint_partition_motion_evidence import _coordinates
from tools.analysis.held_fusion_population import _canonical_tracks, validate_measurement_archive


GAP_S = 120.
CENTER_S = 60.
FOLDS = 3
MIN_FOLD_BLOCKS = 10
MIN_FOLD_COMPONENTS = 2
MIN_REMAINING_BLOCKS = 30
MIN_REMAINING_COMPONENTS = 4


def write_gzip(path, value):
    with gzip.open(path, 'wt', encoding='utf8') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)


def report_triples(times):
    gaps = np.diff(times)
    first = np.flatnonzero((gaps[:-1] <= GAP_S) & (gaps[1:] <= GAP_S))
    return [(int(i), float(times[i]), float(times[i+1]), float(times[i+2])) for i in first]


def match_triples(left, right):
    """Take at most two chronological matches with disjoint reports in each source."""
    i = j = 0
    matches = []
    while i < len(left) and j < len(right) and len(matches) < 2:
        a, start_a, center_a, end_a = left[i]
        b, start_b, center_b, end_b = right[j]
        overlap = min(end_a, end_b) - max(start_a, start_b)
        if abs(center_a-center_b) <= CENTER_S and overlap > 0.:
            matches.append((a, b))
            while i < len(left) and left[i][0] < a+3:
                i += 1
            while j < len(right) and right[j][0] < b+3:
                j += 1
        elif end_a <= start_b or center_a < center_b-CENTER_S:
            i += 1
        else:
            j += 1
    return matches


def changed_cross_pairs(old, sources, block, candidate):
    nodes = np.asarray(block['nodes'])
    left, right = np.triu_indices(len(nodes), 1)
    a, b = nodes[left], nodes[right]
    p = np.asarray(candidate['partition'])
    before = old[a] == old[b]
    after = p[left] == p[right]
    cross = sources[a] != sources[b]
    added = zip(a[cross & ~before & after].tolist(), b[cross & ~before & after].tolist(), strict=True)
    removed = zip(a[cross & before & ~after].tolist(), b[cross & before & ~after].tolist(), strict=True)
    return list(added), list(removed)


def velocity_delta(times, xy, i):
    first = (xy[i+1]-xy[i])/(times[i+1]-times[i])
    second = (xy[i+2]-xy[i+1])/(times[i+2]-times[i+1])
    return second-first


def native_evidence(archive, sid, motion, old, blocks):
    requests = []
    relevant = set()
    for block in blocks:
        options = []
        for candidate in block['candidates']:
            added, removed = changed_cross_pairs(old, motion['sources'], block, candidate)
            options.append((added, removed))
            if candidate['anchor_exact']:
                relevant.update(added)
                relevant.update(removed)
        requests.append(options)
    if not relevant:
        return requests, []
    tracks = _canonical_tracks(archive._rows(sid, 'measurements'), sid, motion['node_ids'], motion['sources'])
    times = [t['canonical_times'] for t in tracks]
    lengths = list(map(len, times))
    np.testing.assert_array_equal(lengths, motion['native_count'])
    offsets = np.r_[0, np.cumsum(lengths)]
    _, xy, _ = _coordinates(np.concatenate([t['canonical_values'][:, :2] for t in tracks]), offsets)
    positions = np.split(xy, offsets[1:-1])
    triples = list(map(report_triples, times))
    evidence = []
    for a, b in sorted(relevant):
        matches = match_triples(triples[a], triples[b])
        rows = []
        for i, j in matches:
            va = velocity_delta(times[a], positions[a], i)
            vb = velocity_delta(times[b], positions[b], j)
            rows.append(dict(start_indices=[i, j], center_gap_s=float(abs(times[a][i+1]-times[b][j+1])),
                             delta_difference_mps=float(np.linalg.norm(va-vb))))
        evidence.append(dict(pair=[a, b], matches=rows))
    return requests, evidence


def observed_changes(added, removed, levels):
    add = [levels[tuple(pair)] for pair in added]
    cut = [levels[tuple(pair)] for pair in removed]
    bilateral = bool(add and cut and max(add) == 2 and max(cut) == 2)
    return bilateral, bool(bilateral and min(add) == 2 and min(cut) == 2)


def preference_counts(safe, adverse, counts, suffix=''):
    if not safe or not adverse:
        return
    counts[f'preference_blocks{suffix}'] += 1
    counts[f'preference_pairs{suffix}'] += len(safe)*len(adverse)
    wrong = tied = 0
    ordered = sorted(adverse)
    for score in safe:
        right = bisect_right(ordered, score+c98.EPS)
        left = bisect_left(ordered, score-c98.EPS)
        wrong += len(ordered)-right
        tied += right-left
    counts[f'j_reversed_pairs{suffix}'] += wrong
    counts[f'j_tied_pairs{suffix}'] += tied
    counts[f'j_unresolved_blocks{suffix}'] += int(wrong+tied > 0)


def evaluate_scene(motion, identity, old, blocks, requests, evidence):
    counts = Counter()
    choices = []
    candidate_labels = []
    levels = {tuple(row['pair']): len(row['matches']) for row in evidence}
    for level in (0, 1, 2):
        counts[f'changed_cross_pairs_native_{level}'] = sum(n == level for n in levels.values())
    oracle_expected = Counter()
    for block, changes in zip(blocks, requests, strict=True):
        labels = c98.label_block(block, old, identity, motion['support'], motion['sources'])
        counts.update(blocks=1, large_blocks=int(block['cells'] > c98.MAX_CELLS),
                      non_ws_candidates=len(labels)-1)
        safe, adverse, native_safe, native_adverse, best = [], [], [], [], 0
        for i, (candidate, label, (added, removed)) in enumerate(
                zip(block['candidates'], labels, changes, strict=True)):
            if not i or not candidate['anchor_exact']:
                continue
            counts['preserve_head_candidates'] += 1
            cross = label['delta']['mstsa']
            dual = label['qualifying'] and cross['fn'] < 0 and cross['fp'] < 0
            worse = label['same_ship_lost'] > 0 or any(
                value > 0 for scope in label['delta'].values() for value in scope.values())
            bilateral, fully = observed_changes(added, removed, levels)
            score = candidate['mixed_gain']
            label.update(dual=dual, adverse=worse, native_bilateral=bilateral, native_all_changed_cross=fully)
            if dual:
                safe.append(score)
                counts['dual_candidates'] += 1
                counts['dual_positive_mixed_j'] += int(score > c98.EPS)
                counts['dual_strict_supply_certificate'] += int(candidate['supplier_unsafe_edges'] == 0)
                if bilateral:
                    native_safe.append(score)
                    counts['dual_native_bilateral'] += 1
                    counts['dual_native_all_changed_cross'] += int(fully)
                if label['value'] > labels[best]['value']:
                    best = i
            if worse:
                adverse.append(score)
                counts['adverse_candidates'] += 1
                if bilateral:
                    native_adverse.append(score)
                    counts['adverse_native_bilateral'] += 1
        counts['dual_blocks'] += int(bool(safe))
        counts['dual_native_bilateral_blocks'] += int(bool(native_safe))
        preference_counts(safe, adverse, counts)
        preference_counts(native_safe, native_adverse, counts, '_native_bilateral')
        for scope, delta in labels[best]['delta'].items():
            oracle_expected.update({f'{scope}_{metric}_delta': value for metric, value in delta.items()})
        choices.append(best)
        candidate_labels.append(labels)
    chosen = c98.materialize(old, blocks, choices)
    coverage = tail.coverage(old, chosen, identity, motion['support'])
    c98.require(not (coverage['same_ship_lost'] or coverage['identities_lost'] or coverage['duplicate_increase']),
                'oracle failed full-scene protection')
    left, right = np.triu_indices(len(old), 1)
    truth = identity[left] == identity[right]
    same = motion['sources'][left] == motion['sources'][right]
    base = c98.edge_rows(truth, old[left] == old[right], same)
    result = c98.edge_rows(truth, chosen[left] == chosen[right], same)
    for key in ('itsa', 'mstsa'):
        for metric in ('fn', 'fp'):
            name = f'{key}_{metric}_delta'
            c98.require(result[key][metric]-base[key][metric] == oracle_expected[name],
                        'oracle full-scene confusion differs from summed changes')
    return dict(counts), dict(coverage=coverage, baseline=base, oracle=result, delta=dict(oracle_expected)), candidate_labels


def pool(rows):
    result = Counter()
    components = set()
    scenes = 0
    for row in rows:
        scenes += 1
        components.add(row['component'])
        result.update(row['counts'])
        result.update(row['oracle']['delta'])
        result['same_ship_recovered'] += row['oracle']['coverage']['same_ship_recovered']
        result['same_ship_lost'] += row['oracle']['coverage']['same_ship_lost']
    return dict(result, scenes=scenes, components=len(components))


def population_summary(rows, order):
    groups = defaultdict(list)
    for row in rows:
        groups[row['component']].append(row)
    per_component = {k: pool(groups[k]) for k in order}
    fold_of = {key: i % FOLDS for i, key in enumerate(order)}
    largest = order[0]
    removed = [row for row in rows if row['component'] != largest]
    without = pool(removed)
    feature = 'preference_blocks_native_bilateral'
    folds = []
    for i in range(FOLDS):
        keys = [k for k in order if fold_of[k] == i]
        summary = pool(row for k in keys for row in groups[k])
        summary['native_preference_components'] = sum(per_component[k].get(feature, 0) > 0 for k in keys)
        folds.append(summary)
    remaining_components = sum(per_component[k].get(feature, 0) > 0 for k in order[1:])
    gates = dict(each_fold=all(f.get(feature, 0) >= MIN_FOLD_BLOCKS and
                              f['native_preference_components'] >= MIN_FOLD_COMPONENTS for f in folds),
                 without_largest=without.get(feature, 0) >= MIN_REMAINING_BLOCKS and
                 remaining_components >= MIN_REMAINING_COMPONENTS,
                 unresolved_j_without_largest=without.get('j_unresolved_blocks_native_bilateral', 0) > 0)
    return dict(full=pool(rows), largest_scene_component=largest, without_largest=without,
                native_preference_components_without_largest=remaining_components, components=per_component,
                fold_of=fold_of, folds=folds, gates=gates, capacity_qualified=all(gates.values()))


def run(args, start):
    original = args.runs / tail.audit.RUNS['056'] / 'association'
    data = args.runs / c98.DATA
    prep = args.runs / c98.PREP
    mixed = args.runs / c98.C95
    read = tail.audit.read_json
    manifest = read(args.manifest)
    roles = read(prep / 'frozen-split.json')
    index = read(data / 'private/index.json')['T_core']
    require = c98.require
    for parent in (data, prep, mixed):
        require(read(parent / 'run.json')['status'] == 'COMPLETED', f'incomplete frozen parent {parent.name}')
    require(roles == tail.scene_split(manifest['splits']['T_core'], manifest['components']), 'frozen fit split changed')
    ids = roles['fit']
    components = {sid: manifest['components'][sid] for sid in ids}
    require(len(ids) == len(set(ids)) == 1748 and len(set(components.values())) == 76, 'fit population changed')
    require(all(index[sid]['component_id'] == components[sid] for sid in ids), 'component binding changed')
    count = Counter(components.values())
    order = sorted(count, key=lambda key: (-count[key], key))
    thresholds = {k: read(mixed / 'summary.json')['fixed_thresholds'][k] for k in ('056', 'M_BCE')}
    _, checkpoint = tail.load_checkpoint(original / 'final.pt', 'cpu')
    require(checkpoint['update'] == 1000, 'original 056 checkpoint changed')
    head = tail.EvidenceAssociation()
    head.load_state_dict(torch.load(mixed / 'private/heads.pt', map_location='cpu', weights_only=True)['M_BCE'])
    head.eval().requires_grad_(False)
    blind = args.output / 'private' / 'blind'
    labelled = args.output / 'private' / 'labelled'
    blind.mkdir(parents=True)
    labelled.mkdir()
    pair_count, archive_counts = 0, Counter()
    with MTADArchive(read(data / 'run.json')['archive']) as archive:
        validate_measurement_archive(args.runs, archive)
        for number, sid in enumerate(ids, 1):
            motion, pairs, scores = tail.score_scene(index[sid], sid, checkpoint['normalizer'],
                                                    {'M_BCE': head}, 'cpu', prep, None)
            old, blocks, _ = catalog_from_scores(motion, pairs, scores, thresholds)
            requests, evidence = native_evidence(archive, sid, motion, old, blocks)
            write_gzip(blind / f'{sid}.json.gz', dict(ws=old.tolist(), blocks=blocks,
                                                      changes=requests, native=evidence))
            pair_count += len(pairs)
            if number % 50 == 0:
                tail.summary_state(args.output, start, status='RUNNING', phase='freeze', scenes=number,
                                   forward_pairs=pair_count)
        for row in archive.members_read:
            require(row['kind'] == 'measurements' and row['scene_id'] in components, 'archive role changed')
            archive_counts[row['kind']] += 1
    require(len(list(blind.iterdir())) == len(ids), 'blind population incomplete before identity phase')
    tail.write(args.output / 'freeze.json', dict(scene_count=len(ids), component_count=len(order),
               scene_order_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
               thresholds=thresholds, counts_by_archive_member=dict(archive_counts), forward_pairs=pair_count,
               heads=['C95 M_BCE'], trained_heads=0, old_056_fit_scores_reused=len(ids),
               all_candidates_and_native_evidence_before_identity=True, other_splits_opened=False))
    rows = []
    for number, sid in enumerate(ids, 1):
        motion = tail.read_motion(index[sid])
        identity = tail.read_identity(index[sid], motion)
        frozen = read(blind / f'{sid}.json.gz')
        counts, oracle, labels = evaluate_scene(motion, identity, np.asarray(frozen['ws']), frozen['blocks'],
                                                frozen['changes'], frozen['native'])
        write_gzip(labelled / f'{sid}.json.gz', labels)
        rows.append(dict(scene=sid, component=components[sid], counts=counts, oracle=oracle))
        if number % 50 == 0:
            tail.summary_state(args.output, start, status='RUNNING', phase='evaluate', scenes=number)
    summary = population_summary(rows, order)
    write_gzip(args.output / 'fit-scenes.json.gz', rows)
    tail.write(args.output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    torch.set_num_threads(1)
    start = time.monotonic()
    tail.summary_state(args.output, start, status='RUNNING', phase='preflight')
    try:
        run(args, start)
    except Exception as exc:
        tail.summary_state(args.output, start, status='FAILED', error=repr(exc), optimizer_updates=0, test_opened=False)
        raise
    tail.summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, test_opened=False)


if __name__ == '__main__':
    main()
