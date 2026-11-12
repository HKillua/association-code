"""C113: two-anchor validation on disjoint native observations in the frozen C107 pool."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import association_registered_holdout_fusion as c107
from tools.analysis import association_registered_supplier_exclusion as c104
from tools.analysis.offline_association_tail import summary_state, write
from tools.analysis.same_source_swap_pilot import read

C107 = 'LEVEL1-OFFLINE-REGISTERED-HOLDOUT-FUSION-20261112-attempt-1'
ARMS = ('WS', 'STRICT', 'C104', 'REG')
REQUIRED = (72, 69, 246)


def exactly_two(row):
    return row['reason'] == 'anchor_count' and row['anchor_groups'] == 2


def native_geometry(sid, motion, archive):
    tracks = c104._canonical_tracks(archive._rows(sid, 'measurements'), sid, motion['node_ids'], motion['sources'])
    sizes = [len(track['canonical_times']) for track in tracks]
    np.testing.assert_array_equal(sizes, motion['native_count'])
    offsets = np.r_[0, np.cumsum(sizes)]
    times = np.concatenate([track['canonical_times'] for track in tracks])
    _, xy, _ = c104._coordinates(np.concatenate([track['canonical_values'][:, :2] for track in tracks]), offsets)
    return times, xy, offsets, times[offsets[:-1]], times[offsets[1:]-1]


def bracket_indices(node, query, times, offsets):
    start, end = offsets[node:node + 2]
    native = times[start:end]
    upper = np.searchsorted(native, query).clip(0, len(native) - 1)
    interpolated = native[upper] != query
    lower = (upper[interpolated] - 1).clip(0, len(native) - 1)
    return set(np.r_[upper, lower].tolist())


def disjoint_native_extents(times, offsets, midpoint):
    """Restrict interpolation to original rows belonging to each time half."""
    early_start, late_start = (np.full(len(offsets) - 1, np.inf) for _ in range(2))
    early_end, late_end = (np.full(len(offsets) - 1, -np.inf) for _ in range(2))
    for node, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        split = start + np.searchsorted(times[start:end], midpoint)
        if start < split:
            early_start[node], early_end[node] = times[start], times[split - 1]
        if split < end:
            late_start[node], late_end[node] = times[split], times[end - 1]
    return ((early_start, early_end), (late_start, late_end))


def temporal_evidence(query, displacement, outside, geometry):
    times, xy, offsets, starts, ends = geometry
    first, last = query[[0, -1]]
    mid = (first + last) / 2.
    whole = [(pair, sampled) for pair in outside
             if (sampled := c104.interval_sample(*pair, (first, last), *geometry)) is not None]
    c104.c98.require(len(whole) == 2, 'frozen two-anchor membership differs')
    extents = disjoint_native_extents(times, offsets, mid)
    limits = ((first, mid), (mid, last))
    spans, clock_spans, counts, halves = [], [], [], []
    reused = 0
    for pair, _ in whole:
        start = max(first, *(starts[i] for i in pair))
        end = min(last, *(ends[i] for i in pair))
        clock_spans.append([max(0., min(mid, end) - start), max(0., end - max(start, mid))])
        samples, pair_spans = [], []
        for bounds, (begin, finish) in zip(limits, extents, strict=True):
            left = max(bounds[0], *(begin[i] for i in pair))
            right = min(bounds[1], *(finish[i] for i in pair))
            pair_spans.append(max(0., right-left))
            samples.append(c104.interval_sample(*pair, bounds, times, xy, offsets, begin, finish))
        spans.append(pair_spans)
        counts.append([sample is not None for sample in samples])
        if all(sample is not None for sample in samples):
            early, late = samples
            reused += sum(len(bracket_indices(node, early[0], times, offsets) &
                              bracket_indices(node, late[0], times, offsets)) for node in pair)
            halves.append((early[1], late[1]))
    result = dict(anchors=[list(pair) for pair, _ in whole], duration=float(last-first),
                  half_spans=[[float(span) for span in pair] for pair in spans], half_samples=counts,
                  calendar_half_spans=[[float(span) for span in pair] for pair in clock_spans],
                  reused_observations=reused)
    if len(halves) != 2:
        return dict(result, reason='half_overlap' if min(min(pair) for pair in spans) < c104.CROSS_MIN_OVERLAP_S
                    else 'half_bracket')
    if reused:
        return dict(result, reason='reused_native_observation')
    differences = np.stack(halves, axis=1)
    centers = np.median(np.median(differences, axis=2), axis=1)
    errors = np.linalg.norm(differences - centers[::-1, None, None, :], axis=-1)
    radius = float(errors.max())
    result.update(radius=radius, directional_radius=errors.max(axis=(1, 2)).tolist(),
                  center_difference=float(np.linalg.norm(centers[0] - centers[1])))
    if radius > c104.ANCHOR_LIMIT:
        return dict(result, reason='cross_half_spread')
    distances = np.median(np.linalg.norm(displacement[None] - centers[:, None], axis=2), axis=1)
    full_shift = np.median([np.median(sample[1], axis=0) for _, sample in whole], axis=0)
    return dict(result, reason='registered', registered=max(0., float(distances.min())-radius),
                unregistered=float(np.median(np.linalg.norm(displacement, axis=1))),
                same_window_fit=float(np.median(np.linalg.norm(displacement-full_shift, axis=1))))


def freeze_scene(sid, motion, prior, archive, calibration, threshold):
    blocks, evidence = prior['blocks'], prior['evidence']
    old = motion['group_WS']
    groups, choices = c104.choose_scene(old, blocks, motion['support'], evidence, calibration)
    assert json.loads(json.dumps(choices)) == prior['choices'], 'fixed C104 decisions differ from the saved run'
    np.testing.assert_array_equal(groups['REG_MIX'], motion['group_REG'])
    strict = c104.c98.materialize(old, blocks, [c104.c98.choose(b, 'SUPPLIER') for b in blocks])
    np.testing.assert_array_equal(strict, motion['group_STRICT'])
    eligible = [row for row in evidence if exactly_two(row)]
    additions = {}
    if eligible:
        geometry = native_geometry(sid, motion, archive)
        _, source = np.unique(motion['sources'], return_inverse=True)
        pairs = np.column_stack(np.triu_indices(len(old), 1))
        anchors = c104.anchor_groups(old, pairs, source, motion['score_056'] - threshold)
        excluded_by_edge = {}
        for block in blocks:
            nodes = np.asarray(block['nodes'])
            for candidate in block['candidates'][1:]:
                if candidate['anchor_exact']:
                    for pair in c104.cut_pairs(old, nodes, motion['support'], candidate):
                        excluded_by_edge[pair] = set(block['nodes'])
        for row in eligible:
            pair = tuple(row['pair'])
            a, b = sorted(pair, key=lambda node: source[node])
            outside = [anchor for anchor in anchors[int(source[a]), int(source[b])]
                       if excluded_by_edge[pair].isdisjoint(anchor)]
            query, displacement = c104.interval_sample(a, b, (-np.inf, np.inf), *geometry)
            assert c104.cross_scores(query, displacement, outside, geometry) == {
                key: value for key, value in row.items() if key not in ('pair', 'same', 'exposure')}
            additions[pair] = temporal_evidence(query, displacement, outside, geometry)
    augmented = [dict(row, registered=additions[tuple(row['pair'])]['registered'])
                 if additions.get(tuple(row['pair']), {}).get('reason') == 'registered' else row
                 for row in evidence]
    alternatives, alternative_choices = c104.choose_scene(old, blocks, motion['support'], augmented, calibration)
    partition = alternatives['REG_MIX']
    return partition, choices, alternative_choices, additions


def candidate_audit(old, blocks, evidence, support, sources, ids, before, after, trials, calibration):
    ordinal = {tuple(row['pair']): index for index, row in enumerate(evidence)}
    rejected = {name: {tuple(pair) for pair in result['rejected']['REG_MIX']}
                for name, result in (('C104', before), ('REG', after))}
    counts, candidates = Counter(), []
    for k, block in enumerate(blocks):
        nodes = np.asarray(block['nodes'])
        labels = c104.c98.label_block(block, old, ids, support, sources)
        for j, item in enumerate(block['candidates'][1:], 1):
            counts['nonbase_candidates'] += 1
            if not item['anchor_exact']:
                continue
            counts['anchor_exact'] += 1
            positive = item['mixed_gain'] > c104.c98.EPS
            counts['anchor_exact_positive'] += positive
            cuts = c104.cut_pairs(old, nodes, support, item)
            known = labels[j]
            joint = known['qualifying'] and all(known['delta']['mstsa'][key] < 0 for key in ('fn', 'fp'))
            counts['safe_joint_positive'] += joint and positive
            admits = {name: set(cuts) <= denied for name, denied in rejected.items()}
            for name, admissible in admits.items():
                counts[f'{name}_admitted'] += admissible and positive
                counts[f'{name}_admitted_safe_joint'] += admissible and joint and positive
            if cuts and (any(pair in trials for pair in cuts) or (joint and positive)):
                candidates.append(dict(block=k, candidate=j, mixed_gain=item['mixed_gain'],
                    delta=known['delta'], same_ship_lost=known['same_ship_lost'],
                    admitted=admits, required=[dict(edge=ordinal[pair], same_ship=bool(ids[pair[0]] == ids[pair[1]]),
                        C104_rejected=pair in rejected['C104'], REG_rejected=pair in rejected['REG'],
                        original_reason=evidence[ordinal[pair]]['reason'], temporal_reason=trials.get(pair, {}).get('reason'))
                        for pair in cuts]))
    for name, decision in (('C104', before), ('REG', after)):
        counts[f'{name}_edited_blocks'] = sum(choice > 0 for choice in decision['selections']['REG_MIX'])
    rows = []
    for pair, trial in trials.items():
        row = evidence[ordinal[pair]]
        truth = bool(ids[pair[0]] == ids[pair[1]])
        counts[f'two_anchor_{"true" if truth else "false"}'] += 1
        counts[f'temporal_{trial["reason"]}'] += 1
        if trial['reason'] == 'registered':
            counts[f'temporal_eligible_{"true" if truth else "false"}'] += 1
            for name, field, cap in (('split', 'registered', 'registered'), ('raw', 'unregistered', 'raw'),
                                     ('same_window_fit', 'same_window_fit', 'registered')):
                counts[f'{name}_reject_{"true" if truth else "false"}'] += trial[field] > calibration[cap]['upper']
        public = {key: value for key, value in trial.items() if key != 'anchors'}
        rows.append(dict(edge=ordinal[pair], same_ship=truth, original_anchors=row['anchor_groups'],
                         original_anchor_pool=row['anchor_pool'], affected_slots=row['exposure'],
                         anchor_wrong=sum(ids[a] != ids[b] for a, b in trial['anchors']), **public))
    return dict(counts={key: int(value) for key, value in counts.items()}, edges=rows, candidates=candidates)


def run(args, start):
    manifest = read(args.manifest)
    roster, isolated, _ = c107.holdout(manifest)
    origin = args.runs / C107
    assert read(origin/'run.json')['status'] == 'COMPLETED'
    assert read(origin/'roster.json') == dict(new=roster, isolated=isolated, sealed_opened=False)
    assert read(args.runs/c107.C104/'run.json')['status'] == 'COMPLETED'
    calibration = read(args.runs/c107.C104/'calibration.json')
    assert {key: row['upper'] for key, row in calibration.items()} == dict(same=None, motion=4., raw=4205.475, registered=420.168)
    threshold = read(args.runs/c104.c98.C95/'summary.json')['fixed_thresholds']['056']
    archives = read(args.runs/c104.c98.DATA/'run.json')['archive']
    old_public = read(origin/'summary.json')
    private = args.output/'private'
    private.mkdir()
    nblocks = ncandidates = 0
    with MTADArchive(archives) as archive:
        c104.validate_measurement_archive(args.runs, archive)
        for sid in roster:
            with np.load(origin/'private'/f'{sid}.npz', allow_pickle=False) as saved:
                keys = ('node_ids', 'sources', 'states', 'support', 'nearest_seconds', 'native_count',
                        'group_WS', 'group_STRICT', 'group_REG', 'score_056')
                motion = {key: saved[key] for key in keys}
            prior = read(origin/'private'/f'{sid}.json')
            nblocks += len(prior['blocks'])
            ncandidates += sum(len(block['candidates'])-1 for block in prior['blocks'])
            partition, before, after, trials = freeze_scene(sid, motion, prior, archive, calibration, threshold)
            np.savez_compressed(private/f'{sid}.npz', partition=partition,
                prediction=np.asarray([item['states'] for item in c107.mean_groups(motion, partition)]).reshape(-1, 64, 4))
            write(private/f'{sid}.json', dict(before=before, after=after, trials=[
                  dict(pair=pair, **row) for pair, row in trials.items()]))
        reads = archive.members_read
        assert all(row['kind'] == 'measurements' and row['scene_id'] in roster for row in reads)
        read_scenes = [row['scene_id'] for row in reads]
        assert len(read_scenes) == len(set(read_scenes))
    assert (len(roster), nblocks, ncandidates) == REQUIRED
    write(args.output/'freeze.json', dict(scenes=len(roster), isolated=len(isolated), blocks=nblocks,
        nonbase_candidates=ncandidates, measurement_members=len(read_scenes), measurement_scenes=read_scenes,
        labels_read=0, references_read=0, evaluation_before_prediction=0, recalibrated=False, optimizer_updates=0))
    summary_state(args.output, start, status='RUNNING', phase='evaluation')
    scenes, matrices = [], []
    for sid in roster:
        with np.load(origin/'private'/f'{sid}.npz', allow_pickle=False) as saved, \
             np.load(private/f'{sid}.npz', allow_pickle=False) as new:
            motion = {key: saved[key] for key in ('states', 'support', 'sources', 'nearest_seconds')}
            groups = {key: saved[f'group_{key}'] for key in ('WS', 'STRICT')}
            groups.update(C104=saved['group_REG'], REG=new['partition'])
            np.testing.assert_array_equal(groups['WS'], groups['C104'])
            predicted = {key: saved[f'prediction_{key}'] for key in ('WS', 'STRICT')}
            predicted.update(C104=saved['prediction_REG'], REG=new['prediction'])
            outputs = {name: [dict(node_indices=group.tolist(), states=states)
                        for group, states in zip(c104.c98.group_nodes(groups[name]), predicted[name], strict=True)] for name in ARMS}
        with np.load(origin/'private'/f'{sid}-evaluation.npz', allow_pickle=False) as saved:
            ev = {key: saved[key] for key in ('target_ids', 'target_states', 'target_valid', 'identities')}
        prior = read(origin/'private'/f'{sid}.json')
        selected = read(private/f'{sid}.json')
        trials = {tuple(row['pair']): row for row in selected['trials']}
        fusion, matrix = c107.fusion(motion, groups, outputs, ev)
        audit = candidate_audit(groups['WS'], prior['blocks'], prior['evidence'], motion['support'], motion['sources'],
              ev['identities'], selected['before'], selected['after'], trials, calibration)
        scenes.append(dict(scene=sid, component=manifest['components'][sid], isolated=sid in isolated,
            clock_supported=sid not in old_public['all']['clock_unsupported'], exclusion=audit['counts'], audit=audit,
            association=c107.association_metrics(groups, motion['support'], motion['sources'], ev['identities']), fusion=fusion))
        matrices.append(matrix)
    summary = {}
    for name, indexes in (('all', list(range(len(scenes)))),
          ('isolated', [i for i, row in enumerate(scenes) if row['isolated']]),
          ('historically_shared', [i for i, row in enumerate(scenes) if not row['isolated']])):
        result = c107.summarize([scenes[i] for i in indexes], [matrices[i] for i in indexes], arms=ARMS, baseline='C104')
        baseline = old_public[name]
        for arm, old_arm in (('WS', 'WS'), ('STRICT', 'STRICT'), ('C104', 'REG')):
            assert result['association'][arm] == baseline['association'][old_arm]
            assert result['fusion']['coverage'][arm] == baseline['fusion']['coverage'][old_arm]
            for attribute in c107.FEATURES:
                assert result['fusion']['errors']['all'][attribute]['own'][arm] == baseline['fusion']['errors']['all'][attribute]['own'][old_arm]
        result['joint_pass'] = result['fusion_pass'] and not result['fusion']['nodes']['added_wrong']
        summary[name] = result
    write(args.output/'summary.json', summary)
    write(args.output/'two-anchor-evidence.json', [dict(scene=row['scene'], isolated=row['isolated'], **row['audit'])
          for row in scenes if row['audit']['edges'] or row['audit']['candidates']])
    with gzip.open(args.output/'scenes.json.gz', 'wt') as stream:
        json.dump(scenes, stream, separators=(',', ':'), allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args, start)
    except Exception as error:
        summary_state(args.output, start, status='FAILED', error=repr(error))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, model_forwards=0, test_opened=False)


if __name__ == '__main__':
    main()
