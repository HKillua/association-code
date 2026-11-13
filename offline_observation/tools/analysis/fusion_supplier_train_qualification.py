"""C121: qualify newly grouped position observations on the frozen training-side split."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from tools.analysis import association_registered_holdout_fusion as c107
from tools.analysis import association_registered_supplier_exclusion as c104
from tools.analysis import fusion_added_supplier_attributes as c118
from tools.analysis.disjoint_partition_motion_evidence import _coordinates
from tools.analysis.held_fusion_population import _canonical_tracks, validate_measurement_archive
from tools.analysis.offline_association_tail import read_motion, summary_state, write
from tools.analysis.same_source_swap_pilot import read
from tools.analysis.supplier_external_geometry import _sample_at


ARMS = ('MEAN', 'KEEP', 'C118_POSITION', 'NATIVE_AGE')
BENEFIT_EPS = 1e-6


def native_fields(motion, old, current, tracks, margins):
    support, q = motion['support'], motion['query_seconds']
    size = len(old)
    positions = np.concatenate([row['canonical_values'][:, :2] for row in tracks])
    times = np.concatenate([row['canonical_times'] for row in tracks])
    lengths = [len(row['canonical_times']) for row in tracks]
    np.testing.assert_array_equal(lengths, motion['native_count'])
    offsets = np.r_[0, np.cumsum(lengths)]
    _, xy, _ = _coordinates(positions, offsets)
    local, bracket = _sample_at(np.arange(size), q, times, xy, offsets)
    age = np.full_like(support, np.inf, dtype=float)
    for node, (left, right) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        t = times[left:right]
        hi = np.searchsorted(t, q).clip(0, len(t)-1)
        lo = (hi-1).clip(0)
        age[node] = np.minimum(abs(q-t[lo]), abs(q-t[hi]))
    sources = np.unique(motion['sources'], return_inverse=True)[1]
    pairs = np.column_stack(np.triu_indices(size, 1))
    anchors = c104.anchor_groups(old, pairs, sources, margins)
    selected = support.copy()
    eligible, timed = np.zeros_like(support), np.zeros_like(support)
    original_gap = np.full(support.shape, np.nan)
    registered_gap = np.full(support.shape, np.nan)
    anchor_uses = []
    for group in c104.c98.group_nodes(current):
        first = group[0]
        retained = group[old[group] == old[first]]
        additions = group[old[group] != old[first]]
        if not len(additions):
            continue
        original_active = support[retained].any(axis=0)
        original_bracket = (support[retained] & bracket[retained]).any(axis=0)
        nearest_original = np.where(support[retained], age[retained], np.inf).min(axis=0)
        excluded = set(old[group])
        for donor in additions:
            cross = sources[donor] != sources[first]
            eligible[donor] = support[donor] & original_bracket & bracket[donor] & cross
            timed[donor] = eligible[donor] & (age[donor] < nearest_original)
            selected[donor] = support[donor] & (~original_active | timed[donor])
            if not eligible[donor].any():
                continue
            a, b = sorted((sources[first], sources[donor]))
            available = [(i, j) if a == sources[first] else (j, i)
                         for i, j in anchors[a, b] if old[i] not in excluded and old[j] not in excluded]
            for point in np.flatnonzero(eligible[donor] & support[first] & bracket[first]):
                active = [(i, j) for i, j in available if bracket[i, point] and bracket[j, point]]
                if len(active) < 3:
                    continue
                i, j = np.array(active).T
                shifts = local[j, point]-local[i, point]
                centers = np.stack([np.median(np.delete(shifts, k, axis=0), axis=0)
                                    for k in range(len(shifts))])
                if np.linalg.norm(shifts-centers, axis=1).max() > c104.ANCHOR_LIMIT:
                    continue
                delta = local[donor, point]-local[first, point]
                original_gap[donor, point] = np.linalg.norm(delta)
                registered_gap[donor, point] = np.linalg.norm(delta-np.median(shifts, axis=0))
                anchor_uses.extend((donor, point, i, j) for i, j in active)
    for group in c104.c98.group_nodes(current):
        np.testing.assert_array_equal(selected[group].any(0), support[group].any(0))
    return dict(selected=selected, eligible=eligible, timed=timed, original_gap=original_gap,
                registered_gap=registered_gap, anchor_uses=np.array(anchor_uses, dtype=int).reshape(-1, 4))


def freeze(motion, old, current, archive, sid, score):
    prior = c118.predict(motion, old, current)
    mean = prior['MEAN']
    keep, c118_position, native = (mean.copy() for _ in range(3))
    keep[..., :2] = prior['KEEP_ORIGINAL'][..., :2]
    c118_position[..., :2] = prior['ATTRIBUTE'][..., :2]
    evidence = dict(eligible=np.zeros_like(motion['support']), timed=np.zeros_like(motion['support']),
                    original_gap=np.full(motion['support'].shape, np.nan),
                    registered_gap=np.full(motion['support'].shape, np.nan),
                    anchor_uses=np.empty((0, 4), int))
    changed = not np.array_equal(old, current)
    if changed:
        tracks = _canonical_tracks(archive._rows(sid, 'measurements'), sid, motion['node_ids'], motion['sources'])
        evidence = native_fields(motion, old, current, tracks, score)
        native[..., :2] = c118.states_for(motion, current, evidence.pop('selected'))[..., :2]
    return dict(MEAN=mean, KEEP=keep, C118_POSITION=c118_position, NATIVE_AGE=native, changed=changed,
                current=current, node_ids=motion['node_ids'], query=motion['query_seconds'],
                active_new=prior['active_new'], old_active=prior['old_active'], **evidence)


def evaluate(sid, motion, saved, truth):
    for key, value in (('node_ids', truth['node_ids']), ('query', truth['query_seconds'])):
        np.testing.assert_array_equal(saved[key], value)
    current = saved['current']
    outputs = {arm: c118.emit(current, saved[arm]) for arm in ARMS}
    reports, detail = evaluate_fusion_methods(outputs, prediction_speed_units=dict.fromkeys(ARMS, 'kn'),
        node_identities=truth['identities'], target_ids=truth['target_ids'], target_states=truth['target_states'],
        target_valid=truth['target_valid'], support=motion['support'], nearest_seconds=motion['nearest_seconds'],
        target_speed_unit='mps', split='T_core')
    nt = len(truth['target_ids'])
    base = detail['MEAN']
    errors = {}
    for arm, row in detail.items():
        for key in ('aligned_group_indices', 'aligned_prediction_valid'):
            np.testing.assert_array_equal(row[key], base[key])
        np.testing.assert_array_equal(row['aligned_prediction'][..., 2:], base['aligned_prediction'][..., 2:])
        errors[arm] = c107.feature_errors(row['aligned_prediction'][:nt], truth['target_states'],
                     row['aligned_prediction_valid'][:nt] & truth['target_valid'])
    matrix = {key: np.stack([errors[arm][key] for arm in ARMS], axis=-1) for key in c107.FEATURES}
    group_nodes = c104.c98.group_nodes(current)
    ids = truth['identities']
    counts, points = Counter(), []
    eligible, timed = saved['eligible'], saved['timed']
    gaps, registered = saved['original_gap'], saved['registered_gap']
    anchors = defaultdict(list)
    for donor, point, i, j in saved['anchor_uses']:
        anchors[donor, point].append(ids[i] == ids[j])
    for target, gi in enumerate(base['aligned_group_indices'][:nt]):
        if gi < 0:
            continue
        nodes = group_nodes[gi]
        new, original = saved['active_new'][nodes], saved['old_active'][nodes]
        for point in np.flatnonzero(new.any(0)):
            observed = nodes[new[:, point]]
            same = ids[observed] == ids[nodes[0]]
            old_here = bool(original[:, point].any())
            counts['new_target_q'] += 1
            counts['old_new_target_q'] += old_here
            counts['only_new_target_q'] += not old_here
            e = matrix['position'][target, point]
            donor_rows = []
            for donor, same_id in zip(observed, same, strict=True):
                exact = anchors.get((donor, point), [])
                donor_rows.append(dict(same=bool(same_id), native_eligible=bool(eligible[donor, point]),
                    native_selected=bool(timed[donor, point]), anchors=len(exact), wrong_anchors=int(len(exact)-sum(exact)),
                    raw_gap_m=float(gaps[donor, point]) if exact else None,
                    registered_gap_m=float(registered[donor, point]) if exact else None))
            points.append(dict(scene=sid, target=target, q=int(point), old_present=old_here, donors=donor_rows,
                               error_m=e.tolist() if np.isfinite(e).all() else None))
    coverage = {name: {key: value['coverage'] for key, value in report['paper_fusion_metrics']['attributes'].items()}
                for name, report in reports.items()}
    return matrix, dict(scene=sid, changed=bool(saved['changed']), targets=nt, coverage=coverage,
                        groups=len(group_nodes), counts=dict(counts)), points


def stats(values):
    common = values[np.isfinite(values).all(axis=-1)]
    return {arm: c107.attribute_stats(common[:, i]) for i, arm in enumerate(ARMS)}


def local_stats(rows):
    values = np.array([row['error_m'] for row in rows if row['error_m'] is not None]).reshape(-1, len(ARMS))
    delta = values[:, 0]-values[:, 1]
    donor_rows = [donor for row in rows for donor in row['donors']]
    valid = [row for row in rows if row['error_m'] is not None]
    helpful = [row for row in valid if row['error_m'][0] < row['error_m'][1]-BENEFIT_EPS]
    harmful = [row for row in valid if row['error_m'][0] > row['error_m'][1]+BENEFIT_EPS]
    return dict(q=len(rows), valid=len(valid), metrics=stats(values),
                helpful=len(helpful), harmful=len(harmful), ties=int((abs(delta) <= BENEFIT_EPS).sum()),
                helpful_over_250=int((delta < -250).sum()), harmful_over_250=int((delta > 250).sum()),
                native_eligible_slots=sum(row['native_eligible'] for row in donor_rows),
                native_selected_slots=sum(row['native_selected'] for row in donor_rows),
                native_selected_on_help=sum(any(d['native_selected'] for d in row['donors']) for row in helpful),
                native_selected_on_harm=sum(any(d['native_selected'] for d in row['donors']) for row in harmful),
                anchor_available_slots=sum(bool(row['anchors']) for row in donor_rows),
                anchor_uses=sum(row['anchors'] for row in donor_rows),
                wrong_anchor_uses=sum(row['wrong_anchors'] for row in donor_rows),
                anchor_reduced_gap_slots=sum(row['raw_gap_m'] > row['registered_gap_m']
                                             for row in donor_rows if row['anchors']),
                oracle=c107.attribute_stats(values[:, :2].min(axis=1)))


def run(args):
    runs, out = args.runs, args.output
    manifest = read(args.manifest)
    old_run, new_run, score_run, data_run = [runs / path for path in
        (c104.C98, c107.C104, c104.c98.C95, c104.c98.DATA)]
    assert all(read(path/'run.json')['status'] == 'COMPLETED' for path in (old_run, new_run, score_run))
    roles = read(runs/c104.c98.PREP/'frozen-split.json')
    scenes, components = roles['held'], manifest['components']
    assert len(scenes) == 177 and set(scenes) <= set(manifest['splits']['T_core'])
    assert {components[s] for s in scenes}.isdisjoint(components[s] for s in roles['calibration'])
    index = read(data_run/'private/index.json')['T_core']
    threshold = read(score_run/'summary.json')['fixed_thresholds']['056']
    private = out/'private'
    private.mkdir()
    with MTADArchive(read(data_run/'run.json')['archive']) as archive:
        validate_measurement_archive(runs, archive)
        for sid in scenes:
            motion = read_motion(index[sid])
            with np.load(old_run/'private/held'/f'{sid}.npz', allow_pickle=False) as a, \
                 np.load(new_run/'private/held'/f'{sid}.npz', allow_pickle=False) as b:
                old, current = a['WS'], b['REG_MIX']
            score = None
            if not np.array_equal(old, current):
                with np.load(score_run/'private/held'/f'{sid}.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['group_056'], old)
                    np.testing.assert_array_equal(data['pairs'], np.column_stack(np.triu_indices(len(old), 1)))
                    score = data['score_056']-threshold
            np.savez_compressed(private/f'{sid}.npz', **freeze(motion, old, current, archive, sid, score))
        assert len(archive.members_read) == 7
        assert all(row['kind'] == 'measurements' and row['scene_id'] in scenes for row in archive.members_read)
        write(out/'freeze.json', dict(scenes=len(scenes),
                                     native_measurement_scenes=[row['scene_id'] for row in archive.members_read],
                                     all_predictions_before_evaluation=True, raw_reference_reads=0))
    blocks, scene_rows, points = [], [], []
    for sid in scenes:
        with np.load(private/f'{sid}.npz', allow_pickle=False) as saved, \
             np.load(index[sid]['evaluation'], allow_pickle=False) as truth:
            block, row, active = evaluate(sid, read_motion(index[sid]), saved, truth)
        blocks.append(block)
        scene_rows.append(row)
        points.extend(active)
    mixed = [row for row in points if row['old_present']]
    true = [row for row in mixed if all(d['same'] for d in row['donors'])]
    by_component = defaultdict(list)
    for row in true:
        by_component[components[row['scene']]].append(row)
    write(out/'summary.json', dict(scenes=len(scenes), components=len({components[s] for s in scenes}),
        metrics={key: stats(np.concatenate([row[key] for row in blocks])) for key in c107.FEATURES},
        all_new=local_stats(points), both=local_stats(mixed), both_true=local_stats(true),
        both_contains_wrong=local_stats([row for row in mixed if any(not d['same'] for d in row['donors'])]),
        only_new=local_stats([row for row in points if not row['old_present']]),
        both_true_components={key: local_stats(rows) for key, rows in by_component.items()},
        scene_rows=scene_rows))
    write(out/'points.json', points)


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
