"""C134: isolate continuous native distances under fixed C133 evidence quotas."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from tools.analysis import fusion_native_donor_evidence as c133
from tools.analysis.offline_association_tail import read_motion, summary_state, write


PRIOR = 'LEVEL1-OFFLINE-NATIVE-DONOR-EVIDENCE-20261113-R1-attempt-1'
CONTROL, DUAL, BRIDGE, BOTH = ARMS = ('AVAILABILITY', 'DUAL_DISTANCE', 'BRIDGE_DISTANCE', 'BOTH_DISTANCE')
TYPES = ('dual', 'bridge', 'unavailable')
CELLS = [(source, kind) for source in ('same', 'cross') for kind in TYPES]
RERANK = {CONTROL: (), DUAL: ('dual',), BRIDGE: ('bridge',), BOTH: ('dual', 'bridge')}


def evidence_type(evidence):
    if len(evidence['witness']) == 2:
        return 'dual', min(evidence['witness'])
    if evidence['bridge_m'] is not None:
        return 'bridge', evidence['bridge_m']
    return 'unavailable', None


def select(rows, j_order, original):
    assert j_order[:len(original)] == original
    by_cell = defaultdict(list)
    scores = {}
    cells = {}
    for i in j_order:
        row = rows[i]
        kind, distance = evidence_type(row['evidence'])
        cell = ('same' if row['same_source'] else 'cross', kind)
        by_cell[cell].append(i)
        cells[i] = cell
        if distance is not None:
            assert np.isfinite(distance) and distance >= 0.
            scores[i] = distance
    quota = Counter(cells[i] for i in original)
    selected = {arm: [] for arm in ARMS}
    for cell in CELLS:
        ranked = by_cell[cell]
        reranked = sorted(ranked, key=scores.__getitem__) if cell[1] != 'unavailable' else ranked
        for arm in ARMS:
            order = reranked if cell[1] in RERANK[arm] else ranked
            selected[arm].extend(order[:quota[cell]])
    rank = {i: j for j, i in enumerate(j_order)}
    for arm in ARMS:
        selected[arm].sort(key=rank.__getitem__)
        assert len(selected[arm]) == len(set(selected[arm])) == len(original)
        assert Counter(cells[i] for i in selected[arm]) == quota
    assert selected[CONTROL] == original
    cell_summary = {'/'.join(cell): dict(pool=len(by_cell[cell]), quota=quota[cell],
                    exchangeable=min(quota[cell], len(by_cell[cell])-quota[cell]),
                    swapped={arm: sum(cells[i] == cell for i in set(selected[arm])-set(original)) for arm in ARMS})
                    for cell in CELLS}
    return selected, cell_summary


def read_candidates(runs, sid, saved, data, thresholds):
    c130, c127 = c133.c130, c133.c130.c127
    c98 = c127.c98
    read = c127.read
    rows = saved['rows']
    if not rows:
        return []
    motion = read_motion(data[sid])
    with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz', allow_pickle=False) as base, \
         np.load(runs/c98.C95/'private/held'/f'{sid}.npz', allow_pickle=False) as scores:
        original = np.load(runs/c133.C127/'private'/f'{sid}-selected.npy', allow_pickle=False)[0]
        _, candidates = c130.visible_candidates(motion, base, scores,
            read(runs/c127.c104.C98/'private/held'/f'{sid}.json.gz'),
            read(runs/c127.c107.C104/'private/held'/f'{sid}-choices.json'), thresholds, original)
        protected = set(saved['protected'])
        fixed = [r for r in candidates if r['group'] not in protected]
        assert len(fixed) == len(rows)
        for row, r in zip(rows, fixed, strict=True):
            for key in ('group', 'q', 'donor', 'donor_group', 'same_source'):
                assert row[key] == r[key]
            assert r['donor_group'] in protected
            group, q, donor = (r[k] for k in ('group', 'q', 'donor'))
            old, new = base['KEEP'][group,q,:2], motion['states'][donor,q,:2]
            assert np.isfinite(old).all() and np.isfinite(new).all() and np.any(old != new)
    return sorted(range(len(fixed)), key=lambda i: -fixed[i]['gain'])


def freeze(args, scenes, data, thresholds):
    read = c133.c130.c127.read
    private = args.output/'private'
    private.mkdir()
    counts = defaultdict(Counter)
    by_scene = {}
    protected_groups = 0
    for sid in scenes:
        original = read(args.runs/PRIOR/'private'/f'{sid}.json')
        order = read_candidates(args.runs, sid, original, data, thresholds)
        rows = original['rows']
        assert len(order) == len(rows)
        selected, cells = select(rows, order, original['selected']['J'])
        protected_groups += len(original['protected'])
        for key, cell in cells.items():
            counts[key].update({k: cell[k] for k in ('pool', 'quota', 'exchangeable')})
            counts[key].update({'swapped_'+arm: n for arm, n in cell['swapped'].items()})
        write(private/f'{sid}.json', dict(selected=selected))
        by_scene[sid] = cells
    assert sum(c['pool'] for c in counts.values()) == 4133
    assert sum(c['quota'] for c in counts.values()) == 2090
    assert protected_groups == 290
    write(args.output/'freeze.json', dict(scenes=len(scenes), protected_scene_groups=protected_groups,
          cells=counts, evaluation_before_freeze=False, native_archive_reads=0))
    return by_scene


def run(args):
    runs, out = args.runs, args.output
    c130, c127 = c133.c130, c133.c130.c127
    read = c127.read
    c98 = c127.c98
    prior = runs/PRIOR
    assert read(prior/'run.json')['status'] == 'COMPLETED'
    scenes = read(runs/c98.PREP/'frozen-split.json')['held']
    manifest = read(args.manifest)
    assert len(set(scenes)) == 177 and set(scenes) <= set(manifest['splits']['T_core'])
    data = read(runs/c98.DATA/'private/index.json')['T_core']
    thresholds = read(runs/c98.C95/'summary.json')['fixed_thresholds']
    by_scene = freeze(args, scenes, data, thresholds)

    previous = read(prior/'summary.json')
    previous_scenes = dict(zip(scenes, read(prior/'scenes.json'), strict=True))
    threshold = previous['baseline_threshold_m']
    totals = defaultdict(Counter)
    components = defaultdict(lambda: defaultdict(Counter))
    errors_by_component = defaultdict(list)
    all_errors, all_events, scene_rows = [], [], []
    for sid in scenes:
        component = manifest['components'][sid]
        original = read(prior/'private'/f'{sid}.json')
        original['selected'] = read(out/'private'/f'{sid}.json')['selected']
        motion = read_motion(data[sid])
        with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz', allow_pickle=False) as base, \
             np.load(data[sid]['evaluation'], allow_pickle=False) as truth, \
             np.load(runs/c98.C95/'private/held'/f'{sid}.npz', allow_pickle=False) as scores:
            groups, _, reach = c127.frozen_scene(motion, base, scores,
                read(runs/c127.c104.C98/'private/held'/f'{sid}.json.gz'),
                read(runs/c127.c107.C104/'private/held'/f'{sid}-choices.json'), thresholds)
            errors, events, scene = c133.evaluate(sid, motion, base, truth, groups, original, reach, threshold)
        old = previous_scenes[sid]
        assert scene['coverage'] == old['coverage'] and component == old['component']
        assert scene['counts'][CONTROL] == old['counts']['J']
        for arm, counts in scene['counts'].items():
            totals[arm].update(counts)
            components[component][arm].update(counts)
        all_errors.append(errors)
        errors_by_component[component].append(errors)
        all_events.extend(events)
        scene_rows.append(dict(scene, scene=sid, component=component, cells=by_scene[sid]))

    def metrics(arrays):
        error = np.concatenate(arrays, axis=1)
        return {arm: c130.STATS(error[i]) for i, arm in enumerate(ARMS)}

    result = metrics(all_errors)
    for key, value in previous['metrics']['J'].items():
        np.testing.assert_allclose(result[CONTROL][key], value, rtol=0., atol=2e-8)
    coverage = dict(zip(('lat', 'lon', 'speed', 'heading'), np.sum([s['coverage'] for s in scene_rows], axis=0).tolist(), strict=True))
    assert coverage == previous['coverage']
    assert len(components) == 4
    leave_one_out = {removed: metrics([a for key, arrays in errors_by_component.items() if key != removed for a in arrays])
                     for removed in errors_by_component}
    write(out/'summary.json', dict(scenes=len(scenes), coverage=coverage, metrics=result, totals=totals, components=components,
          leave_one_component_out=leave_one_out, baseline_threshold_m=threshold, opportunity_slots=4133))
    write(out/'scenes.json', scene_rows)
    write(out/'events.json', all_events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args)
    except Exception as error:
        summary_state(args.output, start, status='FAILED', error=repr(error))
        raise
    summary_state(args.output, start, status='COMPLETED', native_archive_reads=0, optimizer_updates=0,
                  model_forwards=0, test_opened=False)


if __name__ == '__main__':
    main()
