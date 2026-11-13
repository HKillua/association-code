"""C135: test a local position check without undoing the fixed identity quotas."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import fusion_continuous_witness as c134
from tools.analysis.offline_association_tail import read_motion, summary_state, write


PRIOR = 'LEVEL1-OFFLINE-CONTINUOUS-WITNESS-20261113-attempt-1'
J, DUAL, STRICT, AVOID = ARMS = ('J', 'DUAL_DISTANCE', 'LOCAL_STRICT', 'LOCAL_AVOID')
c133 = c134.c133
c130 = c133.c130
c127 = c130.c127
read = c127.read


def position_class(excess):
    if excess is None:
        return 'unavailable'
    if excess > c133.HARM_M:
        return 'risk'
    if excess < -c133.HARM_M:
        return 'benefit'
    return 'neutral'


def local_evidence(rows, motion, base, tracks):
    lengths = [len(track['canonical_times']) for track in tracks]
    np.testing.assert_array_equal(lengths, motion['native_count'])
    offsets = np.r_[0, np.cumsum(lengths)]
    _, xy, scale = c133._coordinates(np.concatenate([t['canonical_values'][:, :2] for t in tracks]), offsets)
    points = np.split(xy, offsets[1:-1])
    times = [track['canonical_times'] for track in tracks]
    groups = c127.c98.group_nodes(base['current'])
    local = []
    for r in rows:
        group, q, donor = (r[key] for key in ('group', 'q', 'donor'))
        when = motion['query_seconds'][q]
        dt = times[donor]
        actual = c133.bracket(dt, when, max_gap=np.inf)
        assert actual and motion['support'][donor,q] and not motion['support'][groups[group],q].any()
        native = tracks[donor]['canonical_values'][:, :2]
        borrowed = motion['states'][donor,q,:2]
        geo = native[actual[0]].copy()
        if len(actual) == 2:
            a,b = actual
            delta = native[b]-native[a]
            delta[1] = (delta[1]+180.) % 360.-180.
            geo += ((when-dt[a])/(dt[b]-dt[a]))*delta
        offset = borrowed-geo
        offset[1] = (offset[1]+180.) % 360.-180.
        np.testing.assert_allclose(offset, 0., rtol=0., atol=2e-9)
        new = c133.at(dt, points[donor], when, actual)+offset*scale
        old_offset = base['KEEP'][group,q,:2]-borrowed
        old_offset[1] = (old_offset[1]+180.) % 360.-180.
        old = new+old_offset*scale
        span = float(dt[actual[-1]]-dt[actual[0]])
        outer = c133.bracket(dt, when, actual) if span <= c133.SHORT_S else ()
        assert not set(actual).intersection(outer)
        disagreement = float(np.linalg.norm(c133.at(dt, points[donor], when, outer)-new)) if outer else None
        endpoints = []
        for node in groups[group]:
            t = times[node]
            assert when < t[0] or when > t[-1]
            before = when > t[-1]
            prediction, check = c133.endpoint(t, points[node], before, when)
            if check is None or check > c133.HARM_M:
                continue
            assert node != donor
            chosen = (len(t)-1, len(t)-2, len(t)-3) if before else (0,1,2)
            excess = float(np.linalg.norm(new-prediction)-np.linalg.norm(old-prediction))
            endpoints.append((abs(when-t[chosen[0]]), int(node), chosen, check, excess))
        closest = min(endpoints) if endpoints else None
        excess = closest[-1] if closest is not None else None
        for actual_value, previous in ((span, r['evidence']['donor_bracket_s']),
                                      (disagreement, r['evidence']['donor_leave_bracket_m']),
                                      (excess, r['evidence']['receiver_excess_m'])):
            if previous is None:
                assert actual_value is None
            else:
                np.testing.assert_allclose(actual_value, previous, rtol=0., atol=1e-7)
        local.append(dict(position_class=position_class(excess), receiver_excess_m=excess,
            receiver_rows=(closest[1], closest[2]) if closest is not None else None,
            receiver_is_anchor=bool(closest[1] == groups[group][0]) if closest is not None else None,
            receiver_age_s=float(closest[0]) if closest is not None else None,
            receiver_step_check_m=closest[3] if closest is not None else None,
            donor_actual_rows=actual, donor_outer_rows=outer,
            donor_actual_offsets_s=(dt[list(actual)]-when).tolist(),
            donor_outer_offsets_s=(dt[list(outer)]-when).tolist(),
            donor_disagreement_m=disagreement))
    return local


def choose(rows, local, order, j, dual):
    rank = {i:k for k,i in enumerate(order)}
    classes = [p['position_class'] for p in local]
    cells = {}
    dual_dist = {}
    for i,r in enumerate(rows):
        kind, distance = c134.evidence_type(r['evidence'])
        cells[i] = ('same' if r['same_source'] else 'cross',kind)
        if kind == 'dual':
            dual_dist[i] = distance
    quota = Counter(cells[i] for i in j)
    assert Counter(cells[i] for i in dual) == quota
    dual_set = set(dual)
    chosen = {J:j, DUAL:dual, STRICT:dual_set.copy(), AVOID:dual_set.copy()}
    offered, swaps = {}, {}
    for cell in c134.CELLS:
        key = '/'.join(cell)
        inside = [i for i in dual if cells[i] == cell and classes[i] == 'risk' and i in dual_dist]
        outside = [i for i in order if i not in dual_set and cells[i] == cell and i in dual_dist]
        # Best identity support is retained when there is more than one candidate to move.
        inside.sort(key=lambda i:(-dual_dist[i], -rank[i]))
        outside.sort(key=lambda i:(dual_dist[i], rank[i]))
        offered[key] = dict(quota=quota[cell], risk_selected=len(inside),
                            benefit_available=sum(classes[i] == 'benefit' for i in outside),
                            without_risk_available=sum(classes[i] != 'risk' for i in outside))
        swaps[key] = {}
        for arm, allow in ((STRICT, ('benefit',)), (AVOID, ('benefit','neutral','unavailable'))):
            replace = [i for i in outside if classes[i] in allow][:len(inside)]
            chosen[arm].difference_update(inside[:len(replace)])
            chosen[arm].update(replace)
            swaps[key][arm] = len(replace)
    for arm in (STRICT,AVOID):
        chosen[arm] = sorted(chosen[arm], key=rank.__getitem__)
    for arm in ARMS:
        assert len(chosen[arm]) == len(set(chosen[arm])) == len(j)
        assert Counter(cells[i] for i in chosen[arm]) == quota
    return chosen, offered, swaps


def freeze(args, scenes, data, thresholds):
    runs, out = args.runs, args.output
    private = out/'private'
    private.mkdir()
    totals, populations = Counter(), defaultdict(Counter)
    evidence_stats, protected_groups = {}, 0
    c98 = c127.c98
    with MTADArchive(read(runs/c98.DATA/'run.json')['archive']) as archive:
        c133.validate_measurement_archive(runs, archive)
        for sid in scenes:
            prior = read(runs/c134.PRIOR/'private'/f'{sid}.json')
            previous = read(runs/PRIOR/'private'/f'{sid}.json')['selected']
            order = c134.read_candidates(runs, sid, prior, data, thresholds)
            rows = prior['rows']
            local = []
            if rows:
                motion = read_motion(data[sid])
                tracks = c133._canonical_tracks(archive._rows(sid,'measurements'), sid, motion['node_ids'], motion['sources'])
                with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz', allow_pickle=False) as base:
                    local = local_evidence(rows, motion, base, tracks)
            j, dual = previous[c134.CONTROL], previous[c134.DUAL]
            assert j == prior['selected']['J']
            selected, cells, swaps = choose(rows, local, order, j, dual)
            for arm in ('pool',*ARMS):
                for i in (range(len(rows)) if arm == 'pool' else selected[arm]):
                    kind, _ = c134.evidence_type(rows[i]['evidence'])
                    t = local[i]
                    populations[arm].update((kind+'_'+t['position_class'], 'all'))
                    populations[arm]['donor_outer_available'] += bool(t['donor_outer_rows'])
                    populations[arm]['receiver_anchor_available'] += t['receiver_is_anchor'] is True
                    populations[arm]['receiver_non_anchor_available'] += t['receiver_is_anchor'] is False
            protected_groups += len(prior['protected'])
            totals.update(pool=len(rows), actions=len(j))
            evidence_stats[sid] = dict(cells=cells, swaps=swaps)
            write(private/f'{sid}.json', dict(selected=selected, local=local))
        members = [r['scene_id'] for r in archive.members_read]
    assert len(members) == len(set(members)) == 85 and set(members) <= set(scenes)
    assert totals == dict(pool=4133,actions=2090) and protected_groups == 290
    write(out/'freeze.json', dict(scenes=len(scenes), measurement_scenes=members, protected_scene_groups=protected_groups,
          totals=totals, populations=populations, evaluation_before_freeze=False))
    return evidence_stats


def run(args):
    runs, out, c98 = args.runs, args.output, c127.c98
    prior = runs/PRIOR
    assert read(prior/'run.json')['status'] == 'COMPLETED'
    scenes = read(runs/c98.PREP/'frozen-split.json')['held']
    manifest = read(args.manifest)
    assert len(set(scenes)) == 177 and set(scenes) <= set(manifest['splits']['T_core'])
    data = read(runs/c98.DATA/'private/index.json')['T_core']
    thresholds = read(runs/c98.C95/'summary.json')['fixed_thresholds']
    audit = freeze(args, scenes, data, thresholds)
    previous = read(prior/'summary.json')
    old_scenes = {d['scene']:d for d in read(prior/'scenes.json')}
    threshold = previous['baseline_threshold_m']
    totals = defaultdict(Counter)
    components = defaultdict(lambda:defaultdict(Counter))
    errors_by_component = defaultdict(list)
    events, scene_rows = [], []
    for sid in scenes:
        component = manifest['components'][sid]
        original = read(runs/c134.PRIOR/'private'/f'{sid}.json')
        saved = read(out/'private'/f'{sid}.json')
        original['selected'] = saved['selected']
        for r, p in zip(original['rows'],saved['local'],strict=True):
            r['evidence'].update(local_position_class=p['position_class'], local_receiver_age_s=p['receiver_age_s'],
                                 local_receiver_step_check_m=p['receiver_step_check_m'],
                                 local_receiver_is_anchor=p['receiver_is_anchor'])
        motion = read_motion(data[sid])
        with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz', allow_pickle=False) as base, \
             np.load(data[sid]['evaluation'], allow_pickle=False) as truth, \
             np.load(runs/c98.C95/'private/held'/f'{sid}.npz', allow_pickle=False) as scores:
            groups, _, reach = c127.frozen_scene(motion, base, scores,
                read(runs/c127.c104.C98/'private/held'/f'{sid}.json.gz'),
                read(runs/c127.c107.C104/'private/held'/f'{sid}-choices.json'), thresholds)
            error, rows, scene = c133.evaluate(sid, motion, base, truth, groups, original, reach, threshold)
        old = old_scenes[sid]
        assert scene['coverage'] == old['coverage'] and component == old['component']
        assert scene['counts'][J] == old['counts'][c134.CONTROL]
        assert scene['counts'][DUAL] == old['counts'][c134.DUAL]
        for key,cell in old['cells'].items():
            assert cell['quota'] == audit[sid]['cells'][key]['quota']
        for arm, count in scene['counts'].items():
            totals[arm].update(count)
            components[component][arm].update(count)
        errors_by_component[component].append(error)
        events.extend(rows)
        scene_rows.append(dict(scene, scene=sid, component=component, **audit[sid]))
    coverage = dict(zip(('lat','lon','speed','heading'),np.sum([s['coverage'] for s in scene_rows], axis=0).tolist(),strict=True))
    assert coverage == previous['coverage'] and len(components) == 4

    def metrics(arrays):
        errors = np.concatenate(arrays,axis=1)
        return {arm:c130.STATS(errors[i]) for i,arm in enumerate(ARMS)}

    metrics_all = metrics([a for arrays in errors_by_component.values() for a in arrays])
    for arm, name in ((J,c134.CONTROL),(DUAL,c134.DUAL)):
        for key, val in previous['metrics'][name].items():
            np.testing.assert_allclose(metrics_all[arm][key],val,rtol=0.,atol=2e-8)
    leave_one = {k:metrics([a for g,arrays in errors_by_component.items() if k != g for a in arrays])
                 for k in errors_by_component}
    write(out/'summary.json', dict(scenes=len(scenes),coverage=coverage,metrics=metrics_all,totals=totals,
          components=components,leave_one_component_out=leave_one,baseline_threshold_m=threshold,opportunity_slots=4133))
    write(out/'events.json',events)
    write(out/'scenes.json',scene_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs','manifest','output'):
        parser.add_argument(f'--{key}',type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args)
    except Exception as error:
        summary_state(args.output,start,status='FAILED',error=repr(error))
        raise
    summary_state(args.output,start,status='COMPLETED',native_archive_reads=85,optimizer_updates=0,model_forwards=0,test_opened=False)


if __name__ == '__main__':
    main()
