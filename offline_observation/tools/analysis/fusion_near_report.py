"""C138: evaluate directional position evidence from nearby native reports."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import fusion_swap_opportunity as c137
from tools.analysis.offline_association_tail import read_motion, summary_state, write


c136, c134, c133 = c137.c136, c137.c134, c137.c133
c130, c127, read = c137.c130, c137.c127, c137.read
PRIOR = 'LEVEL1-OFFLINE-SWAP-OPPORTUNITY-20261113-attempt-1'
J, BOUND, GUARD = BASELINES = (c137.J, c137.BOUND, c137.GUARD)
RAW, BUFFER = NEW = ('NEAR_REPORT', 'BUFFERED_REPORT')
ARMS = BASELINES + NEW
COMPARE = tuple((arm, base) for arm in ARMS for base in BASELINES if arm != base)
HARM = c137.HARM
POOL = c137.POOL


def turn_degrees(points):
    lat, lon = np.deg2rad(points).T
    dlon = np.diff(lon)
    angles = np.arctan2(np.sin(dlon)*np.cos(lat[1:]),
        np.cos(lat[:-1])*np.sin(lat[1:])-np.sin(lat[:-1])*np.cos(lat[1:])*np.cos(dlon))
    moves = c130.DISTANCE(points[1:], points[:-1])
    if (moves == 0.).any():
        return None
    return float(np.max(np.abs((np.rad2deg(np.diff(angles))+180.) % 360.-180.)))


def interval(report, movement, error_m):
    if report is None:
        return -movement, movement
    delta = report['direction_m']
    radius = 2.*report['speed_mps']*report['age_s']+error_m
    return max(-movement, delta-2.*radius), min(movement, delta+2.*radius)


def evidence(rows, prior, motion, base, tracks):
    groups = c127.c98.group_nodes(base['current'])
    times = [t['canonical_times'] for t in tracks]
    geo = [t['canonical_values'][:,:2] for t in tracks]
    np.testing.assert_array_equal([len(t) for t in times], motion['native_count'])
    keys = [tuple(c136.observation_key(t,i) for i in range(len(times[n]))) for n,t in enumerate(tracks)]
    keys_set = [set(k) for k in keys]
    reports, duplicates = [], 0
    for i, r in enumerate(rows):
        group, q, donor = (r[k] for k in ('group','q','donor'))
        anchor = groups[group][0]
        query = motion['query_seconds'][q]
        assert not motion['support'][groups[group],q].any() and motion['support'][donor,q]
        actual = c133.bracket(times[donor],query,max_gap=np.inf)
        assert actual
        borrowed = motion['states'][donor,q,:2]
        keep = base['KEEP'][group,q,:2]
        np.testing.assert_array_equal(keep,motion['states'][anchor,q,:2])
        blocked = keys_set[anchor] | {keys[donor][j] for j in actual}
        options = []
        for node in groups[group][1:]:
            t = times[node]
            assert query < t[0] or query > t[-1]
            ix = tuple(range(5)) if query < t[0] else tuple(range(len(t)-1,len(t)-6,-1))
            age = abs(query-t[ix[0]])
            if len(t)<5 or age>c133.SHORT_S:
                continue
            if any(keys[node][j] in blocked for j in ix):
                duplicates += 1
                continue
            when, positions = t[list(ix)], geo[node][list(ix)]
            vmax = float(np.max(c130.DISTANCE(positions[1:],positions[:-1])/abs(np.diff(when))))
            direction = float(c130.DISTANCE(keep,positions[0])-c130.DISTANCE(borrowed,positions[0]))
            movement = prior['movement_m'][i]
            assert abs(direction)<=movement+1e-7
            options.append((float(age),int(node),dict(node=int(node),rows=ix,age_s=float(age),
                speed_mps=vmax,turn_deg=turn_degrees(positions),direction_m=float(np.clip(direction,-movement,movement)),
                same_report_source=motion['sources'][node]==motion['sources'][donor])))
        reports.append(dict(nonfirst=min(options,key=lambda x:x[:2])[-1] if options else None,third=None))
    clusters, pairs = c136.cluster_evidence(rows,reports)
    items = [r['nonfirst'] for r in reports]
    representatives = {}
    for i, r in enumerate(items):
        if r is None:
            continue
        key = r['cluster']
        if key not in representatives or (r['age_s'],i)<(items[representatives[key]]['age_s'],representatives[key]):
            representatives[key] = i
    for i,r in enumerate(items):
        if r is not None:
            r['representative'] = representatives[r['cluster']]==i
    return items, len(clusters), pairs, duplicates


def select(rows, reports, movement, old):
    cells = [(r['same_source'], c134.evidence_type(r['evidence'])[0]) for r in rows]
    quota = Counter(cells[i] for i in old[J])
    chosen = {arm:old[arm] for arm in BASELINES}
    base = set(chosen[GUARD])
    opportunities = {}
    swaps = {}
    for arm in NEW:
        limits = [interval(r,m,250.) if arm==BUFFER else
                  (r['direction_m'],r['direction_m']) if r else (-m,m)
                  for r,m in zip(reports,movement,strict=True)]
        pairs = []
        inside = defaultdict(list)
        for i in base:
            inside[cells[i]].append(i)
        for j,(low,_) in enumerate(limits):
            if j in base or low<=HARM:
                continue
            for i in inside[cells[j]]:
                margin = low-limits[i][1]
                if margin>0.:
                    pairs.append((-margin,i,j))
        opportunities[arm] = len(pairs)
        selected = set(base)
        used = set()
        swaps[arm] = []
        for neg,i,j in sorted(pairs):
            if i not in selected or j in selected:
                continue
            clusters = {reports[k]['cluster'] for k in (i,j) if reports[k] is not None}
            if clusters & used:
                continue
            used.update(clusters)
            selected.remove(i)
            selected.add(j)
            swaps[arm].append(dict(drop=i,add=j,margin_m=-neg,clusters=sorted(clusters)))
        chosen[arm] = sorted(selected)
    for arm in ARMS:
        assert len(chosen[arm])==len(set(chosen[arm]))==len(old[J])
        assert Counter(cells[i] for i in chosen[arm])==quota
    return chosen,swaps,opportunities


def freeze(runs, out, scenes, data):
    c98 = c127.c98
    private = out/'private'
    private.mkdir()
    totals = Counter()
    frozen_counts = {}
    with MTADArchive(read(runs/c98.DATA/'run.json')['archive']) as archive:
        c133.validate_measurement_archive(runs,archive)
        for sid in scenes:
            original = read(runs/c134.PRIOR/'private'/f'{sid}.json')
            prior = read(runs/PRIOR/'private'/f'{sid}.json')
            rows = original['rows']
            reports, clusters, pairs, duplicates = [],0,0,0
            if rows:
                motion = read_motion(data[sid])
                tracks = c133._canonical_tracks(archive._rows(sid,'measurements'),sid,motion['node_ids'],motion['sources'])
                with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz',allow_pickle=False) as base:
                    reports,clusters,pairs,duplicates = evidence(rows,prior,motion,base,tracks)
            chosen,swaps,opportunities = select(rows,reports,prior['movement_m'],prior['selected'])
            counts = Counter(pool=len(rows),planned=len(chosen[J]),protected=len(original['protected']),
                reports=sum(r is not None for r in reports),clusters=clusters,pairs=pairs,blocked_duplicates=duplicates)
            for arm in NEW:
                counts['swaps_'+arm] = len(swaps[arm])
                counts['opportunities_'+arm] = opportunities[arm]
            totals.update(counts)
            frozen_counts[sid] = dict(counts)
            write(private/f'{sid}.json',dict(selected=chosen,reports=reports,swaps=swaps))
        members = [r['scene_id'] for r in archive.members_read]
        assert all(r['kind']=='measurements' for r in archive.members_read)
    assert len(members)==len(set(members))==85 and set(members)<=set(scenes)
    assert (totals['pool'],totals['planned'],totals['protected'])==(4133,2090,290)
    write(out/'freeze.json',dict(scenes=len(scenes),totals=totals,measurement_scenes=members,evaluation_before_freeze=False))
    return frozen_counts


def diagnose(events, scene_components):
    counts = defaultdict(Counter)
    targets = defaultdict(set)
    for e in events:
        report = e['near_report']
        if report is None:
            continue
        movement = e['movement_m']
        actual = e['before_m']-e['after_m']
        d = report['direction_m']
        turn = report['turn_deg']
        age = report['age_s']
        keys = ('all', 'age/'+('0-30' if age<=30 else '30-60' if age<=60 else '60-120'),
            'turn/'+('unknown' if turn is None else '0-30' if turn<=30 else '30-90' if turn<=90 else '90-180'),
            'source/'+('same' if e['same_source'] else 'cross'),
            'report_source/'+('same' if report['same_report_source'] else 'cross'),
            scene_components[e['scene']])
        scopes = ['all_slots']
        if report['representative']:
            scopes.append('deduplicated')
        if e['in_c137_guard']:
            scopes.extend('selected_'+s for s in tuple(scopes))
        for scope in scopes:
            for key in keys:
                label = scope+'/'+key
                counts[label].update(n=1,true=int(e['true']),gain250=int(actual>HARM),loss250=int(actual<-HARM),
                    direction_correct=int(np.sign(d)==np.sign(actual)),
                    positive_margin=int(d>HARM),positive_margin_gain250=int(d>HARM and actual>HARM))
                targets[label].add((e['scene'],e['target']))
                for error in (0,250,1000):
                    low, high = interval(report,movement,error)
                    counts[label]['contains_'+str(error)] += low<=actual<=high
                    counts[label]['positive_'+str(error)] += low>HARM
                    counts[label]['positive_correct_'+str(error)] += low>HARM and actual>HARM
    return {k:dict(v,targets=len(targets[k])) for k,v in counts.items()}


def run(args):
    runs, out, c98 = args.runs,args.output,c127.c98
    prior = runs/PRIOR
    assert read(prior/'run.json')['status']=='COMPLETED'
    scenes = read(runs/c98.PREP/'frozen-split.json')['held']
    manifest = read(args.manifest)
    assert len(set(scenes))==177 and set(scenes)<=set(manifest['splits']['T_core'])
    data = read(runs/c98.DATA/'private/index.json')['T_core']
    thresholds = read(runs/c98.C95/'summary.json')['fixed_thresholds']
    counts = freeze(runs,out,scenes,data)
    previous = read(prior/'summary.json')
    old_scenes = {r['scene']:r for r in read(prior/'scenes.json')}
    threshold = previous['baseline_threshold_m']
    totals = defaultdict(Counter)
    components = defaultdict(lambda:defaultdict(Counter))
    errors = defaultdict(list)
    all_events,pool,scene_rows = [],[],[]
    for sid in scenes:
        component = manifest['components'][sid]
        original = read(runs/c134.PRIOR/'private'/f'{sid}.json')
        saved = read(out/'private'/f'{sid}.json')
        prior_scene = read(prior/'private'/f'{sid}.json')
        original['selected'] = dict(saved['selected'],**{POOL:list(range(len(original['rows'])))})
        guard = set(saved['selected'][GUARD])
        for i,(row,report) in enumerate(zip(original['rows'],saved['reports'],strict=True)):
            public = {k:v for k,v in report.items() if k not in ('node','rows')} if report is not None else None
            row['evidence'].update(near_report=public,movement_m=prior_scene['movement_m'][i],
                stable_unknown=prior_scene['stable_unknown'][i],in_c137_guard=i in guard)
        motion = read_motion(data[sid])
        with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz',allow_pickle=False) as base, \
             np.load(data[sid]['evaluation'],allow_pickle=False) as truth, \
             np.load(runs/c98.C95/'private/held'/f'{sid}.npz',allow_pickle=False) as scores:
            groups,_,reach = c127.frozen_scene(motion,base,scores,
                read(runs/c127.c104.C98/'private/held'/f'{sid}.json.gz'),
                read(runs/c127.c107.C104/'private/held'/f'{sid}-choices.json'),thresholds)
            err,events,scene = c133.evaluate(sid,motion,base,truth,groups,original,reach,threshold)
        old = old_scenes[sid]
        assert scene['coverage']==old['coverage'] and component==old['component']
        for arm in BASELINES:
            assert scene['counts'][arm]==old['counts'][arm]
        scene['counts'].pop(POOL)
        for arm,n in scene['counts'].items():
            totals[arm].update(n)
            components[component][arm].update(n)
        errors[component].append(err[:len(ARMS)])
        all_events.extend(e for e in events if e['arm']!=POOL)
        pool.extend(e for e in events if e['arm']==POOL)
        scene_rows.append(dict(scene,scene=sid,component=component,frozen=counts[sid]))
    coverage = dict(zip(('lat','lon','speed','heading'),
        np.sum([s['coverage'] for s in scene_rows],axis=0).tolist(),strict=True))
    assert coverage==previous['coverage'] and len(components)==4

    def metrics(arrays):
        joined = np.concatenate(arrays,axis=1)
        return {arm:c130.STATS(joined[i]) for i,arm in enumerate(ARMS)}

    full = metrics([a for seq in errors.values() for a in seq])
    for arm in BASELINES:
        for k,v in previous['metrics'][arm].items():
            np.testing.assert_allclose(full[arm][k],v,rtol=0.,atol=2e-8)
    leave = {k:metrics([a for g,seq in errors.items() if k!=g for a in seq]) for k in errors}
    differences = c137.compare(all_events,COMPARE)
    comparisons = {}
    for arm,base in COMPARE:
        key = arm+'/'+base
        rows = differences[key]
        total = c137.delta_summary(rows,threshold)
        np.testing.assert_allclose(total['net_delta_m'],(full[arm]['mae']-full[base]['mae'])*full[arm]['n'],rtol=0.,atol=1e-5)
        comparisons[key] = dict(total=total,directions={d:c137.delta_summary([r for r in rows if r['direction']==d],threshold)
                                                        for d in ('added','dropped')})
    write(out/'summary.json',dict(scenes=len(scenes),coverage=coverage,metrics=full,totals=totals,components=components,
        leave_one_component_out=leave,baseline_threshold_m=threshold,comparisons=comparisons,
        opportunity_slots=4133,diagnostic=diagnose(pool,manifest['components'])))
    write(out/'events.json',all_events)
    write(out/'scenes.json',scene_rows)
    write(out/'swap-effects.json',differences)
    write(out/'pool-events.json',pool)


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
    summary_state(args.output,start,status='COMPLETED',native_archive_reads=85,optimizer_updates=0,
                  model_forwards=0,test_opened=False)


if __name__=='__main__':
    main()
