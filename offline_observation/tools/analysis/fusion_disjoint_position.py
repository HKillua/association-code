"""C136: select borrowing slots using disjoint, deduplicated position evidence."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import fusion_local_position_witness as c135
from tools.analysis.offline_association_tail import read_motion, summary_state, write


c134, c133, c130, c127, read = c135.c134, c135.c133, c135.c130, c135.c127, c135.read
PRIOR = 'LEVEL1-OFFLINE-LOCAL-POSITION-WITNESS-20261113-attempt-1'
J, DUAL, AVOID, NONFIRST, EXTENDED = ARMS = (c135.J, c135.DUAL, c135.AVOID, 'NONFIRST', 'THIRD_FRAGMENT')
CLASSES = {'risk': -1, 'unavailable': 0, 'neutral': 0, 'benefit': 1}


def observation_key(track, index):
    return (track['source'], float(track['canonical_times'][index]),
            *map(float, track['canonical_values'][index,:2]))


def short_position(times, points, when):
    bracket = c133.bracket(times, when)
    if bracket:
        return c133.at(times, points, when, bracket), bracket, 0., 'bracket'
    if times[0] <= when <= times[-1]:
        return None
    before = when > times[-1]
    position, check = c133.endpoint(times, points, before, when)
    if check is None or check > c133.HARM_M:
        return None
    rows = (len(times)-1, len(times)-2, len(times)-3) if before else (0,1,2)
    return position, rows, check, 'endpoint'


def scene_evidence(rows, prior_local, motion, base, scores, groups, reach, thresholds, tracks):
    lengths = [len(track['canonical_times']) for track in tracks]
    np.testing.assert_array_equal(lengths, motion['native_count'])
    offsets = np.r_[0, np.cumsum(lengths)]
    _, xy, scale = c133._coordinates(np.concatenate([t['canonical_values'][:,:2] for t in tracks]), offsets)
    points = np.split(xy, offsets[1:-1])
    times = [t['canonical_times'] for t in tracks]
    observation_keys = [tuple(observation_key(t, i) for i in range(len(times[j]))) for j,t in enumerate(tracks)]
    raw_sets = [set(keys) for keys in observation_keys]
    both = np.zeros((len(tracks),len(tracks)), bool)
    a,b = scores['pairs'].T
    both[a,b] = both[b,a] = ((scores['score_056'] > thresholds['056']) &
                            (scores['score_M_BCE'] > thresholds['M_BCE']))
    output = []
    for r, saved in zip(rows, prior_local, strict=True):
        group, q, donor = (r[k] for k in ('group','q','donor'))
        members = groups[group]
        anchor = int(members[0])
        when = motion['query_seconds'][q]
        assert not motion['support'][members,q].any() and motion['support'][donor,q]
        old_geo = base['KEEP'][group,q,:2]
        np.testing.assert_array_equal(old_geo, motion['states'][anchor,q,:2])
        actual = c133.bracket(times[donor], when, max_gap=np.inf)
        assert list(actual) == saved['donor_actual_rows']
        donor_geo = motion['states'][donor,q,:2]
        new = c133.at(times[donor], points[donor], when, actual)
        offset = old_geo-donor_geo
        offset[1] = (offset[1]+180.) % 360.-180.
        old = new + offset*scale
        blocked = raw_sets[anchor] | {observation_keys[donor][i] for i in actual}
        donor_group = set(groups[r['donor_group']].tolist())
        outside = [i for i in reach[group]['first'] if i not in donor_group and i not in members
                   and both[anchor,i] and both[donor,i]]
        evidence = {}
        duplicates = 0
        for kind, nodes in (('nonfirst', members[1:]), ('third', outside)):
            options = []
            for node in nodes:
                assert node != anchor and node != donor
                local = short_position(times[node],points[node],when)
                if local is None:
                    continue
                pred, indices, check, method = local
                if any(observation_keys[node][i] in blocked for i in indices):
                    duplicates += 1
                    continue
                distance_new, distance_old = (float(np.linalg.norm(pred-p)) for p in (new,old))
                excess = distance_new-distance_old
                age = min(abs(when-times[node][list(indices)]))
                options.append((float(age), int(node), dict(node=int(node), rows=indices,
                    age_s=float(age), step_m=check, method=method, distance_borrow_m=distance_new,
                    distance_keep_m=distance_old, excess_m=excess, quality=c135.position_class(excess),
                    relative_s=(times[node][list(indices)]-when).tolist(), kind=kind)))
            evidence[kind] = min(options, key=lambda x:x[:2])[-1] if options else None
        output.append(dict(**evidence, blocked_duplicate_options=duplicates, third_candidates=len(outside)))
    return output


def cluster_evidence(rows, evidence):
    """Merge overlapping native intervals for each recipient/donor/witness triple."""
    spans = defaultdict(list)
    for i,(r,e) in enumerate(zip(rows,evidence,strict=True)):
        for kind, witness in e.items():
            if kind not in ('nonfirst','third') or witness is None:
                continue
            key = (r['group'],r['donor'],witness['node'])
            spans[key].append((min(witness['rows']),max(witness['rows']),i,kind))
    pairs, clusters = set(), {}
    for key, intervals in sorted(spans.items()):
        right = -1
        for lo, hi, i, kind in sorted(intervals):
            if lo > right:
                cluster = len(clusters)
                clusters[cluster] = [*key, lo, hi, 0]
            right = max(hi,right)
            clusters[cluster][4] = right
            clusters[cluster][5] += 1
            pairs.add(key[:2])
            evidence[i][kind]['cluster'] = cluster
    return clusters, len(pairs)


def choose(rows, evidence, order, previous):
    rank = {i:k for k,i in enumerate(order)}
    distance = {i:c134.evidence_type(r['evidence'])[1] for i,r in enumerate(rows)}
    cells = {i:('same' if r['same_source'] else 'cross',c134.evidence_type(r['evidence'])[0])
             for i,r in enumerate(rows)}
    quota = Counter(cells[i] for i in previous[J])
    chosen = {k:previous[k] for k in (J,DUAL,AVOID)}
    swaps, opportunities = {}, {}
    for arm in (NONFIRST,EXTENDED):
        witnesses = [e['nonfirst'] or (e['third'] if arm==EXTENDED else None) for e in evidence]
        quality = [w['quality'] if w is not None else 'unavailable' for w in witnesses]
        base = set(previous[AVOID])
        actions = set(base)
        all_pairs = []
        for cell in (('same','dual'),('cross','dual')):
            inside = [i for i in base if cells[i]==cell]
            outside = [i for i in order if i not in base and cells[i]==cell]
            for a in inside:
                for b in outside:
                    if CLASSES[quality[b]] > CLASSES[quality[a]] and (
                            quality[b]=='benefit' or (quality[a]=='risk' and quality[b]=='neutral')):
                        all_pairs.append((-(CLASSES[quality[b]]-CLASSES[quality[a]]),
                            distance[b],-distance[a],rank[b],-rank[a],a,b))
        used_clusters = set()
        changes = []
        for _, _, _, _, _, a,b in sorted(all_pairs):
            if a not in actions or b in actions:
                continue
            used = {witnesses[i]['cluster'] for i in (a,b) if witnesses[i] is not None}
            if used & used_clusters:
                continue
            used_clusters.update(used)
            actions.remove(a)
            actions.add(b)
            changes.append(dict(remove=a,add=b,old=quality[a],new=quality[b],clusters=sorted(used),
                                source=cells[a][0]))
        chosen[arm] = sorted(actions,key=rank.__getitem__)
        swaps[arm] = changes
        opportunities[arm] = len(all_pairs)
    for arm in ARMS:
        assert len(set(chosen[arm])) == len(chosen[arm]) == len(previous[J])
        assert Counter(cells[i] for i in chosen[arm]) == quota
    return chosen, swaps, opportunities


def freeze(args, scenes, data, thresholds):
    runs, out, c98 = args.runs,args.output,c127.c98
    private = out/'private'
    private.mkdir()
    totals = Counter()
    scene_counts = {}
    with MTADArchive(read(runs/c98.DATA/'run.json')['archive']) as archive:
        c133.validate_measurement_archive(runs, archive)
        for sid in scenes:
            original = read(runs/c134.PRIOR/'private'/f'{sid}.json')
            prior = read(runs/PRIOR/'private'/f'{sid}.json')
            order = c134.read_candidates(runs,sid,original,data,thresholds)
            rows = original['rows']
            evidence = []
            if rows:
                motion = read_motion(data[sid])
                tracks = c133._canonical_tracks(archive._rows(sid,'measurements'),sid,motion['node_ids'],motion['sources'])
                with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz',allow_pickle=False) as base, \
                     np.load(runs/c98.C95/'private/held'/f'{sid}.npz',allow_pickle=False) as scores:
                    groups, _, reach = c127.frozen_scene(motion,base,scores,
                        read(runs/c127.c104.C98/'private/held'/f'{sid}.json.gz'),
                        read(runs/c127.c107.C104/'private/held'/f'{sid}-choices.json'),thresholds)
                    evidence = scene_evidence(rows,prior['local'],motion,base,scores,groups,reach,thresholds,tracks)
            clusters, pairs = cluster_evidence(rows,evidence)
            chosen, swaps, opportunities = choose(rows,evidence,order,prior['selected'])
            counts = Counter(pool=len(rows), actions=len(chosen[J]), protected=len(original['protected']),
                             intervals=len(clusters), evidence_pairs=pairs,
                             third_screened_slots=sum(e['third_candidates']>0 for e in evidence),
                             raw_duplicates=sum(e['blocked_duplicate_options'] for e in evidence))
            for e in evidence:
                for k in ('nonfirst','third'):
                    if e[k] is not None:
                        counts.update((k, k+'_'+e[k]['quality']))
                if e['nonfirst'] is None and e['third'] is not None:
                    counts['third_increment'] += 1
            totals.update(counts)
            scene_counts[sid] = dict(evidence_counts=counts, swap_counts={a:len(x) for a,x in swaps.items()},
                opportunities=opportunities, swap_types={a:Counter(f"{x['source']}/{x['old']}/{x['new']}" for x in v)
                                                         for a,v in swaps.items()})
            write(private/f'{sid}.json',dict(selected=chosen,evidence=evidence,swaps=swaps,clusters=clusters))
        members = [r['scene_id'] for r in archive.members_read]
        assert all(r['kind']=='measurements' for r in archive.members_read)
    assert len(members) == len(set(members)) == 85 and set(members) <= set(scenes)
    assert (totals['pool'],totals['actions'],totals['protected']) == (4133,2090,290)
    write(out/'freeze.json',dict(scenes=len(scenes),measurement_scenes=members,totals=totals,evaluation_before_freeze=False))
    return scene_counts


def run(args):
    runs, out, c98 = args.runs,args.output,c127.c98
    previous = runs/PRIOR
    assert read(previous/'run.json')['status']=='COMPLETED'
    scenes = read(runs/c98.PREP/'frozen-split.json')['held']
    manifest = read(args.manifest)
    assert len(set(scenes)) == 177 and set(scenes) <= set(manifest['splits']['T_core'])
    data = read(runs/c98.DATA/'private/index.json')['T_core']
    thresholds = read(runs/c98.C95/'summary.json')['fixed_thresholds']
    checks = freeze(args,scenes,data,thresholds)
    previous_summary = read(previous/'summary.json')
    old_scenes = {d['scene']:d for d in read(previous/'scenes.json')}
    threshold = previous_summary['baseline_threshold_m']
    totals = defaultdict(Counter)
    components = defaultdict(lambda:defaultdict(Counter))
    errors_by_component = defaultdict(list)
    events, scene_rows = [], []
    for sid in scenes:
        component = manifest['components'][sid]
        original = read(runs/c134.PRIOR/'private'/f'{sid}.json')
        saved = read(out/'private'/f'{sid}.json')
        original['selected'] = saved['selected']
        for r,e in zip(original['rows'],saved['evidence'],strict=True):
            r['evidence'].update({k+'_position_class':e[k]['quality'] if e[k] is not None else 'unavailable'
                                 for k in ('nonfirst','third')})
        motion = read_motion(data[sid])
        with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz',allow_pickle=False) as base, \
             np.load(data[sid]['evaluation'],allow_pickle=False) as truth, \
             np.load(runs/c98.C95/'private/held'/f'{sid}.npz',allow_pickle=False) as scores:
            groups, _, reach = c127.frozen_scene(motion,base,scores,
                read(runs/c127.c104.C98/'private/held'/f'{sid}.json.gz'),
                read(runs/c127.c107.C104/'private/held'/f'{sid}-choices.json'),thresholds)
            err,rows,scene = c133.evaluate(sid,motion,base,truth,groups,original,reach,threshold)
        old = old_scenes[sid]
        assert scene['coverage']==old['coverage'] and component==old['component']
        for arm in (J,DUAL,AVOID):
            assert scene['counts'][arm]==old['counts'][arm]
        for arm,count in scene['counts'].items():
            totals[arm].update(count)
            components[component][arm].update(count)
        errors_by_component[component].append(err)
        events.extend(rows)
        scene_rows.append(dict(scene,scene=sid,component=component,**checks[sid]))
    coverage = dict(zip(('lat','lon','speed','heading'),
                        np.sum([s['coverage'] for s in scene_rows],axis=0).tolist(),strict=True))
    assert coverage==previous_summary['coverage'] and len(components)==4

    def metrics(arrays):
        error = np.concatenate(arrays,axis=1)
        return {arm:c130.STATS(error[i]) for i,arm in enumerate(ARMS)}

    full = metrics([x for arrays in errors_by_component.values() for x in arrays])
    for arm in (J,DUAL,AVOID):
        for key,value in previous_summary['metrics'][arm].items():
            np.testing.assert_allclose(full[arm][key],value,rtol=0.,atol=2e-8)
    leave_one = {k:metrics([x for g,arrays in errors_by_component.items() if k!=g for x in arrays])
                 for k in errors_by_component}
    write(out/'summary.json',dict(scenes=len(scenes),coverage=coverage,metrics=full,totals=totals,
          components=components,leave_one_component_out=leave_one,baseline_threshold_m=threshold))
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
    summary_state(args.output,start,status='COMPLETED',native_archive_reads=85,optimizer_updates=0,
                  model_forwards=0,test_opened=False)


if __name__=='__main__':
    main()
