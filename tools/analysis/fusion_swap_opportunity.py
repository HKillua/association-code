"""C137: compare slot exchanges against the cost of removing an active borrow."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from tools.analysis import fusion_disjoint_position as c136
from tools.analysis.offline_association_tail import read_motion, summary_state, write


c135, c134, c133, c130, c127, read = c136.c135, c136.c134, c136.c133, c136.c130, c136.c127, c136.read
PRIOR = 'LEVEL1-OFFLINE-DISJOINT-POSITION-20261113-attempt-1'
J, AVOID, OLD = c136.J, c136.AVOID, c136.NONFIRST
BOUND, GUARD = 'REMOVAL_BOUND', 'STABILITY_GUARD'
BASELINES, NEW = (J, AVOID, OLD), (BOUND, GUARD)
ARMS = BASELINES + NEW
COMPARISONS = [(arm, base) for arm in ARMS for base in BASELINES if arm != base]
HARM = c133.HARM_M
POOL = 'ALL_CANDIDATES'


def stable_unknown(witness, outer, movement):
    return witness is None and outer is not None and outer <= HARM and movement > HARM


def choose(rows, evidence, stable, movement, order, old):
    rank = {i:k for k,i in enumerate(order)}
    base = set(old[AVOID])
    sources = {i:r['same_source'] for i,r in enumerate(rows)}
    distances = {i:d for i,r in enumerate(rows)
                 for k,d in [c134.evidence_type(r['evidence'])] if k=='dual'}
    cells = [(r['same_source'],c134.evidence_type(r['evidence'])[0]) for r in rows]
    quota = Counter(cells[i] for i in old[J])
    chosen = {k:old[k] for k in BASELINES}
    witnesses = [e['nonfirst'] or e['third'] for e in evidence]
    opportunities = Counter()
    candidates = []
    for same in (True,False):
        inside = [i for i in distances if i in base and sources[i]==same and
                  (witnesses[i] is None or witnesses[i]['quality']!='benefit')]
        outside = [i for i in distances if i not in base and sources[i]==same and
                   witnesses[i] is not None and witnesses[i]['quality']=='benefit']
        for i in inside:
            for j in outside:
                opportunities['position_pairs'] += 1
                margin = -witnesses[j]['excess_m']-movement[i]
                if margin <= 0.:
                    continue
                opportunities['positive_bound_pairs'] += 1
                opportunities['protected_pairs'] += stable[i]
                candidates.append((-margin, distances[j], -distances[i], rank[j], -rank[i],i,j))
    swaps = {}
    for arm in NEW:
        actions = set(base)
        used = set()
        swaps[arm] = []
        for neg_margin, _, _, _, _, i,j in sorted(candidates):
            if i not in actions or j in actions or (arm==GUARD and stable[i]):
                continue
            clusters = {witnesses[k]['cluster'] for k in (i,j) if witnesses[k] is not None}
            if clusters & used:
                continue
            used.update(clusters)
            actions.remove(i)
            actions.add(j)
            swaps[arm].append(dict(drop=i,add=j,margin_m=-neg_margin,drop_bound_m=movement[i],
                                   add_proxy_gain_m=-witnesses[j]['excess_m'],protected_drop=stable[i],
                                   clusters=sorted(clusters)))
        chosen[arm] = sorted(actions,key=rank.__getitem__)
    for arm in ARMS:
        assert len(chosen[arm])==len(set(chosen[arm]))==len(old[J])
        assert Counter(cells[i] for i in chosen[arm])==quota
    return chosen, swaps, opportunities


def freeze(args, scenes, data, thresholds):
    runs, out = args.runs,args.output
    private = out/'private'
    private.mkdir()
    counts = Counter()
    by_scene = {}
    for sid in scenes:
        prior = read(runs/PRIOR/'private'/f'{sid}.json')
        local = read(runs/c136.PRIOR/'private'/f'{sid}.json')['local']
        original = read(runs/c134.PRIOR/'private'/f'{sid}.json')
        order = c134.read_candidates(runs,sid,original,data,thresholds)
        rows, evidence = original['rows'], prior['evidence']
        movement = []
        if rows:
            motion = read_motion(data[sid])
            groups, queries, donors = np.array([(r['group'],r['q'],r['donor']) for r in rows]).T
            with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz',allow_pickle=False) as base:
                movement = c130.DISTANCE(base['KEEP'][groups,queries,:2],motion['states'][donors,queries,:2]).tolist()
            assert all(np.isfinite(x) and x>0. for x in movement)
        stable = [stable_unknown(e['nonfirst'] or e['third'],p['donor_disagreement_m'],m)
                  for e,p,m in zip(evidence,local,movement,strict=True)]
        chosen, swaps, opportunities = choose(rows,evidence,stable,movement,order,prior['selected'])
        scene = Counter(pool=len(rows), planned=len(chosen[J]), protected_groups=len(original['protected']),
                        stable_unknown=sum(stable), avoid_stable_unknown=sum(stable[i] for i in chosen[AVOID]))
        scene.update(opportunities)
        for arm in NEW:
            scene['swaps_'+arm] = len(swaps[arm])
        counts.update(scene)
        by_scene[sid] = dict(scene)
        write(private/f'{sid}.json',dict(selected=chosen,swaps=swaps,stable_unknown=stable,movement_m=movement))
    assert (counts['pool'],counts['planned'],counts['protected_groups'])==(4133,2090,290)
    write(out/'freeze.json',dict(scenes=len(scenes),totals=counts,evaluation_before_freeze=False,native_archive_reads=0))
    return by_scene


def delta_summary(events, threshold):
    counts = Counter(added=0,dropped=0,harm250=0,non_tail_harm250=0,improved250=0,
                     true_harm250=0,wrong_harm250=0)
    harm_sum = benefit_sum = 0.
    for row in events:
        delta = row['after_m']-row['before_m']
        counts[row['direction']] += 1
        if delta > 0.:
            harm_sum += delta
        if delta < 0.:
            benefit_sum -= delta
        if delta > HARM:
            counts['harm250'] += 1
            counts['non_tail_harm250'] += row['before_m'] < threshold
            counts['true_harm250' if row['true'] else 'wrong_harm250'] += 1
        counts['improved250'] += delta < -HARM
    return dict(counts, harm_sum_m=harm_sum, benefit_sum_m=benefit_sum, net_delta_m=harm_sum-benefit_sum)


def compare(events):
    by_arm = defaultdict(dict)
    for row in events:
        arm = row['arm']
        key = (row['scene'],row['target'],row['q'])
        assert key not in by_arm[arm]
        by_arm[arm][key] = row
    changes = defaultdict(list)
    for arm, baseline in COMPARISONS:
        key = arm+'/'+baseline
        a,b = by_arm[arm],by_arm[baseline]
        for k in a.keys() & b.keys():
            assert a[k]['after_m']==b[k]['after_m']
        for direction, chosen, other in (('added',a,b),('dropped',b,a)):
            for k in sorted(chosen.keys()-other.keys()):
                row = chosen[k]
                before,after = (row['before_m'],row['after_m']) if direction=='added' else (row['after_m'],row['before_m'])
                changes[key].append(dict(scene=row['scene'],target=row['target'],q=row['q'],direction=direction,
                    true=row['true'],before_m=before,after_m=after,stable_unknown=row['stable_unknown'],
                    movement_m=row['movement_m']))
                assert abs(after-before) <= row['movement_m']+1e-7
    return changes


def run(args):
    runs, out, c98 = args.runs,args.output,c127.c98
    prior = runs/PRIOR
    assert read(prior/'run.json')['status']=='COMPLETED'
    scenes = read(runs/c98.PREP/'frozen-split.json')['held']
    manifest = read(args.manifest)
    assert len(set(scenes))==177 and set(scenes)<=set(manifest['splits']['T_core'])
    data = read(runs/c98.DATA/'private/index.json')['T_core']
    thresholds = read(runs/c98.C95/'summary.json')['fixed_thresholds']
    frozen_counts = freeze(args,scenes,data,thresholds)
    previous = read(prior/'summary.json')
    previous_scenes = {d['scene']:d for d in read(prior/'scenes.json')}
    threshold = previous['baseline_threshold_m']
    totals = defaultdict(Counter)
    components = defaultdict(lambda:defaultdict(Counter))
    errors_by_component = defaultdict(list)
    all_events, pool_events, scene_rows = [], [], []
    for sid in scenes:
        component = manifest['components'][sid]
        original = read(runs/c134.PRIOR/'private'/f'{sid}.json')
        frozen = read(out/'private'/f'{sid}.json')
        previous_selection = read(prior/'private'/f'{sid}.json')['selected']
        original['selected'] = dict(frozen['selected'], **{POOL:list(range(len(original['rows'])))})
        avoid = set(previous_selection[AVOID])
        for i,(r,stable,movement) in enumerate(zip(original['rows'],frozen['stable_unknown'],frozen['movement_m'],strict=True)):
            r['evidence'].update(stable_unknown=stable,movement_m=movement,in_c135_avoid=i in avoid)
        motion = read_motion(data[sid])
        with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz',allow_pickle=False) as base, \
             np.load(data[sid]['evaluation'],allow_pickle=False) as truth, \
             np.load(runs/c98.C95/'private/held'/f'{sid}.npz',allow_pickle=False) as scores:
            groups,_,reach = c127.frozen_scene(motion,base,scores,
                read(runs/c127.c104.C98/'private/held'/f'{sid}.json.gz'),
                read(runs/c127.c107.C104/'private/held'/f'{sid}-choices.json'),thresholds)
            errors,events,scene = c133.evaluate(sid,motion,base,truth,groups,original,reach,threshold)
        old = previous_scenes[sid]
        assert scene['coverage']==old['coverage'] and component==old['component']
        for arm in BASELINES:
            assert scene['counts'][arm]==old['counts'][arm]
        scene['counts'].pop(POOL)
        for arm,count in scene['counts'].items():
            totals[arm].update(count)
            components[component][arm].update(count)
        errors_by_component[component].append(errors[:len(ARMS)])
        all_events.extend(e for e in events if e['arm']!=POOL)
        pool_events.extend(e for e in events if e['arm']==POOL)
        scene_rows.append(dict(scene,scene=sid,component=component,frozen=frozen_counts[sid]))
    coverage = dict(zip(('lat','lon','speed','heading'),
                        np.sum([s['coverage'] for s in scene_rows],axis=0).tolist(),strict=True))
    assert coverage==previous['coverage'] and len(components)==4

    def metrics(arrays):
        error = np.concatenate(arrays,axis=1)
        return {arm:c130.STATS(error[i]) for i,arm in enumerate(ARMS)}

    full = metrics([x for arrays in errors_by_component.values() for x in arrays])
    for arm in BASELINES:
        for key,value in previous['metrics'][arm].items():
            np.testing.assert_allclose(full[arm][key],value,rtol=0.,atol=2e-8)
    leave_one = {k:metrics([x for g,arrays in errors_by_component.items() if k!=g for x in arrays])
                 for k in errors_by_component}
    differences = compare(all_events)
    comparisons = {}
    for arm, baseline in COMPARISONS:
        key = arm+'/'+baseline
        rows = differences[key]
        aggregate = delta_summary(rows,threshold)
        np.testing.assert_allclose(aggregate['net_delta_m'],
            (full[arm]['mae']-full[baseline]['mae'])*full[arm]['n'],rtol=0.,atol=1e-5)
        comparisons[key] = dict(total=aggregate,directions={d:delta_summary([r for r in rows if r['direction']==d],threshold)
                                                          for d in ('added','dropped')})
    diagnostic = defaultdict(Counter)
    for e in pool_events:
        if not e['in_c135_avoid']:
            continue
        key = 'stable_unknown' if e['stable_unknown'] else 'other'
        d = e['after_m']-e['before_m']
        diagnostic[key].update(evaluated=1,true=int(e['true']),wrong=int(not e['true']),
            saved_true250=int(e['true'] and d < -HARM),saved_any250=int(d < -HARM),harm250=int(d > HARM),
            true_tail_exit=int(e['true'] and e['tail'] and e['after_m'] < threshold))
    write(out/'summary.json',dict(scenes=len(scenes),coverage=coverage,metrics=full,totals=totals,
        components=components,leave_one_component_out=leave_one,baseline_threshold_m=threshold,
        comparisons=comparisons,avoid_stability_diagnostic=diagnostic,opportunity_slots=4133))
    write(out/'events.json',all_events)
    write(out/'scenes.json',scene_rows)
    write(out/'swap-effects.json',differences)
    write(out/'pool-events.json',pool_events)


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
    summary_state(args.output,start,status='COMPLETED',native_archive_reads=0,optimizer_updates=0,
                  model_forwards=0,test_opened=False)


if __name__=='__main__':
    main()
