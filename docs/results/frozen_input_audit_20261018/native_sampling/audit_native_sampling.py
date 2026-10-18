"""One bounded, CPU-only pass over frozen T_core/V_select measurements.

No manifest, reference, model, training, pair matrix, or new split is created.
Native selection uses the public selector; cap128 input assembly uses the
public timed adapter with a pair-free metadata view. FP32 checks reproduce
only pack_timed_scene's time guards, without fabricating a normalizer.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[4]
ROOT = Path(__file__).parent
if len(sys.argv) == 3 and sys.argv[1] == '--output':
    ROOT = Path(sys.argv[2])
    ROOT.mkdir(parents=True, exist_ok=False)
ARCHIVE = Path('/Users/a1/Downloads/数据集level2.zip')
LISTS = REPO/'docs/results/frozen_clock_audit_20261018/frozen_scene_lists_20261018.json'
sys.path.insert(0, str(REPO))
from association.data.archive import MTADArchive
from association.data.native import uniform_observation_indices
from association.data.timed import prepare_timed_scene
from association.training.sampling import BalancedScenePairSampler

CAPS = (64, 128, 256, None)
Q = (0, .01, .05, .5, .9, .95, .99, 1)
QN = ('min', 'p01', 'p05', 'median', 'p90', 'p95', 'p99', 'max')


def write(name, value):
    path = ROOT/name
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def quantiles(values):
    arrays = [np.asarray(x).reshape(-1) for x in values if np.size(x)]
    if not arrays:
        return {'count': 0}
    a = np.concatenate(arrays)
    return {'count': len(a), **dict(zip(QN, np.quantile(a, Q).tolist()))}


def capname(cap):
    return 'full' if cap is None else str(cap)


def check():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError('Single audit total CPU wall-time budget 300 seconds reached')


class CompactPools:
    """Rank-select in NumPy triu order using O(N) identity metadata."""
    def __init__(self, identities):
        self.ids = np.asarray(identities, dtype=str)
        self.n = len(self.ids)
        self.byid = {key: np.flatnonzero(self.ids == key) for key in dict.fromkeys(identities)}
        pc = np.asarray([len(self.byid[key])-np.searchsorted(self.byid[key], i, side='right')
                         for i, key in enumerate(self.ids)], dtype=np.int64)
        self.ends = {1: np.cumsum(pc), 0: np.cumsum(self.n-np.arange(self.n)-1-pc)}
        self.count = {c: int(a[-1]) if len(a) else 0 for c, a in self.ends.items()}

    def select(self, label, ranks):
        ranks = np.asarray(ranks, dtype=np.int64)
        ends = self.ends[label]
        rows = np.searchsorted(ends, ranks, side='right')
        before = np.where(rows, ends[np.maximum(rows-1, 0)], 0)
        result = []
        for i, offset in zip(rows, ranks-before):
            members = self.byid[self.ids[i]]
            same = members[members > i]
            if label:
                j = int(same[offset])
            else:
                # Find the offset-th allowed right endpoint without an N^2 pool.
                lo, hi = int(i+1), self.n-1
                while lo < hi:
                    mid = (lo+hi)//2
                    allowed = mid-i-int(np.searchsorted(same, mid, side='right'))
                    if allowed >= offset+1:
                        hi = mid
                    else:
                        lo = mid+1
                j = lo
            result.append(int(i*(2*self.n-i-1)//2+(j-i-1)))
        return np.asarray(result, dtype=np.int64)

    def endpoints(self, pair_indices):
        starts = np.arange(self.n, dtype=np.int64)
        starts = starts*(2*self.n-starts-1)//2
        i = np.searchsorted(starts, pair_indices, side='right')-1
        j = np.asarray(pair_indices)-starts[i]+i+1
        return np.column_stack((i, j))


class CompactSampler:
    def __init__(self, pools, seed):
        self.pools = pools
        self.rng = np.random.default_rng(seed)
        self.eligible = {label: np.asarray([i for i, pool in enumerate(pools) if pool.count[label]], dtype=np.int64)
                         for label in (1, 0)}

    def sample(self, size=64):
        chosen = {}
        for label in (1, 0):
            si = int(self.rng.choice(self.eligible[label]))
            pool = self.pools[si]
            ranks = self.rng.choice(pool.count[label], size=size//2, replace=pool.count[label] < size//2)
            chosen.setdefault(si, []).append(pool.select(label, ranks))
        groups = [(si, self.rng.permutation(np.concatenate(chunks))) for si, chunks in chosen.items()]
        return [groups[int(i)] for i in self.rng.permutation(len(groups))]


def selfcheck():
    scenes, compact = [], []
    for i, ids in enumerate((['a','b','a','c','b'], ['x','x','x','y'], ['a','b'], ['q','q'])):
        pairs = np.column_stack(np.triu_indices(len(ids), 1))
        labels = np.asarray([ids[a] == ids[b] for a, b in pairs], dtype=np.float32)
        scenes.append(dict(scene_id=str(i), pairs=pairs, labels=labels))
        cp = CompactPools(ids)
        for label in (1, 0):
            expected = np.flatnonzero(labels == label)
            actual = cp.select(label, np.arange(len(expected)))
            assert np.array_equal(actual, expected)
            assert np.array_equal(cp.endpoints(actual), pairs[expected])
        compact.append(cp)
    for seed in (20, 21):
        dense, small = BalancedScenePairSampler(scenes, seed), CompactSampler(compact, seed)
        for _ in range(100):
            a, b = dense.sample(64), small.sample(64)
            assert len(a) == len(b)
            assert all(x[0] == y[0] and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
        assert dense._rng.integers(2**32) == small.rng.integers(2**32)
    return 'PASS: compact rank-select, 200 public-sampler batch comparisons and final RNG states'


def accumulator():
    return dict(counts=Counter(), lengths=[], durations=[], support=[], grid_steps=[],
                dt=defaultdict(list), caps={capname(c): dict(counts=Counter(), gaps=[], maxgaps=[],
                    fractions=[], rounding=[], examples=[]) for c in CAPS}, scenes=[])


def audit_scene(archive, split, sid, acc):
    check()
    scene = archive.load_scene(sid)
    tracks = scene.tracks
    identities = [t.identity_id for t in tracks]
    ic = Counter(identities)
    n = len(tracks)
    positive = sum(c*(c-1)//2 for c in ic.values())
    total = n*(n-1)//2
    counts = acc['counts']
    counts.update(scenes=1, valid_tracks=n, canonical_points=sum(len(t.times) for t in tracks),
        quarantined_tracks=scene.audit['quarantined_nodes'], candidate_pairs=total,
        positive_pairs=positive, negative_pairs=total-positive, scene_identities=len(ic),
        positive_eligible_scenes=int(positive > 0), negative_eligible_scenes=int(total-positive > 0),
        positive_eligible_tracks=sum(c for c in ic.values() if c >= 2),
        negative_eligible_tracks=sum(c for c in ic.values() if c < n),
        positive_eligible_identities=sum(c >= 2 for c in ic.values()),
        negative_eligible_identities=sum(c < n for c in ic.values()))
    row = dict(scene_id=sid, split=split, valid_tracks=n, positive_pairs=positive,
               negative_pairs=total-positive, quarantine=scene.audit['quarantine_reasons'],
               identities=[dict(identity_id=k, valid_tracks=v, positive_pairs=v*(v-1)//2)
                           for k,v in ic.items()], tracks=[])
    acc['scenes'].append({k: row[k] for k in ('scene_id','valid_tracks','positive_pairs','negative_pairs')})
    if not n:
        return row
    # Pair-free native metadata. No invented manifest or pair candidate contents.
    native = dict(scene_id=sid, split=split, node_ids=[t.node_id for t in tracks],
        identities=identities, sources=[t.source for t in tracks],
        intervals=np.asarray([[t.times[0],t.times[-1]] for t in tracks], dtype=np.float64).reshape(n,2),
        tracks=[dict(node_id=t.node_id, identity_id=t.identity_id, source=t.source,
                     canonical_times=t.times, canonical_values=t.values) for t in tracks])
    timed = prepare_timed_scene(native, length=64, max_observations=128, time_unit_seconds=20)
    grid = timed['grid']; origin = timed['time_origin_seconds']
    grid32 = ((grid-origin)/20).astype(np.float32)
    counts['public_timed_adapter_scenes'] += 1
    counts['query_float32_nonincreasing_steps'] += int((np.diff(grid32) <= 0).sum())
    acc['grid_steps'].append([grid[1]-grid[0]])
    valid_and_bad = [(t.source,t.batch) for t in tracks] + [(q['source'],q['batch']) for q in scene.audit['quarantined']]
    candidate_order = {key: i for i,key in enumerate(sorted(set(valid_and_bad)))}
    for i, t in enumerate(tracks):
        if i % 128 == 0:
            check()
        times = np.asarray(t.times)
        native_count = len(times)
        support = (grid >= times[0]) & (grid <= times[-1])
        assert np.array_equal(support, timed['observed'][i])
        cast_support = (grid32 >= np.float32((times[0]-origin)/20)) & (grid32 <= np.float32((times[-1]-origin)/20))
        support_changed = int((support != cast_support).sum())
        ns = int(support.sum())
        counts['zero_query_support_tracks'] += int(ns == 0)
        counts['one_query_support_tracks'] += int(ns == 1)
        counts['float32_support_changed_tracks'] += int(support_changed > 0)
        counts['float32_support_changed_points'] += support_changed
        acc['lengths'].append([native_count]); acc['durations'].append([times[-1]-times[0]])
        acc['support'].append([ns]); acc['dt'][t.source].append(np.diff(times))
        row['tracks'].append(dict(canonical_index=i, original_candidate_index=candidate_order[t.source,t.batch],
            node_id=t.node_id, identity_id=t.identity_id, source=t.source, native_points=native_count,
            support_start=float(times[0]), support_stop=float(times[-1]), query_support_points=ns))
        for cap in CAPS:
            name = capname(cap); item = acc['caps'][name]; c = item['counts']
            selected = uniform_observation_indices(native_count, cap)
            chosen = times[selected]; tau = (chosen-origin)/20; fp = tau.astype(np.float32)
            delta = np.diff(chosen)
            collisions = int((np.diff(fp) <= 0).sum())
            where = np.searchsorted(grid32, fp)
            keep = where < len(grid32)
            false_equal = int(((fp[keep] == grid32[where[keep]]) & (chosen[keep] != grid[where[keep]])).sum())
            c.update(selected_points=len(selected), affected_tracks=int(len(selected)<native_count),
                omitted_points=native_count-len(selected), endpoints_lost=int(selected[0] != 0 or selected[-1] != native_count-1),
                native_float32_nonincreasing_steps=collisions, native_float32_affected_tracks=int(collisions>0),
                native_query_false_equal_points=false_equal,
                native_query_false_equal_tracks=int(false_equal>0))
            item['gaps'].append(delta); item['maxgaps'].append([float(delta.max())])
            item['fractions'].append([len(selected)/native_count])
            item['rounding'].append([float(np.max(np.abs(fp.astype(np.float64)*20+origin-chosen)))])
            if (collisions or false_equal or support_changed) and len(item['examples']) < 12:
                item['examples'].append(dict(scene_id=sid,node_id=t.node_id,native_time_collisions=collisions,
                    query_aliases=false_equal,support_changed_points=support_changed))
    del timed, native, scene
    return row


def summarize(acc):
    count = dict(acc['counts'])
    result = dict(counts=count, native_points_per_track=quantiles(acc['lengths']),
        duration_seconds=quantiles(acc['durations']), query_support_points=quantiles(acc['support']),
        query_grid_step_seconds=quantiles(acc['grid_steps']),
        native_delta_seconds_by_source={k: quantiles(v) for k,v in acc['dt'].items()}, caps={},
        scenes=acc['scenes'])
    for name,item in acc['caps'].items():
        c = dict(item['counts'])
        c.update(point_retention_fraction=c.get('selected_points',0)/max(count.get('canonical_points',0),1),
            affected_track_fraction=c.get('affected_tracks',0)/max(count.get('valid_tracks',0),1))
        result['caps'][name] = dict(counts=c, retained_delta_seconds=quantiles(item['gaps']),
            per_track_max_gap_seconds=quantiles(item['maxgaps']),
            per_track_retention_fraction=quantiles(item['fractions']),
            per_track_max_float32_time_rounding_seconds=quantiles(item['rounding']),
            collision_examples=item['examples'])
    return result


def steps_for_scene(s, q):
    return 1 if s == 1 else math.ceil(math.log1p(-q)/math.log1p(-1/s))


def coverage(stage, groups, pools, state):
    for si, pairs in groups:
        state['scenes'].add(si)
        for p,(a,b) in zip(pairs,pools[si].endpoints(pairs)):
            state['pairs'].add((si,int(p)))
            state['tracks'].update(((si,int(a)),(si,int(b))))
            state['identities'].update(((si,str(pools[si].ids[a])),(si,str(pools[si].ids[b]))))
            state['global_identities'].update((str(pools[si].ids[a]),str(pools[si].ids[b])))
            label = int(pools[si].ids[a] == pools[si].ids[b])
            state['class_scenes'][label].add(si)
            state['presentations'][label] += 1


def fresh():
    return dict(scenes=set(),tracks=set(),pairs=set(),identities=set(),global_identities=set(),
                class_scenes={1:set(),0:set()},presentations=Counter())


def covjson(state, scene_ids):
    return dict(unique_scenes=len(state['scenes']), unique_tracks=len(state['tracks']),
        unique_scene_identities=len(state['identities']), unique_global_AB_identities=len(state['global_identities']),
        unique_pairs=len(state['pairs']), positive_presentations=state['presentations'][1],
        negative_presentations=state['presentations'][0],
        positive_scenes=len(state['class_scenes'][1]), negative_scenes=len(state['class_scenes'][0]),
        scene_ids=[scene_ids[i] for i in sorted(state['scenes'])])


def simulate(rows):
    scene_ids = [r['scene_id'] for r in rows]
    pools = [CompactPools([t['identity_id'] for t in r['tracks']]) for r in rows]
    tracks = [(s,t) for s,p in enumerate(pools) for t in range(p.n)]
    n = len(tracks)
    eligible = {c: sum(p.count[c]>0 for p in pools) for c in (1,0)}
    qs = (.9,.95,.99)
    required = {str(q): dict(TCM=math.ceil(math.log1p(-q)/math.log1p(-64/n)),
        positive_scene=steps_for_scene(eligible[1],q),negative_scene=steps_for_scene(eligible[0],q)) for q in qs}
    plans = {'current_cost_measurement_proposal':dict(TCM=2,TAM_FROZEN=2,TFM_FROZEN=2,JOINT=2)}
    for q in qs:
        r = required[str(q)]
        k = max(r['positive_scene'],r['negative_scene'])
        plans[f'coverage_{int(q*100)}_candidate_not_authorized'] = dict(TCM=r['TCM'],TAM_FROZEN=k,TFM_FROZEN=r['positive_scene'],JOINT=k)
    output = dict(seed=20, tfm_seed=21, batch_tracks=64, pair_batch_size=64,
        policy='exact current sampler distribution and RNG call order; no model or training',
        eligible_scene_counts={str(k):v for k,v in eligible.items()}, required_updates_for_expected_coverage=required,
        warning='Candidates illustrate sampling coverage, not convergence or an approved training plan. Joint sampler continues after TAM.', plans={})
    for name, updates in plans.items():
        check()
        track_rng, tfm_rng = np.random.default_rng(20), np.random.default_rng(21)
        sampler = CompactSampler(pools,20)
        cumulative = fresh(); stages = {}
        for stage,k in updates.items():
            current = fresh()
            for step in range(k):
                if step%64 == 0:
                    check()
                if stage == 'TCM':
                    selected = track_rng.choice(n,64,replace=n<64)
                    for idx in selected:
                        si,t = tracks[int(idx)]; ident = str(pools[si].ids[t])
                        current['scenes'].add(si); current['tracks'].add((si,t))
                        current['identities'].add((si,ident)); current['global_identities'].add(ident)
                else:
                    if stage == 'TFM_FROZEN':
                        si = int(tfm_rng.choice(sampler.eligible[1])); pool=pools[si]
                        ranks=tfm_rng.choice(pool.count[1],64,replace=pool.count[1]<64)
                        groups=[(si,pool.select(1,ranks))]
                    else:
                        groups=sampler.sample(64)
                    coverage(stage,groups,pools,current); coverage(stage,groups,pools,cumulative)
            stages[stage]=dict(updates=k,actual=covjson(current,scene_ids))
            if stage=='TCM':
                stages[stage]['expected_track_fraction']=1-(1-64/n)**k
            else:
                stages[stage]['expected_positive_scene_fraction']=1-(1-1/eligible[1])**k
                if stage!='TFM_FROZEN':
                    stages[stage]['expected_negative_scene_fraction']=1-(1-1/eligible[0])**k
        output['plans'][name]=dict(stages=stages,all_pair_stages_cumulative=covjson(cumulative,scene_ids))
    return output


def main():
    global DEADLINE
    start=time.monotonic(); DEADLINE=start+300
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('300 second hard CPU audit alarm')))
    signal.alarm(300)
    result=dict(status='RUNNING',started_at=datetime.now(timezone.utc).isoformat(),
        scope='frozen T_core512 and V_select64 only; no references/T_test/hash checks/models/GPU',
        archive=str(ARCHIVE),scene_list=str(LISTS),cpu_thread_limit=2,budget_seconds=300,
        time_unit_seconds=20,query_length=64,cap_candidates=[64,128,256,'full'],
        fp32_method='time-only arithmetic from pack_timed_scene; no normalizer fabricated or model inference',
        fixture_check=selfcheck())
    write('results.json',result)
    accum={s:accumulator() for s in ('T_core','V_select')}; core_rows=[]; completed=0
    try:
        lists=json.loads(LISTS.read_text())['scene_ids']
        assert len(lists['T_core'])==512 and len(lists['V_select'])==64
        assert len(set(lists['T_core']+lists['V_select']))==576
        tasks=[(s,sid) for s in ('T_core','V_select') for sid in lists[s]]
        with MTADArchive(ARCHIVE) as archive, (ROOT/'scene_metadata.jsonl').open('x') as handle:
            for split,sid in tasks:
                row=audit_scene(archive,split,sid,accum[split])
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
                if split=='T_core': core_rows.append(row)
                completed+=1
                if completed==8:
                    elapsed=time.monotonic()-start
                    estimate=dict(first_scene_ids=[x[1] for x in tasks[:8]],seconds=elapsed,
                        projected_full_scan_seconds=elapsed/8*576,
                        note='First eight frozen scenes in existing order; not representative sampling or runtime guarantee')
                    write('first8_estimate.json',estimate)
                    result['first8_estimate']=estimate
                    print(json.dumps({'first8':estimate}),flush=True)
                    if estimate['projected_full_scan_seconds']>255:
                        raise TimeoutError('First-eight estimate leaves insufficient room under 300s; no new scan')
                if completed%64==0:
                    print(json.dumps(dict(completed_scenes=completed,seconds=time.monotonic()-start)),flush=True)
            result['member_reads']=dict(Counter(r['kind'] for r in archive.members_read))
            result['measurement_scene_ids']=[r['scene_id'] for r in archive.members_read if r['kind']=='measurements']
            assert result['measurement_scene_ids']==[sid for _,sid in tasks]
        result['scan_seconds']=time.monotonic()-start
        result['splits']={s:summarize(a) for s,a in accum.items()}
        result['sampling']=simulate(core_rows)
        result['status']='COMPLETED'
    except BaseException as exc:
        result.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        result.update(completed_scenes=completed,elapsed_seconds=time.monotonic()-start,
                      ended_at=datetime.now(timezone.utc).isoformat())
        write('results.json',result)
        print(json.dumps({k:result[k] for k in ('status','completed_scenes','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    main()
