"""Join existing JSON only: no ZIP, reference, model, or resampling policy change."""
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('native_sampling_audit',ROOT/'audit_native_sampling.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
start=time.monotonic()
rows=[json.loads(line) for line in (ROOT/'attempt2/scene_metadata.jsonl').read_text().splitlines()]
core=[r for r in rows if r['split']=='T_core']
source=ROOT.parent/'supervision_gap_zero_targets_20261018.json'
sup=json.loads(source.read_text())
assert sup['status']=='COMPLETED'
zero=sup['zero_feature_targets']['180']
assert all(r['valid_count']['180']==[0,0,0,0] for r in zero)
assert not sup['zero_feature_targets']['600']
by_scene={r['scene_id']:r for r in core}
bad=defaultdict(set);linked=[]
for item in zero:
    sid,identity=item['scene_id'],item['identity_id']
    scene=by_scene[sid]
    count=sum(t['identity_id']==identity for t in scene['tracks'])
    pairs=count*(count-1)//2
    linked.append(dict(scene_id=sid,identity_id=identity,valid_tracks=count,positive_pairs=pairs,
                       supervised_queries=item['valid_count']))
    if pairs:bad[sid].add(identity)

def all_bad_probability(z,p,b):
    if not z:return 0.
    if p<b:return (z/p)**b
    if z<b:return 0.
    return math.prod((z-i)/(p-i) for i in range(b))

scene_stats=[]
for scene in core:
    sid=scene['scene_id'];p=scene['positive_pairs']
    z=sum(i['positive_pairs'] for i in scene['identities'] if i['identity_id'] in bad[sid])
    if z:
        scene_stats.append(dict(scene_id=sid,zero_supervision_positive_pairs=z,total_positive_pairs=p,
            zero_pair_fraction=z/p,tfm64_all_bad_probability=all_bad_probability(z,p,64),
            joint32_all_bad_probability=all_bad_probability(z,p,32)))
eligible=sum(s['positive_pairs']>0 for s in core)
prob={stage:sum(r[key] for r in scene_stats)/eligible for stage,key in
      (('TFM_FROZEN','tfm64_all_bad_probability'),('JOINT','joint32_all_bad_probability'))}
pools=[audit.CompactPools([t['identity_id'] for t in row['tracks']]) for row in core]
base=json.loads((ROOT/'attempt2/results.json').read_text())
actual={}
for name,plan in base['sampling']['plans'].items():
    sampler=audit.CompactSampler(pools,20)
    tfm_rng=np.random.default_rng(21)
    result={}
    for stage in ('TAM_FROZEN','TFM_FROZEN','JOINT'):
        k=plan['stages'][stage]['updates']
        exposed=set();bad_presentations=positive_presentations=zero_batches=0;events=[]
        for step in range(k):
            if stage=='TFM_FROZEN':
                si=int(tfm_rng.choice(sampler.eligible[1]));pool=pools[si]
                ranks=tfm_rng.choice(pool.count[1],64,replace=pool.count[1]<64)
                groups=[(si,pool.select(1,ranks))]
            else:groups=sampler.sample(64)
            batch_bad=batch_positive=0
            for si,indices in groups:
                p=pools[si];sid=core[si]['scene_id']
                for a,b in p.endpoints(indices):
                    identity=str(p.ids[a])
                    if identity!=p.ids[b]:continue
                    batch_positive+=1
                    if identity in bad[sid]:
                        batch_bad+=1;exposed.add((sid,identity))
            bad_presentations+=batch_bad;positive_presentations+=batch_positive
            if batch_positive and batch_bad==batch_positive:
                zero_batches+=1;events.append(dict(update=step+1,scene_ids=[core[si]['scene_id'] for si,_ in groups]))
        if stage!='TAM_FROZEN':
            q=prob[stage]
            result[stage]=dict(updates=k,zero_supervision_positive_presentations=bad_presentations,
                positive_presentations=positive_presentations,exposed_zero_identities=len(exposed),
                all_zero_supervision_batches=zero_batches,all_zero_batch_events=events,
                exact_distribution_all_zero_probability_per_update=q,
                probability_at_least_one_all_zero_batch=(-math.expm1(k*math.log1p(-q)) if q<1 else 1))
    actual[name]=result
out=dict(status='COMPLETED',elapsed_seconds=time.monotonic()-start,scope='JSON-only static join and original-distribution sampling simulation',
    source=str(source),native_metadata=str(ROOT/'attempt2/scene_metadata.jsonl'),
    gap180_zero_label_identities=len(zero),gap180_zero_identities_with_valid_tracks=sum(x['valid_tracks']>0 for x in linked),
    gap180_zero_identities_with_positive_pairs=sum(x['positive_pairs']>0 for x in linked),
    valid_tracks_of_zero_identities=sum(x['valid_tracks'] for x in linked),positive_pairs_of_zero_identities=sum(x['positive_pairs'] for x in linked),
    positive_eligible_scene_count=eligible,zero_identities=linked,affected_positive_scenes=scene_stats,
    all_zero_probability_per_update=prob,seed20_plans=actual,
    boundary='All 57 zero-feature targets have all four counts zero; probability is for all sampled positive pairs having zero targets. Mixed good/bad batches can remain valid; no filtering/resampling is authorized. gap600 already has no zero-feature identity and was not reaudited.',
    model_forward_calls=0,archive_payload_reads=0,production_policy_modified=False)
(ROOT/'attempt2/gap180_sampling_risk.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('zero_identities','affected_positive_scenes','seed20_plans')},ensure_ascii=False))
print(json.dumps(actual,ensure_ascii=False))
