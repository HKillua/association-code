import gzip,json,math,time
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
started=time.monotonic()
root=Path(__file__).resolve().parent
source=root.parents[2]/'docs/results/frozen_input_audit_20261018/native_sampling/attempt2/scene_metadata.jsonl.gz'
a=json.loads((root/'train_metadata_analysis.json').read_text()); b=json.loads((root/'window_population_comparison.json').read_text())
classes=np.zeros(4,dtype=np.int64); kept=np.zeros(3,dtype=np.int64); copies=np.zeros(3,dtype=np.int64); nonempty=np.zeros(3,dtype=np.int64)
macro=[[],[],[]]; tracks_total=positive_total=0
with gzip.open(source,'rt') as f:
 for _ in range(512):
  row=json.loads(next(f)); assert row['split']=='T_core'
  ts=row['tracks']; tracks_total+=len(ts)
  starts=np.array([x['support_start'] for x in ts]); stops=np.array([x['support_stop'] for x in ts])
  q=np.linspace(starts.min(),stops.max(),64); masks=(starts[:,None]<=q)&(q<=stops[:,None])
  groups=defaultdict(list)
  for i,t in enumerate(ts): groups[t['identity_id']].append(i)
  pos=[(i,j) for ids in groups.values() for i,j in combinations(ids,2)]; positive_total+=len(pos)
  for ids in groups.values():
   own=masks[ids]; any_support=own.any(axis=0); envelope=(q>=starts[ids].min())&(q<=stops[ids].max())
   classes+=np.array([own.sum(),((~own)&any_support).sum(),len(ids)*((~any_support)&envelope).sum(),len(ids)*(~envelope).sum()])
  origin=20*math.floor(starts.min()/20)
  for k,scheme in enumerate(b['schemes']):
   width=scheme['width_seconds']; step=scheme['step_seconds']
   n=max(1,math.ceil((stops.max()-origin-width)/step)+1)
   wstarts=origin+np.arange(n)*step
   member=(stops[:,None]>=wstarts)&(starts[:,None]<=wstarts+width)
   assert member.any(axis=1).all()
   copies[k]+=member.sum(); nonempty[k]+=member.any(axis=0).sum()
   co=sum(bool((member[i]&member[j]).any()) for i,j in pos)
   kept[k]+=co; macro[k].append(co/len(pos))
assert tracks_total==54792 and positive_total==110733
keys=['self_support','other_same_identity_support','identity_internal_union_gap','outside_identity_envelope']
assert classes.tolist()==[a['full_scene_q64_fragment_geometry']['categories'][key]['count'] for key in keys]
for k,s in enumerate(b['schemes']):
 assert int(kept[k])==s['positive_pairs_co_window_at_least_once']
 assert int(copies[k])==s['track_copies'] and int(nonempty[k])==s['nonempty_windows']
 assert abs(float(np.mean(macro[k]))-s['positive_pair_retention_scene_equal_mean'])<1e-12
result={'status':'PASS','scenes':512,'tracks':tracks_total,'positive_pairs':positive_total,'geometry_counts':dict(zip(keys,map(int,classes))),'window_retained_pairs':kept.tolist(),'window_track_copies':copies.tolist(),'window_nonempty_counts':nonempty.tolist(),'method':'independent boolean query/track and window/track matrices; no imported study routines; first 512 T_core cached metadata records only','zip_opened':False,'reference_opened':False,'test_opened':False,'gpu_used':False,'elapsed_seconds':time.monotonic()-started}
(root/'root_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
