import json,sys,pathlib,collections,csv,io,math
import numpy as np
sys.path.insert(0,'/Users/a1/Projects/association-code')
from association.data import MTADArchive,_name
ROOT=pathlib.Path('/Users/a1/Projects/biye-ronghe/docs/reviews/2026-10-18_dataset_followup_018');ROOT.mkdir(exist_ok=True)
splits=json.loads(pathlib.Path('/Users/a1/Projects/biye-ronghe/docs/reviews/2026-10-18_experiment_reaudit/frozen_scene_lists_20261018.json').read_text())['scene_ids']
S=['2606','1622','1759','4349','4','270','1587','2211'];fields=['time','lat','lon','vel','cou'];out={'scope':{'scenes':{s:next(k for k,v in splits.items() if s in v) for s in S},'T_test_reads':0,'selection':'first four frozen T_core; four previously audited V cases; not representative or random','additional_preview':'scene0 first3 measurement/first2 label rows inspected before subset check; scene0 is previously inspected development anchor, not newly exposed; no reference inspected in this step','units':'official paper kn conflicts official guide m/s; no conversion applied'},'scenes':{},'aggregate':{},'examples':[]}
agg=collections.defaultdict(list)
def q(a):
 a=np.asarray(a);a=a[np.isfinite(a)];return {'n':len(a),'q':dict(zip(['min','p05','p50','p95','p99','max'],np.quantile(a,[0,.05,.5,.95,.99,1]).tolist()))} if len(a) else {'n':0}
def distbear(a,b):
 lat1,lon1,lat2,lon2=map(np.radians,[a[:,1],a[:,2],b[:,1],b[:,2]])
 dp=lat2-lat1;dl=lon2-lon1
 d=2*6371008.8*np.arcsin(np.sqrt(np.clip(np.sin(dp/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dl/2)**2,0,1)))
 bear=np.degrees(np.arctan2(np.sin(dl)*np.cos(lat2),np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dl)))%360
 return d,bear
with MTADArchive('/Users/a1/Downloads/数据集level2.zip') as archive:
 refs={_name(i).split('/')[-1].split('-')[-1].split('.')[0]:i for i in archive._zip.infolist() if '/真实场景/' in _name(i) and _name(i).endswith('.csv')}
 for sid in S:
  assert sid in splits['T_core'] or sid in splits['V_select']
  obs=list(archive._rows(sid,'measurements'));labs=archive.read_labels(sid);payload=archive._zip.read(refs[sid]);rr=list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
  groups=collections.defaultdict(list)
  for i,row in enumerate(obs):groups[(row['source'],row['batch'])].append((i+2,[float(row[k]) for k in fields]))
  labelident={(r['source'],r['batch']):r['mmsi'].rsplit('-',1)[0] for r in labs}
  allowed=set(labelident.values());reference_rows=[r for r in rr if r['MMSI'] in allowed]
  for i,row in enumerate(reference_rows):groups[('reference',row['MMSI'])].append((i+2,[float(row[k]) for k in fields]))
  stats={};
  for source in ['9001','9002','reference']:
   rows=[]; nraw=sum(len(v) for (src,k),v in groups.items() if src==source);dups=0
   for (src,key),values in groups.items():
    if src!=source:continue
    values=sorted(values,key=lambda v:v[1][0]);line=np.array([x[0] for x in values]);a=np.array([x[1] for x in values]);
    if len(a)<2:continue
    valid=np.isfinite(a[:-1]).all(1)&np.isfinite(a[1:]).all(1)&(np.diff(a[:,0])>0)
    dups+=int((np.diff(a[:,0])==0).sum());left=a[:-1][valid];right=a[1:][valid];dt=right[:,0]-left[:,0];d,b=distbear(left,right);geom=d/dt
    data=np.c_[dt,geom,left[:,3],right[:,3],(left[:,3]+right[:,3])/2,np.abs((left[:,4]-b+180)%360-180),np.abs((right[:,4]-b+180)%360-180)]
    rows.extend(data.tolist())
    for ii in np.argsort(-right[:,3])[:2]:
     if right[ii,3]>100:out['examples'].append({'scene':sid,'source':source,'key':key,'prev':dict(zip(fields,left[ii].tolist())),'next':dict(zip(fields,right[ii].tolist())),'geo_speed_mps':float(geom[ii]),'bearing_deg':float(b[ii]),'stored_next_div_geo':float(right[ii,3]/geom[ii]) if geom[ii]>0 else None})
   a=np.array(rows);agg[source].extend(rows); stats[source]={'raw_rows':nraw,'adjacent_valid_pairs':len(a),'zero_dt_pairs':dups,'dt':q(a[:,0]),'geo_speed_mps':q(a[:,1]),'stored_next_speed':q(a[:,3]),'ratio_next_div_geo_all_positive':q(a[a[:,1]>1e-6,3]/a[a[:,1]>1e-6,1]),'heading_error_next_deg':q(a[:,6]),'heading_error_prev_deg':q(a[:,5])}
  out['scenes'][sid]=stats
for source,rows in agg.items():
 a=np.array(rows);good=(a[:,0]>=1)&(a[:,0]<=600)&(a[:,1]>.5)&(a[:,1]<100)&(a[:,3]>0)
 # diagnostic slice only, all counts/results retained separately; does not alter train/eval population
 b=a[good]; per={}
 for col,label in [(2,'previous'),(3,'next'),(4,'mean_endpoints')]:
  ratios=b[:,col]/b[:,1];per[label]={'ratio':q(ratios),'if_stored_mps_abs_error':q(np.abs(b[:,col]-b[:,1])),'if_stored_knots_abs_error':q(np.abs(b[:,col]*(1852/3600)-b[:,1]))}
 out['aggregate'][source]={'all_adjacent_pairs':len(a),'dt':q(a[:,0]),'geo_speed_mps':q(a[:,1]),'stored_next_speed':q(a[:,3]),'heading_error_next_deg':q(a[:,6]),'heading_error_previous_deg':q(a[:,5]),'diagnostic_slice':'1<=dt<=600 seconds, .5<geometry speed<100 m/s, next vel>0; statistics only, not cleaning','diagnostic_pairs':len(b),'unit_checks':per}
out['examples']=sorted(out['examples'],key=lambda x:-x['next']['vel'])[:24]
(ROOT/'source_semantics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out['aggregate'],ensure_ascii=False,indent=2));print('TOP',json.dumps(out['examples'][:3],ensure_ascii=False))
for source,rows in agg.items():
 a=np.array(rows);g=a[:,1]>1e-9;b=a[g]
 out['aggregate'][source]['all_positive_geometry_checks']={'pairs':len(b),'ratio_previous_vel_div_geom':q(b[:,2]/b[:,1]),'approx_knots_factor_0_514_error_mps':q(np.abs(b[:,2]*.514-b[:,1])),'within_0_1_mps':int((np.abs(b[:,2]*.514-b[:,1])<.1).sum()),'within_1_percent':int((np.abs(b[:,2]*.514/b[:,1]-1)<.01).sum()),'geo_speed_gt100_mps':int((b[:,1]>100).sum()),'geo_speed_gt1000_mps':int((b[:,1]>1000).sum()),'source_course_before_interval_error_lt_1deg':int((b[:,5]<1).sum()),'source_course_after_interval_error_lt_1deg':int((b[:,6]<1).sum())}
(ROOT/'source_semantics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('ADDITIONAL',json.dumps({s:out['aggregate'][s]['all_positive_geometry_checks'] for s in agg},ensure_ascii=False,indent=2))
