"""177 prescribed supported/all-fragment split, independently recounted on saved harm cases."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis import observation_common as c


def distance(a,b):
    x,y=np.deg2rad(a[...,0]),np.deg2rad(b[...,0]);d=np.deg2rad((a[...,1]-b[...,1]+180.)%360.-180.)
    h=np.clip(np.sin((x-y)/2)**2+np.cos(x)*np.cos(y)*np.sin(d/2)**2,0,1)
    return 2*6371000.*np.arctan2(np.sqrt(h),np.sqrt(1-h))


def run(root):
    began=time.monotonic();record=c.old.frozen.read(root/'run.json');assert record['status']=='COMPLETED';parent=Path(record['parent_run']);origin=c.old.frozen.read(parent/'run.json');native=Path(origin['native_run']);summary=c.old.frozen.read(root/'summary.json');results={}
    for sid in c.old.frozen.read(origin['manifest'])['splits']['T_core'][:32]:
        m=c.load_npz(native/'private/native-motion/T_core'/f'scene-{sid}.npz');ev=c.load_npz(native/'private/native-evaluation/T_core'/f'scene-{sid}.npz');z=c.load_npz(parent/'private'/f'states-T_core-{sid}.npz');row=c.old.frozen.read(root/'scenes'/f'T_core-{sid}.json.gz')
        for key,values in row['comparisons'].items():
            new,old=key.split('_vs_');a={k[len(new+'_NN_'):]:v for k,v in z.items() if k.startswith(new+'_NN_')};b={k[len(old+'_NN_'):]:v for k,v in z.items() if k.startswith(old+'_NN_')}
            groups=[a['group_node_indices'][l:r] for l,r in zip(a['group_offsets'][:-1],a['group_offsets'][1:])];counts=results.setdefault(key,{})
            for example in values['examples']:
                t,j=example['target_index'],example['query_index'];g=int(a['aligned_group_indices'][t]);members=groups[g];truth=ev['target_states'][t,j]
                before=float(distance(b['aligned_prediction'][t,j],truth));after=float(distance(a['aligned_prediction'][t,j],truth));assert abs(after-before-example['increase_m'])<1e-6
                all_errors=distance(m['states'][members,j],truth);visible=m['support'][members,j];supported=float(all_errors[visible].min()) if visible.any() else np.inf
                any_fragment=float(all_errors.min());compatible=[i for i,v in enumerate(groups) if str(ev['identities'][v[0]])==str(ev['target_ids'][t])]
                best_output=float(distance(a['group_states'][compatible,j],truth).min());assert (best_output-before<=1000)==example['compatible_output_oracle_removes_new_harm']
                active=supported if example['new_sources']==1 else any_fragment;assert (active-before<=1000)==example['fragment_oracle_removes_new_harm']
                regions=['all','pure' if example['candidate_group_pure'] else 'mixed','new_zero' if example['new_sources']==0 else 'new_one' if example['new_sources']==1 else 'new_multi']
                if example['candidate_group_pure']:regions.append('pure_zero' if example['new_sources']==0 else 'pure_one' if example['new_sources']==1 else 'pure_multi')
                for region in regions:
                    item=counts.setdefault(region,dict(harm_points=0,all_group_fragment_oracle_removes_harm=0,supported_fragment_oracle_removes_harm=0,compatible_output_oracle_removes_harm=0,supported_fragment_candidate_count_zero=0))
                    item['harm_points']+=1;item['all_group_fragment_oracle_removes_harm']+=int(any_fragment-before<=1000);item['supported_fragment_oracle_removes_harm']+=int(supported-before<=1000);item['compatible_output_oracle_removes_harm']+=int(best_output-before<=1000);item['supported_fragment_candidate_count_zero']+=int(not visible.any())
    for key,v in results.items():assert v['all']['harm_points']==summary['comparisons'][key]['strata']['all']['new_harm_over_1km']
    result=dict(status='COMPLETED',exit_code=0,scope='T-only saved harm cases independent distance recount plus preplanned supported/all-fragment diagnostic bounds; no new deployment or fit; oracle not deployable',comparisons=results,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=c.old.frozen.peak_rss_bytes())
    dest=root/'supported-all-fragment-accounting.json'
    if dest.exists():raise FileExistsError(dest)
    c.old.frozen.write(dest,result);print(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();run(a.run.resolve())
