"""Read-only complete physical/paper attribute reports from frozen mps artifacts."""
from __future__ import annotations
import argparse,json,gzip,sys,time,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation.scene import _regions
from association.evaluation.paper_fusion_metrics import paper_fusion_scene
from association.evaluation.tmtaff import aggregate_fusion_reports

def read(path):return json.loads(path.read_text())
def account(root,out):
    detail=out.with_name(out.stem+'-scenes.json.gz')
    if out.exists() or detail.exists():raise FileExistsError('new accounting output required')
    start=time.monotonic();record=read(root/'run.json');names=record['methods'];scenes=[];result=dict(status='COMPLETED',exit_code=0,run=str(root),new_predictions=0,new_fits=0,splits={})
    for split in record.get('evaluation_splits',['T_core','V_select']):
        ids=record['training_scene_ids'] if split=='T_core' else record['scene_ids'];reports={n:[] for n in names}
        for sid in ids:
            with np.load(root/'private'/f'states-{split}-{sid}.npz',allow_pickle=False) as z:
                row=dict(scene_id=sid,split=split,methods={})
                for name in names:
                    prefix=name+'_'
                    if str(z[prefix+'state_speed_unit'])!='mps':raise ValueError('explicit canonical saved mps required')
                    prediction,truth,valid,available,observed=[z[prefix+k] for k in ['aligned_prediction','aligned_truth','aligned_truth_valid','aligned_prediction_valid','aligned_observed']];nt=len(z[prefix+'target_ids'])
                    value=dict(full_target_metrics=_regions(prediction,truth,valid,available,observed,speed_unit='mps'),paper_fusion_metrics=paper_fusion_scene(prediction[:nt],truth[:nt],valid[:nt],prediction_valid=available[:nt],prediction_speed_unit='mps',target_speed_unit='mps'))
                    row['methods'][name]=value;reports[name].append(value)
                scenes.append(row)
        pooled={n:aggregate_fusion_reports(v) for n,v in reports.items()};summary=read(root/('T-summary.json' if split=='T_core' else 'summary.json'))
        for n,v in pooled.items():
            for k in ['mae','rmse','intersection_count','available_target_count']:
                if not np.isclose(v['all']['position'][k],summary['all_components']['methods'][n]['position'][k],rtol=1e-10,atol=1e-8):raise ValueError('position reaggregation changed')
        result['splits'][split]=dict(scenes=len(ids),methods=pooled)
    with gzip.open(detail,'wt',encoding='utf8') as f:json.dump(scenes,f)
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='post-run same-evaluator saved-array full four-attribute accounting; not independent metric audit, new prediction or author ESD validation')
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();account(a.run.resolve(),a.output.resolve())
