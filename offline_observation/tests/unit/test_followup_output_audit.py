import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.analysis import audit_followup_outputs as a


class FollowupAuditTests(unittest.TestCase):
    def test_complete_io_and_corrupt_reference_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run, native = root/'run', root/'native'
            (run/'private').mkdir(parents=True)
            for key in ('native-motion', 'native-evaluation'):
                (native/'private'/key/'T_core').mkdir(parents=True)
                (native/'private'/key/'V_select').mkdir(parents=True)
            names = ('WS_SHARED_Q_HARD', 'WS_GLOBAL_ACTUAL_MIX'); pair = names[1]+'_vs_'+names[0]
            for split in ('T_core', 'V_select'):
                rows=[]; methods={}; per={}
                for sid in ('0', '1'):
                    truth=np.zeros((1, 2, 4)); valid=np.ones_like(truth, bool)
                    ev=dict(target_ids=np.array(['a']), target_states=truth, target_valid=valid, identities=np.array(['a','a']))
                    np.savez(native/'private/native-evaluation'/split/f'scene-{sid}.npz', **ev)
                    np.savez(native/'private/native-motion'/split/f'scene-{sid}.npz', states=np.zeros((2,2,4)),support=np.ones((2,2),bool),sources=np.array(['x','y']))
                    payload={}; row=dict(scene_id=sid,component_id='c'+sid,methods={})
                    for name, longitude in zip(names,(1.,.5)):
                        prediction=truth.copy(); prediction[...,1]=longitude
                        art=dict(target_ids=ev['target_ids'],aligned_truth=truth,aligned_truth_valid=valid,
                            aligned_prediction=prediction,aligned_prediction_valid=valid,group_states=prediction,
                            group_node_indices=np.array([0,1]),group_offsets=np.array([0,2]),aligned_group_indices=np.array([0]))
                        payload.update({name+'_'+k:v for k,v in art.items()})
                        per[name]=a.distance(prediction,truth).ravel()
                        methods[name]=dict(position={**a.stats(per[name]),**a.coverage(art)})
                        row['methods'][name]=dict(position=dict(all=methods[name]['position']))
                    np.savez(run/'private'/f'states-{split}-{sid}.npz', **payload)
                    rows.append(row)
                delta=per[names[1]]-per[names[0]]; change=float(delta.mean())
                component=dict(methods=methods)
                full={}
                for name in names:
                    pop=methods[name]['position']
                    full[name]=dict(position={**a.stats(np.tile(per[name],2)), **{k:2*pop[k] for k in a.coverage(art)}})
                summary=dict(all_components=dict(methods=full),by_component={'c0':component,'c1':component},
                    without_largest_component=component,primary_bootstrap={pair:dict(point_estimate_m=change,percentile95_m=[change,change])},
                    additional_actual_harm=dict(all_components={pair:a.harm(np.tile(delta,2))},
                        by_component={c:{pair:a.harm(delta)} for c in ('c0','c1')},without_largest_component={pair:a.harm(delta)}))
                (run/('T-summary.json' if split=='T_core' else 'summary.json')).write_text(json.dumps(summary))
                with gzip.open(run/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),'wt') as f: json.dump(rows,f)
            (run/'run.json').write_text(json.dumps(dict(status='COMPLETED',exit_code=0,native_run=str(native),training_scene_ids=['0','1'],scene_ids=['0','1'])))
            a.audit(run,run/'audit.json')
            result=json.loads((run/'audit.json').read_text());self.assertEqual(result['status'],'COMPLETED')
            self.assertAlmostEqual(result['splits']['V_select']['primary'][pair]['delta_mae_m'],-6371000*np.pi/360)
            path=run/'private/states-T_core-0.npz'; data=a.load(path);data[names[0]+'_aligned_truth'][0,0,0]=.1
            np.savez(path,**data)
            with self.assertRaisesRegex(ValueError,'native reference'): a.audit(run,run/'corrupt.json')


if __name__=='__main__': unittest.main()
