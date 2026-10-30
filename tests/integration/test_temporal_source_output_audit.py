from pathlib import Path
import tempfile,unittest
import numpy as np
from tools.analysis import temporal_source_pilot as p
from tools.analysis import audit_temporal_source_outputs as audit
from tests.integration.test_source_risk_pilot import scene_arrays

class TemporalOutputAuditTests(unittest.TestCase):
    def test_explicit_T_and_V_artifacts_and_wrong_split_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);native=root/'native';run=root/'science';(run/'private').mkdir(parents=True)
            names=(p.BASE,*p.ZERO)
            record=dict(status='COMPLETED',exit_code=0,mode='zero',methods=list(names),native_run=str(native),training_scene_ids=['1','2'],scene_ids=['3','4'])
            for split,ids,sf,rf in [('T_core',['1','2'],'T-summary.json','T-scene-summary.json.gz'),('V_select',['3','4'],'summary.json','scene-summary.json.gz')]:
                rows=[];arrays=[]
                for sid,component in zip(ids,['A','B']):
                    motion,ev,membership,artifact=scene_arrays(sid)
                    groups=p.old.groups_from_membership(membership,4)
                    pack=p.risk.build_source_features(**{k:motion[k] for k in p.source.MOTION_KEYS},groups=groups)
                    prediction=artifact['group_states']
                    row,arr,artifacts=p.common.evaluate_predictions(motion,groups,pack,ev,{n:prediction for n in names},{},method_names=names,split=split,prediction_speed_unit='kn')
                    row.update(scene_id=sid,component_id=component);p.add_harm(row,arr)
                    p.common.save_artifacts(run/'private'/f'states-{split}-{sid}.npz',artifacts)
                    for category,data in [('native-motion',motion),('native-evaluation',ev)]:
                        directory=native/'private'/category/split;directory.mkdir(parents=True,exist_ok=True);np.savez_compressed(directory/f'scene-{sid}.npz',**data)
                    rows.append(row);arrays.append(arr)
                p.old.frozen.write(run/sf,p.summarize(rows,arrays,names));p.old.frozen.write(run/rf,rows)
            p.old.frozen.write(run/'run.json',record)
            audit.audit(run,run/'audit.json')
            result=p.old.frozen.read(run/'audit.json')
            self.assertEqual(result['status'],'COMPLETED')
            self.assertEqual(set(result['splits']),{'T_core','V_select'})
            file=run/'private/states-T_core-1.npz';z=audit.load(file)
            for key in z:
                if key.endswith('_split'):z[key]=np.asarray('V_select')
            np.savez_compressed(file,**z)
            with self.assertRaisesRegex(ValueError,'policy'):audit.audit(run,run/'wrong.json')
            self.assertTrue((run/'wrong.failed.json').exists())
