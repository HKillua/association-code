from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,unittest
import numpy as np
from association.data.motion_evidence import build_motion_evidence
from tools.analysis import within_source_forecast_pilot as p
from tools.analysis.audit_temporal_source_outputs import audit
from tests.integration.test_source_risk_pilot import scene_arrays

class ForecastPilotIOTests(unittest.TestCase):
    def test_actual_saved_T_V_pipeline_before_reference_and_output_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=root/'native';value=root/'value';partition=root/'partition';source=root/'133';candidate=root/'149';out=root/'156'
            for d in [value/'private',partition/'private',source/'private',candidate/'private']:d.mkdir(parents=True)
            manifest=root/'manifest.json';p.common.old.frozen.write(manifest,dict(components={'1':'T1','3':'T2','2':'V1','4':'V2'}))
            paths=dict(native_run=native,value_run=value,partition_run=partition,source_risk_run=source)
            for split,sid in [('T_core','1'),('T_core','3'),('V_select','2'),('V_select','4')]:
                old,ev,membership,_=scene_arrays(sid);query=old['query_seconds'];t=np.tile([10.,620.],(4,1));values=np.stack([old['states'][:,1],old['states'][:,-2]],axis=1)
                motion=build_motion_evidence(values,t,np.ones((4,2),bool),query,old['sources']);motion['node_ids']=old['node_ids']
                for name in ['native-motion','native-evaluation','native-events']:(native/'private'/name/split).mkdir(parents=True,exist_ok=True)
                np.savez_compressed(native/'private/native-motion'/split/f'scene-{sid}.npz',**motion);np.savez_compressed(native/'private/native-evaluation'/split/f'scene-{sid}.npz',**ev)
                np.savez_compressed(native/'private/native-events'/split/f'scene-{sid}.npz',node_ids=motion['node_ids'],sources=motion['sources'],query_seconds=query,event_offsets=np.arange(5)*2,event_times_seconds=t.reshape(-1),event_values=values.reshape(-1,4))
                if split=='T_core':np.savez_compressed(value/'private'/f'T-features-{sid}.npz',membership=membership)
                else:p.common.old.frozen.write(partition/'private'/f'scene-{sid}.json.gz',dict(memberships={'WS':membership.tolist()}))
                m,g,mean,pack,_=p.physical(paths,split,sid);scores=np.zeros(pack['available'].shape);base=p.recover(m,g,pack,scores,mean)
                np.savez_compressed(source/'private'/(f'oof-{sid}.npz' if split=='T_core' else f'policy-{sid}.npz'),**{('QUADRATIC__scores' if split=='T_core' else p.BASE+'__scores'):scores})
                nearest,_=p.unsupported_candidates(m,g,base,{},'nearest');np.savez_compressed(candidate/'private'/f'raw-{split}-{sid}.npz',**{p.BASE:base,'WS_Q_ZERO_NEAREST_OLS':nearest})
            original=p.common.evaluation
            def guarded(native,split,sid,m):
                self.assertTrue((out/'private'/f'raw-{split}-{sid}.npz').exists());return original(native,split,sid,m)
            a=SimpleNamespace(**paths,manifest=manifest,output=out,candidate_run=candidate,time_policy='advisory')
            with patch.object(p.common,'metadata',return_value=(['1','3'],['2','4'])),patch.object(p.common,'evaluation',side_effect=guarded):p.run(a)
            audit(out,out/'independent-output-audit.json')
            result=p.common.old.frozen.read(out/'independent-output-audit.json');self.assertEqual(result['status'],'COMPLETED')
            self.assertEqual(result['splits']['T_core']['methods'],6)

if __name__=='__main__':unittest.main()
