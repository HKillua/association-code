from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,unittest
import numpy as np
from tools.analysis import local_expression_pilot as p
from tools.analysis.audit_temporal_source_outputs import audit
from tools.analysis.saved_attribute_accounting import account
from tests.integration.test_source_risk_pilot import scene_arrays

class LocalExpressionIOTests(unittest.TestCase):
    def test_frozen_heads_predict_before_reference_and_same_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=root/'127';value=root/'125';partition=root/'114';source=root/'133';head=root/'155';data=root/'151';out=root/'159'
            for d in [value/'private',partition/'private',source/'private',head/'private',data/'private']:d.mkdir(parents=True)
            manifest=root/'manifest.json';p.common.old.frozen.write(manifest,dict(splits={'T_core':['1','3']},components={'1':'a','3':'b','2':'v1','4':'v2'}))
            paths=dict(native_run=native,value_run=value,partition_run=partition,source_risk_run=source);oldmodel={}
            for fold in ['full','leave_a','leave_b']:
                oldmodel.update({fold+'__mean':np.zeros(10),fold+'__std':np.ones(10),fold+'__beta':np.zeros(66)})
            np.savez_compressed(head/'private/models.npz',**oldmodel);p.common.old.frozen.write(head/'run.json',dict(status='COMPLETED',analytic_head_fits=13,fit_training_scene_ids=['1','3']));p.common.old.frozen.write(data/'private/index.json',{'T_core':{}})
            for split,sid in [('T_core','1'),('T_core','3'),('V_select','2'),('V_select','4')]:
                m,ev,membership,_=scene_arrays(sid)
                for d in ['native-motion','native-evaluation']:(native/'private'/d/split).mkdir(parents=True,exist_ok=True)
                np.savez_compressed(native/'private/native-motion'/split/f'scene-{sid}.npz',**m);np.savez_compressed(native/'private/native-evaluation'/split/f'scene-{sid}.npz',**ev)
                if split=='T_core':np.savez_compressed(value/'private'/f'T-features-{sid}.npz',membership=membership)
                else:p.common.old.frozen.write(partition/'private'/f'scene-{sid}.json.gz',dict(memberships={'WS':membership.tolist()}))
                m,g,mean,pack,_=p.physical(paths,split,sid);scores=np.zeros(pack['available'].shape);raw=p.recover(m,g,pack,scores,mean)
                np.savez_compressed(source/'private'/(f'oof-{sid}.npz' if split=='T_core' else f'policy-{sid}.npz'),**{('QUADRATIC__scores' if split=='T_core' else p.BASE+'__scores'):scores})
                np.savez_compressed(head/'private'/f'raw-{split}-{sid}.npz',**{p.NEW:raw})
                if split=='T_core':np.savez_compressed(source/'private'/f'training-{sid}.npz',**{k:v for k,v in p.source.training_labels(pack,g,ev).items() if isinstance(v,np.ndarray)})
            original=p.common.evaluation
            def guard(native,split,sid,m):
                self.assertTrue((out/'private'/f'raw-{split}-{sid}.npz').exists());return original(native,split,sid,m)
            a=SimpleNamespace(**paths,manifest=manifest,output=out,head_run=head,data_run=data,device='cpu')
            with patch.object(p.common,'metadata',return_value=(['1','3'],['2','4'])),patch.object(p.common,'evaluation',side_effect=guard):p.run(a)
            audit(out,out/'independent-output-audit.json');account(out,out/'four-attribute-accounting.json')
            self.assertEqual(p.common.old.frozen.read(out/'run.json')['optimizer_updates'],1536)

if __name__=='__main__':unittest.main()
