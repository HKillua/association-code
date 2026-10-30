from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,unittest
import numpy as np
from tools.analysis import t128_frozen_full_scope as p
from tools.analysis.audit_temporal_source_outputs import audit
from tools.analysis.saved_attribute_accounting import account
from tests.integration.test_source_risk_pilot import scene_arrays
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.query_contribution import query_contribution

class FrozenFullScopeTests(unittest.TestCase):
    def test_full_reference_registry_mps_controls_and_independent_V_only_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'151';head=root/'155';base=root/'147';out=root/'157';index={}
            for d in [data/'private/motion/V_select',data/'private/evaluation/V_select',head/'private',base/'private/native-motion']:d.mkdir(parents=True)
            np.savez_compressed(head/'private/models.npz',full__mean=np.zeros(10),full__std=np.ones(10),full__beta=np.zeros(66))
            for sid,c in [('1','a'),('2','b'),('3','a'),('4','b')]:
                m,ev,membership,_=scene_arrays(sid);g=p.common.old.groups_from_membership(membership,4);first=[v[0] for v in g]
                pack=p.risk.build_source_features(**{k:m[k] for k in p.source.MOTION_KEYS},groups=g);mean=ordinary_fusion(m['states'],m['nearest_seconds'],m['support'],g)['predictions']['mean_supported'][first]
                hard=p.risk.recover_source_policy(states=m['states'],support=m['support'],groups=g,pack=pack,scores=np.zeros(pack['available'].shape),kind='hard',mean_states=mean)['group_states']
                raw={p.BASE:hard,'WS_MEAN':mean,'WS_NEAREST_SUPPORTED':query_contribution(m['states'],m['nearest_seconds'],m['support'],m['sources'],g,method='nearest_supported')['group_states']}
                # Real kn->mps boundary creates the frozen147 artifacts.
                _,_,art=p.common.evaluate_predictions(m,g,pack,ev,raw,{},method_names=p.METHODS[:-1],split='V_select',prediction_speed_unit='kn')
                p.common.save_artifacts(base/'private'/f'states-{sid}.npz',art);np.savez_compressed(base/'private/native-motion'/f'scene-{sid}.npz',**m)
                np.savez_compressed(base/'private'/f'policy-{sid}.npz',**{k:v for k,v in pack.items() if isinstance(v,np.ndarray)},membership=membership,node_ids=m['node_ids'],query_seconds=m['query_seconds'],scores=np.zeros(pack['available'].shape))
                if sid in ['1','2']:np.savez_compressed(head/'private'/f'raw-V_select-{sid}.npz',**{p.NEW:hard})
                nm=dict(m,states=m['states'].copy());nm['states'][...,2]*=1852/3600;mp=data/'private/motion/V_select'/f'scene-{sid}.npz';ep=data/'private/evaluation/V_select'/f'scene-{sid}.npz'
                np.savez_compressed(mp,**nm);np.savez_compressed(ep,**ev);index[sid]=dict(evaluation=str(ep),component_id=c)
            original=p.common.load_npz
            def guard(path,*args):
                if '/evaluation/' in str(path):
                    sid=Path(path).stem.split('-')[-1];self.assertTrue((out/'private'/f'raw-V_select-{sid}.npz').exists())
                return original(path,*args)
            a=SimpleNamespace(data_run=data,head_run=head,baseline_run=base,output=out,time_policy='advisory')
            with patch.object(p,'population',return_value=(index,['1','2'])),patch.object(p.common,'load_npz',side_effect=guard):p.run(a)
            audit(out,out/'independent-output-audit.json');result=p.common.old.frozen.read(out/'independent-output-audit.json')
            self.assertEqual(result['splits']['V_select']['scenes'],4);self.assertNotIn('T_core',result['splits'])
            summary=p.common.old.frozen.read(out/'summary.json');self.assertGreater(summary['all_components']['methods'][p.NEW]['position']['missing_prediction_count'],0)
            account(out,out/'four-attribute-accounting.json');full=p.common.old.frozen.read(out/'four-attribute-accounting.json')
            self.assertEqual(full['splits']['V_select']['methods'][p.NEW]['all']['speed']['unit'],'mps')

if __name__=='__main__':unittest.main()
