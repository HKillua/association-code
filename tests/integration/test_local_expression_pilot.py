from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,unittest
import numpy as np
from tools.analysis import local_expression_pilot as p
from tools.analysis.audit_temporal_source_outputs import audit
from tools.analysis.saved_attribute_accounting import account
from tools.analysis import objective_difficulty_pilot as objective
from tools.analysis import group_reference_counterfactual as counterfactual
from tools.analysis.audit_group_reference_population import audit as audit_counterfactual
from tools.analysis import set_conditioned_risk_pilot as context
from tools.analysis import truth_free_partition_pilot as regroup
from tools.analysis.audit_variable_group_outputs import audit as audit_regroup
from tools.analysis import association_context_pilot as assoc_context
from tools.analysis import audit_association_context_outputs as assoc_audit
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
                np.savez_compressed(source/'private'/(f'oof-{sid}.npz' if split=='T_core' else f'policy-{sid}.npz'),**{k:v for k,v in pack.items() if isinstance(v,np.ndarray)},membership=membership,**{('QUADRATIC__scores' if split=='T_core' else p.BASE+'__scores'):scores})
                np.savez_compressed(head/'private'/f'raw-{split}-{sid}.npz',**{p.NEW:raw})
                if split=='T_core':np.savez_compressed(source/'private'/f'training-{sid}.npz',**{k:v for k,v in {**pack,**p.source.training_labels(pack,g,ev)}.items() if isinstance(v,np.ndarray)})
            original=p.common.evaluation
            def guard(native,split,sid,m):
                self.assertTrue((out/'private'/f'raw-{split}-{sid}.npz').exists());return original(native,split,sid,m)
            a=SimpleNamespace(**paths,manifest=manifest,output=out,head_run=head,data_run=data,device='cpu')
            with patch.object(p.common,'metadata',return_value=(['1','3'],['2','4'])),patch.object(p.common,'evaluation',side_effect=guard):p.run(a)
            audit(out,out/'independent-output-audit.json');account(out,out/'four-attribute-accounting.json')
            self.assertEqual(p.common.old.frozen.read(out/'run.json')['optimizer_updates'],1536)
            b=SimpleNamespace(**paths,manifest=manifest,output=root/'163',expression_run=out)
            with patch.object(p.common,'metadata',return_value=(['1','3'],['2','4'])):counterfactual.run(b)
            audit_counterfactual(root/'163',root/'163'/'independent-population-audit.json')
            self.assertTrue(p.common.old.frozen.read(root/'163'/'summary.json')['oracle_diagnostic_only'])
            expression=out;out=root/'162';b=SimpleNamespace(**paths,manifest=manifest,output=out,head_run=head,expression_run=expression,device='cpu')
            with patch.object(p.common,'metadata',return_value=(['1','3'],['2','4'])),patch.object(p.common,'evaluation',side_effect=guard):objective.run(b)
            audit(out,out/'independent-output-audit.json');account(out,out/'four-attribute-accounting.json')
            self.assertEqual(p.common.old.frozen.read(out/'run.json')['optimizer_updates'],1536)
            out=root/'166';b=SimpleNamespace(**paths,manifest=manifest,output=out,head_run=head,expression_run=expression,device='cpu')
            with patch.object(p.common,'metadata',return_value=(['1','3'],['2','4'])),patch.object(p.common,'evaluation',side_effect=guard):context.run(b)
            audit(out,out/'independent-output-audit.json');account(out,out/'four-attribute-accounting.json')
            self.assertEqual(p.common.old.frozen.read(out/'run.json')['optimizer_updates'],768)
            association=root/'056';association.mkdir();rawscores={}
            p.common.old.frozen.write(partition/'run.json',dict(source_association=str(association)))
            for split,sid in [('T_core','1'),('T_core','3'),('V_select','2'),('V_select','4')]:
                m,ev,mem,_=scene_arrays(sid);pairs=m['pairs'];scores=np.where(mem[pairs[:,0]]==mem[pairs[:,1]],3.,0.)
                if split=='T_core':np.savez_compressed(value/'private'/f'T-features-{sid}.npz',membership=mem,scores=scores)
                else:rawscores.update({sid+'_pairs':pairs,sid+'_scores':scores,sid+'_node_ids':m['node_ids']})
            np.savez_compressed(association/'final-scores.npz',**rawscores)
            out=root/'165';b=SimpleNamespace(**paths,manifest=manifest,output=out,expression_run=expression,device='cpu')
            original_agglomerate=regroup.partition.agglomerate
            def coverage_control(n,pairs,scores,method,*args,**kwargs):
                # Force a legitimate merging control to exercise lost targets;
                # this is wiring coverage, not an assertion about actual CL.
                if method=='CL':return np.zeros(n,np.int64),{},[]
                return original_agglomerate(n,pairs,scores,method,*args,**kwargs)
            with patch.object(p.common,'metadata',return_value=(['1','3'],['2','4'])),patch.object(p.common,'evaluation',side_effect=guard),patch.object(regroup.partition,'agglomerate',side_effect=coverage_control):regroup.run(b)
            audit_regroup(out,out/'independent-output-audit.json');account(out,out/'four-attribute-accounting.json')
            self.assertGreater(p.common.old.frozen.read(out/'summary.json')['all_components']['pairwise']['CL_NN_vs_WS_NN']['coverage_transitions']['lost_target_points'],0)
            refinement=out;out=root/'169'
            b=SimpleNamespace(**paths,manifest=manifest,output=out,expression_run=expression,refinement_run=refinement,data_run=data,device='cpu',mode='support')
            with patch.object(p.common,'metadata',return_value=(['1','3'],['2','4'])),patch.object(p.common,'evaluation',side_effect=guard):assoc_context.run(b)
            audit_regroup(out,out/'independent-output-audit.json',extra_scene=assoc_audit.scene,extra_summary=assoc_audit.summary)
            account(out,out/'four-attribute-accounting.json')

if __name__=='__main__':unittest.main()
