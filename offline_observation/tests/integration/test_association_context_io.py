from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from association.models.evidence_association import EvidenceAssociation,FEATURE_NAMES
from association.evaluation.local_source_evidence import ResidualScore
from tools.analysis import association_context_pilot as p
from tools.analysis import audit_association_context_outputs as audit
from tools.analysis.saved_attribute_accounting import account
from tests.integration.test_source_risk_pilot import scene_arrays

class AssociationContextIOTests(unittest.TestCase):
    def test_frozen_teacher_to_real_head_saved_variable_group_metrics_and_audit(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'168';out.mkdir();(out/'private').mkdir();native=root/'127'
            torch.manual_seed(4);teacher=EvidenceAssociation();teacher.network[-1].bias.data.fill_(3.)
            normalizer=dict(feature_names=list(FEATURE_NAMES),fitted_split='T_core',mean=[0.]*22,std=[1.]*22)
            m,ev,mem,_=scene_arrays('1');cache=p.prepare_edges(teacher,normalizer,m,m['pairs'])
            cache.update(scene_id='1',labels=ev['identities'][m['pairs'][:,0]]==ev['identities'][m['pairs'][:,1]])
            bound=p.prepare_edges(teacher,normalizer,m,m['pairs'],cache['scores']);self.assertEqual(bound['teacher_binding_max_abs'],0.)
            with self.assertRaisesRegex(ValueError,'binding gap'):p.prepare_edges(teacher,normalizer,m,m['pairs'],cache['scores']+1.)
            heads,norms,curve,draws=p.train_heads([cache],'cpu',lambda **kw:None,steps=2)
            backend=dict(dimension=10,mean=np.zeros(10),std=np.ones(10),base=dict(mean=np.zeros(10),std=np.ones(10),beta=np.zeros(66)),weights=ResidualScore(10).state_dict())
            methods=tuple(k+'_'+s for k in ('WS','OWN','CTX') for s in ('MEAN','NN'));primary=('CTX_NN_vs_WS_NN','CTX_NN_vs_OWN_NN')
            record=dict(status='COMPLETED',exit_code=0,mode='front',methods=methods,native_run=str(native),training_scene_ids=['1','3'],scene_ids=['2','4'])
            p.common.old.frozen.write(out/'run.json',record)
            for split,ids,fn,rf in [('T_core',['1','3'],'T-summary.json','T-scene-summary.json.gz'),('V_select',['2','4'],'summary.json','scene-summary.json.gz')]:
                rows=[];arrays=[]
                for sid in ids:
                    m,ev,mem,_=scene_arrays(sid)
                    for name,values in [('native-motion',m),('native-evaluation',ev)]:
                        folder=native/'private'/name/split;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/f'scene-{sid}.npz',**values)
                    cache=p.prepare_edges(teacher,normalizer,m,m['pairs']);scores=p.corrected_scores(cache,heads,norms,'cpu')
                    members={k:p.partition.agglomerate(4,m['pairs'],s,'WS')[0] for k,s in scores.items()}
                    groups,packs,raw=p.predictions(m,members,backend,'cpu')
                    np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw)
                    np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',pairs=m['pairs'],scores=scores['WS'],**members,**{k+'_scores':v for k,v in scores.items()})
                    row,arr,art=p.evaluate(m,ev,groups,packs,raw,split,method_names=methods)
                    rec=dict(scene_id=sid,component_id=sid,pairs=m['pairs'],node_ids=m['node_ids'],sources=m['sources'],labels=ev['identities'][m['pairs'][:,0]]==ev['identities'][m['pairs'][:,1]],joint_support_queries=(m['support'][m['pairs'][:,0]]&m['support'][m['pairs'][:,1]]).sum(1))
                    assoc={}
                    for k in scores:
                        assoc[k+'_raw']=p.partition.frozen.evaluate_scenes([dict(rec,scores=scores[k])],p.partition.THRESHOLD)
                        co=members[k][m['pairs'][:,0]]==members[k][m['pairs'][:,1]]
                        assoc[k+'_decoded']=p.partition.frozen.evaluate_scenes([dict(rec,scores=co.astype(float))],.5)
                    row.update(scene_id=sid,component_id=sid,search={},association=assoc);rows.append(row);arrays.append(arr)
                    p.common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art)
                summary=p.finish(rows,arrays,methods,primary);p.common.old.frozen.write(out/fn,summary);p.common.old.frozen.write(out/rf,rows)
            audit.parent.audit(out,out/'independent-output-audit.json',extra_scene=audit.scene,extra_summary=audit.summary)
            account(out,out/'four-attribute-accounting.json')
            self.assertEqual(p.common.old.frozen.read(out/'independent-output-audit.json')['status'],'COMPLETED')

if __name__=='__main__':unittest.main()
