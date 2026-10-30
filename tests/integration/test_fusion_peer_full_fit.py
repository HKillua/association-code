from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from unittest.mock import patch
import numpy as np
import torch
from tools.experiments import fusion_peer_full_fit as p
from tests.integration.test_source_risk_pilot import scene_arrays

class FullPeerFitTests(unittest.TestCase):
    def test_real_backward_both_ptfm_branches_and_deepaf_PV(self):
        motion,ev,_,_=scene_arrays('1');motion['state_speed_unit']=np.asarray('mps')
        group=dict(members=[0,1],target_index=0,segments=[s['query_indices'].tolist() for s in p.sensor_segments(motion,np.array([0,1]))])
        r=p.row(motion,ev,group)
        norm=dict(mean=[0.]*4,scale=[1.]*4,split='T_core',truth_used=False)
        for method,model,prefixes in [('ptfm',p.PTFM(norm,sensor_branches=True),('anchor','helper')),
                                      ('deepaf',p.deep.DeepAFPeer(),('position_network','velocity_network'))]:
            before={k:v.clone() for k,v in model.state_dict().items()}
            result=p.train_batch(model,[r],method,torch.optim.Adam(model.parameters(),lr=.001),'cpu')
            self.assertTrue(np.isfinite(result['loss']))
            for name in prefixes:self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith(name)))

    def test_full_registry_and_core_missing_kept_at_common_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);index={};partition=root/'partition';hard=root/'147';output=root/'out'
            (partition/'private').mkdir(parents=True);(hard/'private').mkdir(parents=True)
            (output/'private').mkdir(parents=True);(output/'scenes').mkdir()
            for sid,component in [('1','A'),('2','B')]:
                motion,ev,membership,_=scene_arrays(sid);motion['state_speed_unit']=np.asarray('mps')
                for k in ('states','support','nearest_seconds'):motion['cap128_'+k]=motion[k].copy()
                mp=root/(sid+'-motion.npz');ep=root/(sid+'-ev.npz');np.savez_compressed(mp,**motion);np.savez_compressed(ep,**ev)
                index[sid]=dict(motion=str(mp),evaluation=str(ep),component_id=component)
                groups=p.common.old.groups_from_membership(membership,4);first=[g[0] for g in groups]
                mean=p.ordinary_fusion(motion['states'],motion['nearest_seconds'],motion['support'],groups)['predictions']['mean_supported'][first]
                np.savez_compressed(hard/'private'/f'states-{sid}.npz',WS_SHARED_Q_HARD_node_group_indices=membership,
                    WS_SHARED_Q_HARD_state_speed_unit=np.asarray('mps'),WS_SHARED_Q_HARD_group_states=mean,WS_MEAN_group_states=mean)
                p.common.old.frozen.write(partition/'private'/f'scene-{sid}.json.gz',dict(memberships=dict(WS=membership.tolist())))
            args=SimpleNamespace(method='ptfm',partition_run=partition,qhard_run=hard,device='cpu')
            model=p.PTFM(dict(mean=[0.]*4,scale=[1.]*4,split='T_core',truth_used=False),sensor_branches=True)
            original=p.load
            def guarded(path):
                if str(path).endswith('-ev.npz'):
                    sid=Path(path).name.split('-')[0]
                    self.assertTrue((output/'private'/f'raw-V_select-{sid}.npz').exists())
                return original(path)
            with patch.object(p,'load',guarded):p.evaluate(model,args,index,output,lambda **kw:None)
            result=p.common.old.frozen.read(output/'summary.json');methods=result['all_components']['methods']
            self.assertEqual(result['all_components']['scene_count'],2)
            self.assertLess(methods['WS_PTFM_CORE']['position']['intersection_count'],methods['WS_PTFM_PLUS_MEAN']['position']['intersection_count'])
            self.assertGreater(methods['WS_PTFM_PLUS_MEAN']['position']['missing_prediction_count'],0)
            self.assertEqual(methods['WS_MEAN']['position']['available_target_count'],methods['WS_PTFM_CORE']['position']['available_target_count'])
