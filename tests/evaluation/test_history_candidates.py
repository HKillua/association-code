import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from association.evaluation import history_candidates as h
from association.evaluation.native_temporal import fit_quadratic
from association.evaluation.local_source_evidence import fit_residual,predict_residual
from tests.integration.test_source_risk_pilot import scene_arrays


class HistoryCandidateTests(unittest.TestCase):
    def fixture(self):
        m,ev,_,_=scene_arrays('1');m['sources']=np.array(['red','green','blue','red'])
        m['support'][:]=True;m['support'][:,0]=False;m['support'][1:,1]=False
        return m,ev,[[0,1,2],[3]]

    def test_features_source_rename_permutation_k3_and_speed_units(self):
        m,ev,g=self.fixture();x,c=h.descriptors(m,g)
        renamed=dict(m,sources=np.array(['9','a','00','9']),states=m['states'].copy());renamed['states'][...,2]*=1852/3600
        xx,cc=h.descriptors(renamed,g);np.testing.assert_array_equal(c,cc);np.testing.assert_allclose(x,xx,atol=1e-12)
        order=np.array([2,0,3,1]);inverse=np.argsort(order);mm={k:(v[order] if isinstance(v,np.ndarray) and len(v)==4 and k!='query_seconds' else v) for k,v in m.items()}
        gg=[inverse[np.array(group)].tolist() for group in g];xxx,ccc=h.descriptors(mm,gg)
        np.testing.assert_array_equal(c,ccc);np.testing.assert_allclose(x,xxx[inverse],atol=1e-12)
        self.assertEqual(x.shape,(4,64,12));self.assertEqual(c[0,2],3)

    def test_actions_only_change_position_in_declared_low_support_domain(self):
        m,_,g=self.fixture();_,count=h.descriptors(m,g);score=np.ones((4,64));score[1]=-1
        base=np.stack([m['states'][0],m['states'][3]]);out,w=h.readouts(m,g,count,score,base)
        np.testing.assert_array_equal(out['M'][0,0],base[0,0]);np.testing.assert_array_equal(out['M'][0,1,:2],m['states'][0,1,:2])
        np.testing.assert_array_equal(out['O'][0,:2,:2],m['states'][1,:2,:2]);np.testing.assert_array_equal(out['Z'][:,1],base[:,1]);np.testing.assert_array_equal(out['1'][:,0],base[:,0])
        for key in out:
            np.testing.assert_array_equal(out[key][...,2:],base[...,2:]);np.testing.assert_array_equal(out[key][0,2:],base[0,2:])
        self.assertEqual(w['M'][1,1],0);self.assertEqual(w['O'][1,1],1)
        # Exact ties share mass per source, even when one source has two fragments.
        m['sources']=np.array(['red','green','red','blue']);g=[[0,1,2,3]];_,count=h.descriptors(m,g)
        _,w=h.readouts(m,g,count,np.zeros((4,64)),base[:1]);np.testing.assert_allclose(w['O'][:,0],[1/6,1/3,1/6,1/3])

    def test_labels_keep_first_target_and_account_duplicate_unknown_invalid(self):
        m,ev,_=self.fixture();groups=[[0],[1],[2],[3]];_,c=h.descriptors(m,groups)
        ev['identities']=np.array(['t0','t0','unknown','t1']);z=h.labels(m,groups,c,ev)
        self.assertEqual(z['audit']['duplicate_groups_without_labels'],1);self.assertEqual(z['audit']['unknown_groups'],1)
        self.assertFalse(z['labeled'][1:3].any());self.assertFalse(z['labeled'][0,2]);self.assertEqual(z['audit']['claimed_targets'],2)
        self.assertEqual(z['audit']['group_queries'],127);self.assertAlmostEqual(z['w'].sum(),127.)

    def test_real_fit_checkpoint_inference_without_labels(self):
        torch.set_num_threads(1);m,ev,g=self.fixture();x,c=h.descriptors(m,g);z=h.labels(m,g,c,ev);use=z['labeled']
        base=fit_quadratic(x[use],z['y'][use],z['w'][use]);model=fit_residual(x[use],z['y'][use],z['w'][use],base,device='cpu',steps=2)
        self.assertEqual(len(base['beta']),91);self.assertEqual(model['parameter_count'],1505);self.assertEqual(model['updates'],2)
        with TemporaryDirectory() as t:
            path=Path(t)/'model.pt';torch.save(model,path);loaded=torch.load(path,weights_only=False)
            np.testing.assert_array_equal(predict_residual(model,x.reshape(-1,12),'cpu'),predict_residual(loaded,x.reshape(-1,12),'cpu'))

if __name__=='__main__':unittest.main()
