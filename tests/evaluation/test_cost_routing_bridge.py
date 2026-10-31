from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from association.training.cost_routing import objective,fit_heads
from association.evaluation.local_source_evidence import ResidualScore,predict_residual
from association.evaluation.native_history_bridge import controls

class CostRoutingTests(unittest.TestCase):
    def test_cost_gradient_mask_permutation_and_tail_cost(self):
        torch.set_num_threads(1);risk=torch.zeros((1,3),requires_grad=True);y=torch.log1p(torch.tensor([[.1,3.,999.]]));mask=torch.tensor([[True,True,False]]);mass=torch.tensor([[.5,.5,0.]])
        loss=objective('COST',risk,y,mask,mass);loss.backward()
        self.assertGreater(risk.grad[0,0],0);self.assertLess(risk.grad[0,1],0);self.assertEqual(float(risk.grad[0,2]),0)
        loss=loss.detach();risk=risk.detach();order=[1,0,2];self.assertAlmostEqual(float(loss),float(objective('COST',risk[:,order],y[:,order],mask[:,order],mass[:,order])),places=6)
        self.assertGreater(float(objective('TAIL',risk,y,mask,mass)),float(loss));self.assertTrue(torch.isfinite(loss))
        equal=torch.full((1,3),2.);self.assertEqual(float(objective('COST',risk,equal,mask,mass)),0.)
        with self.assertRaises(ValueError):objective('COST',risk,y,torch.tensor([[True,False,False]]),mass)

    def test_real_matched_fit_keeps_base_norm_and_label_free_checkpoint(self):
        torch.set_num_threads(1);torch.manual_seed(20);initial=dict(dimension=12,parameter_count=1505,mean=np.zeros(12),std=np.ones(12),base=dict(mean=np.zeros(12),std=np.ones(12),beta=np.zeros(91)),weights=ResidualScore(12).state_dict())
        bags=[dict(x=np.vstack([np.zeros(12),np.ones(12)]),y=np.log1p([.1,3.]),w=np.array([.5,.5])),dict(x=np.zeros((3,12)),y=np.log1p([1.,.1,2.]),w=np.array([.25,.25,.5]))]
        models,draws=fit_heads(bags,initial,'cpu',steps=2);self.assertEqual(draws.shape,(2,128));self.assertEqual(set(models),{'MSE','COST','TAIL'})
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'heads.pt';torch.save(models,path);loaded=torch.load(path,weights_only=False)
            for k,m in models.items():
                self.assertEqual(m['updates'],2);np.testing.assert_array_equal(m['base']['beta'],initial['base']['beta']);np.testing.assert_array_equal(m['mean'],initial['mean'])
                np.testing.assert_array_equal(predict_residual(m,bags[0]['x'],'cpu'),predict_residual(loaded[k],bags[0]['x'],'cpu'))
                from tools.analysis.history_followup_closeout import numpy_score
                np.testing.assert_allclose(numpy_score(m,bags[0]['x']),predict_residual(m,bags[0]['x'],'cpu'),atol=1e-5)

class NativeBridgeTests(unittest.TestCase):
    def test_pilot_primary_pairs_are_available_in_declared_order(self):
        from tools.analysis import cost_routing_pilot,native_history_bridge_pilot
        for pilot in (cost_routing_pilot,native_history_bridge_pilot):
            for key in pilot.PRIMARY:
                right,left=key.split('_vs_');self.assertLess(pilot.METHODS.index(left),pilot.METHODS.index(right))

    def fixture(self):
        q=np.array([5.,10.,30.]);states=np.zeros((5,3,4));states[...,0]=30+np.arange(5)[:,None]*.01;states[...,1]=179.;states[...,2:]=[10.,359.]
        m=dict(states=states,sources=np.array(['A','A','B','B','C']),query_seconds=q)
        times=np.array([0.,1.,19.,20.,0.,1.,19.,20.,0.,1.]);values=np.zeros((10,4));values[:,0]=30.;values[:,1]=[179.,179.,-179.,-179.,170.,170.,172.,172.,100.,100.]
        ev=dict(event_offsets=np.arange(6)*2,event_times_seconds=times,event_values=values,event_valid=np.ones((10,4),bool))
        base=states[:1].copy();base[...,0]=31.;scores=np.broadcast_to(np.array([0.,.5,1.,1.5,-1.])[:,None],(5,3));count=np.array([[0,1,0]])
        return m,ev,base,scores,count

    def test_k3_native_brackets_and_unchanged_domains(self):
        m,ev,base,scores,count=self.fixture();out,flags,diag=controls(m,ev,[[0,1,2,3,4]],count,scores,base)
        self.assertTrue(flags['eligible'][0,0]);self.assertFalse(flags['eligible'][0,2]);self.assertEqual(flags['eligible_source_count'][0,0],2)
        for k in out:
            np.testing.assert_array_equal(out[k][0,1:],base[0,1:]);np.testing.assert_array_equal(out[k][...,2:],base[...,2:])
        np.testing.assert_array_equal(out['R'][0,0,:2],m['states'][0,0,:2]);self.assertAlmostEqual(out['N'][0,0,1],179.);self.assertGreater(abs(out['B'][0,0,1]),179.)
        mm=dict(m,sources=np.array(['99','99','00','00','A']));renamed,_,_=controls(mm,ev,[[4,3,2,1,0]],count,scores,base)
        for k in out:np.testing.assert_allclose(out[k],renamed[k],atol=1e-12)

    def test_no_cross_source_bracketing(self):
        m,ev,base,scores,count=self.fixture();m['sources']=np.array(['left','right','left','right','left'])
        out,flags,_=controls(m,ev,[[0,1,2,3,4]],count,scores,base)
        self.assertFalse(flags['eligible'].any())
        for k in out:np.testing.assert_array_equal(out[k],base)

    def test_independent_native_reconstruction_and_tied_source_weights(self):
        from tools.analysis.history_followup_closeout import native_expected,selected_weights,weighted_positions
        m,ev,base,scores,count=self.fixture();out,flags,_=controls(m,ev,[[0,1,2,3,4]],count,scores,base)
        expected,eligible,sourcecount=native_expected(m,ev,[np.arange(5)],count,scores,base)
        for k in out:np.testing.assert_allclose(out[k],expected[k],atol=1e-10)
        np.testing.assert_array_equal(eligible,flags['eligible']);np.testing.assert_array_equal(sourcecount,flags['eligible_source_count'])
        tied=np.zeros((5,3));w=selected_weights(tied,np.arange(5),m['sources'],np.array([True,False,False]))
        np.testing.assert_allclose(w[:,0],[1/6,1/6,1/6,1/6,1/3]);np.testing.assert_array_equal(w[:,1:],0)
        pos=weighted_positions(m['states'],w);self.assertTrue(np.isfinite(pos).all())

if __name__=='__main__':unittest.main()
