import unittest
import numpy as np
import torch
from association.evaluation import native_phase_gate as p,native_compatibility as nc
from association.training import temporal_gate as t

class TemporalGateTests(unittest.TestCase):
    def data(self,angle=0):
        n=3;tt=np.arange(8.);values=np.zeros((n*8,4));values[:,2]=np.tile(np.arange(1,9),n)*3600/1852;values[:,3]=angle;events=dict(event_offsets=np.arange(n+1)*8,event_times_seconds=np.tile(tt,n),event_values=values,event_valid=np.ones(values.shape,bool));q=np.array([8.,9.]);v=np.tile([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))],(n,1));states=np.zeros((n,2,4))
        for i in range(n):states[i,:,:2]=nc.advance(np.zeros(2),v[i],q-7)
        m=dict(states=states,support=np.zeros((n,2),bool),sources=np.array(['sourceC','sourceA','sourceB']),query_seconds=q,ols_velocity_mps=v,native_count=np.full(n,8),span_seconds=np.full(n,7.),fit_rmse_m=np.full(n,10.));return m,events
    def test_order_bag_and_time_shift(self):
        m,e=self.data();a,am,_=p.node_context(m,e);reversed_events={**e,'event_values':e['event_values'].reshape(3,8,4)[:,::-1].reshape(-1,4).copy()};b,bm,_=p.node_context(m,reversed_events);self.assertFalse(np.allclose(a[0]['phase'],b[0]['phase']));np.testing.assert_allclose(a[0]['bag'],b[0]['bag']);np.testing.assert_allclose(am,bm)
        shifted={**e,'event_times_seconds':e['event_times_seconds']+12345};c,cm,_=p.node_context({**m,'query_seconds':m['query_seconds']+12345},shifted);np.testing.assert_allclose(a[0]['phase'],c[0]['phase']);np.testing.assert_allclose(am,cm)
    def test_anonymous_k3_and_rotation(self):
        m,e=self.data();a,am,_=p.node_context(m,e);case=p.cases(m,a,am,[np.arange(3)],np.zeros((1,2),int),np.zeros((3,2)));other=p.cases({**m,'sources':np.array(['x','z','y'])},a,am,[np.array([2,0,1])],np.zeros((1,2),int),np.zeros((3,2)))
        for k in p.KINDS:np.testing.assert_allclose(case['x'][k],other['x'][k],atol=1e-10)
        r,re=self.data(90);rc,rm,_=p.node_context(r,re);rot=p.cases(r,rc,rm,[np.arange(3)],np.zeros((1,2),int),np.zeros((3,2)))
        for k in p.KINDS:np.testing.assert_allclose(case['x'][k],rot['x'][k],atol=1e-8)
    def test_circular_loss_and_degenerate_gradient(self):
        old=np.array([[0,179.9],[0,0.]]);mc=np.array([[0,-179.9],[0,180.]]);target=np.array([[0,179.95],[0,1.]]);logits=torch.zeros(2,requires_grad=True);loss=t.physical_loss(logits,torch.tensor(old),torch.tensor(mc),torch.tensor(target));expected=nc.distance(p.blend(old,mc,np.array([.5,.5])),target);np.testing.assert_allclose(loss.detach().numpy(),expected,atol=2e-5);loss.sum().backward();self.assertTrue(torch.isfinite(logits.grad).all());np.testing.assert_array_equal(p.blend(old,mc,np.zeros(2)),old);np.testing.assert_array_equal(p.blend(old,mc,np.ones(2)),mc)
    def test_matched_learning_and_numpy_checkpoint(self):
        n=4;data=dict(old=np.tile([0.,0.],(n,1)),mc=np.tile([0.,.01],(n,1)),target=np.tile([0.,.01],(n,1)),w=np.full(n,.5));data.update({'x_'+k:np.zeros((n,32)) for k in p.KINDS});models,draws,initial=t.fit(data,'cpu',steps=3);self.assertEqual(draws.shape,(3,1024));self.assertTrue(torch.equal(initial['net.4.weight'],torch.zeros_like(initial['net.4.weight'])))
        for k,model in models.items():
            prediction=t.predict(model,data['x_'+k],'cpu');self.assertTrue((prediction>.5).all());self.assertEqual(model['parameter_count'],2145);np.testing.assert_allclose(prediction,t.numpy_predict(model,data['x_'+k]),atol=1e-6)
    def test_first_claim_labels(self):
        case=dict(indices=np.array([[0,0],[0,1],[1,0]]));ev=dict(target_ids=np.array(['object']),identities=np.array(['object','object']),target_valid=np.array([[[True]*4,[False]*4]]),target_states=np.zeros((1,2,4)));ix,y=p.labels(case,[np.array([0]),np.array([1])],ev);np.testing.assert_array_equal(ix,[0]);self.assertEqual(y.shape,(1,2))
