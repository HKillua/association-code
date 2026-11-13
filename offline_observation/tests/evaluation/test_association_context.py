import unittest
import numpy as np
import torch
from association.models.association_context import context_cache,edge_inputs,EdgeResidual
from association.evaluation.partition_refinement import refine_partition
from tools.analysis.association_context_pilot import train_heads

class AssociationContextTests(unittest.TestCase):
    def test_leave_edge_out_and_node_permutation(self):
        rng=np.random.default_rng(4);pairs=np.column_stack(np.triu_indices(4,1));h=rng.normal(size=(6,64)).astype(np.float32);s=np.zeros(6)
        c=context_cache(h,pairs,s,4,0.)
        x=edge_inputs(c,[0],'CTX')[0]
        # Edge0=(0,1): all other edges at each endpoint have weight .5.
        a=h[[1,2]].mean(0);b=h[[3,4]].mean(0)
        np.testing.assert_allclose(x[64:128],a+b,atol=2e-7)
        changed=h.copy();changed[0]+=1000
        np.testing.assert_allclose(edge_inputs(context_cache(changed,pairs,s,4,0.),[0],'CTX')[0,64:],x[64:],atol=2e-7)
        perm=np.array([3,1,0,2]);newpairs=np.sort(perm[pairs],axis=1)
        cp=context_cache(h,newpairs,s,4,0.)
        np.testing.assert_allclose(edge_inputs(cp,np.arange(6),'CTX'),edge_inputs(c,np.arange(6),'CTX'),atol=1e-7)
        c2=context_cache(h[:1],np.array([[0,1]]),s[:1],2,0.)
        np.testing.assert_array_equal(edge_inputs(c2,[0],'CTX')[0,64:],0)

    def test_zero_residual_and_matched_real_updates(self):
        torch.manual_seed(20);model=EdgeResidual();self.assertEqual(sum(p.numel() for p in model.parameters()),6209)
        self.assertEqual(model(torch.randn(3,192)).tolist(),[0.,0.,0.])
        rng=np.random.default_rng(6);p=np.column_stack(np.triu_indices(4,1))
        c=context_cache(rng.normal(size=(6,64)),p,np.zeros(6),4,0.)
        c.update(scene_id='1',pairs=p,scores=np.zeros(6),labels=np.array([1,0,0,0,0,1],bool))
        heads,norms,curve,sampling=train_heads([c],'cpu',lambda **kw:None,steps=2)
        self.assertEqual(sampling['completed_batch_count'],2)
        self.assertEqual(sampling['sampled_pair_count'],512)
        self.assertEqual(set(heads),{'OWN','CTX'})
        self.assertTrue(any(float(v.abs().sum())>0 for v in heads['CTX'].network[-1].parameters()))

    def test_anchor_and_support_constraints_are_separate(self):
        p=np.column_stack(np.triu_indices(4,1))
        # Node1 cannot move before destination anchor2; node3 can move to anchor0.
        w=np.array([2.,-4.,3.,-3.,6.,5.]);m=np.array([0,0,1,1])
        a,stats,trace=refine_partition(4,p,w,m,threshold=0.,move_policy='anchors')
        self.assertTrue(trace);self.assertEqual(a[0],0);self.assertEqual(a[2],1);self.assertEqual(a[3],a[0])
        support=np.ones((4,3),bool);sources=np.array(['a','b','a','b'])
        b,ss,_=refine_partition(4,p,w,m,threshold=0.,move_policy='support',support=support,sources=sources)
        np.testing.assert_array_equal(b,m);self.assertGreater(ss['support_rejections'],0)
        # Redundant same-source support permits the legal move.
        sources[2:]='a'
        b,ss,_=refine_partition(4,p,w,m,threshold=0.,move_policy='support',support=support,sources=sources)
        self.assertEqual(b[3],b[0]);self.assertGreater(ss['final_J'],ss['initial_J'])
        renamed=np.array(['z' if s=='a' else 'q' for s in sources])
        b2,_,_=refine_partition(4,p,w,m,threshold=0.,move_policy='support',support=support,sources=renamed)
        np.testing.assert_array_equal(b,b2)

if __name__=='__main__':unittest.main()
