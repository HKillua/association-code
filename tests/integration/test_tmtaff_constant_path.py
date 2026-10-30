import unittest
import numpy as np
import torch
from association.models.fusion import PairFusion,PairFusionConfig
from tools.analysis.tmtaff_constant_path import trace,double_readout

class ConstantPathTests(unittest.TestCase):
    def test_constant_projection_and_intervention_hooks_are_separate(self):
        torch.manual_seed(20);model=PairFusion(PairFusionConfig(fc_hidden=8,length=4,d_model=8,nhead=2,num_layers=1,dim_feedforward=16,dropout=0.,output_parameterization='train_position_affine'),dict(mean=[30.,120.,1.,1.],std=[1.,1.,1.,1.])).eval()
        x=torch.randn(2,4,8);y=torch.randn(2,4,8)
        with torch.no_grad():model.state_projection.weight[:2].zero_();model.state_projection.bias[:2].zero_()
        states,cap=trace(model,x,y);zero,z=trace(model,x,y,zero_memory=True);again,_=trace(model,x,y)
        self.assertEqual(len(np.unique(states[...,:2].reshape(-1,2),axis=0)),1)
        self.assertGreater(np.ptp(cap['fc2']),0);self.assertEqual(np.count_nonzero(z['fc2']),0)
        np.testing.assert_array_equal(again,states);np.testing.assert_array_equal(zero[...,:2],states[...,:2])
        double,_=double_readout(model,cap['projection_input'],states);np.testing.assert_array_equal(double[...,:2],states[...,:2])

if __name__=='__main__':unittest.main()
