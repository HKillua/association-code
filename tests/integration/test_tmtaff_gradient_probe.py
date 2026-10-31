import unittest
import torch
from association.models.fusion import PairFusion,PairFusionConfig,FusionLoss
from tools.analysis.tmtaff_gradient_flow_probe import gradient_record

class GradientProbeTests(unittest.TestCase):
    def test_original_loss_has_finite_gradients_and_never_updates_weights(self):
        torch.set_num_threads(2);torch.manual_seed(20);f=PairFusion(PairFusionConfig(fc_hidden=8,length=4,d_model=8,nhead=2,num_layers=2,dim_feedforward=16,dropout=0.)).eval();objective=FusionLoss(position_loss_mode='standard_smooth_l1');old={k:v.clone() for k,v in f.state_dict().items()};x=torch.randn(2,4,8);y=torch.randn(2,4,8);target=torch.ones(2,4,4);valid=torch.ones_like(target,dtype=torch.bool)
        p,r=gradient_record(f,objective,x,y,target,valid,training=False,clip=1.)
        self.assertEqual(p.shape,(2,4,4));self.assertGreater(r['gradients']['total']['total_parameter_gradient_l2'],0.);self.assertTrue(r['weights_unchanged']);self.assertFalse(f.training)
        for k,v in old.items():self.assertTrue(torch.equal(v,f.state_dict()[k]))

if __name__=='__main__':unittest.main()
