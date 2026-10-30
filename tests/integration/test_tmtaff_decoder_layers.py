import unittest
import torch
import numpy as np
from association.models.fusion import PairFusion,PairFusionConfig
from tools.analysis.tmtaff_decoder_layers import layer_trace

class DecoderLayerTests(unittest.TestCase):
    def test_layer_hooks_preserve_predictions_and_remove_themselves(self):
        torch.set_num_threads(2);torch.manual_seed(20);model=PairFusion(PairFusionConfig(fc_hidden=8,length=4,d_model=8,nhead=2,num_layers=2,dim_feedforward=16,dropout=0.)).eval();x=torch.randn(2,4,8);y=torch.randn(2,4,8)
        with torch.no_grad():expected=model(x,y)['states'].numpy()
        original,path,attention=layer_trace(model,x,y,1.);zero,_,_=layer_trace(model,x,y,0.)
        np.testing.assert_array_equal(original,expected);self.assertEqual(path['1_norm3'].shape,(2,4,8));self.assertEqual(len(attention),2);self.assertGreater(float(np.max(np.abs(original-zero))),0)
        with torch.no_grad():np.testing.assert_array_equal(model(x,y)['states'].numpy(),expected)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

if __name__=='__main__':unittest.main()
