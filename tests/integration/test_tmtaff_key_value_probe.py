import unittest
import numpy as np
import torch
from association.models.fusion import PairFusion,PairFusionConfig
from tools.analysis.tmtaff_decoder_layers import layer_trace
from tools.analysis.tmtaff_key_value_probe import kv_trace

class KeyValueTests(unittest.TestCase):
    def test_only_value_keeps_first_attention_and_hooks_are_removed(self):
        torch.set_num_threads(2);torch.manual_seed(20);m=PairFusion(PairFusionConfig(fc_hidden=8,length=4,d_model=8,nhead=2,num_layers=2,dim_feedforward=16,dropout=0.)).eval();x=torch.randn(2,4,8);y=torch.randn(2,4,8)
        with torch.no_grad():m.bos.copy_(torch.randn_like(m.bos))
        base,_,attn=kv_trace(m,x,y,1.,1.);original,_,_=layer_trace(m,x,y,1.);np.testing.assert_array_equal(base,original)
        _,_,v=kv_trace(m,x,y,1.,.01);_,_,k=kv_trace(m,x,y,.01,1.)
        self.assertEqual(attn['0']['entropy_mean'],v['0']['entropy_mean']);self.assertNotEqual(attn['0']['logit_range_max'],k['0']['logit_range_max'])
        both,_,_=kv_trace(m,x,y,.01,.01);prior,_,_=layer_trace(m,x,y,.01);np.testing.assert_array_equal(both,prior)
        self.assertTrue(all(not z._forward_hooks and not z._forward_pre_hooks for z in m.modules()))

if __name__=='__main__':unittest.main()
