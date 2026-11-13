import unittest
import numpy as np
import torch
from association.evaluation.objective_difficulty import paired_training_data,fit_factorial
from association.evaluation.local_source_evidence import predict_residual

class ObjectiveDifficultyTests(unittest.TestCase):
    def test_train_only_error_balance_without_source_role_labels(self):
        x=np.zeros((2,6,10));x[1,:,0]=1.;e=np.array([[0.,0.,0.,0.,2.,2.],[2.,2.,2.,2.,0.,0.]])
        base=dict(mean=np.zeros(10),std=np.ones(10),beta=np.r_[0.,1.,np.zeros(64)])
        p=dict(features=x,group_offsets=np.array([0,2]),labeled=np.ones((2,6),bool),errors_m=e,y=np.log1p(e))
        data,account=paired_training_data([p],base)
        self.assertEqual(account['hard_query_mass'],2.);self.assertEqual(account['easy_query_mass'],4.)
        self.assertEqual(account['balanced_hard_mass'],3.);self.assertEqual(account['balanced_easy_mass'],3.)
        np.testing.assert_allclose(data['weight'],np.ones(6))
        # Reversing both source rows preserves difficulty and pair weights.
        reverse=dict(p,features=x[::-1].copy(),errors_m=e[::-1].copy(),y=p['y'][::-1].copy())
        other,counts=paired_training_data([reverse],base);np.testing.assert_array_equal(data['hard'],other['hard']);self.assertEqual(account,counts)
    def test_four_independent_heads_receive_real_updates(self):
        torch.set_num_threads(2);rng=np.random.default_rng(20);x=rng.normal(size=(2,32,10));e=np.abs(x[...,0])*100
        p=dict(features=x,group_offsets=np.array([0,2]),labeled=np.ones((2,32),bool),errors_m=e,y=np.log1p(e/1000.));base=dict(mean=np.zeros(10),std=np.ones(10),beta=np.zeros(66));normal=dict(mean=np.zeros(10),std=np.ones(10),base=base)
        models,account=fit_factorial([p],normal,device='cpu',steps=4);self.assertEqual(len(models),4)
        for model in models.values():
            self.assertEqual(model['updates'],4);self.assertTrue(np.isfinite(predict_residual(model,x.reshape(-1,10),'cpu')).all());self.assertTrue(any(v['gradient_norm']>0 for v in model['curve']))

if __name__=='__main__':unittest.main()
