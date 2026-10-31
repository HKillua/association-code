from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from association.models.association_context import EdgeResidual,context_cache,fit_context_normalizer
from tools.analysis.structured_association_pilot import train,predict,balanced_bce


class StructuredAssociationTests(unittest.TestCase):
    def test_matched_exposure_optimization_and_label_free_checkpoint_inference(self):
        torch.set_num_threads(1);torch.manual_seed(20);pairs=np.column_stack(np.triu_indices(4,1));ids=np.array([0,0,1,1])
        caches=[]
        for i in range(2):
            c=context_cache(np.random.default_rng(i).normal(size=(6,64)).astype(np.float32),pairs,np.array([1.,8.,3.,2.,7.,4.]),4,1.7)
            c.update(scores=np.array([1.,8.,3.,2.,7.,4.]),scene_id=str(i),identities=ids,labels=ids[pairs[:,0]]==ids[pairs[:,1]])
            caches.append(c)
        norm=fit_context_normalizer(caches,'CTX');initial=EdgeResidual().state_dict();heads,curve,sampling=train(caches,initial,norm,'cpu',lambda **kw:None,epochs=1)
        self.assertEqual(sampling['per_scene_passes'],[1,1]);self.assertEqual(sampling['pair_presentations_per_head'],12)
        self.assertEqual(sampling['updates_per_head'],2);self.assertTrue(any(r['heads']['STRUCT']['fy']>0 for r in curve))
        self.assertTrue(all(r['heads']['BCE']['fy']==0 for r in curve))
        self.assertTrue(any(not torch.equal(heads['BCE'].state_dict()[k],heads['STRUCT'].state_dict()[k]) for k in initial))
        cache={k:v for k,v in caches[0].items() if k not in ('identities','labels')};before=predict(cache,heads,norm,'cpu')
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'heads.pt';torch.save(dict(heads={k:h.state_dict() for k,h in heads.items()},normalizer=norm),path)
            ckpt=torch.load(path,weights_only=False);restored={k:EdgeResidual() for k in heads}
            for k,h in restored.items():h.load_state_dict(ckpt['heads'][k])
            after=predict(cache,restored,ckpt['normalizer'],'cpu')
        for k in before:np.testing.assert_array_equal(before[k],after[k])

    def test_class_balance_and_rank_zero_do_not_drop_examples(self):
        x=torch.tensor([1.,-1.,-2.],dtype=torch.float64);y=torch.tensor([1.,0.,0.],dtype=torch.float64)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(x,y,reduction='none')
        self.assertAlmostEqual(balanced_bce(x,y).item(),((loss[0]+loss[1:].mean())/2).item())
        self.assertAlmostEqual(balanced_bce(x,torch.zeros_like(y)).item(),torch.nn.functional.binary_cross_entropy_with_logits(x,torch.zeros_like(y)).item())

if __name__=='__main__':unittest.main()
