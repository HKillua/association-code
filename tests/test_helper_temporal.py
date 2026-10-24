"""087 contracts plus real tiny original/pre-pool/temporal fit-to-cache chain."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from association.models.helper_temporal import HelperTemporalPosition,initialize_from_reference,permutation_plan,validate_plan
from association.models.helper_sequence import helper_sequence_inputs
from association.models.sequence_position import SequencePosition,sequence_objectives
from association.utils.runtime import read_json,write_json
from tests.test_sequence_position import motion_fixture
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_train
from tools.experiments import helper_temporal_fit as entry
from tools.analysis import compare_helper_temporal as compare


class HelperTemporalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(20261023)
        self.reference=SequencePosition(backbone='transformer').state_dict()
    def model(self,mode):
        model=HelperTemporalPosition(correspondence=mode)
        initialize_from_reference(model,self.reference,20348223)
        return model
    def test_inputs_initialization_permutations_and_empty_support(self):
        models=[self.model(m) for m in ('shuffled','preserved')]
        self.assertEqual([sum(p.numel() for p in m.parameters()) for m in models],[109042]*2)
        self.assertTrue(all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items()))
        self.assertTrue(all(not torch.count_nonzero(p) for name,p in models[0].temporal.named_parameters() if 'bias' in name))
        before=torch.get_rng_state().clone();initialize_from_reference(models[0],self.reference,20348223)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for members in ([0],[0,1],[0,1,2]):
            args=helper_sequence_inputs(motion_fixture(),members)
            plan=permutation_plan(len(members)-1,len(args[0]),np.random.default_rng(7))
            for model in models:
                result=model(*args,plan=plan)
                self.assertTrue(torch.equal(result['states'],args[0]))
                self.assertTrue(torch.equal(result['helper_embedding'][~args[-1].any(0)],torch.zeros_like(result['helper_embedding'][~args[-1].any(0)])))
            if len(members)<3:
                self.assertTrue(torch.equal(models[0](*args,plan=plan)['helper_embedding'],models[1](*args,plan=plan)['helper_embedding']))
        features=torch.randn(4,5,27);support=torch.rand(4,5)>.3;times=torch.randn(5,3)
        plan=permutation_plan(4,5,np.random.default_rng(2))
        b=models[1];expected=b.temporal_representation(features,support,times,plan)[0]
        p=torch.tensor([3,1,0,2])
        torch.testing.assert_close(expected,b.temporal_representation(features[p],support[p],times,plan)[0],rtol=1e-6,atol=1e-6)
        corrupted=features.clone();corrupted[~support]=torch.nan
        self.assertTrue(torch.equal(expected,b.temporal_representation(corrupted,support,times,plan)[0]))
        a=models[0].temporal_representation(features,support,times,plan)[0]
        self.assertFalse(torch.equal(a,expected))
        # One supported row per query does not imply whole-group H=1:
        # cross-query identity remains observable to the shared GRU.
        sparse=torch.tensor([[True,False,True,False,True],[False,True,False,True,False]])
        swapped=np.array([[0,1],[1,0],[0,1],[1,0],[0,1]],np.int64)
        single_supported=features[:2]
        preserve=b.temporal_representation(single_supported,sparse,times,swapped)
        shuffle=models[0].temporal_representation(single_supported,sparse,times,swapped)
        self.assertTrue(torch.equal(preserve[2],torch.ones(5,dtype=torch.int64)))
        self.assertFalse(torch.equal(preserve[0],shuffle[0]))
        np.testing.assert_array_equal(preserve[3],np.tile([0,1],(5,1)))
        np.testing.assert_array_equal(shuffle[3],swapped)
        with self.assertRaises(ValueError):validate_plan(np.zeros((5,4),int),4,5)
    def test_delayed_branch_gradient_and_empty_adam_state(self):
        for members in ([0],[0,1,2]):
            model=self.model('preserved');args=helper_sequence_inputs(motion_fixture(),members)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            target=args[0].clone();target[:,0]+=.01;valid=torch.ones_like(target,dtype=torch.bool)
            temporal_seen=False
            for step in range(8):
                optimizer.zero_grad(set_to_none=True)
                result=model(*args,plan=permutation_plan(len(members)-1,len(args[0]),np.random.default_rng(2)))
                loss=sequence_objectives(result,target,valid)[0]['linear'];loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                nonzero=any(bool(torch.count_nonzero(p.grad)) for p in model.temporal.parameters())
                if step==0:self.assertFalse(nonzero)
                temporal_seen|=nonzero;optimizer.step()
            self.assertEqual(temporal_seen,len(members)>1)
            self.assertEqual(len(optimizer.state),len(list(model.parameters())))
    def test_frozen_production_contract(self):
        manifest=read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment=read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        for filename in entry.CONFIGS.values():
            c=read_json(entry.ROOT/'configs'/filename);entry.production_contract(c,manifest,alignment)
            c['training']['updates']+=1
            with self.assertRaises(ValueError):entry.production_contract(c,manifest,alignment)
    def test_actual_four_fits_endpoints_routes_and_cross_table(self):
        from tools.experiments import sequence_position_fit as original,helper_sequence_fit as preentry
        from tools.analysis import compare_helper_sequence as precompare
        with TemporaryDirectory() as directory:
            root=Path(directory);source,metadata,validation=support_gate_cache_fixture(root)
            train=tiny_train();manifest=dict(splits={'T_core':[str(s['scene_id']) for s in train],'V_select':['3035']},components={'3035':'shared'})
            base=read_json(entry.ROOT/'configs/sequence-position-transformer-extended-level1.json');base['training'].update(updates=4,batch_size=2)
            for name,value in (('config',base),('manifest',manifest),('alignment',{})):write_json(root/(name+'.json'),value)
            args=Namespace(source_system=str(source),output=str(root/'linear'),paired_control=None,
                archive=['unread.zip'],manifest=str(root/'manifest.json'),alignment=str(root/'alignment.json'),
                config=str(root/'config.json'),device='cpu',cpu_threads=4,time_policy='advisory',profile='extended4000',pre_pool_control=str(root/'pre_pool'))
            dataset=lambda *a,**k:dict(train=train,validation=validation,report={'fixture':True})
            def meta(path,arm=None):
                path=Path(path);final=read_json(path/'final/summary.json')
                return dict(path=path,method='learned_sequence',source_system=str(source.resolve()),scene_ids=['3035'],components={'3035':'shared'},summary=final,configuration=read_json(path/'run.json')['configuration'])
            with patch('association.evaluation.sequence_position.frozen_motion',side_effect=lambda raw,*_:raw):
                with patch.object(original,'production_contract'),patch.object(original,'_bind_source'),patch.object(original,'_load_metadata',return_value=metadata),patch.object(original,'load_timed_dataset',side_effect=dataset):
                    original.run(args)
                config=read_json(entry.ROOT/'configs/helper-sequence-pre-pool-level1.json');config['training'].update(updates=4,batch_size=2)
                write_json(root/'config.json',config);args.output=str(root/'pre_pool');args.paired_control=str(root/'linear');args.profile='screen078'
                with patch.object(preentry,'production_contract'),patch.object(preentry,'_bind_source'),patch.object(preentry,'_load_metadata',return_value=metadata),patch.object(preentry,'load_timed_dataset',side_effect=dataset):preentry.run(args)
                for mode,filename in entry.CONFIGS.items():
                    config=read_json(entry.ROOT/'configs'/filename);config['training'].update(updates=4,batch_size=2)
                    write_json(root/'config.json',config);args.output=str(root/mode)
                    with patch.object(entry,'production_contract'),patch.object(entry,'_bind_source'),patch.object(entry,'_load_metadata',return_value=metadata),patch.object(entry,'load_timed_dataset',side_effect=dataset),patch.object(precompare,'helper_metadata',return_value=('pre_pool',meta(root/'pre_pool'))),patch.object(compare,'_metadata',side_effect=meta),patch.object(compare,'pre_metadata',return_value=('pre_pool',meta(root/'pre_pool'))),patch.object(compare,'temporal_metadata',side_effect=lambda p:(Path(p).name,meta(p))):entry.run(args)
            params=Namespace(shuffled_run=str(root/'shuffled'),preserved_run=str(root/'preserved'),linear_run=str(root/'linear'),pre_pool_run=str(root/'pre_pool'),output=str(root/'compare'),cpu_threads=4)
            with patch.object(compare,'_metadata',side_effect=meta),patch.object(compare,'pre_metadata',return_value=('pre_pool',meta(root/'pre_pool'))),patch.object(compare,'temporal_metadata',side_effect=lambda p:(Path(p).name,meta(p))):
                result=compare.run(params)
            self.assertEqual(result['status'],'COMPLETED');self.assertEqual(result['pairing']['matched_updates'],4)
            self.assertEqual(result['support_helper_cross']['point_count'],127)
            self.assertEqual(sum(c['point_count'] for c in result['support_helper_cross']['cells'].values()),127)
            for method in result['methods']:
                total=sum(c['methods'][method]['signed_delta_vs_linear_m'] for c in result['support_helper_cross']['cells'].values())
                expected=(result['position_weightings']['all'][method]['point_equal']['mae']-result['position_weightings']['all']['linear']['point_equal']['mae'])*127
                self.assertAlmostEqual(total,expected,places=7)
            for mode in entry.CONFIGS:compare.replay_routes(meta(root/mode))
            self.assertEqual(result['pairing']['arms']['preserved']['routing_exposure']['applied_changed_slots'],0)
            for endpoint in ('initial','final'):
                self.assertEqual(result['routing']['preserved'][endpoint]['exposure']['applied_changed_slots'],0)
            # V cache route audit must reject corruption before metric comparison.
            cache=root/'preserved/final/predictions/states-3035.npz'
            with np.load(cache,allow_pickle=False) as z: original_cache={key:z[key] for key in z.files}
            changed={key:value.copy() for key,value in original_cache.items()}
            changed['group_helper_query_applied_route'][0,0]=999
            np.savez_compressed(cache,**changed)
            with self.assertRaisesRegex(ValueError,'permutation'):compare.replay_routes(meta(root/'preserved'))
            np.savez_compressed(cache,**original_cache)
            arms={name:meta(root/name) for name in ('linear','pre_pool','shuffled','preserved')}
            path=root/'shuffled/fit/training.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
            rows[0]['temporal_rng_before']['has_uint32']=1-rows[0]['temporal_rng_before']['has_uint32']
            path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaisesRegex(ValueError,'RNG discontinuity'):compare.verify_pairing(arms)

if __name__=='__main__':unittest.main()
