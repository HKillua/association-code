"""076B single-factor PTFM initialization, native geometry and paired exposure."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
import torch
from association.models.ptfm import (PTFM,ptfm_inputs,native_states,objectives,
    RESIDUAL_SCOPE,RESIDUAL_ROUTING_SEMANTICS)
from association.models.sequence_position import sequence_inputs
from association.training.ptfm import configuration,train_ptfm,load_checkpoint
from association.utils.runtime import read_json,write_json
from tests.test_ptfm import config,normalizer,motion,sequence_reference
from tests.training.test_support_gate import tiny_train


def residual_config():
    value=config();value.update(scope=RESIDUAL_SCOPE,routing_semantics=RESIDUAL_ROUTING_SEMANTICS)
    value['model']['position_output']='prior095_residual'
    return value


class PTFMResidualTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_same_rng_all_initial_tensors_except_zero_position_rows(self):
        norm=normalizer();norm['mean'][:2]=[.7,-.4]
        torch.manual_seed(91);direct=PTFM(norm);rng=torch.get_rng_state().clone()
        torch.manual_seed(91);residual=PTFM(norm,position_mode='prior095_residual')
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for key,value in direct.state_dict().items():
            expected=value.clone()
            if key in ('output.weight','output.bias'):expected[:2].zero_()
            self.assertTrue(torch.equal(residual.state_dict()[key],expected),key)
        inputs=ptfm_inputs(motion(),[0,1,2],include_prior=True)
        old=direct(*inputs[:4]);new=residual(*inputs)
        self.assertTrue(torch.equal(old['states'][:,2:],new['states'][:,2:]))
        self.assertTrue(torch.equal(old['task_weights'],new['task_weights']))
        self.assertTrue(torch.equal(new['states'][:,:2],inputs[4][:,:2]))
        self.assertTrue(torch.equal(new['position_residual_deg'],torch.zeros(3,2,dtype=torch.float64)))

    def test_prior_exact_for_all_queries_missing_helpers_dateline_and_poles(self):
        for members in ([0,1,2],[2,1,0],[0]):
            for latitude in (20.123456789,89.999999999,-90.,90.):
                m=motion();m['states'][...,0]=latitude
                m['states'][...,1]=np.array([[179.99,-179.99,180.],[179.98,-179.98,-180.],[179.97,-179.97,179.99]])
                args=ptfm_inputs(m,members,include_prior=True);old=ptfm_inputs(m,members)
                self.assertEqual(len(old),4)
                for before,after in zip(old,args):self.assertTrue(torch.equal(before,after))
                expected=sequence_inputs(m,members)[0]
                self.assertTrue(torch.equal(expected,args[4]))
                model=PTFM(normalizer(),position_mode='prior095_residual');result=model(*args)
                self.assertTrue(torch.equal(result['states'][:,:2],expected[:,:2]))
                inverse=native_states(result['local_states'],args[3],position_base=result['base_states'],
                    position_residual_deg=result['position_residual_deg'])
                self.assertTrue(torch.equal(result['states'],inverse))
                result['states'][:,:2].sum().backward()
                self.assertTrue(torch.isfinite(model.output.bias.grad).all())
                self.assertTrue((model.output.bias.grad[:2].abs()>0).all())

    def test_degree_scale_no_mean_offset_and_original_native_reflection(self):
        norm=normalizer();norm['mean'][:2]=[.7,-.4]
        model=PTFM(norm,position_mode='prior095_residual')
        with torch.no_grad():model.output.bias[:2].copy_(torch.tensor([30.,50.]))
        m=motion();m['states'][...,0]=89.;args=ptfm_inputs(m,[0,1,2],include_prior=True)
        result=model(*args)
        expected=torch.tensor([[3.,5.]]*3,dtype=torch.float64)
        torch.testing.assert_close(result['position_residual_deg'],expected,rtol=0,atol=1e-12)
        self.assertTrue((result['states'][:,0].abs()<=90.).all())
        self.assertTrue(((result['states'][:,1]>=-180.)&(result['states'][:,1]<180.)).all())
        old_inverse=native_states(result['local_states'],args[3])
        self.assertTrue(torch.equal(result['states'],old_inverse))
        target=result['states'].detach().clone();target[:,0]+=.001
        valid=torch.ones_like(target,dtype=torch.bool);valid[1,:2]=False;target[1,:2]=torch.nan
        loss,diag=objectives(model,result,target,valid);loss.backward()
        self.assertEqual(diag['task_valid_counts'],[2,3,3]);self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        result=model(*args);valid[:]=False;target[:]=torch.nan
        loss,diag=objectives(model,result,target,valid)
        self.assertEqual(float(loss.detach()),0.);self.assertEqual(diag['task_valid_counts'],[0,0,0])
        loss.backward()

    def test_configuration_and_explicit_prior_routing_reject_mixing(self):
        self.assertEqual(configuration(config()),config());self.assertEqual(configuration(residual_config()),residual_config())
        wrong=residual_config();wrong['scope']=config()['scope']
        with self.assertRaises(ValueError):configuration(wrong)
        wrong=residual_config();wrong['model']['position_output']='direct'
        with self.assertRaises(ValueError):configuration(wrong)
        args=ptfm_inputs(motion(),[0,1,2],include_prior=True)
        with self.assertRaisesRegex(ValueError,'only for residual'):PTFM(normalizer())(*args)
        model=PTFM(normalizer(),position_mode='prior095_residual')
        with self.assertRaisesRegex(ValueError,'only for residual'):model(*args[:4])
        bad=list(args);bad[4]=bad[4].float()
        with self.assertRaisesRegex(ValueError,'physical prior'):model(*bad)
        with self.assertRaisesRegex(ValueError,'both base and residual'):
            native_states(args[0],args[3],position_base=args[4])
        result=model(*args);corrupt=result['local_states'].clone();corrupt[0,0]+=1.
        with self.assertRaisesRegex(ValueError,'does not equal base'):
            native_states(corrupt,args[3],position_base=args[4],position_residual_deg=result['position_residual_deg'])

    def test_real_training_pairs_direct_and_sequence_and_checkpoint_routes(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);control=sequence_reference(root/'sequence')
            original=train_ptfm(tiny_train(),config(),root/'direct',paired_reference_fit=control/'fit')
            trained=train_ptfm(tiny_train(),residual_config(),root/'residual',
                paired_reference_fit=control/'fit',paired_ptfm_fit=root/'direct')
            self.assertEqual(trained['exposure'],original['exposure'])
            self.assertEqual(trained['paired_control']['status'],'PAIR_VERIFIED')
            pair=trained['paired_ptfm_control'];self.assertEqual(pair['status'],'PAIR_VERIFIED')
            for key in ('initial_parameters_exact_except_position_rows','position_rows_zero','normalizer_exact',
                    'initial_sampler_rng_exact','initial_torch_rng_exact','actual_successful_batches_exact',
                    'all_task_counts_exact','full_exposure_exact'):self.assertTrue(pair[key],key)
            self.assertEqual(pair['matched_updates'],3);self.assertEqual(pair['matched_anchors'],6)
            model,saved=load_checkpoint(root/'residual/final.pt')
            self.assertEqual(model.position_mode,'prior095_residual');self.assertEqual(saved['actual_updates'],3)
            self.assertEqual(load_checkpoint(root/'direct/final.pt')[0].position_mode,'direct')
            initial=load_checkpoint(root/'residual/step-0.pt')[1]
            self.assertFalse(torch.equal(saved['model']['output.weight'][:2],initial['model']['output.weight'][:2]))
            saved['routing_semantics']=config()['routing_semantics'];torch.save(saved,root/'bad.pt')
            with self.assertRaisesRegex(ValueError,'semantics'):load_checkpoint(root/'bad.pt')

    def test_pairing_refuses_incomplete_changed_parameters_and_batches_before_update(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'direct';train_ptfm(tiny_train(),config(),source)
            with self.assertRaisesRegex(ValueError,'requires the completed'):
                train_ptfm(tiny_train(),residual_config(),root/'no-source')
            run=read_json(source/'run.json');run['status']='FAILED';write_json(source/'run.json',run)
            with self.assertRaisesRegex(ValueError,'not completed'):
                train_ptfm(tiny_train(),residual_config(),root/'failed-source',paired_ptfm_fit=source)
            run['status']='COMPLETED';write_json(source/'run.json',run)
            initial=torch.load(source/'step-0.pt',weights_only=False);original=deepcopy(initial)
            initial['model']['output.bias'][2]+=1.;torch.save(initial,source/'step-0.pt')
            with self.assertRaisesRegex(ValueError,'initial parameters differ/output.bias'):
                train_ptfm(tiny_train(),residual_config(),root/'bad-parameters',paired_ptfm_fit=source)
            failed=read_json(root/'bad-parameters/run.json');self.assertEqual(failed['optimizer_updates'],0)
            torch.save(original,source/'step-0.pt')
            path=source/'training.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertNotEqual(rows[0]['batch'][0],rows[0]['batch'][1]);rows[0]['batch'].reverse()
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'actual batch/task masks differ'):
                train_ptfm(tiny_train(),residual_config(),root/'bad-batch',paired_ptfm_fit=source)
            failed=read_json(root/'bad-batch/run.json')
            self.assertEqual(failed['optimizer_updates'],0);self.assertEqual(failed['exposure']['sampled_anchors'],0)
            self.assertFalse((root/'bad-batch/final.pt').exists())


if __name__=='__main__':unittest.main()
