"""074C geometry, masked dual inputs, T-only statistics and real full-state chain."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import gzip,json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from association.models.ptfm import PTFM,ptfm_inputs,local_states,native_states,objectives
from association.training.ptfm import fit_normalizer,train_ptfm,load_checkpoint
from association.training.evidence_fusion import prepare_evidence_scenes,training_members,TRAIN_MODES
from association.training.sequence_position import train_sequence_position,load_checkpoint as sequence_checkpoint
from association.evaluation.sequence_position import evaluate_sequence_position
from association.evaluation.ptfm import evaluate_ptfm,compare_endpoints,compare_sequence_control,METHODS
from association.utils.runtime import read_json,write_json
from tests.fixtures.learning import fixture_motion,tiny_train
from tests.fixtures.cache import support_gate_cache_fixture
from tools.experiments import ptfm_fit as entry


def normalizer():
    return dict(mean=[0.,0.,3.,0.],scale=[.1,.1,10.,30.],split='T_core',truth_used=False)


def config():
    value=read_json(entry.ROOT/'configs/ptfm-adapted-level1.json');value['training'].update(updates=3,batch_size=2)
    return value


def motion():
    value=fixture_motion();value['query_seconds']=np.array([0.,20.,40.]);return value


def sequence_reference(root,source=None,metadata=None,validation=None):
    cfg=read_json(entry.ROOT/'configs/sequence-position-transformer-extended-level1.json')
    cfg['training']=config()['training'];root.mkdir()
    train_sequence_position(tiny_train(),cfg,root/'fit')
    if source is not None:
        model,_=sequence_checkpoint(root/'fit/final.pt')
        with patch('association.evaluation.sequence_position.frozen_motion',side_effect=lambda raw,*_:raw):
            evaluate_sequence_position(model,validation,source,metadata,root/'final')
    return root


class PTFMTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_dual_pool_dimensions_missing_helper_and_order(self):
        m=motion();args=ptfm_inputs(m,[0,1,2]);reverse=ptfm_inputs(m,[0,2,1])
        for a,b in zip(args,reverse):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(args[0].shape,(3,4));self.assertEqual(args[2].tolist(),[[1.,1.],[1.,0.],[0.,0.]])
        self.assertEqual(args[1][0,2],8.);self.assertEqual(args[1][1,2],7.)
        self.assertEqual(args[1][0,1],2.)  # Circular179->-179, not358deg.
        self.assertEqual(ptfm_inputs(m,[0])[2][:,0].sum(),0)
        model=PTFM(normalizer()).eval();result=model(*args)
        self.assertEqual(result['states'].shape,(3,4));self.assertEqual(result['task_weights'].shape,(3,3))
        altered=list(args);altered[1]=args[1].clone();altered[1][2]+=100.
        with torch.no_grad():
            torch.testing.assert_close(model(*altered)['states'],model(*args)['states'],atol=0,rtol=0)
        self.assertEqual(model.attention[0].in_features,144);self.assertEqual(model.projection.in_features,208)

    def test_native_roundtrip_dateline_course_units_and_gradients(self):
        states=np.array([[89.,179.5,12345.,359.],[89.1,-179.5,99999.,1.]])
        origin=states[0];local=torch.tensor(local_states(states,origin),requires_grad=True)
        native=native_states(local,torch.tensor(origin))
        np.testing.assert_allclose(native.detach(),states,rtol=0,atol=1e-12)
        native.sum().backward();self.assertTrue(torch.isfinite(local.grad).all())
        crossing=torch.tensor([[2.,2.,3.,5.]],dtype=torch.float64,requires_grad=True)
        value=native_states(crossing,torch.tensor([89.,179.,2.,359.],dtype=torch.float64))
        self.assertTrue((-90<=value[0,0]<=90));self.assertTrue(-180<=value[0,1]<180)
        self.assertEqual(float(value[0,3].detach()),4.)
        negative=torch.tensor([[0.,0.,-7.,-359.]],dtype=torch.float64,requires_grad=True)
        inverse=native_states(negative,torch.tensor([0.,0.,0.,359.]))
        self.assertEqual(float(inverse[0,2].detach()),7.);self.assertEqual(float(inverse[0,3].detach()),0.)
        inverse.sum().backward();self.assertTrue(torch.isfinite(negative.grad).all())

    def test_invalid_truth_fullstate_and_circular_objective(self):
        model=PTFM(normalizer());result=model(*ptfm_inputs(motion(),[0,1,2]))
        truth=torch.full_like(result['states'],torch.nan);valid=torch.zeros_like(truth,dtype=torch.bool)
        loss,d=objectives(model,result,truth,valid);self.assertEqual(float(loss.detach()),0);self.assertEqual(d['task_valid_counts'],[0,0,0])
        loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        result=model(*ptfm_inputs(motion(),[0,1,2]));truth=result['states'].detach().clone();truth[:,3]=(truth[:,3]+359.)%360
        valid[:]=True;_,d=objectives(model,result,truth,valid)
        self.assertAlmostEqual(d['task_losses'][2],1/30.,places=12)
        truth[0,2]=torch.nan;valid[0,2]=False
        loss,d=objectives(model,result,truth,valid);self.assertEqual(d['task_valid_counts'],[3,2,3]);self.assertTrue(torch.isfinite(loss))
        truth=result['states'].detach().clone();truth[:,3]+=360.
        _,d=objectives(model,result,truth,torch.ones_like(truth,dtype=torch.bool));self.assertAlmostEqual(d['task_losses'][2],0.,places=12)

    def test_normalizer_is_T_visible_only_with_fixed_exposure(self):
        prepared=prepare_evidence_scenes(tiny_train(),motion_observation_policy='canonical_all');cfg=config()
        first=fit_normalizer(prepared,cfg)
        rng=np.random.default_rng(cfg['training']['seed']);visible=[];allrows=0
        for step in range(cfg['training']['updates']):
            for _ in range(cfg['training']['batch_size']):
                scene=prepared[int(rng.integers(len(prepared)))];anchor=int(rng.integers(len(scene['identities'])))
                members=training_members(scene,anchor,TRAIN_MODES[step%3]);a,b,c,_=ptfm_inputs(scene['motion'],members)
                visible.extend((a.numpy(),b.numpy()[c[:,0].numpy().astype(bool)]));allrows+=len(a)*2
        values=np.concatenate(visible)
        self.assertEqual(first['count'],len(values));self.assertLess(first['count'],allrows)
        np.testing.assert_allclose(first['mean'],values.mean(0),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(first['scale'],np.maximum(values.std(0),1e-6),rtol=1e-12,atol=1e-12)
        for scene in prepared:scene['target'][:]=np.nan;scene['valid'][:]=False
        second=fit_normalizer(prepared,cfg);self.assertEqual(first,second)
        self.assertEqual(first['sampled_anchors'],6);self.assertFalse(first['truth_used'])
        self.assertTrue(np.all(np.array(first['scale'])>0))
        with TemporaryDirectory() as d:
            wrong=tiny_train();wrong[0]['split']='V_select'
            with self.assertRaisesRegex(ValueError,'T_core'):train_ptfm(wrong,cfg,Path(d)/'bad')

    def test_real_training_checkpoint_and_full_population_evaluation(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);cfg=config();trained=train_ptfm(tiny_train(),cfg,root/'fit')
            self.assertEqual(trained['optimizer_updates'],3);self.assertEqual(trained['exposure']['sampled_anchors'],6)
            source,metadata,validation=support_gate_cache_fixture(root)
            for label,step,file in [('initial',0,'step-0.pt'),('final',3,'final.pt')]:
                model=entry.endpoint_model(root/'fit'/file,step,cfg,trained)
                with patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw):
                    summary=evaluate_ptfm(model,validation,source,metadata,root/label,initial=label=='initial')
                self.assertEqual(summary['model_forwards'],4);self.assertEqual(summary['method_names'],list(METHODS))
                for method in METHODS:
                    self.assertEqual(summary['methods'][method]['population']['duplicate_prediction_group_count'],1)
                    self.assertEqual(summary['methods'][method]['full_target_metrics']['all']['position']['intersection_count'],127)
                self.assertGreater(summary['helper_availability']['all_output_groups']['supported_helper_absent'],0)
            comparison=compare_endpoints(root/'initial',root/'final');self.assertTrue(comparison['population_and_masks_exact'])
            with np.load(root/'final/system-predictions/states-3035.npz') as data:
                self.assertFalse(np.array_equal(data['ptfm_adapted_group_states'][...,2:],data['mean_supported_group_states'][...,2:]))
            with self.assertRaisesRegex(ValueError,'mismatch'):entry.endpoint_model(root/'fit/final.pt',0,cfg,trained)
            payload=torch.load(root/'fit/final.pt',weights_only=False)
            payload['normalizer']['mean'][0]+=1.;payload['model']['mean'][0]+=1.;torch.save(payload,root/'other-normalizer.pt')
            load_checkpoint(root/'other-normalizer.pt')  # Internally valid, but not this run's fitted statistics.
            with self.assertRaisesRegex(ValueError,'mismatch'):entry.endpoint_model(root/'other-normalizer.pt',3,cfg,trained)
            payload=torch.load(root/'fit/final.pt',weights_only=False);payload['actual_updates']=2;torch.save(payload,root/'corrupt.pt')
            with self.assertRaisesRegex(ValueError,'actual updates'):load_checkpoint(root/'corrupt.pt')
            payload=torch.load(root/'fit/final.pt',weights_only=False);payload['model']['mean'][0]+=1;torch.save(payload,root/'corrupt.pt')
            with self.assertRaisesRegex(ValueError,'normalizer buffers'):load_checkpoint(root/'corrupt.pt')
            with self.assertRaises(FileExistsError):train_ptfm(tiny_train(),cfg,root/'fit')

    def test_failure_does_not_commit_optimizer_exposure(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)/'fit'
            with patch('torch.optim.Adam.step',side_effect=RuntimeError('injected')),self.assertRaisesRegex(RuntimeError,'injected'):
                train_ptfm(tiny_train(),config(),output)
            run=read_json(output/'run.json');self.assertEqual(run['status'],'FAILED');self.assertEqual(run['optimizer_updates'],0)
            self.assertEqual(run['exposure']['sampled_anchors'],0);self.assertFalse((output/'final.pt').exists())
            self.assertEqual(load_checkpoint(output/'interrupted.pt')[1]['step'],0)

    def test_sparse_truth_does_not_filter_forward_and_bad_output_fails(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source,metadata,raw=support_gate_cache_fixture(root,sparse=True);model=PTFM(normalizer())
            with patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw):
                result=evaluate_ptfm(model,raw,source,metadata,root/'result')
            self.assertEqual(result['model_forwards'],4)
            for attr in ('position','speed','heading'):
                self.assertEqual(result['methods']['ptfm_adapted']['attribute_quantiles'][attr]['all']['point_count'],1)
            self.assertIsNone(result['methods']['ptfm_adapted']['attribute_quantiles']['position']['no_predicted_group']['p99'])
            forward=model.forward
            def bad(*args):
                result=forward(*args);result['states'][0,0]=torch.nan;return result
            with patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw),patch.object(model,'forward',side_effect=bad),self.assertRaises(ValueError):
                evaluate_ptfm(model,raw,source,metadata,root/'bad')
            self.assertEqual(read_json(root/'bad/run.json')['status'],'FAILED')

    def test_real_tiny_cli_train_and_both_endpoints(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source,metadata,validation=support_gate_cache_fixture(root);training=tiny_train();cfg=config()
            manifest=dict(splits={'T_core':[str(s['scene_id']) for s in training],'V_select':['3035']},components={'3035':'shared'})
            paired=sequence_reference(root/'control',source,metadata,validation)
            for name,obj in [('config',cfg),('manifest',manifest),('alignment',{})]:write_json(root/(name+'.json'),obj)
            args=Namespace(source_system=str(source),output=str(root/'out'),archive=['not-opened.zip'],
                paired_control=str(paired),manifest=str(root/'manifest.json'),alignment=str(root/'alignment.json'),config=str(root/'config.json'),device='cpu',cpu_threads=4,time_policy='advisory')
            with patch.object(entry,'production_contract'),patch.object(entry,'_bind_source'),patch.object(entry,'bind_paired_control'),patch.object(entry,'_load_metadata',return_value=metadata),\
                    patch.object(entry,'load_timed_dataset',return_value=dict(train=training,validation=validation,report={'fixture':True})),\
                    patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw):
                report=entry.run(args)
            self.assertEqual(report['status'],'COMPLETED');self.assertEqual(report['optimizer_updates'],3)
            self.assertEqual(read_json(root/'out/run.json')['completed_endpoints'],['initial','final'])
            self.assertFalse(read_json(root/'out/run.json')['validation_selection'])
            self.assertEqual(report['paired_control']['status'],'PAIR_VERIFIED')
            self.assertEqual(report['paired_control']['matched_anchors'],6)
            with gzip.open(root/'out/transformer-comparison.json.gz','rt') as handle:comparison=json.load(handle)
            self.assertTrue(comparison['full_group_population_controls_and_masks_exact'])
            self.assertEqual(comparison['metrics']['all']['position']['ptfm_adapted']['unit_count'],127)
            self.assertEqual(set(comparison['metrics']['all']),{'position','speed','heading'})
            datafile=paired/'final/system-predictions/states-3035.npz'
            with np.load(datafile) as d:corrupt={k:d[k].copy() for k in d.files}
            corrupt['learned_sequence_group_states'][0,0,2]+=1.;np.savez_compressed(datafile,**corrupt)
            with self.assertRaisesRegex(ValueError,'cached group output'):compare_sequence_control(root/'out/final',paired/'final')

    def test_pairing_mismatch_prevents_first_optimizer_exposure(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);paired=sequence_reference(root/'control')
            file=paired/'fit/training.jsonl';rows=[json.loads(line) for line in file.read_text().splitlines()]
            self.assertNotEqual(rows[0]['batch'][0],rows[0]['batch'][1])
            rows[0]['batch'].reverse();file.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'actual batch differs'):
                train_ptfm(tiny_train(),config(),root/'bad',paired_reference_fit=paired/'fit')
            failed=read_json(root/'bad/run.json');self.assertEqual(failed['status'],'FAILED')
            self.assertEqual(failed['optimizer_updates'],0);self.assertEqual(failed['exposure']['sampled_anchors'],0)
            self.assertEqual(failed['paired_control']['matched_anchors'],0);self.assertFalse((root/'bad/final.pt').exists())

if __name__=='__main__':unittest.main()
