"""091 objective geometry, T-only calibration and real paired train/cache chain."""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import gzip,json,math,unittest
import numpy as np
import torch
from association.models.ptfm import PTFM,objectives,ptfm_inputs,position_objective_values,POSITION_OBJECTIVES
from association.training.ptfm import configuration,load_checkpoint,fit_normalizer
from association.training.evidence_fusion import prepare_evidence_scenes,training_members,TRAIN_MODES
from association.evaluation.ptfm_position_objectives import calibrate_initial_inputs,validate_calibration,fixed_training_probe,ERRORS
from association.utils.runtime import read_json,write_json
from tests.test_ptfm import normalizer,motion,tiny_train
from tests.test_ptfm_residual_fit import old_fixture,run_fixture
from tools.experiments import ptfm_objective_fit as old_entry,ptfm_position_objective_fit as entry
from tools.analysis import ptfm_position_objective_compare as comparison


def fixture_config(kind):
    cfg=read_json(entry.ROOT/f'configs/ptfm-{kind}-level1.json');cfg['training'].update(updates=3,batch_size=2)
    return cfg


def explicit_rows(prepared,config):
    """Every mode observes one supervised and one unsupervised anchor."""
    scene=prepared[0];rows=[]
    for step,mode in enumerate(TRAIN_MODES,1):
        batch=[]
        for anchor in (0,1):
            members=training_members(scene,anchor,mode);valid=scene['valid'][anchor]
            eligible=~scene['motion']['support'][anchor]&scene['motion']['support'][members[1:]].any(0)
            counts=[int(valid[:,:2].all(-1).sum()),int(valid[:,2].sum()),int(valid[:,3].sum())]
            batch.append(dict(scene_id=str(scene['scene_id']),anchor_index=anchor,node_id=str(scene['node_ids'][anchor]),
                members=[int(x) for x in members],valid_position_points=counts[0],eligible_query_points=int(eligible.sum()),
                eligible_valid_position_points=int((eligible&valid[:,:2].all(-1)).sum()),task_valid_counts=counts))
        rows.append(dict(step=step,mode=mode,batch=batch))
    return rows


class PTFMPositionObjectiveTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_radial_dateline_geodesic_zero_gradients_and_scope_isolation(self):
        states=torch.tensor([[0.,179.,4.,1.],[0.,0.,2.,359.]],dtype=torch.float64,requires_grad=True)
        target=torch.tensor([[0.,-179.,float('nan'),float('nan')],[0.,0.,2.,359.]],dtype=torch.float64)
        valid=torch.tensor([[True,True,False,False],[True,True,True,True]])
        values,mask=position_objective_values(states,target,valid,torch.tensor([.5,2.,10.,30.],dtype=torch.float64))
        self.assertEqual(values['normalized_squared'].tolist(),[1.,0.]);self.assertEqual(values['radial-linear'].tolist(),[1.,0.])
        self.assertAlmostEqual(float(values['geodesic-linear'][0].detach()),6371*math.radians(2),places=10)
        for value in values.values():
            gradient=torch.autograd.grad(value.sum(),states,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all());self.assertFalse(gradient[1].any())
        model=PTFM(normalizer(),position_mode='prior095_residual');args=ptfm_inputs(motion(),[0,1,2],include_prior=True)
        for kind in POSITION_OBJECTIVES:
            result=model(*args);truth=result['states'].detach().clone();truth[:,0]+=.1
            use=torch.ones_like(truth,dtype=torch.bool)
            loss,diag=objectives(model,result,truth,use,mode='position-only',position_objective=kind,position_loss_scale=2.)
            self.assertAlmostEqual(float(loss.detach()),2*diag['raw_position_sample_means'][kind]/3+math.log(3),places=11)
            model.zero_grad(set_to_none=True);loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertFalse(model.output.weight.grad[2:].any());self.assertFalse(model.task_weights.weight.grad.any())
            result=model(*args);loss,diag=objectives(model,result,torch.full_like(truth,torch.nan),torch.zeros_like(use),
                mode='position-only',position_objective=kind,position_loss_scale=2.)
            self.assertEqual(float(loss.detach()),0.);self.assertEqual(diag['actual_task_coefficients'],[0.,0.,0.])
            model.zero_grad(set_to_none=True);loss.backward();self.assertTrue(all(p.grad is not None and not p.grad.any() for p in model.parameters()))
        for change in (dict(scope='ptfm-prior095-objective-control-v1'),dict(objective_mode='fixed-equal'),
                       dict(position_loss_normalization='V-selected'),dict(position_objective='squared')):
            bad=fixture_config('radial-linear');bad.update(change)
            with self.assertRaises(ValueError):configuration(bad)
        with self.assertRaises(ValueError):objectives(model,model(*args),truth,use,mode='fixed-equal',position_objective='radial-linear',position_loss_scale=1.)

    def test_full_T_calibration_preserves_zero_samples_rng_and_fixed_probe(self):
        cfg=fixture_config('radial-linear');prepared=prepare_evidence_scenes(tiny_train(),motion_observation_policy='canonical_all')[:1]
        prepared[0]['valid'][1]=False;prepared[0]['target'][1]=np.nan
        rows=explicit_rows(prepared,cfg);norm=fit_normalizer(prepared,cfg);before=torch.get_rng_state().clone()
        with patch.object(PTFM,'forward',side_effect=AssertionError('calibration must not forward model')):
            calibration=calibrate_initial_inputs(prepared,rows,cfg,norm,'/tmp/091-source')
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(calibration['sample_count'],6)
        self.assertEqual(calibration['zero_supervision_samples'],3);self.assertTrue(calibration['truth_used']);self.assertFalse(norm['truth_used'])
        expected={key:[] for key in ERRORS}
        for row in rows:
            sample=row['batch'][0];scene=prepared[0];base=ptfm_inputs(scene['motion'],sample['members'],include_prior=True)[4].numpy()
            truth=scene['target'][0];use=scene['valid'][0,:,:2].all(-1)
            lat=(base[:,0]-truth[:,0])/norm['scale'][0];lon=((base[:,1]-truth[:,1]+180)%360-180)/norm['scale'][1]
            squared=lat**2+lon**2
            a=np.sin(np.deg2rad(base[:,0]-truth[:,0])/2)**2+np.cos(np.deg2rad(base[:,0]))*np.cos(np.deg2rad(truth[:,0]))*np.sin(np.deg2rad((base[:,1]-truth[:,1]+180)%360-180)/2)**2
            for key,value in zip(ERRORS,(squared,np.sqrt(squared),2*6371*np.arcsin(np.sqrt(np.clip(a,0,1))))):expected[key].extend([float(value[use].mean()),0.])
        for key in ERRORS:self.assertAlmostEqual(calibration['initial_sample_equal_means'][key],float(np.mean(expected[key])),places=8)
        for kind in POSITION_OBJECTIVES:
            self.assertAlmostEqual(calibration['alpha'][kind]*calibration['initial_sample_equal_means'][kind],calibration['initial_sample_equal_means']['normalized_squared'],places=10)
        wrong=deepcopy(prepared);wrong[0]['split']='V_select'
        with self.assertRaisesRegex(ValueError,'T_core'):calibrate_initial_inputs(wrong,rows,cfg,norm,'/tmp/091-source')
        with self.assertRaisesRegex(ValueError,'full original'):calibrate_initial_inputs(prepared,rows[:1],cfg,norm,'/tmp/091-source')
        corrupted=deepcopy(rows);corrupted[0]['batch'][0]['members']=[99]
        with self.assertRaisesRegex(ValueError,'members/task masks'):calibrate_initial_inputs(prepared,corrupted,cfg,norm,'/tmp/091-source')
        model=PTFM(norm,position_mode='prior095_residual');parameters=deepcopy(model.state_dict());before=torch.get_rng_state().clone()
        probe=fixed_training_probe(model,prepared,rows,cfg,source_fit='/tmp/probe',label='initial')
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in parameters.items()))
        self.assertEqual(probe['sample_count'],6);self.assertEqual(probe['metrics']['zero_supervision_samples'],3)
        for key in ERRORS:self.assertAlmostEqual(probe['metrics']['conditional_sample_means'][key],calibration['initial_sample_equal_means'][key],places=8)
        self.assertEqual(sum(r['valid_point_count'] for r in probe['by_original_region'].values()),probe['metrics']['valid_point_count'])

    def test_real_076B_086B_091_pair_complete_cache_chain_and_rejections(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);args,source,metadata,validation,train,manifest,residual,frozen=old_fixture(root)
            run_fixture(args,metadata,validation,train,frozen);write_json(frozen/'configs/ptfm-residual-level1.json',residual)
            cfg086=read_json(entry.ROOT/'configs/ptfm-position-only-level1.json');cfg086['training']=deepcopy(residual['training'])
            write_json(frozen/'configs/ptfm-position-only-level1.json',cfg086)
            request=Namespace(**vars(args));request.config=str(frozen/'configs/ptfm-position-only-level1.json')
            request.output=str(root/'086B');request.paired_residual=str(root/'new')
            with ExitStack() as stack:
                for name in ('production_contract','_bind_source','bind_paired_control'):stack.enter_context(patch.object(old_entry,name))
                stack.enter_context(patch.object(old_entry,'ROOT',frozen))
                stack.enter_context(patch.object(old_entry,'_load_metadata',return_value=metadata))
                stack.enter_context(patch.object(old_entry,'load_timed_dataset',return_value=dict(train=train,validation=validation,report={'fixture':True})))
                stack.enter_context(patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw))
                old_entry.run(request)
            outputs={};configs={}
            for kind in POSITION_OBJECTIVES:
                cfg=fixture_config(kind);configs[kind]=cfg;config_path=root/(kind+'.json');write_json(config_path,cfg)
                request=Namespace(**vars(args));request.config=str(config_path);request.output=str(root/kind)
                request.paired_residual=str(root/'new');request.paired_position_only=str(root/'086B')
                with ExitStack() as stack:
                    for name in ('production_contract','_bind_source','bind_paired_control'):stack.enter_context(patch.object(entry,name))
                    stack.enter_context(patch.object(entry,'ROOT',frozen));stack.enter_context(patch.object(old_entry,'ROOT',frozen))
                    stack.enter_context(patch.object(entry,'_load_metadata',return_value=metadata))
                    stack.enter_context(patch.object(entry,'load_timed_dataset',return_value=dict(train=train,validation=validation,report={'fixture':True})))
                    stack.enter_context(patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw))
                    result=entry.run(request)
                output=root/kind;outputs[kind]=output
                self.assertEqual(result['status'],'COMPLETED');self.assertEqual(result['optimizer_updates'],3)
                self.assertEqual(result['paired_position_only']['status'],'PAIR_VERIFIED')
                initial=load_checkpoint(output/'fit/step-0.pt')[1];final=load_checkpoint(output/'fit/final.pt')[1]
                original=load_checkpoint(root/'086B/fit/step-0.pt')[1]
                self.assertEqual(initial['loss_calibration'],final['loss_calibration']);self.assertEqual(initial['loss_calibration'],result['loss_calibration'])
                self.assertEqual(initial['numpy_rng'],original['numpy_rng']);self.assertTrue(torch.equal(initial['torch_rng'],original['torch_rng']))
                self.assertTrue(all(torch.equal(v,original['model'][k]) for k,v in initial['model'].items()))
                self.assertEqual(len(final['optimizer']['state']),len(final['optimizer']['param_groups'][0]['params']))
                for key in ('output.weight','output.bias'):self.assertTrue(torch.equal(initial['model'][key][2:],final['model'][key][2:]))
                end=read_json(output/'final/summary.json')
                self.assertEqual(end['methods']['ptfm_adapted']['full_target_metrics']['all']['position']['intersection_count'],127)
                self.assertEqual(end['methods']['ptfm_adapted']['population']['duplicate_prediction_group_count'],1)
                with gzip.open(output/'initial-position-only-comparison.json.gz','rt') as handle:self.assertTrue(json.load(handle)['all_initial_group_states_exact'])
                with patch.object(comparison,'bind_paired_control'):
                    bound=comparison.bind_arm(output,kind,expected_config=cfg,expected_manifest=manifest)
                self.assertTrue(bound['training']['calibrated_sample_mean_objective_verified'])
                self.assertEqual(bound['probe']['source086B_final']['sample_count'],6)
            request=Namespace(radial_linear=str(outputs['radial-linear']),geodesic_linear=str(outputs['geodesic-linear']),output=str(root/'compare'),cpu_threads=4)
            with patch.object(comparison,'bind_paired_control'),patch.object(PTFM,'forward',side_effect=AssertionError('cache closeout must not forward')):
                result=comparison.run(request,expected_configs=configs,expected_manifest=manifest)
            self.assertTrue(result['initial_replay']);self.assertEqual(result['model_forwards'],0);self.assertTrue(result['matched_actual_samples'])
            self.assertEqual(result['scene_count'],1);self.assertEqual(result['paired_successful_updates'],3)
            output=outputs['geodesic-linear']
            def bind():
                with patch.object(comparison,'bind_paired_control'):
                    return comparison.bind_arm(output,'geodesic-linear',expected_config=configs['geodesic-linear'],expected_manifest=manifest)
            cases=[('fit/loss-calibration.json',lambda x:x['alpha'].update({'geodesic-linear':x['alpha']['geodesic-linear']*2})),
                   ('fit/fixed-training-probe-plan.json',lambda x:x['rows'][0]['batch'][0].update(anchor_index=99)),
                   ('fit/fixed-training-probe.json',lambda x:x['endpoints']['final'].update(selection=True)),
                   ('final/run.json',lambda x:x['checkpoint'].update(path='selected.pt')),
                   ('final/summary.json',lambda x:x.update(scene_ids=['wrong']))]
            for filename,mutate in cases:
                path=output/filename;original=read_json(path);bad=deepcopy(original);mutate(bad);write_json(path,bad)
                with self.subTest(filename=filename),self.assertRaises(ValueError):bind()
                write_json(path,original)
            path=output/'fit/training.jsonl';original=path.read_text();rows=[json.loads(line) for line in original.splitlines()]
            rows[0]['raw_position_sample_means']['geodesic-linear']*=2;path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'calibrated position loss'):bind()
            path.write_text(original)
            path=output/'fit/final.pt';payload=torch.load(path,weights_only=False);payload['loss_calibration']['source_fit']='/tmp/different086B';torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'checkpoint calibration'):bind()
            with self.assertRaises(FileExistsError):entry.run(Namespace(**dict(vars(args),output=str(output),paired_residual=str(root/'new'),paired_position_only=str(root/'086B'))))

if __name__=='__main__':unittest.main()
