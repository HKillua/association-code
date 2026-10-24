"""086 objective math, exact source binding and real paired train/evaluate chain."""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import gzip,json,math,unittest
import numpy as np
import torch
from association.models.ptfm import PTFM,objectives,ptfm_inputs
from association.training.ptfm import load_checkpoint,configuration
from association.utils.runtime import read_json,write_json
from tests.test_ptfm import normalizer,motion
from tests.test_ptfm_residual_fit import old_fixture,run_fixture
from tools.experiments import ptfm_objective_fit as entry
from tools.analysis import ptfm_objective_compare as comparison


class PTFMObjectiveTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_fixed_coefficients_masks_circular_values_and_zero_gradients(self):
        model=PTFM(normalizer());args=ptfm_inputs(motion(),[0,1,2])
        for mode in ('fixed-equal','position-only'):
            result=model(*args);truth=result['states'].detach().clone();truth[:,0]+=.1;truth[:,1]-=359.9;truth[:,3]=(truth[:,3]+359.)%360
            valid=torch.ones_like(truth,dtype=torch.bool);valid[0,2]=False
            loss,diag=objectives(model,result,truth,valid,mode=mode)
            use=range(3) if mode=='fixed-equal' else range(1)
            self.assertAlmostEqual(float(loss.detach()),sum(diag['task_losses'][i]/3+math.log(3) for i in use),places=10)
            self.assertEqual(diag['actual_task_coefficients'],[1/3,1/3,1/3] if mode=='fixed-equal' else [1/3,0.,0.])
            self.assertAlmostEqual(diag['task_losses'][2],1/30,places=12)
            model.zero_grad(set_to_none=True);loss.backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
            self.assertEqual(float(model.task_weights.weight.grad.abs().sum()),0.)
            if mode=='position-only':self.assertEqual(float(model.output.weight.grad[2:].abs().sum()),0.)
            result=model(*args);zero,diag=objectives(model,result,torch.full_like(truth,torch.nan),torch.zeros_like(valid),mode=mode)
            self.assertEqual(float(zero.detach()),0.);self.assertEqual(diag['actual_task_coefficients'],[0.,0.,0.])
            model.zero_grad(set_to_none=True);zero.backward();self.assertTrue(all(p.grad is not None and not p.grad.any() for p in model.parameters()))
        bad=read_json(entry.ROOT/'configs/ptfm-fixed-equal-level1.json');bad['objective_mode']='adaptive'
        with self.assertRaises(ValueError):configuration(bad)

    def test_real_076b_and_both_086_arms_full_population_pair_and_rejections(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);args,source,metadata,validation,train,manifest,residual,frozen=old_fixture(root)
            run_fixture(args,metadata,validation,train,frozen)
            write_json(frozen/'configs/ptfm-residual-level1.json',residual)
            outputs={};configs={}
            for mode in ('fixed-equal','position-only'):
                cfg=read_json(entry.ROOT/f'configs/ptfm-{mode}-level1.json');cfg['training']=deepcopy(residual['training']);configs[mode]=cfg
                config_path=root/(mode+'.json');write_json(config_path,cfg)
                output=root/mode;request=Namespace(**vars(args));request.config=str(config_path);request.output=str(output);request.paired_residual=str(root/'new')
                with ExitStack() as stack:
                    for name in ('production_contract','_bind_source','bind_paired_control'):stack.enter_context(patch.object(entry,name))
                    stack.enter_context(patch.object(entry,'ROOT',frozen))
                    stack.enter_context(patch.object(entry,'_load_metadata',return_value=metadata))
                    stack.enter_context(patch.object(entry,'load_timed_dataset',return_value=dict(train=train,validation=validation,report={'fixture':True})))
                    stack.enter_context(patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw))
                    report=entry.run(request)
                self.assertEqual(report['status'],'COMPLETED');self.assertEqual(report['optimizer_updates'],3)
                self.assertEqual(report['paired_ptfm']['status'],'PAIR_VERIFIED');self.assertTrue(report['paired_ptfm']['initial_parameters_exact'])
                initial=load_checkpoint(output/'fit/step-0.pt')[1];final=load_checkpoint(output/'fit/final.pt')[1]
                original=load_checkpoint(root/'new/fit/step-0.pt')[1]
                for key in initial['model']:self.assertTrue(torch.equal(initial['model'][key],original['model'][key]))
                for key in ('task_weights.weight','task_weights.bias'):self.assertTrue(torch.equal(initial['model'][key],final['model'][key]))
                if mode=='position-only':
                    for key in ('output.weight','output.bias'):self.assertTrue(torch.equal(initial['model'][key][2:],final['model'][key][2:]))
                self.assertEqual(len(final['optimizer']['state']),len(final['optimizer']['param_groups'][0]['params']))
                end=read_json(output/'final/summary.json');self.assertEqual(end['methods']['ptfm_adapted']['full_target_metrics']['all']['position']['intersection_count'],127)
                self.assertEqual(end['methods']['ptfm_adapted']['population']['duplicate_prediction_group_count'],1)
                with gzip.open(output/'initial-ptfm-comparison.json.gz','rt') as handle:self.assertTrue(json.load(handle)['all_initial_group_states_exact'])
                rows=[json.loads(line) for line in (output/'fit/training.jsonl').read_text().splitlines()]
                for row in rows:self.assertAlmostEqual(sum(row['actual_task_contributions']),row['loss'],places=12)
                window=read_json(output/'fit/training-windows.json')['windows'][0]
                self.assertIn('actual_task_contributions_mean',window);self.assertIn('l2_mean',window)
                bound=comparison.bind_arm(output,mode,expected_config=cfg,expected_manifest=manifest)
                self.assertTrue(bound['training']['actual_fixed_objective_verified'])
                self.assertEqual(bound['training']['optimized_task_valid_counts'][1:],
                    read_json(output/'fit/run.json')['exposure']['task_valid_counts'][1:] if mode=='fixed-equal' else [0,0])
                outputs[mode]=output
            request=Namespace(fixed_equal=str(outputs['fixed-equal']),position_only=str(outputs['position-only']),output=str(root/'compare'),cpu_threads=4)
            result=comparison.run(request,expected_configs=configs,expected_manifest=manifest);self.assertTrue(result['initial_replay']);self.assertEqual(result['paired_successful_updates'],3)
            self.assertEqual(result['scene_count'],1);self.assertTrue(result['matched_actual_samples']);self.assertEqual(result['model_forwards'],0)
            def bind_b():return comparison.bind_arm(outputs['position-only'],'position-only',expected_config=configs['position-only'],expected_manifest=manifest)
            with self.assertRaisesRegex(ValueError,'frozen production configuration'):
                comparison.bind_arm(outputs['position-only'],'position-only')
            # A changed successful sample cannot pass because aggregate counters match.
            path=outputs['position-only']/'fit/training.jsonl';original=path.read_text();rows=[json.loads(line) for line in original.splitlines()];rows[0]['batch'][0]['anchor_index']+=1
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'successful batches'):bind_b()
            path.write_text(original)
            # Logged optimized terms must be supported by the actual sample masks.
            for key in ('actual_task_coefficients','actual_task_regularizers','actual_task_contributions'):
                rows=[json.loads(line) for line in original.splitlines()];rows[0][key][2]=.1
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.assertRaisesRegex(ValueError,'objective diagnostic'):bind_b()
                path.write_text(original)
            path=outputs['position-only']/'final/summary.json';source=read_json(path);bad=deepcopy(source);bad['scene_ids']=['not-full-V'];write_json(path,bad)
            with self.assertRaisesRegex(ValueError,'frozen V population'):bind_b()
            write_json(path,source)
            # Invalid endpoint provenance and modified initial parameters fail closed.
            path=outputs['position-only']/'final/run.json';original=read_json(path);bad=deepcopy(original);bad['checkpoint']['path']='selected.pt';write_json(path,bad)
            with self.assertRaisesRegex(ValueError,'provenance'):bind_b()
            write_json(path,original)
            path=outputs['position-only']/'fit/step-0.pt';cp=torch.load(path,weights_only=False);cp['model']['output.weight'][0,0]+=1;torch.save(cp,path)
            with self.assertRaisesRegex(ValueError,'initial parameters'):bind_b()

if __name__=='__main__':unittest.main()
