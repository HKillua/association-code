"""076B real tiny old/new training, frozen complete population and failure binding."""
from argparse import Namespace
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import gzip,json,unittest
import numpy as np
import torch
from association.utils.runtime import read_json,write_json
from association.evaluation.ptfm import compare_ptfm_control,evaluate_ptfm
from tests.test_ptfm import config,sequence_reference,tiny_train,support_gate_cache_fixture
from tools.experiments import ptfm_fit as old_entry,ptfm_residual_fit as entry


def old_fixture(root):
    source,metadata,validation=support_gate_cache_fixture(root)
    train=tiny_train();cfg=config()
    manifest=dict(splits={'T_core':[str(s['scene_id']) for s in train],'V_select':['3035']},components={'3035':'shared'})
    paired=sequence_reference(root/'sequence',source,metadata,validation)
    frozen=root/'frozen';(frozen/'configs').mkdir(parents=True)
    write_json(frozen/'configs/ptfm-adapted-level1.json',cfg)
    residual=read_json(entry.ROOT/'configs/ptfm-residual-level1.json');residual['training']=deepcopy(cfg['training'])
    for name,obj in [('direct-config',cfg),('residual-config',residual),('manifest',manifest),('alignment',{})]:write_json(root/(name+'.json'),obj)
    args=Namespace(source_system=str(source),output=str(root/'old'),archive=['not-opened.zip'],paired_control=str(paired),
        manifest=str(root/'manifest.json'),alignment=str(root/'alignment.json'),config=str(root/'direct-config.json'),device='cpu',cpu_threads=4,time_policy='advisory')
    with ExitStack() as stack:
        for name in ('production_contract','_bind_source','bind_paired_control'):stack.enter_context(patch.object(old_entry,name))
        stack.enter_context(patch.object(old_entry,'_load_metadata',return_value=metadata))
        stack.enter_context(patch.object(old_entry,'load_timed_dataset',return_value=dict(train=train,validation=validation,report={'fixture':True})))
        stack.enter_context(patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw))
        old_entry.run(args)
    args.output=str(root/'new');args.paired_ptfm=str(root/'old');args.config=str(root/'residual-config.json')
    return args,source,metadata,validation,train,manifest,residual,frozen


def run_fixture(args,metadata,validation,train,frozen):
    with ExitStack() as stack:
        for name in ('production_contract','_bind_source','bind_paired_control'):stack.enter_context(patch.object(entry,name))
        stack.enter_context(patch.object(entry,'ROOT',frozen))
        stack.enter_context(patch.object(entry,'_load_metadata',return_value=metadata))
        stack.enter_context(patch.object(entry,'load_timed_dataset',return_value=dict(train=train,validation=validation,report={'fixture':True})))
        stack.enter_context(patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw))
        return entry.run(args)


class PTFMResidualEntryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_real_old_new_and_transformer_pair_full_population(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);args,source,metadata,validation,train,manifest,cfg,frozen=old_fixture(root)
            result=run_fixture(args,metadata,validation,train,frozen)
            self.assertEqual(result['status'],'COMPLETED');self.assertEqual(result['optimizer_updates'],3)
            self.assertEqual(result['paired_control']['status'],'PAIR_VERIFIED');self.assertEqual(result['paired_ptfm']['status'],'PAIR_VERIFIED')
            self.assertTrue(result['paired_ptfm']['initial_parameters_exact_except_position_rows'])
            self.assertEqual(read_json(root/'new/run.json')['completed_endpoints'],['initial','final'])
            for label in ('initial','final'):
                report=read_json(root/'new'/label/'summary.json')
                self.assertEqual(report['position_mode'],'prior095_residual')
                self.assertEqual(report['methods']['ptfm_adapted']['population']['duplicate_prediction_group_count'],1)
                self.assertEqual(report['methods']['ptfm_adapted']['full_target_metrics']['all']['position']['intersection_count'],127)
                with np.load(root/'new'/label/'predictions/states-3035.npz') as data:
                    self.assertIn('group_base_states',data.files);self.assertIn('group_position_residual_deg',data.files)
                    if label=='initial':
                        np.testing.assert_array_equal(data['ptfm_adapted_group_states'][...,:2],data['fixed_prior095_group_states'][...,:2])
                        self.assertFalse(data['group_position_residual_deg'].any())
            with gzip.open(root/'new/initial-ptfm-comparison.json.gz','rt') as handle:initial=json.load(handle)
            self.assertTrue(initial['nonposition_initial_replay']['all_output_groups_verified'])
            self.assertEqual(initial['nonposition_initial_replay']['maximum_native_difference'],{'speed':0.,'heading':0.})
            with gzip.open(root/'new/ptfm-comparison.json.gz','rt') as handle:comparison=json.load(handle)
            self.assertTrue(comparison['full_group_population_controls_and_masks_exact'])
            self.assertEqual(comparison['metrics']['all']['speed']['ptfm_adapted']['unit_count'],127)
            delta=comparison['residual_minus_ptfm']['all']['position']
            self.assertAlmostEqual(delta['harm_sum']-delta['improvement_sum'],delta['signed_error_change_sum'])
            self.assertIn('improvement_tail',delta);self.assertIn('absolute_change_tail',delta)
            self.assertEqual(len(comparison['position_weightings']['all']['ptfm_adapted']),4)
            with self.assertRaises(FileExistsError):run_fixture(args,metadata,validation,train,frozen)
            args.output=str(source/'forbidden')
            with self.assertRaisesRegex(ValueError,'preserve source'):run_fixture(args,metadata,validation,train,frozen)

    def test_binding_rejects_selected_wrong_population_and_incomplete(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);args,source,metadata,validation,train,manifest,cfg,frozen=old_fixture(root);old=root/'old'
            with patch.object(entry,'ROOT',frozen):
                bound=entry.bind_ptfm_control(old,source,manifest,cfg);self.assertEqual(bound['status'],'BOUND')
                cases=[('final/run.json',lambda d:d['checkpoint'].update(path=str(old/'fit/selected.pt'))),
                       ('run.json',lambda d:d.update(source_system=str(root/'different'))),
                       ('fit/run.json',lambda d:d.update(training_scene_ids=['wrong'])),
                       ('final/summary.json',lambda d:d.update(scene_ids=['wrong'])),
                       ('summary.json',lambda d:d.update(status='RUNNING'))]
                for filename,change in cases:
                    original=read_json(old/filename);bad=deepcopy(original);change(bad);write_json(old/filename,bad)
                    with self.subTest(filename=filename),self.assertRaises(ValueError):entry.bind_ptfm_control(old,source,manifest,cfg)
                    write_json(old/filename,original)
            original=read_json(old/'final/run.json');original['checkpoint']['path']=str(old/'fit/selected.pt');write_json(old/'final/run.json',original)
            with self.assertRaises(ValueError):run_fixture(args,metadata,validation,train,frozen)
            self.assertEqual(read_json(root/'new/run.json')['status'],'FAILED');self.assertFalse((root/'new/fit').exists())

    def test_all_group_initial_nonposition_output_is_checked(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);args,source,metadata,validation,train,manifest,cfg,frozen=old_fixture(root)
            run_fixture(args,metadata,validation,train,frozen)
            # Corrupt a group that may be duplicate/unselected: all-group consistency
            # must reject it independently of which targets enter the MAE.
            path=root/'new/initial/system-predictions/states-3035.npz'
            with np.load(path) as arrays:data={k:arrays[k].copy() for k in arrays.files}
            data['ptfm_adapted_group_states'][-1,0,2]+=1
            np.savez_compressed(path,**data)
            with self.assertRaisesRegex(ValueError,'cached group output'):
                compare_ptfm_control(root/'new/initial',root/'old/initial',initial=True)

if __name__=='__main__':unittest.main()
