"""083 production binding, real tiny train/evaluation and cached comparison."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from association.utils.runtime import read_json, write_json
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_train
from tools.experiments import sequence_position_fit as entry
from tools.analysis import compare_helper_dropout as compare


class DropoutPipelineTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_production_configuration_requires_fixed_profile_and_reference_before_data(self):
        paths=[entry.ROOT/'configs'/name for name in ('sequence-position-transformer-helper-dropout-level1.json',
            'data/mtad-level1-scene3000.json','data/mtad-level1-alignment.json')]
        config,manifest,alignment=map(read_json,paths)
        entry.production_contract(config,manifest,alignment,profile='helper-dropout4000')
        for change in ('seed','updates'):
            bad=deepcopy(config);bad['training'][change]+=1
            if change=='seed':bad['augmentation']['seed']+=1
            with self.assertRaises(ValueError):entry.production_contract(bad,manifest,alignment,profile='helper-dropout4000')
        with TemporaryDirectory() as temp:
            args=Namespace(source_system=temp+'/source/system',output=temp+'/out',paired_control=None,
                config=str(paths[0]),manifest=str(paths[1]),alignment=str(paths[2]),archive=['unread.zip'],
                profile='helper-dropout4000',device='cpu',cpu_threads=4,time_policy='advisory')
            with patch.object(entry,'load_timed_dataset') as loader:
                with self.assertRaisesRegex(ValueError,'paired-control'):entry.run(args)
                loader.assert_not_called()

    def test_actual_fit_initial_final_comparison_and_tampered_augmentation_rejection(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source,metadata,validation=support_gate_cache_fixture(root)
            train=tiny_train();manifest=dict(splits={'T_core':[str(s['scene_id']) for s in train],'V_select':['3035']},components={'3035':'shared'})
            config=read_json(entry.ROOT/'configs/sequence-position-transformer-extended-level1.json')
            config['training'].update(updates=3,batch_size=2)
            for name,value in (('config',config),('manifest',manifest),('alignment',{})):write_json(root/(name+'.json'),value)
            args=Namespace(source_system=str(source),output=str(root/'linear'),paired_control=None,
                archive=['unread.zip'],manifest=str(root/'manifest.json'),alignment=str(root/'alignment.json'),
                config=str(root/'config.json'),device='cpu',cpu_threads=4,time_policy='advisory',profile='extended4000')
            dataset=lambda *a,**k:dict(train=train,validation=validation,report={'fixture':True})
            with patch.object(entry,'production_contract'),patch.object(entry,'_bind_source'), \
                    patch.object(entry,'_load_metadata',return_value=metadata),patch.object(entry,'load_timed_dataset',side_effect=dataset), \
                    patch('association.evaluation.sequence_position.frozen_motion',side_effect=lambda raw,*_:raw):
                entry.run(args)
                config.update(scope='sequence-position-helper-dropout-v1',augmentation=dict(kind='helper_fragment_dropout',probability=.5,seed=20344023))
                write_json(root/'config.json',config)
                args.profile='helper-dropout4000';args.output=str(root/'dropout');args.paired_control=str(root/'linear')
                result=entry.run(args)
            self.assertEqual(result['status'],'COMPLETED')
            self.assertTrue(result['augmentation_applied_only_to_training'])
            self.assertEqual(result['augmentation_exposure']['successful_updates'],3)
            self.assertEqual(compare.verify_dropout_pair(root/'dropout/fit',root/'linear/fit')['matched_updates'],3)
            def meta(path,arm):
                final=read_json(Path(path)/'final/summary.json')
                return dict(path=Path(path),method='learned_sequence',source_system=str(source),scene_ids=['3035'],components={'3035':'shared'},summary=final)
            with patch.object(compare,'_metadata',side_effect=meta),patch.object(entry,'bind_linear_control'):
                summary=compare.run(Namespace(dropout_run=str(root/'dropout'),linear_run=str(root/'linear'),output=str(root/'comparison'),cpu_threads=4))
            self.assertEqual(summary['position_population']['covered_target_points'],127)
            self.assertEqual(summary['population']['predicted_group_count'],4)
            self.assertEqual(summary['population']['duplicate_prediction_group_count'],1)
            self.assertEqual(sum(summary['helper_cohorts'][k]['dropout']['point_equal']['unit_count'] for k in ('zero','one','two_or_more')),127)
            self.assertFalse(summary['scientific_screen']['worth_matched_replication'])  # One component: no outside population.
            log=root/'dropout/fit/training.jsonl';text=log.read_text();rows=[json.loads(x) for x in text.splitlines()]
            rows[0]['augmentation_batch'][0]['retained_members']=[999]
            log.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaisesRegex(ValueError,'retained members'):compare.verify_dropout_pair(root/'dropout/fit',root/'linear/fit')
            log.write_text(text)
            bad=[json.loads(x) for x in text.splitlines()]
            bad[0]['augmentation_batch'][0]['retained_eligible_valid_position_points']=999
            log.write_text(''.join(json.dumps(x)+'\n' for x in bad))
            with self.assertRaisesRegex(ValueError,'eligible counts'):compare.verify_dropout_pair(root/'dropout/fit',root/'linear/fit')
            log.write_text(text)
            with self.assertRaises(FileExistsError):compare.run(Namespace(dropout_run=str(root/'dropout'),linear_run=str(root/'linear'),output=str(root/'comparison'),cpu_threads=4))

    def test_screen_requires_magnitude_tail_and_population_breadth(self):
        def row(new=95.,tail=190.):
            old=dict(mae=100.,rmse=200.,p95=300.,p99=400.)
            revised=dict(mae=new,rmse=tail,p95=290.,p99=390.)
            return dict(position_weightings={'all':{name:{k:deepcopy(v) for k in ('point_equal','scene_target_equal','scene_equal_point_mean','scene_equal_target_mean')} for name,v in (('linear',old),('dropout',revised))}})
        summary=row();summary.update(components={'a':row(),'b':row()},remaining_components=row())
        self.assertTrue(compare.scientific_screen(summary,[row(),row()])['worth_matched_replication'])
        for change in ('small','tail','scenes','outside'):
            bad=deepcopy(summary);scenes=[row(),row()]
            if change=='small':bad['position_weightings']['all']['dropout']['point_equal']['mae']=99.
            if change=='tail':bad['position_weightings']['all']['dropout']['point_equal']['rmse']=201.
            if change=='scenes':scenes=[row(),row(101.)]
            if change=='outside':bad['remaining_components']=row(101.)
            self.assertFalse(compare.scientific_screen(bad,scenes)['worth_matched_replication'])

if __name__=='__main__':unittest.main()
