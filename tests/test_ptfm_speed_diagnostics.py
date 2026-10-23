"""076A full-population replay and constant-vs-dynamics diagnostics."""
from argparse import Namespace
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from association.evaluation.ptfm_speed_diagnostics import dynamics, pearson, summarize, load_scene
from association.utils.runtime import read_json, write_json
from tests.test_ptfm import config, sequence_reference
from tests.training.test_support_gate import tiny_train
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tools.experiments import ptfm_fit
from tools.analysis import ptfm_speed_diagnostics as entry


def completed_source(root, sparse=False):
    source,metadata,validation=support_gate_cache_fixture(root,sparse=sparse)
    training=tiny_train();cfg=config()
    manifest=dict(splits={'T_core':[str(s['scene_id']) for s in training],'V_select':['3035']},components={'3035':'shared'})
    paired=sequence_reference(root/'control',source,metadata,validation)
    for name,obj in [('config',cfg),('manifest',manifest),('alignment',{})]:write_json(root/(name+'.json'),obj)
    args=Namespace(source_system=str(source),output=str(root/'source074c'),archive=['not-opened.zip'],
        paired_control=str(paired),manifest=str(root/'manifest.json'),alignment=str(root/'alignment.json'),
        config=str(root/'config.json'),device='cpu',cpu_threads=4,time_policy='advisory')
    with patch.object(ptfm_fit,'production_contract'),patch.object(ptfm_fit,'_bind_source'),patch.object(ptfm_fit,'bind_paired_control'),\
            patch.object(ptfm_fit,'_load_metadata',return_value=metadata),\
            patch.object(ptfm_fit,'load_timed_dataset',return_value=dict(train=training,validation=validation,report={'fixture':True})),\
            patch('association.evaluation.ptfm.frozen_motion',side_effect=lambda raw,*_:raw):
        ptfm_fit.run(args)
    frozen=root/'frozen';(frozen/'configs/data').mkdir(parents=True);write_json(frozen/'configs/ptfm-adapted-level1.json',cfg)
    write_json(frozen/'configs/data/mtad-level1-scene3000.json',manifest)
    return root/'source074c',frozen


class PTFMSpeedDiagnosticsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_constant_correlation_and_actual_seconds_without_gap_bridging(self):
        self.assertIsNone(pearson(np.full(11,1.4),np.arange(11)))
        self.assertIsNone(pearson([1],[2]))
        self.assertAlmostEqual(pearson([1,2,3],[3,5,7]),1.)
        prediction=np.array([[0.,4.,999.,10.,15.],[9.,9.,9.,9.,9.]])
        truth=np.array([[0.,2.,np.nan,7.,11.],[5.,5.,5.,5.,5.]])
        valid=np.isfinite(truth)
        result=dynamics(prediction,truth,valid,np.array([0.,2.,5.,10.,20.]))
        self.assertEqual(result['adjacent_increment_error']['point_count'],6)
        self.assertAlmostEqual(result['adjacent_increment_error']['mae'],.5)
        self.assertAlmostEqual(result['adjacent_rate_error_per_second']['mae'],1.1/6)
        self.assertEqual(result['target_pearson']['undefined_target_count'],1)
        self.assertEqual(result['prediction_within_target_std']['min'],0.)
        with self.assertRaisesRegex(ValueError,'seconds'):dynamics(prediction,truth,valid,[0,2,2,10,20])

    def test_fixed_bins_are_exhaustive_and_no_truth_is_not_zero_truth(self):
        truth=np.array([[0.,1.,5.,20.,21.,np.nan]])
        valid=np.isfinite(truth);pred=np.array([[1.,1.,1.,1.,1.,100.]])
        record=dict(scene_id='x',population=dict(predicted_group_count=2),truth=truth,valid=valid,
            truth_valid=valid,helper=np.zeros_like(valid),represented=np.ones_like(valid),query_seconds=np.arange(6)[None,:],
            methods={m:pred.copy() for m in ('anchor','mean_supported','ptfm_adapted','zero_speed')},
            group_speed={m:np.array([[1.,2.,3.,4.,5.,6.],[99.,99.,99.,99.,99.,99.]]) for m in ('anchor','mean_supported','ptfm_adapted','zero_speed')},
            native_speed=np.ones((3,6)),group_helper=np.zeros((2,6),bool))
        record['methods']['zero_speed'][:]=0
        result=summarize([record])
        self.assertEqual([b['point_count'] for b in result['fixed_truth_speed_bins'].values()],[1]*5)
        self.assertEqual(result['strata']['all']['methods']['zero_speed']['point_count'],5)
        self.assertEqual(result['strata']['supported_helper_present']['point_count'],0)
        self.assertIsNone(result['strata']['supported_helper_present']['methods']['ptfm_adapted']['pearson'])
        self.assertEqual(result['all_output_group_distributions']['ptfm_adapted']['count'],12)

    def test_real_completed_source_cache_only_cli_and_duplicate_population(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source,frozen=completed_source(root)
            watched={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
            args=Namespace(source_074c=str(source),output=str(root/'diagnostics'),cpu_threads=4)
            with patch.object(entry,'ROOT',frozen),patch('torch.load',side_effect=AssertionError('no checkpoint')),\
                    patch('association.models.ptfm.PTFM.forward',side_effect=AssertionError('no forward')):
                result=entry.run(args)
            self.assertTrue(result['source_population_and_mapping_verified'])
            self.assertTrue(result['zero_control_all_groups_verified'])
            self.assertEqual(result['pooled']['population']['duplicate_prediction_group_count'],1)
            self.assertEqual(result['pooled']['all_output_group_distributions']['zero_speed']['count'],256)
            self.assertEqual(result['pooled']['strata']['all']['methods']['zero_speed']['prediction_distribution']['max'],0.)
            self.assertGreater(result['pooled']['population']['reference_unavailable_target_count'],0)
            self.assertEqual(result['pooled']['common_speed_point_count'],read_json(source/'final/summary.json')['methods']['anchor']['full_target_metrics']['all']['speed']['intersection_count'])
            self.assertEqual(read_json(root/'diagnostics/run.json')['status'],'COMPLETED')
            self.assertIn('零速', (root/'diagnostics/report.md').read_text())
            for p,value in watched.items():self.assertEqual(p.read_bytes(),value)
            with self.assertRaises(FileExistsError):entry.run(args)
            args.output=str(source/'bad')
            with self.assertRaisesRegex(ValueError,'preserve'):entry.run(args)

    def test_sparse_truth_and_mapping_corruption_rejected_without_forward(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source,frozen=completed_source(root,sparse=True)
            with patch.object(entry,'ROOT',frozen):
                result=entry.run(Namespace(source_074c=str(source),output=str(root/'ok'),cpu_threads=4))
            self.assertEqual(result['pooled']['common_speed_point_count'],1)
            dynamic=result['pooled']['strata']['all']['methods']['ptfm_adapted']['dynamics']
            self.assertEqual(dynamic['targets_with_at_least_two_points'],0)
            self.assertEqual(dynamic['adjacent_increment_error']['point_count'],0)
            path=source/'final/system-predictions/states-3035.npz'
            with np.load(path) as d:payload={k:d[k].copy() for k in d.files}
            payload['ptfm_adapted_aligned_group_indices'][0]=1
            np.savez_compressed(path,**payload)
            with patch.object(entry,'ROOT',frozen),self.assertRaisesRegex(ValueError,'cached'):
                entry.run(Namespace(source_074c=str(source),output=str(root/'bad'),cpu_threads=4))
            self.assertEqual(read_json(root/'bad/run.json')['status'],'FAILED')

    def test_source_metadata_and_original056_binding_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source,frozen=completed_source(root)
            path=source/'summary.json';original=read_json(path)
            changed=deepcopy(original);changed['optimizer_updates']-=1;write_json(path,changed)
            with patch.object(entry,'ROOT',frozen),self.assertRaisesRegex(ValueError,'update'):
                entry.bind_source(source)
            write_json(path,original)
            final=read_json(source/'final/summary.json')
            path=Path(final['source_system'])/'predictions/states-3035.npz'
            with np.load(path) as d:payload={k:d[k].copy() for k in d.files}
            payload['native_states'][0,0,2]+=1;np.savez_compressed(path,**payload)
            with patch.object(entry,'ROOT',frozen),self.assertRaisesRegex(ValueError,'frozen056'):
                entry.run(Namespace(source_074c=str(source),output=str(root/'bad'),cpu_threads=4))
            self.assertEqual(read_json(root/'bad/run.json')['status'],'FAILED')

if __name__=='__main__':unittest.main()
