"""Real native ZIP -> frozen TCM -> TFM probe, with no reference/test reads."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import io
import json
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from association.models.fusion import PairFusion, PairFusionConfig, FusionLoss
from association.models.timed import TimedAssociationModel
from association.models.tmtaff import ModelConfig
from association.training.common import fit_normalizer
from association.training.tmtaff import _configuration, _scenes, _states
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.analysis import tfm_information_probe as probe


class TFMInformationProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(53)
        self.fixture = make_native_zip_fixture(self)
        self.root = self.fixture.root
        self.fixture.data['length'] = 64
        dataset = self.fixture.load()
        model_config = ModelConfig(length=64, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0)
        fusion_config = PairFusionConfig(fc_hidden=8,length=64,d_model=8,nhead=2,num_layers=1,dim_feedforward=16,dropout=0)
        model = TimedAssociationModel(model_config, 'ce', max_observations=4)
        self.fusion = PairFusion(fusion_config).eval()
        states = _states((model, self.fusion, FusionLoss(position_loss_mode='standard_smooth_l1')))
        config = _configuration(self.fixture.config['training'])
        config['updates'] = dict(TCM=0,TAM_FROZEN=0,TFM_FROZEN=300,JOINT=0)
        self.saved = dict(format_version=1,phase='tfm',input_adapter='native_time_v1',
            max_observations=4,model_config=asdict(model_config),fusion_config=asdict(fusion_config),
            normalizer=fit_normalizer(dataset['train']), config=config, selected=None,
            data_contract=_scenes(dataset['train'],dataset['validation'],64),states=states)
        self.paths=[]
        # Synthetic snapshots exercise metadata/loader contracts; no claimed training.
        for step in (0,100,300):
            path=self.root/f'update-{step:06d}.pt'
            saved={**self.saved,'cursor':dict(stage=2,update=step,completed_updates=step),
                   'history':[dict(stage='TFM_FROZEN',update=i) for i in range(1,step+1)]}
            torch.save(saved,path); self.paths.append(path)
        self.tcm=self.root/'033.pt'
        torch.save({**self.saved,'phase':'tcm','cursor':dict(stage=0,update=4000,completed_updates=4000)},self.tcm)
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps(self.fixture.manifest))

    def args(self,name='probe'):
        args=['--archive',str(self.fixture.path),'--manifest',str(self.manifest),
              '--scene-ids','2','--tcm-checkpoint',str(self.tcm),
              '--output',str(self.root/name),'--cpu-threads','1']
        for path in self.paths: args.extend(['--checkpoint',str(path)])
        return args

    def test_native_export_and_real_models_preserve_sealed_boundaries_and_responses(self):
        original_open=zipfile.ZipFile.open
        read=[]
        def guarded_open(archive,name,*args,**kwargs):
            member=name.filename if isinstance(name,zipfile.ZipInfo) else str(name)
            if '真实场景' in member or member.endswith('-3.csv'):
                raise AssertionError('reference/test member opened by frozen probe')
            if member.endswith('.csv'): read.append(member)
            return original_open(archive,name,*args,**kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(zipfile.ZipFile,'open',guarded_open))
            for target in ('torch.optim.Adam','association.training.common.fit_normalizer',
                           'association.data.timed.fit_timed_normalizer'):
                stack.enter_context(patch(target,side_effect=AssertionError('probe must not fit/train')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(probe.main(self.args()),0)
        report=json.loads((self.root/'probe/report.json').read_text())
        self.assertEqual(report['status'],'COMPLETED')
        self.assertFalse(report['test_opened']); self.assertFalse(report['reference_read'])
        self.assertEqual(report['optimizer_updates'],0)
        self.assertEqual(report['source_tcm']['cursor']['update'],4000)
        self.assertEqual([r['cursor']['update'] for r in report['checkpoints']],[0,100,300])
        self.assertEqual(report['input_provenance']['scene_ids'],['2'])
        self.assertEqual(len(report['input_provenance']['pairs']),2)
        self.assertTrue(read); self.assertTrue(all(x.endswith('-2.csv') for x in read))
        with np.load(self.root/'probe/pairs.npz') as pairs:
            self.assertEqual(set(pairs.files),{'left','right'})
            self.assertEqual(pairs['left'].shape,(2,64,8))
        responses=report['checkpoints'][0]['controlled_responses']
        self.assertEqual(responses['fixed_left_replace_right']['input_left_rms'],0.)
        self.assertGreater(responses['fixed_left_replace_right']['input_right_rms'],0.)
        self.assertEqual(responses['fixed_right_replace_left']['input_right_rms'],0.)
        self.assertGreater(responses['fixed_right_replace_left']['input_left_rms'],0.)
        self.assertEqual(responses['fixed_left_replace_right']['stages']['left_encoder']['rms'],0.)
        self.assertGreater(responses['fixed_left_replace_right']['stages']['right_encoder']['rms'],0.)

    def test_near_identical_external_latents_are_reported_without_provenance_claim(self):
        pairs=self.root/'constant.npz'
        np.savez(pairs,left=np.zeros((2,64,8),np.float32),right=np.zeros((2,64,8),np.float32))
        args=['--pairs-npz',str(pairs),'--output',str(self.root/'constant'),'--cpu-threads','1']
        for path in self.paths: args.extend(['--checkpoint',str(path)])
        with redirect_stdout(io.StringIO()): self.assertEqual(probe.main(args),0)
        report=json.loads((self.root/'constant/report.json').read_text())
        self.assertTrue(report['first_two_inputs_nearly_identical'])
        self.assertIsNone(report['test_opened'])
        for row in report['checkpoints']:
            self.assertEqual(row['stages']['output_physical_by_attribute']['position_metres']['pair_0_1_rms'],0.)
            self.assertEqual(row['controlled_responses']['fixed_left_replace_right']['stages']['physical']['position_local_metres']['rms'],0.)

    def test_sealed_scene_and_changed_completion_rejected(self):
        args=self.args('sealed'); args[args.index('--scene-ids')+1]='3'
        with self.assertRaisesRegex(ValueError,'T_test is sealed'):
            probe.main(args)
        saved=torch.load(self.paths[1],weights_only=True)
        key=next(iter(saved['states'][0])); saved['states'][0][key].add_(1)
        torch.save(saved,self.paths[1])
        with self.assertRaisesRegex(ValueError,'TCM weights differ'):
            probe.main(self.args('changed'))

    def test_legacy_033_omitted_defaults_loads_but_changed_semantics_fails(self):
        old = torch.load(self.tcm, weights_only=True)
        for key in ('heading_representation', 'tam_dropout', 'contrastive_negative_scale'):
            old['model_config'].pop(key)
        torch.save(old, self.tcm)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(self.args('legacy033')), 0)
        report = json.loads((self.root/'legacy033/report.json').read_text())
        self.assertTrue(report['source_tcm']['frozen_weights_identical'])
        old['model_config']['heading_representation'] = 'circular_sincos'
        torch.save(old, self.tcm)
        with self.assertRaisesRegex(ValueError, 'model_config'):
            probe.main(self.args('wrong_heading'))

    def test_hooks_preserve_exact_output_and_are_removed(self):
        left,right=torch.randn(2,64,8),torch.randn(2,64,8)
        with torch.inference_mode(): before=self.fusion(left,right)['states'].clone()
        result=probe._collect(self.fusion,left,right)
        with torch.inference_mode(): after=self.fusion(left,right)['states']
        self.assertTrue(torch.equal(before,after))
        self.assertEqual(result['decoder_generated_latents']['shape'],[2,64,8])
        self.assertFalse(any(m._forward_hooks for m in self.fusion.modules()))


if __name__=='__main__':
    unittest.main()
