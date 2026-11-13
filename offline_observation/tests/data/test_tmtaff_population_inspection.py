import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.fixtures.native_zip import NativeZipFixture
from tools.analysis.inspect_tmtaff_population import main, summarize_scene, summarize_population
from association.data.physical_units import SPEED_UNIT_POLICY


class PopulationInspectionTests(NativeZipFixture, unittest.TestCase):
    def inspection_config(self):
        return dict(scope='TMTAFF-unit-population-inspection-v1',train_splits=['T_core'],
            data={**self.data,'reference_longitude_policy':'shortest_arc','speed_unit_policy':SPEED_UNIT_POLICY},
            expected={'T_core':{'scene_count':1,'node_count':4,'native_point_count':12,
                'candidate_count':6,'positive_count':2,'negative_count':4},
                'V_select':{'scene_count':2,'node_count':4,'candidate_count':6,'target_count':4}},
            max_member_bytes=1024*1024,max_address_space_bytes=32*1024**3,pair_batch_size=64,
            exposure_step_options=[1000,4000])

    def args(self, output, config=None):
        for filename,value in [('manifest.json',self.manifest),('alignment.json',self.alignment),
                               ('inspection.json',config or self.inspection_config())]:
            (self.root/filename).write_text(json.dumps(value))
        return ['--archive',str(self.path),'--manifest',str(self.root/'manifest.json'),
            '--alignment',str(self.root/'alignment.json'),'--config',str(self.root/'inspection.json'),
            '--output',str(self.root/output),'--cpu-threads','1']

    def read(self, output, filename):
        path=self.root/output/filename
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
            return json.load(stream)

    def test_pair_supervision_uses_target_mask_and_preserves_zero_pairs(self):
        data=self.load(data_config=self.inspection_config()['data'])
        scene=data['train'][0]
        scene['fusion_targets']['target0a-0']['valid'][:]=False
        actual=summarize_scene(scene)
        self.assertEqual(actual['tcm_eligible_tracks'],4)
        self.assertEqual(actual['tcm_bidirectional_transitions'],16)
        self.assertEqual(actual['positive_pairs_without_any_supervision'],1)
        self.assertEqual(actual['positive_pair_valid_query_histograms']['lat'],[1,0,0,0,1])
        self.assertEqual(actual['positive_pair_valid_query_totals']['lat'],4)
        self.assertEqual(actual['target_valid_points']['lat'],8) # includes no-input target c
        summary=summarize_population([actual], [1], 64)['T_core']
        self.assertAlmostEqual(summary['exposure_expectations_not_training'][0]['expected_unique_pairs'],2.)

    def test_cli_only_prepares_allowed_splits_and_exports_counts(self):
        with (patch('association.models.tmtaff.AssociationModel',side_effect=AssertionError('no model')),
                patch('association.models.fusion.PairFusion',side_effect=AssertionError('no fusion')),
                patch('tools.analysis.inspect_tmtaff_population.address_space_limit',return_value=None)):
            self.assertEqual(main(self.args('inspect')),0)
        run=self.read('inspect','run.json')
        self.assertEqual(run['status'],'COMPLETED')
        self.assertEqual(run['read_member_count'],9)
        self.assertEqual(run['completed_observation_scenes'],3)
        self.assertEqual(run['completed_reference_scenes'],3)
        self.assertFalse(run['model_constructed'])
        self.assertFalse(run['normalizer_fitted'])
        for p in (self.root/'inspect').iterdir():
            if not p.is_file(): continue
            with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as stream:
                content=stream.read()
            self.assertNotIn('target0a-0',content)
            self.assertNotIn('9001',content)
        members=self.read('inspect','read-members.json.gz')
        self.assertEqual({x['scene_id'] for x in members},{'0','2','4'})
        with self.assertRaises(FileExistsError):
            main(self.args('inspect'))

    def test_failure_keeps_progress_and_private_error(self):
        with (patch('association.data.dataset.build_scene_fusion_targets',side_effect=ValueError('private-identity-example')),
                patch('tools.analysis.inspect_tmtaff_population.address_space_limit',return_value=None)):
            self.assertEqual(main(self.args('failure')),1)
        run=self.read('failure','run.json')
        self.assertEqual(run['status'],'FAILED')
        self.assertEqual(run['completed_observation_scenes'],3)
        self.assertEqual(run['completed_reference_scenes'],0)
        self.assertEqual(run['last_event'],dict(stage='reference',status='READING',split='T_core',scene_id='0'))
        self.assertNotIn('private-identity-example',json.dumps(run))
        self.assertIn('private-identity-example',(self.root/'failure/private/traceback.txt').read_text())
        self.assertEqual((self.root/'failure/private').stat().st_mode & 0o777,0o700)

    def test_mismatch_returns_all_counts_without_training_or_deletion(self):
        config=self.inspection_config();config['expected']['T_core']['positive_count']=3
        with patch('tools.analysis.inspect_tmtaff_population.address_space_limit',return_value=None):
            self.assertEqual(main(self.args('mismatch',config)),2)
        actual=self.read('mismatch','summary.json.gz')
        self.assertEqual(actual['mismatches'],[dict(split='T_core',field='positive_count',expected=3,actual=2)])
        self.assertEqual(actual['populations']['T_core']['candidate_count'],6)
        self.assertEqual(self.read('mismatch','run.json')['status'],'COMPLETED_WITH_MISMATCHES')

    def test_member_limit_rejects_before_payload_read(self):
        config=self.inspection_config();config['max_member_bytes']=1
        with (patch('zipfile.ZipFile.read',side_effect=AssertionError('payload opened')),
                patch('tools.analysis.inspect_tmtaff_population.address_space_limit',return_value=None)):
            self.assertEqual(main(self.args('oversize',config)),1)
        run=self.read('oversize','run.json')
        self.assertEqual(run['completed_observation_scenes'],0)
        self.assertEqual(run['error_type'],'ValueError')


if __name__=='__main__':
    unittest.main()
