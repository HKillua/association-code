"""Configuration inspection must not accept settings a fit would reject."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from association.complete_training import STAGES
from association.fusion import PairFusionConfig
from association.models import ModelConfig
from tools.experiments import tmtaff_complete_fit as cli


class CompleteCliTests(unittest.TestCase):
    def test_inspection_rejects_invalid_training_before_archive_access(self):
        config = dict(scope='TMTAFF-time-adapted-v1',
            model=asdict(ModelConfig()), fusion=asdict(PairFusionConfig(fc_hidden=128)),
            data=dict(max_observations=128, time_unit_seconds=20), train_splits=['T_core'],
            training=dict(updates=dict.fromkeys(STAGES, 2),
                learning_rates=dict.fromkeys(STAGES, .001), seed=20, branch='contrastive',
                batch_tracks=64, pair_batch_size=64, pair_microbatch=1, track_microbatch=1,
                completion_weight=.1, gradient_clip=1., max_hours=1.,
                validate_every_updates=2, completion_cache_mib=1024,
                position_loss_mode='standard_smooth_l1', lr_schedule='constant',
                singleton_policy='observed', validation_batches=dict(tracks=1, pairs=1024)))
        cases = []
        invalid = copy.deepcopy(config)
        del invalid['training']['position_loss_mode']
        cases.append(invalid)
        invalid = copy.deepcopy(config)
        invalid['training']['updates']['TCM'] = 0
        cases.append(invalid)
        invalid = copy.deepcopy(config)
        invalid['training']['position_loss_mode'] = 'unknown'
        cases.append(invalid)
        invalid = copy.deepcopy(config)
        invalid['training']['margin_policy'] = 'train_negative_median_v1'
        invalid['model']['contrastive_positive_power'] = 2
        cases.append(invalid)
        invalid = copy.deepcopy(config)
        invalid['training']['margin_policy'] = 'train_negative_median_v1'
        invalid['train_splits'] = ['T_core', 'T_head']
        cases.append(invalid)
        with tempfile.TemporaryDirectory() as directory:
            for invalid in cases:
                with self.subTest(training=invalid['training']), \
                        patch.object(cli.runtime, 'read_json', return_value=invalid) as read, \
                        patch.object(cli, 'load_timed_dataset') as load, \
                        patch('builtins.print'):
                    result = cli.main(['--archive', 'unopened.zip', '--manifest', 'unopened.json',
                        '--alignment', 'unopened-alignment.json', '--config', 'config.json',
                        '--output', str(Path(directory) / 'inspection'), '--inspect-only'])
                    self.assertEqual(result, 1)
                    read.assert_called_once_with('config.json')
                    load.assert_not_called()
                    self.assertFalse((Path(directory) / 'inspection').exists())


if __name__ == '__main__':
    unittest.main()
