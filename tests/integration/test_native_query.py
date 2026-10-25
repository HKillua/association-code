"""Tiny production-entry fit/final/cache integration, never a research score."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.data.source_events import source_events
from association.models.sequence_position import SequencePosition, sequence_inputs
from association.utils.runtime import read_json, write_json
from tests.fixtures.cache import support_gate_cache_fixture
from tests.fixtures.learning import native_query_training_fixture
from tools.analysis.frozen_fusion_transfer import groups_from_cache
from tools.experiments import native_query_fit as entry


class NativeQueryIntegrationTests(unittest.TestCase):
    def test_production_fit_and_final_cache_keep_frozen_full_population(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root)
            train, replay = native_query_training_fixture()
            with np.load(source/'predictions/states-3035.npz') as arrays:
                fragment = {key: arrays[key].copy() for key in arrays.files}
            raw = validation[0]
            raw.update(node_ids=fragment['node_ids'].copy(), sources=fragment['sources'].copy(),
                grid=raw['motion']['query_seconds'].copy(), motion_observation_policy='canonical_all')
            raw['motion']['sources'] = raw['sources'].copy()
            # This historical population fixture has no original CSV. Supply
            # two explicit fixture events per node for exercising the new API;
            # only its motion/cache equality reconstruction is substituted below.
            raw['tracks'] = [dict(node_id=str(node_id), source=str(raw['sources'][node]),
                canonical_times=raw['grid'][[0, -1]].copy(),
                canonical_values=raw['motion']['states'][node, [0, -1]].copy())
                for node, node_id in enumerate(raw['node_ids'])]
            raw['motion']['native_events'] = source_events(raw)
            manifest = dict(splits={'T_core': [str(s['scene_id']) for s in train], 'V_select': ['3035']},
                            components={'3035': 'shared'})
            base_config = dict(backbone='transformer', hidden_size=64, num_layers=2, initial_gate=.95)
            torch.manual_seed(17)
            base = SequencePosition(**base_config).eval()
            for name, value in (('manifest', manifest), ('alignment', {})):
                write_json(root/(name+'.json'), value)
            args = Namespace(source_system=str(source), strong_run=str(root/'strong'),
                output='', archive=['not-opened.zip'], config=str(root/'config.json'),
                manifest=str(root/'manifest.json'), alignment=str(root/'alignment.json'),
                device='cpu', cpu_threads=4, time_policy='advisory')
            for arm in ('query', 'budget'):
                with self.subTest(arm=arm):
                    config = read_json(entry.ROOT/'configs'/f'native-query-{arm}-level1.json')
                    config['training'].update(updates=3, batch_size=2)
                    write_json(root/'config.json', config)
                    output = root/arm
                    args.output = str(output)
                    def dataset(*a, **kwargs):
                        self.assertEqual(kwargs['retain_native_events'], arm != 'budget')
                        self.assertEqual(kwargs['motion_observation_policy'], 'canonical_all')
                        return dict(train=deepcopy(train), validation=deepcopy(validation), report={'fixture': True})
                    with patch.object(entry, 'production_contract'), patch.object(entry, '_bind_source'), \
                            patch.object(entry, 'bind_base', return_value=(base, base_config, replay)), \
                            patch.object(entry, '_load_metadata', return_value=deepcopy(metadata)), \
                            patch.object(entry, 'load_timed_dataset', side_effect=dataset), \
                            patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda scene, *a: scene):
                        report = entry.run(args)
                    self.assertEqual(report['status'], 'COMPLETED')
                    self.assertEqual(report['optimizer_updates'], 3)
                    self.assertEqual(report['actual_successful_batches_matched'], 3)
                    self.assertEqual(report['completed_endpoints'], ['final'])
                    self.assertFalse(report['validation_selection'])
                    self.assertFalse(report['test_opened'])
                    self.assertTrue(report['fixed_base_verified'])
                    self.assertFalse((output/'initial').exists())
                    final = read_json(output/'final/summary.json')
                    self.assertEqual(final['status'], 'COMPLETED')
                    self.assertTrue(final['ordinary_source_replay_exact'])
                    self.assertTrue(final['fixed_population_verified'])
                    self.assertEqual(final['checkpoint']['actual_updates'], 3)
                    learned = final['methods']['learned_sequence']
                    self.assertEqual(learned['population'], metadata['expected']['3035']['learned_assoc_anchor']['population'])
                    self.assertEqual(learned['population']['target_identity_count'], 5)
                    self.assertEqual(learned['population']['duplicate_prediction_group_count'], 1)
                    self.assertEqual(learned['population']['reference_unavailable_target_count'], 1)
                    position = learned['full_target_metrics']['all']['position']
                    self.assertEqual(position['intersection_count'], 127)
                    self.assertEqual(position['missing_prediction_count'], 128)
                    with np.load(output/'final/system-predictions/states-3035.npz') as actual, \
                            np.load(source/'system-predictions/states-3035.npz') as old:
                        for method in ('anchor', 'mean_supported', 'attribute_hybrid'):
                            prefix = 'learned_assoc_'+method+'_'
                            for key in old.files:
                                if key.startswith(prefix):
                                    np.testing.assert_array_equal(actual[key.removeprefix('learned_assoc_')], old[key])
                    with np.load(output/'final/predictions/states-3035.npz') as saved:
                        with torch.no_grad():
                            expected_base = np.asarray([base(*sequence_inputs(raw['motion'], group))['states'].numpy()
                                for group in groups_from_cache(fragment, 'learned_assoc')])
                        np.testing.assert_array_equal(saved['group_frozen_base_states'], expected_base)
                        self.assertEqual(saved['group_native_increment_m'].shape, (4, 64, 2))
                    self.assertFalse(read_json(output/'execution.json')['test_opened'])


if __name__ == '__main__':
    unittest.main()
