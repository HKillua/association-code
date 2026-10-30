"""Small synthetic cache chain only; does not report MTAD performance."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from association.utils import runtime
from tests.integration.test_source_risk_pilot import cached_input
from tools.analysis import source_contribution_readout_pilot as pilot
from tools.analysis import source_risk_pilot as source


def fixture(root):
    """Use the genuine133 writer on tiny software-only source records."""
    original = cached_input(root)
    source.run(original)
    return SimpleNamespace(native_run=original.native_run, value_run=original.value_run,
        partition_run=original.partition_run, source_risk_run=original.output,
        manifest=original.manifest, output=root/'fixture140', cpu_threads=1, time_policy='advisory')


class SourceContributionReadoutPilotTests(unittest.TestCase):
    def test_fixed_endpoint_fold_isolation_and_input_failure_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary).resolve())
            original_fit = pilot.core.fit_risk_model
            seen = []
            def tracked_fit(x, y, weight):
                seen.append((x.copy(), y.copy(), weight.copy()))
                return original_fit(x, y, weight)
            with patch.object(pilot.core, 'fit_risk_model', side_effect=tracked_fit):
                result = pilot.run(args)
            state = runtime.read_json(args.output/'run.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertEqual(state['analytic_head_fits'], 13)
            self.assertEqual(len(seen), 13)
            self.assertEqual(state['completed_training_scene_count'], 32)
            self.assertEqual(state['completed_scene_count'], 32)
            self.assertTrue(state['full_T_models_frozen_before_V'])
            for value, methods, pairs in ((result, 13, 78), (result['training_summary'], 6, 15)):
                self.assertEqual(len(value['all_components']['methods']), methods)
                self.assertEqual(len(value['all_components']['pairwise']), pairs)
                self.assertEqual(set(value['primary_bootstrap']), set(pilot.PRIMARY))
            # Each input to the sole critic is independently reconstructed from
            # the saved, legal training-component list, including normalizer data.
            folds = runtime.read_json(args.output/'T-fold-index.json')
            models = pilot.common.load_npz(args.output/'private/models.npz')
            for entry, observed in zip(folds, seen):
                blocks = [pilot.common.load_npz(args.output/'private'/f'training-{sid}.npz')
                          for sid in entry['training_scene_ids']]
                self.assertNotIn(entry['heldout_component'], entry['training_components'])
                arrays = tuple(np.concatenate([block[key][block['labeled']] for block in blocks])
                               for key in ('features', 'y_chord_km2', 'sample_weight'))
                for actual, expected in zip(observed, arrays):
                    np.testing.assert_array_equal(actual, expected)
                independently_fitted = original_fit(*arrays)
                for key in ('mean', 'std', 'beta'):
                    np.testing.assert_array_equal(models[entry['fold']+'__'+key], independently_fitted[key])
            for split, sid in (('T_core', '0'), ('V_select', '32')):
                policy = pilot.common.load_npz(args.output/'private'/f'policy-{split}-{sid}.npz')
                for name in pilot.NEW:
                    self.assertEqual(policy[name+'__source_weights'].shape, policy['available'].shape)
                    np.testing.assert_array_equal(policy[name+'__node_weights'][:, 0], 0.)
                artifact = pilot.common.load_artifacts(args.output/'private'/f'states-{split}-{sid}.npz',
                    pilot.T_METHODS if split == 'T_core' else pilot.V_METHODS,
                    template='WS_SOURCE_EQUAL')
                for name in pilot.NEW:
                    np.testing.assert_array_equal(artifact[name]['group_states'][..., 2:],
                                                  artifact['WS_SOURCE_EQUAL']['group_states'][..., 2:])
            # A new attempt rejects altered immutable input and records failure;
            # completed outputs are never overwritten or removed.
            original_path = args.source_risk_run/'private/training-0.npz'
            broken_input = pilot.common.load_npz(original_path)
            broken_input['features'][0, 1, 0] += 1.
            np.savez_compressed(original_path, **broken_input)
            broken = SimpleNamespace(**vars(args)); broken.output = Path(temporary)/'broken140'
            with self.assertRaisesRegex(ValueError, 'binding'):
                pilot.run(broken)
            self.assertEqual(runtime.read_json(broken.output/'run.json')['status'], 'FAILED')
            self.assertEqual(runtime.read_json(args.output/'run.json')['status'], 'COMPLETED')
            self.assertFalse((broken.output/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
