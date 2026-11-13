"""Complete139 software cache fixture; no fabricated scientific validation."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from association.utils import runtime
from tests.integration.test_observation_quality_pilot import fixture as quality_fixture
from tools.analysis import source_risk_pilot as previous
from tools.analysis import mixture_opportunity_pilot as mixture
from tools.analysis import observation_quality_pilot as quality
from tools.analysis import source_selection_evidence_pilot as pilot
from tools.analysis import observation_common as common


def fixture(root):
    quality_args = quality_fixture(root)
    source_args = SimpleNamespace(native_run=quality_args.native_run, value_run=quality_args.value_run,
        partition_run=quality_args.partition_run, manifest=quality_args.manifest,
        output=root/'real-fixture133', cpu_threads=1, time_policy='advisory')
    previous.run(source_args)
    quality_args.source_risk_run = source_args.output
    quality.run(quality_args)
    mixture_args = SimpleNamespace(native_run=quality_args.native_run, source_run=source_args.output,
        manifest=quality_args.manifest, output=root/'real-fixture134', cpu_threads=1, time_policy='advisory')
    mixture.run(mixture_args)
    return SimpleNamespace(native_run=quality_args.native_run, value_run=quality_args.value_run,
        partition_run=quality_args.partition_run, source_risk_run=source_args.output,
        quality_run=quality_args.output, mixture_run=mixture_args.output,
        manifest=quality_args.manifest, output=root/'new139', cpu_threads=1, time_policy='advisory')


class SourceSelectionEvidencePilotTests(unittest.TestCase):
    def test_full_pipeline_and_immutable_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary).resolve())
            summary = pilot.run(args)
            state = runtime.read_json(args.output/'run.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertEqual(state['analytic_head_fits'], 13)
            self.assertEqual(state['ridge_system_solves'], 13)
            self.assertEqual(state['completed_training_scene_count'], 32)
            self.assertEqual(state['completed_scene_count'], 32)
            self.assertEqual(state['newly_recovered_old_T_log_scene_count'], 32)
            self.assertEqual(state['newly_recovered_policy_scene_count'], 96)
            self.assertTrue(state['full_T_models_frozen_before_V'])
            self.assertEqual(len(summary['all_components']['methods']), 12)
            self.assertEqual(len(summary['all_components']['pairwise']), 66)
            training = runtime.read_json(args.output/'T-summary.json')
            self.assertEqual(len(training['all_components']['methods']), 4)
            self.assertEqual(len(training['all_components']['pairwise']), 6)
            self.assertEqual(set(summary['primary_bootstrap']), {pilot.V_PRIMARY})
            self.assertEqual(set(training['primary_bootstrap']), {pilot.T_PRIMARY})
            for split, sid, methods in (('T_core', '0', pilot.T_METHODS), ('V_select', '32', pilot.V_METHODS)):
                artifacts = common.load_artifacts(args.output/'private'/f'states-{split}-{sid}.npz', methods, template=methods[0])
                for name in methods:
                    np.testing.assert_array_equal(artifacts[name]['group_states'][..., 2:],
                                                  artifacts[methods[0]]['group_states'][..., 2:])
            # Wrong old labels must fail before any new fit and preserve both
            # the completed output and new failed diagnostic directory.
            path = args.source_risk_run/'private/training-0.npz'
            arrays = common.load_npz(path)
            index = tuple(np.argwhere(arrays['labeled'])[0])
            arrays['errors_m'][index] += 1.
            np.savez_compressed(path, **arrays)
            broken = SimpleNamespace(**vars(args)); broken.output = args.output.parent/'broken139'
            with self.assertRaisesRegex(ValueError, 'original133 label'):
                pilot.run(broken)
            failed = runtime.read_json(broken.output/'run.json')
            self.assertEqual(failed['status'], 'FAILED')
            self.assertEqual(failed['analytic_head_fits'], 0)
            self.assertEqual(runtime.read_json(args.output/'run.json')['status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
