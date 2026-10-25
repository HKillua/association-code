"""096 real source/phase composition, fixed score reuse and observed controls."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.artifacts import load_scene_prediction
from association.evaluation.tmtaff import predicted_observed_controls
from association.utils import runtime
from tests.integration import test_completion_value_fit as fixtures
from tools.analysis import compare_completion_policies as pairing
from tools.analysis.compare_physical_predictions import compare
from tools.experiments import completion_system_evaluate as cli


class CompletionSystemEvaluationTests(unittest.TestCase):
    def invoke(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(argv)
        return code, stdout.getvalue() + stderr.getvalue()

    def test_waits_for_both_wrappers_without_reading_active_weights_or_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for phase in ('tam', 'tfm'):
                (root / phase).mkdir()
                runtime.write_json(root / phase / 'run.json', dict(status='RUNNING', phase=phase))
            args = ['--tam-run', str(root/'tam'), '--tfm-run', str(root/'tfm'),
                '--completion-policy', 'learned', '--manifest', str(root/'manifest.json'),
                '--alignment', str(root/'alignment.json'), '--check-source-only', '--output', str(root/'new')]
            with patch.object(torch, 'load', side_effect=AssertionError('active weights read')):
                code, text = self.invoke(args)
            self.assertEqual(code, 75, text)
            self.assertFalse((root/'new').exists())
            # A terminal trainer is not sufficient while its wrapper verifies.
            runtime.write_json(root/'tam/run.json', dict(status='COMPLETED', phase='tam',
                                                        final_evaluation_status='COMPLETED'))
            with patch.object(torch, 'load', side_effect=AssertionError('unverified weights read')):
                code, text = self.invoke(args)
            self.assertEqual(code, 75, text)
            runtime.write_json(root/'tam/completion-value-verification.json', dict(status='FAILED'))
            code, text = self.invoke(args)
            self.assertEqual(code, 1, text)
            self.assertFalse((root/'new').exists())

    def test_predicted_controls_keep_groups_masks_and_circular_angles(self):
        values = np.array([[[10., 179., 2., 359.], [20., 10., 4., 20.]],
                           [[12., -179., 6., 1.], [99., 99., 99., 99.]],
                           [[30., 40., 5., 60.], [35., 42., 6., 62.]]])
        observed = np.array([[True, True], [True, False], [False, False]])
        controls = predicted_observed_controls(values, observed, [[0, 1], [2]])
        mean = controls['predicted_mean'][0]
        np.testing.assert_allclose(mean['states'][0, :3], [11., -180., 4.], atol=1e-12)
        self.assertLess(abs((mean['states'][0, 3]+180) % 360-180), 1e-12)
        np.testing.assert_array_equal(mean['states'][1], values[0, 1])
        for rows in controls.values():
            self.assertEqual([row['node_indices'] for row in rows], [[0, 1], [2]])
            self.assertFalse(rows[1]['prediction_valid'].any())
        np.testing.assert_array_equal(controls['predicted_anchor'][0]['states'], values[0])
        opposing = np.array([[[1., 0., 1., 90.]], [[2., 180., 2., 270.]]])
        ambiguous = predicted_observed_controls(opposing, np.ones((2, 1), bool), [[0, 1]])
        np.testing.assert_array_equal(ambiguous['predicted_mean'][0]['prediction_valid'],
                                      [[False, False, True, False]])

    def test_real_two_systems_use_trained_modules_and_frozen_scores(self):
        helper = fixtures.CompletionValueFitTests(
            methodName='test_active_source_does_not_load_weights_claim_or_create_output')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.prepare()
        self.enterContext(patch.object(pairing, '_expected_configuration',
            side_effect=lambda phase, policy: helper.targets[phase, policy]))
        for phase in ('tam', 'tfm'):
            for policy in cli.arm_checks.POLICIES:
                code, text = helper.invoke(helper.args(phase, policy))
                self.assertEqual(code, 0, text)
        root = helper.root

        def args(policy, output=None):
            return ['--tam-run', str(root/f'tam-{policy}'), '--tfm-run', str(root/f'tfm-{policy}'),
                '--completion-policy', policy, '--manifest', str(root/'manifest.json'),
                '--alignment', str(root/'alignment.json'), '--archive', str(helper.fixture.path),
                '--output', str(root/(output or f'system-{policy}')), '--device', 'cpu',
                '--cpu-threads', '1', '--max-seconds', '.000001', '--time-policy', 'advisory']

        original = {path: path.read_bytes() for path in root.glob('*/checkpoint.pt')}
        for policy in cli.arm_checks.POLICIES:
            with patch('association.training.tmtaff.load_complete_snapshot',
                       side_effect=AssertionError('source-only constructed inference models')):
                code, text = self.invoke([*args(policy, f'check-{policy}'), '--check-source-only'])
            self.assertEqual(code, 0, text)
            self.assertFalse((root/f'check-{policy}').exists())
            with (patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('evaluation trained')),
                  patch('association.models.timed.TimedAssociationModel.score_pairs',
                        side_effect=AssertionError('TAM scores recomputed'))):
                code, text = self.invoke(args(policy))
            self.assertEqual(code, 0, text)
            output = root/f'system-{policy}'
            run, summary = (runtime.read_json(output/name) for name in ('run.json', 'summary.json'))
            self.assertEqual(run['status'], 'COMPLETED')
            self.assertEqual(run['optimizer_updates'], 0)
            self.assertTrue(run['advisory_reached'])
            self.assertTrue((output/'time_advisory.json').is_file())
            self.assertEqual(run['source_binding'], summary['source_binding'])
            tam = runtime.read_json(root/f'tam-{policy}/summary.json')
            self.assertEqual(summary['evaluation']['threshold'], tam['evaluation']['threshold'])
            pairing._numbers(summary['evaluation']['association'], tam['evaluation']['association'], 'scores retained')
            index = runtime.read_json(output/'predictions/index.json')
            self.assertEqual([row['scene_id'] for row in index['scenes']], ['2', '4'])
            for entry in index['scenes']:
                artifact = load_scene_prediction(output/'predictions'/entry['metadata'])
                methods = artifact['methods']
                for name in ('predicted_anchor', 'predicted_mean'):
                    np.testing.assert_array_equal(methods[name]['node_group_indices'],
                                                  methods['predicted']['node_group_indices'])
                    np.testing.assert_array_equal(methods[name]['aligned_truth_valid'],
                                                  methods['predicted']['aligned_truth_valid'])
                if entry['scene_id'] == '4':
                    self.assertFalse(methods['predicted']['aligned_prediction_valid'].any())
                    self.assertEqual(len(methods['predicted']['target_ids']), 1)
            # Passing original endpoint eligibility is not enough to accept a
            # different requested policy or a changed common frozen tensor.
            wrong = args(policy, f'wrong-{policy}')
            wrong[wrong.index('--completion-policy')+1] = 'state_hold' if policy == 'learned' else 'learned'
            code, text = self.invoke(wrong)
            self.assertEqual(code, 1, text)
            self.assertFalse((root/f'wrong-{policy}').exists())

        comparison = compare(root/'system-learned/predictions', root/'system-state_hold/predictions',
                             root/'system-paired')
        self.assertEqual(comparison['scene_count'], 2)
        self.assertFalse(comparison['test_opened'])
        from tools.analysis import compare_completion_systems as systems
        with patch.object(systems, '_expected_source_updates', return_value={'tam': 2, 'tfm': 2}):
            comparison = systems.compare(root/'system-learned', root/'system-state_hold',
                                         root/'system096-paired')
        self.assertEqual(comparison['scene_count'], 2)
        self.assertTrue(comparison['ordinary_controls_same_predicted_groups'])
        self.assertTrue(comparison['validation_arrays_equal'])
        tcm_output = args('learned', 'source/forbidden-new-run')
        code, text = self.invoke(tcm_output)
        self.assertEqual(code, 1, text)
        self.assertIn('separate from original TCM source', text)
        self.assertFalse((root/'source/forbidden-new-run').exists())
        for path, contents in original.items():
            self.assertEqual(path.read_bytes(), contents, 'source checkpoint changed')
        # Semantic scene contracts alone do not detect altered native floats.
        from association.data import dataset
        original_loader = dataset.load_timed_dataset

        def changed_archive_view(*argv, **kwargs):
            prepared = original_loader(*argv, **kwargs)
            scene = prepared['validation'][0]
            scene['native_values'][0, 0, 0] += .001
            return prepared

        with patch.object(dataset, 'load_timed_dataset', side_effect=changed_archive_view):
            code, text = self.invoke(args('learned', 'changed-input'))
        self.assertEqual(code, 1, text)
        self.assertIn('prepared validation input differs', text)
        self.assertEqual(runtime.read_json(root/'changed-input/run.json')['status'], 'FAILED')
        path = root/'tfm-learned/checkpoint.pt'
        damaged = torch.load(path, map_location='cpu', weights_only=True)
        name = next(key for key in damaged['states'][0] if key.startswith('tcm_encoder.'))
        damaged['states'][0][name] += 1
        torch.save(damaged, path)
        code, text = self.invoke(args('learned', 'changed-frozen'))
        self.assertEqual(code, 1, text)
        self.assertFalse((root/'changed-frozen').exists())
        path.write_bytes(original[path])


if __name__ == '__main__':
    unittest.main()
