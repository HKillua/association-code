"""Real 137 kernel/runner on tiny caches; these are software-only fixtures."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.query_contribution import recover_group_states
from association.evaluation.scene import evaluate_full_scene
from association.utils import runtime
from tests.integration.test_observation_quality_pilot import fixture as quality_fixture
from tools.analysis import observation_common as common
from tools.analysis import observation_scale_pilot as pilot
from tools.analysis import native_cv_state_control as cv
from tools.analysis import audit_observation_scale as auditor


def fixture(root):
    args = quality_fixture(root)
    args.output = root/'fixture137'
    manifest = runtime.read_json(args.manifest)
    rho = 1e-8
    runtime.write_json(args.native_run/'scale-selection.json', dict(selected_rho=rho, fixture_only=True))
    # Construct the real historical127 LOW and CV controls from the original
    # kernel, rather than pretending that all controls are copies of Mean.
    for sid in manifest['splits']['V_select']:
        split = 'V_select'
        members = common.membership(args.value_run, args.partition_run, split, sid)
        motion, groups, mean, _, _ = common.physical(args.native_run, split, sid, members)
        native = common.load_npz(args.native_run/'private/native-events'/split/f'scene-{sid}.npz')
        ev = common.evaluation(args.native_run, split, sid, motion)
        low_weights = cv.low_weights(motion['support'], motion['sources'], groups)
        low = recover_group_states(motion['states'], low_weights, groups)['group_states']
        low[..., 2:] = mean[..., 2:]
        predictions = dict(WS_MEAN=mean, WS_LOW_SOURCE_SUPPORTED=low)
        node, _ = cv.predict_scene(native, rho)
        for direction in ('forward', 'rts'):
            states = motion['states'].copy()
            states[..., :2] = node[direction]
            localmean = ordinary_fusion(states, motion['nearest_seconds'], motion['support'], groups)[
                'predictions']['mean_supported'][[g[0] for g in groups]]
            locallow = recover_group_states(states, low_weights, groups)['group_states']
            for suffix, value in (('MEAN', localmean), ('LOW', locallow)):
                value[..., 2:] = mean[..., 2:]
                predictions['WS_CV_'+direction.upper()+'_'+suffix] = value
        artifacts = {}
        for name, prediction in predictions.items():
            artifacts[name] = evaluate_full_scene(emit_group_outputs(prediction, groups, node_count=len(motion['states'])),
                node_identities=ev['identities'], target_identities=ev['target_ids'], target_states=ev['target_states'],
                target_valid=ev['target_valid'], node_observed=motion['support'], split=split, grouping_kind='predicted',
                correspondence_policy='anchor_identity_first', return_artifact=True,
                node_input_support=dict(support=motion['support'], nearest_seconds=motion['nearest_seconds']))['artifact']
        common.save_artifacts(args.native_run/'private'/f'states-{sid}.npz', artifacts)
        for path, names in ((args.source_risk_run, pilot.source.METHODS), (args.value_run, pilot.source.IMPORTED)):
            old = common.load_artifacts(path/'private'/f'states-{sid}.npz', names)
            old['WS_LOW_SOURCE_SUPPORTED'] = artifacts['WS_LOW_SOURCE_SUPPORTED']
            common.save_artifacts(path/'private'/f'states-{sid}.npz', old)
    return args


class ObservationScalePilotTests(unittest.TestCase):
    def test_full_endpoint_isolation_dedup_and_failure_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary).resolve())
            original_read = common.old.frozen.read
            original_evaluation = common.evaluation
            def guarded_read(path, *extra, **kwargs):
                if Path(path) == args.native_run/'scale-selection.json':
                    self.assertTrue((args.output/'scale-selection.json').is_file())
                    state = runtime.read_json(args.output/'run.json')
                    self.assertEqual(state['phase'], 'selection_frozen')
                    self.assertTrue(state['selection_frozen_before_reference_control'])
                    self.assertEqual(state['completed_selection_scene_count'], 32)
                return original_read(path, *extra, **kwargs)
            def guarded_evaluation(*arguments, **kwargs):
                self.assertTrue((args.output/'scale-selection.json').is_file())
                native, split, sid, _ = arguments
                self.assertTrue((args.output/'private'/f'native-state-{split}-{sid}.npz').is_file())
                return original_evaluation(*arguments, **kwargs)
            with patch.object(common.old.frozen, 'read', side_effect=guarded_read), \
                 patch.object(common, 'evaluation', side_effect=guarded_evaluation):
                result = pilot.run(args)
            state = runtime.read_json(args.output/'run.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertEqual(state['completed_training_scene_count'], 32)
            self.assertEqual(state['completed_scene_count'], 32)
            self.assertEqual(state['completed_selection_scene_count'], 32)
            self.assertEqual(state['selection_eligible_fragment_count'], 32*3)
            self.assertEqual(state['selection_filter_smoother_fits'], 9*32*3)
            self.assertEqual(state['recovery_filter_smoother_fits'], 64*4*len(state['unique_recovery_rhos']))
            self.assertEqual(state['analytic_filter_smoother_fits'], state['selection_filter_smoother_fits']+state['recovery_filter_smoother_fits'])
            self.assertEqual(len(result['all_components']['methods']), 18)
            self.assertEqual(len(result['all_components']['pairwise']), 153)
            self.assertEqual(len(result['training_summary']['all_components']['methods']), 5)
            self.assertEqual(len(result['training_summary']['all_components']['pairwise']), 10)
            self.assertEqual(len(result['primary_bootstrap']), 2)
            self.assertEqual(len(result['by_component']), 9)
            self.assertEqual(len(result['training_summary']['by_component']), 12)
            self.assertFalse(state['test_opened'])
            self.assertEqual(state['target_reference_fit_count'], 0)
            checks = auditor.core.Checks()
            independent = auditor.inspect_run(args.output, args.native_run, args.source_risk_run,
                                              args.manifest, checks)
            self.assertEqual([block['methods'] for block in independent['metric_blocks']], [5, 18])
            self.assertEqual([block['pairs_per_scene'] for block in independent['metric_blocks']], [10, 153])
            self.assertEqual(independent['selection_fit_count'], 9*32*3)
            # No-score/missing fragments persist in both output and selection.
            held = common.load_npz(args.output/'private/heldout-T-0.npz')
            np.testing.assert_array_equal(held['eligible'], [True, False, True, True])
            self.assertEqual(held['rts_positions'].shape, (9, int(held['heldout_offsets'][-1]), 2))
            # Independent geometry detects incorrect stored criterion rather
            # than merely accepting the producer's selection/summary values.
            raw = common.load_npz(args.native_run/'private/native-events/T_core/scene-0.npz')
            altered = {key: value.copy() for key, value in held.items()}
            altered['mse_by_rho'][0, 0] += 100.
            with self.assertRaisesRegex(ValueError, 'MSE'):
                auditor.inspect_heldout(altered, raw, auditor.core.Checks())
            for split, sid, methods in (('T_core', '0', pilot.T_METHODS), ('V_select', '32', pilot.V_METHODS)):
                artifacts = common.load_artifacts(args.output/'private'/f'states-{split}-{sid}.npz', methods,
                                                  template=methods[0])
                for name in methods:
                    np.testing.assert_array_equal(artifacts[name]['group_states'][..., 2:], artifacts[methods[0]]['group_states'][..., 2:])
                    np.testing.assert_array_equal(artifacts[name]['aligned_prediction_valid'], artifacts[methods[0]]['aligned_prediction_valid'])
            # Real identical-rho prediction executes the original filter once
            # per node, supplies aliases, and does not run FWD/RTS separately.
            sid, split = '0', 'T_core'
            members = common.membership(args.value_run, args.partition_run, split, sid)
            m, g, mean, pack, _ = common.physical(args.native_run, split, sid, members)
            native = common.load_npz(args.native_run/'private/native-events'/split/f'scene-{sid}.npz')
            with patch.object(cv, 'predict_scene', wraps=cv.predict_scene) as calls:
                predictions, _, stored, _, _ = pilot.predict(m, g, mean, pack, native, 1e-8, 1e-8)
            self.assertEqual(calls.call_count, 1)
            np.testing.assert_array_equal(stored['unique_rhos'], [1e-8])
            for suffix in ('FORWARD', 'RTS'):
                np.testing.assert_array_equal(predictions['WS_OBS_SCALE_'+suffix+'_EQUAL'], predictions['WS_REFERENCE_SCALE_'+suffix+'_EQUAL'])
            # Corrupt canonical source lineage: fail before ref/evaluation,
            # keep the original completed fixture and the new failed record.
            path = args.native_run/'private/native-events/T_core/scene-0.npz'
            native['event_node_ids'][0] = 'bad'
            np.savez_compressed(path, **native)
            broken = SimpleNamespace(**vars(args)); broken.output = args.output.parent/'broken137'
            with self.assertRaisesRegex(ValueError, 'lineage'):
                pilot.run(broken)
            self.assertEqual(runtime.read_json(broken.output/'run.json')['status'], 'FAILED')
            self.assertFalse((broken.output/'scale-selection.json').exists())
            self.assertEqual(runtime.read_json(args.output/'run.json')['status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
