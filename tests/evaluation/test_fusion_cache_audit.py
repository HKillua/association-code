"""067 original-OLS replay attribution: fixed correspondence and arithmetic."""
from copy import deepcopy
import unittest

import numpy as np

from association.evaluation.fusion_cache_audit import audit_source, aggregate, CATEGORIES, REPLAY_TOLERANCE
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.scene import evaluate_full_scene


class FusionCacheAuditTests(unittest.TestCase):
    @staticmethod
    def fixture(arm='helper_quality', sparse=False, multiple=True):
        groups = [[0, 1, 2], [3, 4], [5]] if multiple else [[0, 1], [2], [3, 4], [5]]
        n, q = 6, 3
        identities = np.array(['a', 'a', 'b', 'a', 'a', 'c'])
        targets = np.array(['a', 'b', 'c', 'd'])
        native = np.broadcast_to(np.array([[30., 120., 10., 30.], [30.1, 120.1, 3000., 40.],
            [30.2, 120.2, 5000., 50.], [30.3, 120.3, 20., 60.],
            [30.4, 120.4, 30., 70.], [31., 121., 40., 80.]])[:, None], (n, q, 4)).copy()
        truth = native[[0, 2, 5, 0]].copy()
        valid = np.ones_like(truth, bool)
        valid[0, 1, 2] = False
        if sparse:
            valid[:] = False
        truth[~valid] = np.nan
        idx = np.array([0, 0, 1, 0, 0, 2])
        support = np.ones((n, q), bool)
        fragment = dict(native_states=native, identities=identities, target_identity_ids=targets,
            node_ids=np.array(['node'+str(i) for i in range(n)]), sources=np.arange(n)%2,
            query_seconds=np.arange(q, dtype=float)*10, target=truth[idx], target_valid=valid[idx],
            support=support, exact_observation=np.zeros((n, q), bool), nearest_seconds=np.zeros((n, q)),
            ols_group_offsets=np.r_[0, np.cumsum([len(g) for g in groups])],
            ols_group_node_indices=np.array([i for g in groups for i in g]))
        controls = ordinary_fusion(native, fragment['nearest_seconds'], support, groups)
        ha = np.array([a for group in groups for a in group for h in group if h != a], np.int64)
        hn = np.array([h for group in groups for a in group for h in group if h != a], np.int64)
        weights = np.zeros((len(ha), q, 3), float)
        own = np.ones((n, q, 3))
        # Zero helper is a valid baseline; new weights are modified in tests.
        route = dict(helper_anchor=ha, helper_node=hn, helper_weights=weights,
            anchor_weights=own, utility=np.zeros_like(weights), membership=controls['membership'])
        old = deepcopy(fragment)
        old.update(ols_learned=native.copy(), ols_mean_supported=controls['predictions']['mean_supported'])
        old.update({'ols_routing_'+k: v.copy() for k, v in route.items()})
        fragment.update({'source057_'+arm+'_routing_'+k: v.copy() for k, v in route.items()})
        fragment.update({'source057_'+arm+'_prediction_learned': native.copy(),
            'source057_'+arm+'_prediction_mean_supported': controls['predictions']['mean_supported'].copy()})
        artifact = evaluate_full_scene([dict(node_indices=g, states=native[g[0]]) for g in groups],
            node_identities=identities, target_identities=targets, target_states=truth, target_valid=valid,
            node_observed=support, split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=support, nearest_seconds=fragment['nearest_seconds']))['artifact']
        return fragment, old, artifact, arm

    @staticmethod
    def change_speed(fragment, arm, edge, time, value):
        prefix = 'source057_'+arm+'_'
        ha = fragment[prefix+'routing_helper_anchor']
        hn = fragment[prefix+'routing_helper_node']
        node, helper = int(ha[edge]), int(hn[edge])
        fragment[prefix+'routing_helper_weights'][edge, time, 1] += value
        fragment[prefix+'routing_anchor_weights'][node, time, 1] -= value
        fragment[prefix+'prediction_learned'][node, time, 2] += value*(
            fragment['native_states'][helper, time, 2]-fragment['native_states'][node, time, 2])

    def test_four_mapping_categories_and_edge_vs_statepoint_counts(self):
        f, old, artifact, arm = self.fixture()
        # anchor0 has two helpers: two violating edge slots, one state point.
        for edge in (0, 1):
            self.change_speed(f, arm, edge, 0, 2e-6)
        self.change_speed(f, arm, 0, 1, 2e-6)  # selected but invalid speed truth
        self.change_speed(f, arm, 2, 0, 2e-6)  # anchor1 is not a system representative
        self.change_speed(f, arm, 6, 0, 2e-6)  # anchor3 is duplicate output for identity a
        row, arrays = audit_source(f, old, artifact, arm, '2535', 'component-a')
        counts = row['violation_mapping_counts']['helper_weight']
        self.assertEqual([counts[c]['record_count'] for c in CATEGORIES], [1, 1, 1, 2])
        self.assertEqual(counts[CATEGORIES[3]]['distinct_anchor_query_attribute_count'], 1)
        self.assertEqual(row['speed_algebra']['exceeds_arithmetic_bound_count'], 0)
        self.assertEqual(row['population']['duplicate_groups'], 1)
        self.assertTrue(row['rejected_output_replay'])
        self.assertEqual(arrays['unique_selected_groups_speed_current_error'].size, 5)
        self.assertEqual(arrays['all_output_groups_speed_current_error'].size, 7)
        self.assertEqual(arrays['unique_selected_groups_speed_state_difference'].size, 6)
        summary = aggregate([row], [arrays])
        self.assertEqual(summary['original066_acceptance'], 'REJECTED')
        self.assertEqual(summary['by_arm'][arm]['scene_count'], 1)
        self.assertEqual(summary['by_component']['component-a'][arm]['rejected_replay_count'], 1)

    def test_scene2535_speed_amplification_may_or_may_not_be_adopted(self):
        for node, expected_category in ((0, CATEGORIES[3]), (1, CATEGORIES[0])):
            with self.subTest(node=node):
                f, old, artifact, arm = self.fixture()
                ha = f['source057_'+arm+'_routing_helper_anchor']
                hn = f['source057_'+arm+'_routing_helper_node']
                edge = next(i for i, pair in enumerate(zip(ha, hn)) if pair == (node, 2))
                self.change_speed(f, arm, edge, 0, 8e-7)
                row, _ = audit_source(f, old, artifact, arm, '2535', 'component-a')
                peak = row['largest_speed_difference']
                self.assertEqual(peak['category'], expected_category)
                self.assertGreater(peak['state_difference'], .001)
                self.assertAlmostEqual(peak['signed_speed_difference'], peak['weight_decomposition'], places=11)
                self.assertLessEqual(abs(peak['residual']), peak['arithmetic_bound'])
                self.assertEqual(sum(x['record_count'] for x in row['violation_mapping_counts']['helper_weight'].values()), 0)
                self.assertEqual(len(row['violation_records']['physical']), 1)
                self.assertTrue(row['rejected_output_replay'])

    def test_zero_helpers_sparse_truth_and_sign_flips_have_no_false_error_denominator(self):
        f, old, artifact, arm = self.fixture(sparse=True)
        f['source057_'+arm+'_routing_utility'][0, 0, 0] = 1e-12
        row, values = audit_source(f, old, artifact, arm, '1', 'c')
        self.assertEqual(len(row['violation_records']['utility_activation_flip']), 1)
        self.assertFalse(row['rejected_output_replay'])
        self.assertEqual(row['speed_algebra']['maximum_absolute_residual'], 0)
        for scope in ('all_output_groups', 'unique_selected_groups'):
            for attr in ('position', 'speed', 'heading'):
                self.assertEqual(row['metrics'][scope][attr]['previous_error']['count'], 0)
                self.assertIsNone(row['metrics'][scope][attr]['mae_change'])
        result = aggregate([row], [values])
        self.assertIsNone(result['by_arm'][arm]['metrics']['unique_selected_groups']['speed']['rmse_change'])

    def test_bad_output_algebra_reported_without_changing_replay_acceptance(self):
        f, old, artifact, arm = self.fixture()
        f['source057_'+arm+'_prediction_learned'][0, 0, 2] += .0001
        row, _ = audit_source(f, old, artifact, arm, '1', 'c')
        self.assertFalse(row['rejected_output_replay'])
        self.assertEqual(row['speed_algebra']['exceeds_arithmetic_bound_count'], 1)
        self.assertFalse(row['speed_algebra']['is066_acceptance_gate'])

    def test_corrupt_duplicate_artifact_nonfinite_and_routing_are_rejected(self):
        for kind in ('duplicate', 'nonfinite', 'membership', 'mass', 'input'):
            f, old, artifact, arm = self.fixture()
            if kind == 'duplicate':
                artifact['group_states'][1, 0, 2] += 1
            elif kind == 'nonfinite':
                f['source057_'+arm+'_routing_helper_weights'][0, 0, 0] = np.nan
            elif kind == 'membership':
                f['source057_'+arm+'_routing_membership'][0] = 5
            elif kind == 'mass':
                f['source057_'+arm+'_routing_anchor_weights'][0, 0, 0] = .5
            else:
                old['native_states'][0, 0, 2] += 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit_source(f, old, artifact, arm, '1', 'c')

    def test_full_prefixed_artifact_and_pooled_component_metrics(self):
        f, old, artifact, arm = self.fixture()
        a, av = audit_source(f, old, {'ols_learned_'+k: v for k, v in artifact.items()}, arm, '1', 'c1')
        self.change_speed(f, arm, 0, 0, 1e-7)
        b, bv = audit_source(f, old, artifact, arm, '2', 'c2')
        result = aggregate([a, b], [av, bv])
        speed = result['by_arm'][arm]['metrics']['unique_selected_groups']['speed']
        values = np.r_[av['unique_selected_groups_speed_current_error'], bv['unique_selected_groups_speed_current_error']]
        self.assertEqual(speed['current_error']['count'], len(values))
        self.assertAlmostEqual(speed['current_error']['p99'], np.quantile(values, .99))
        self.assertEqual(len(result['by_component']), 2)
        with self.assertRaisesRegex(ValueError, 'duplicate scene'):
            aggregate([a, a], [av, av])

    def test_mapped_violating_weight_need_not_change_output_or_error(self):
        f, old, artifact, arm = self.fixture()
        f['native_states'][1, :, 2] = 10.
        old['native_states'][1, :, 2] = 10.
        old['ols_learned'][1, :, 2] = 10.
        f['source057_'+arm+'_prediction_learned'][1, :, 2] = 10.
        self.change_speed(f, arm, 0, 0, 2e-6)
        row, values = audit_source(f, old, artifact, arm, '1', 'c')
        summary = aggregate([row], [values])
        self.assertEqual(summary['rejected_replay_count'], 1)
        self.assertEqual(summary['rejected_replays_with_unique_valid_mapped_violation'], 1)
        self.assertEqual(summary['rejected_replays_with_unique_valid_effect']['state_changed'], 0)
        self.assertEqual(summary['rejected_replays_with_unique_valid_effect']['truth_error_changed'], 0)
        self.assertEqual(summary['by_arm'][arm]['rejected_replay_count_by_mapping_category'][CATEGORIES[3]], 1)

    def test_original_numeric_limits_match_execution_contract(self):
        # Test-only import; the array-only analyzer must not import inference.
        from association.evaluation.evidence_fusion_combo import REPLAY_TOLERANCE as original
        self.assertEqual(REPLAY_TOLERANCE, original)


if __name__ == '__main__':
    unittest.main()
