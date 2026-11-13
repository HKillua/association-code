"""062 report contracts: full outputs, routing denominators and pooled tails."""
from copy import deepcopy
import unittest

import numpy as np

from association.evaluation.frozen_fusion_transfer import METHODS, LEARNED_METHODS, summarize_scene, aggregate
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.scene import evaluate_full_scene


class FrozenFusionTransferTests(unittest.TestCase):
    @staticmethod
    def fixture(sparse=False, singleton=False, cancellation=False):
        groups = ([[i] for i in range(5)] if singleton else
                  [[0, 1, 2], [3], [4]] if cancellation else [[0, 1], [2, 3], [4]])
        nodes, targets = np.array(['a', 'a', 'a', 'a', 'b']), np.array(['a', 'b', 'c'])
        q = 3
        native = np.broadcast_to(np.array([[30.01, 120., 5., 359.], [30., 120., 7., 1.],
            [30.1, 120., 5., 359.], [30.1, 120., 5., 359.], [31., 121., 6., 1.]])[:, None], (5, q, 4)).copy()
        if cancellation:
            native[0, :, 0], native[1, :, 0], native[2, :, 0] = 30., 29.9, 30.1
        truth = np.broadcast_to(np.array([[30., 120., 6., 1.], [31., 121., 7., 359.],
            [32., 122., 8., 1.]])[:, None], (3, q, 4)).copy()
        valid = np.ones_like(truth, bool)
        if sparse:
            truth[0, :, 0] += .2
            valid[:] = False
            valid[0, 0] = True
        truth[~valid] = np.nan
        support = np.array([[True, False, False], [True, True, False],
                            [True, True, True], [True, True, True], [False, False, False]])
        nearest = np.where(support, 10., 2000.)
        controls = ordinary_fusion(native, nearest, support, groups)
        anchor, mean = controls['predictions']['anchor'], controls['predictions']['mean_supported']
        hybrid, switch = mean.copy(), mean.copy()
        hybrid[..., :2] = anchor[..., :2]
        for group in groups:
            for node in group:
                use = ~support[node] & support[[other for other in group if other != node]].any(0)
                switch[node, :, :2] = np.where(use[:, None], mean[node, :, :2], anchor[node, :, :2])
        predictions = dict(anchor=anchor, mean_supported=mean, attribute_hybrid=hybrid,
            support_switch=switch, **{name: native.copy() for name in LEARNED_METHODS})
        artifacts = {}
        for method, prediction in predictions.items():
            artifacts[method] = evaluate_full_scene([
                dict(node_indices=group, states=prediction[group[0]]) for group in groups],
                node_identities=nodes, target_identities=targets, target_states=truth, target_valid=valid,
                node_observed=support, split='V_select', grouping_kind='predicted',
                correspondence_policy='anchor_identity_first', return_artifact=True,
                node_input_support=dict(support=support, nearest_seconds=nearest))['artifact']
        ha = np.array([a for group in groups for a in group for h in group if h != a], np.int64)
        hn = np.array([h for group in groups for a in group for h in group if h != a], np.int64)
        routing = {}
        for name in LEARNED_METHODS:
            weights = np.zeros((len(ha), q, 3))
            own = np.ones((5, q, 3))
            if cancellation and name != 'direct_only':
                weights[ha == 0, :, 0] = .25
                own[0, :, 0] = .5
            elif not singleton and name != 'direct_only':
                # Nonrepresentative anchor1 is deliberately active; it must not
                # contaminate actual-output routing. Duplicate anchor2's helper
                # is active but identical, so positive weight changes no state.
                weights[1] = .25
                weights[2] = .5
                own[1] = .75
                own[2] = .5
            routing[name] = dict(helper_anchor=ha, helper_node=hn, helper_weights=weights,
                anchor_weights=own, utility=np.zeros_like(weights), membership=controls['membership'])
        fragment = dict(native_states=native, support=support, nearest_seconds=nearest,
            identities=nodes, target_identity_ids=targets, direct_only_prediction=predictions['direct_only'],
            learned_assoc_group_offsets=np.r_[0, np.cumsum([len(g) for g in groups])],
            learned_assoc_group_node_indices=np.arange(5))
        return fragment, artifacts, routing

    def test_duplicate_output_and_nonrepresentative_routes_have_distinct_denominators(self):
        fragment, artifacts, routes = self.fixture()
        row, arrays = summarize_scene(fragment, artifacts, routes, 'scene-a', 'component-1')
        population = row['methods']['anchor']['population']
        self.assertEqual(population['target_identity_count'], 3)
        self.assertEqual(population['predicted_group_count'], 3)
        self.assertEqual(population['duplicate_prediction_group_count'], 1)
        metrics = row['methods']['anchor']['full_target_metrics']['all']['position']
        self.assertEqual(metrics['available_target_count'], 9)
        self.assertEqual(metrics['intersection_count'], 6)
        self.assertEqual(metrics['available_prediction_count'], 9)
        self.assertEqual(np.isfinite(arrays['anchor_position_error']).sum(), 6)
        all_routes = row['routing']['helper_quality']['all_output_groups']['position']
        unique_routes = row['routing']['helper_quality']['unique_selected_groups']['position']
        self.assertEqual(all_routes['output_query_count'], 9)
        self.assertEqual(all_routes['helper_candidate_query_count'], 6)
        self.assertEqual(all_routes['eligible_anchor_query_count'], 6)
        self.assertEqual(all_routes['positive_helper_query_count'], 3)
        self.assertEqual(all_routes['accepted_anchor_query_count'], 3)
        self.assertEqual(all_routes['helper_weight_sum'], 1.5)
        self.assertEqual(all_routes['positive_helper_unchanged_state_point_count'], 3)
        self.assertEqual(all_routes['changed_state_point_count'], 0)
        self.assertEqual(unique_routes['output_query_count'], 6)
        self.assertEqual(unique_routes['helper_candidate_query_count'], 3)
        self.assertEqual(unique_routes['positive_helper_query_count'], 0)
        self.assertEqual(unique_routes['helper_weight_sum'], 0)

    def test_direct_only_position_checks_fragment_group_and_unique_views(self):
        fragment, artifacts, routes = self.fixture()
        fragment['direct_only_prediction'][1, :, 0] += .01
        row, _ = summarize_scene(fragment, artifacts, routes, 's', 'c')
        self.assertFalse(row['direct_only_vs_anchor']['fragment']['exact_array_equal'])
        self.assertEqual(row['direct_only_vs_anchor']['fragment']['physical_changed_point_count'], 3)
        self.assertTrue(row['direct_only_vs_anchor']['all_output_groups']['exact_array_equal'])
        self.assertTrue(row['direct_only_vs_anchor']['unique_valid_targets']['exact_array_equal'])
        self.assertEqual(row['direct_only_vs_anchor']['unique_valid_targets']['point_count'], 6)

    def test_no_helper_is_null_and_positive_helpers_can_cancel(self):
        fragment, artifacts, routes = self.fixture(singleton=True)
        row, values = summarize_scene(fragment, artifacts, routes, 'singletons', 'c')
        summary = aggregate([row], [values])
        route = summary['routing']['helper_quality']['all_output_groups']['position']
        self.assertEqual(route['output_query_count'], 15)
        self.assertEqual(route['helper_candidate_query_count'], 0)
        self.assertEqual(route['eligible_anchor_query_count'], 0)
        self.assertIsNone(route['positive_helper_query_fraction'])
        self.assertIsNone(route['accepted_anchor_query_fraction'])
        self.assertIsNone(route['mean_helper_mass_per_eligible_anchor_query'])
        fragment, artifacts, routes = self.fixture(cancellation=True)
        row, _ = summarize_scene(fragment, artifacts, routes, 'cancel', 'c')
        route = row['routing']['helper_quality']['unique_selected_groups']['position']
        self.assertEqual(route['positive_helper_query_count'], 6)
        self.assertEqual(route['accepted_anchor_query_count'], 3)
        self.assertEqual(route['changed_state_point_count'], 0)
        self.assertEqual(route['positive_helper_unchanged_state_point_count'], 3)

    def test_population_mask_and_duplicate_group_alignment_tampering_fail(self):
        fragment, artifacts, routes = self.fixture()
        for key in ('group_valid', 'aligned_prediction_valid'):
            wrong = deepcopy(artifacts)
            wrong['mixed_action'][key][-1, 0, 0] = False
            with self.assertRaisesRegex(ValueError, '062 fixed'):
                summarize_scene(fragment, wrong, routes, 's', 'c')
        wrong = deepcopy(artifacts)
        wrong['mixed_action']['group_states'][1, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'group/row output'):
            summarize_scene(fragment, wrong, routes, 's', 'c')
        wrong = deepcopy(artifacts)
        wrong['anchor']['group_node_indices'][:2] = [1, 0]
        with self.assertRaisesRegex(ValueError, '062 source group_node_indices'):
            summarize_scene(fragment, wrong, routes, 's', 'c')

    def test_routing_requires_all_pairs_and_normalized_nonnegative_weights(self):
        fragment, artifacts, routes = self.fixture()
        wrong = deepcopy(routes)
        wrong['direct_only']['helper_anchor'][0] = 2
        with self.assertRaisesRegex(ValueError, 'every original'):
            summarize_scene(fragment, artifacts, wrong, 's', 'c')
        wrong = deepcopy(routes)
        wrong['direct_only']['helper_weights'][0, 0, 0] = .1
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            summarize_scene(fragment, artifacts, wrong, 's', 'c')

    def test_pooled_quantiles_changes_and_components_use_actual_points(self):
        rows, arrays = [], []
        for index, sparse in enumerate((False, True)):
            fragment, artifacts, routes = self.fixture(sparse=sparse)
            row, values = summarize_scene(fragment, artifacts, routes, str(index), 'shared')
            rows.append(row)
            arrays.append(values)
        summary = aggregate(rows, arrays)
        errors = np.concatenate([x['anchor_position_error'][np.isfinite(x['anchor_position_error'])] for x in arrays])
        quantiles = summary['methods']['anchor']['position_quantiles']['all']
        self.assertEqual(quantiles['point_count'], 7)
        self.assertEqual(quantiles['p95'], np.quantile(errors, .95, method='linear'))
        self.assertNotAlmostEqual(quantiles['p95'], np.mean([
            row['methods']['anchor']['position_quantiles']['all']['p95'] for row in rows]))
        self.assertEqual(summary['components']['shared']['scene_count'], 2)
        self.assertEqual(summary['component_count'], 1)
        self.assertIsNone(summary['methods']['anchor']['position_quantiles']['no_predicted_group']['p99'])
        for attr in ('position', 'speed', 'heading'):
            delta = np.concatenate([(x['mean_supported_'+attr+'_error']-x['anchor_'+attr+'_error'])[
                np.isfinite(x['anchor_'+attr+'_error'])] for x in arrays])
            changes = summary['pairwise']['mean_supported__minus__anchor']['all'][attr]
            self.assertEqual(changes['common_point_count'], 7)
            self.assertAlmostEqual(changes['signed_error_change_sum'], delta.sum())
            self.assertAlmostEqual(changes['harm_sum'], delta[delta > 0].sum())
            self.assertEqual(changes['positive_delta_tail']['point_count'], int((delta > 0).sum()))
            if (delta > 0).any():
                self.assertEqual(changes['positive_delta_tail']['p99'], np.quantile(delta[delta > 0], .99, method='linear'))
        changes = summary['pairwise']['direct_only__minus__anchor']['all']['position']
        self.assertEqual(changes['tied_point_count'], 7)
        self.assertIsNone(changes['positive_delta_tail']['p95'])
        self.assertEqual(summary['routing']['helper_quality']['all_output_groups']['position']['helper_weight_sum'], 3.)
        corrupted = deepcopy(arrays)
        corrupted[0]['anchor_position_error'][0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'pooled point population'):
            aggregate(rows, corrupted)


if __name__ == '__main__':
    unittest.main()
