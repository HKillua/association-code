"""Targeted source-risk software checks, not MTAD performance evidence."""
from copy import deepcopy
import unittest

import numpy as np

from association.evaluation import source_risk as r


def inputs(sources=('a', 'b', 'c'), q=3, groups=None):
    n = len(sources)
    states = np.zeros((n, q, 4), np.float64)
    states[..., 0] = 30.
    states[..., 1] = 120.+np.arange(n)[:, None]*.01
    states[..., 2] = 10.+np.arange(n)[:, None]
    states[..., 3] = np.arange(n)[:, None]*20.
    return dict(states=states, support=np.ones((n, q), bool),
        nearest_seconds=np.broadcast_to(np.arange(n)[:, None]*60., (n, q)).copy(),
        sources=np.asarray(sources), native_count=np.arange(n, dtype=float)+5.,
        span_seconds=np.arange(n, dtype=float)*60.+60., fit_rmse_m=np.arange(n, dtype=float)*100.,
        ols_velocity_mps=np.column_stack((np.arange(n, dtype=float), np.zeros(n))),
        groups=groups if groups is not None else [list(range(n))] if n else [])


class SourceRiskTests(unittest.TestCase):
    def test_variable_source_count_renaming_and_group_structure(self):
        for k in (1, 2, 3, 4):
            data = inputs(tuple(f's{i}' for i in range(k)), q=5)
            original = deepcopy(data)
            pack = r.build_source_features(**data)
            self.assertEqual(pack['features'].shape, (k, 5, 10))
            self.assertEqual(pack['group_offsets'].tolist(), [0, k])
            np.testing.assert_array_equal(pack['node_to_source_row'], np.arange(k))
            renamed = deepcopy(data)
            renamed['sources'] = np.asarray([f'z{100-i}' for i in range(k)])
            other = r.build_source_features(**renamed)
            for key in pack:
                if key != 'audit':
                    np.testing.assert_array_equal(pack[key], other[key])
            for key in original:
                np.testing.assert_array_equal(data[key], original[key])
            if k == 1:
                np.testing.assert_array_equal(pack['features'][..., 7:], 0.)
        data = inputs(('a', 'b', 'a', 'c', 'd', 'c'), groups=[[0, 1, 2], [3, 4, 5]])
        pack = r.build_source_features(**data)
        self.assertEqual(pack['group_offsets'].tolist(), [0, 2, 4])
        self.assertEqual(pack['source_group_indices'].tolist(), [0, 0, 1, 1])
        self.assertEqual(pack['source_member_count'].tolist(), [2, 1, 2, 1])
        self.assertEqual(pack['node_to_source_row'].tolist(), [0, 1, 0, 2, 3, 2])

    def test_source_internal_means_features_and_no_input_mutation(self):
        data = inputs(('a', 'a', 'b'), q=2)
        data['states'][:, :, 0] = np.array([0., 2., 3.])[:, None]
        data['states'][:, :, 1] = 0.
        data['support'][1, 1] = False
        pack = r.build_source_features(**data)
        np.testing.assert_allclose(pack['source_states'][0, :, 0], [1., 0.])
        np.testing.assert_array_equal(pack['supported_counts'][0], [2, 1])
        np.testing.assert_allclose(pack['features'][0, :, 0], [np.log1p(.5), 0.])
        np.testing.assert_allclose(pack['features'][0, :, 2], np.log1p([5.5, 5.]))
        np.testing.assert_allclose(pack['features'][0, :, 5], [1., .5])
        one_degree = 6371000.*np.pi/180.
        self.assertAlmostEqual(pack['features'][0, 0, 6], np.log1p(one_degree/1000.), places=12)
        self.assertEqual(pack['features'][0, 1, 6], 0.)

    def test_unavailable_source_never_enters_defined_context(self):
        data = inputs(q=2)
        data['support'][2] = False
        before = r.build_source_features(**data)
        changed = deepcopy(data)
        changed['states'][2] = [80., -170., 1000., 270.]
        changed['ols_velocity_mps'][2] = 1e9
        changed['native_count'][2] = 1e6
        after = r.build_source_features(**changed)
        np.testing.assert_array_equal(before['features'], after['features'])
        np.testing.assert_array_equal(before['features'][2], 0.)
        np.testing.assert_array_equal(before['available'][2], False)

    def test_circular_states_and_ambiguous_context_fallback(self):
        data = inputs(('a', 'a', 'b'), q=2)
        data['states'][:2, :, 1] = np.array([179., -179.])[:, None]
        data['states'][:2, :, 3] = np.array([350., 10.])[:, None]
        pack = r.build_source_features(**data)
        np.testing.assert_allclose(np.abs(pack['source_states'][0, :, 1]), 180.)
        np.testing.assert_allclose(pack['source_states'][0, :, 3] % 360., 0., atol=1e-12)
        data = inputs(('a', 'b', 'c'), q=1)
        data['states'][:, 0, 1] = [30., 0., 180.]
        data['states'][:, 0, 3] = [20., 0., 180.]
        pack = r.build_source_features(**data)
        self.assertGreaterEqual(pack['audit']['context_longitude_zero_resultant_count'], 1)
        self.assertGreaterEqual(pack['audit']['context_heading_zero_resultant_count'], 1)
        self.assertEqual(pack['features'][0, 0, 9], 0.)

    def test_ambiguous_source_label_matches_hard_recovery_group_anchor(self):
        data = inputs(('a', 'b', 'b'), q=2)
        data['states'][..., 0] = 0.
        data['states'][:, :, 1] = np.array([60., 0., 180.])[:, None]
        data['states'][:, :, 3] = np.array([20., 0., 180.])[:, None]
        data['support'][:, 1] = False
        pack = r.build_source_features(**data)
        self.assertEqual(pack['audit']['source_longitude_zero_resultant_count'], 1)
        self.assertEqual(pack['audit']['source_heading_zero_resultant_count'], 1)
        # The available ambiguous source uses the group angle, but an absent
        # source keeps its own first-member storage and has no scientific mass.
        np.testing.assert_array_equal(pack['source_states'][1, 0, [1, 3]], [60., 20.])
        np.testing.assert_array_equal(pack['source_states'][1, 1, [1, 3]], [0., 0.])
        np.testing.assert_array_equal(pack['features'][:, 1], 0.)
        out = r.recover_source_policy(states=data['states'], support=data['support'],
            groups=data['groups'], pack=pack, scores=np.array([[1., 1.], [0., 0.]]),
            kind='hard', mean_states=data['states'][:1].copy())
        np.testing.assert_array_equal(out['source_weights'][:, 0], [0., 1.])
        np.testing.assert_array_equal(out['group_states'][0, 0, :2], pack['source_states'][1, 0, :2])
        truth = np.zeros((1, 2, 4))
        labels = r.source_labels(pack, data['groups'], np.array(['t']*3), np.array(['t']),
            truth, np.ones_like(truth, bool))
        self.assertAlmostEqual(labels['errors_m'][1, 0], 6371000.*np.pi/3.)
        np.testing.assert_array_equal(labels['labeled'][:, 1], False)

    def test_ambiguous_context_declared_unavailable_anchor_exception_is_counted(self):
        data = inputs(('a', 'b', 'c', 'd'), q=1)
        data['states'][..., 0] = 0.
        data['states'][:, 0, 1] = [60., 30., 0., 180.]
        data['support'][0] = False
        before = r.build_source_features(**data)
        data['states'][0, 0, 1] = -60.
        after = r.build_source_features(**data)
        for pack in (before, after):
            self.assertEqual(pack['audit']['context_longitude_zero_resultant_count'], 1)
            np.testing.assert_array_equal(pack['features'][0], 0.)
            np.testing.assert_array_equal(pack['available'][0], False)
        # The other available sources have an undefined resultant; this is the
        # explicit finite-value fallback exception, not weighted participation.
        self.assertAlmostEqual(before['features'][1, 0, 7], np.log1p(6371000.*np.pi/6./1000.))
        self.assertAlmostEqual(after['features'][1, 0, 7], np.log1p(6371000.*np.pi/2./1000.))

    def test_soft_hard_equal_missing_and_source_row_equivariance(self):
        scores = np.array([[0., 1., 5., 0.], [np.log(2.), 1., 6., 0.], [np.log(4.), 2., 7., 0.]])
        available = np.array([[True, True, False, False], [True, True, True, False], [True, True, False, False]])
        soft = r.source_weights(scores, available, [0, 3], kind='soft')
        np.testing.assert_allclose(soft[:, 0], [4/7, 2/7, 1/7])
        np.testing.assert_array_equal(soft[:, 2], [0., 1., 0.])
        np.testing.assert_array_equal(soft[:, 3], 0.)
        hard = r.source_weights(scores, available, [0, 3], kind='hard')
        np.testing.assert_array_equal(hard[:, 1], [.5, .5, 0.])
        order = [2, 0, 1]
        for kind in ('equal', 'soft', 'hard'):
            original = r.source_weights(scores, available, [0, 3], kind=kind)
            changed = r.source_weights(scores[order], available[order], [0, 3], kind=kind)
            np.testing.assert_allclose(changed, original[order])
            constant = r.source_weights(np.ones_like(scores), available, [0, 3], kind=kind)
            np.testing.assert_array_equal(constant, r.source_weights(scores, available, [0, 3], kind='equal'))
        extreme = r.source_weights(np.array([[-1e308], [1e308]]), np.ones((2, 1), bool), [0, 2], kind='soft')
        np.testing.assert_array_equal(extreme[:, 0], [1., 0.])

    def test_recovery_source_internal_sharing_and_position_only_fallback(self):
        data = inputs(('a', 'a', 'b'), q=3)
        data['support'][:, 2] = False
        data['support'][1, 1] = False
        pack = r.build_source_features(**data)
        mean = np.zeros((1, 3, 4)); mean[..., 0] = 10.; mean[..., 1] = 50.
        mean[..., 2:] = [123., 234.]
        result = r.recover_source_policy(states=data['states'], support=data['support'], groups=data['groups'],
            pack=pack, scores=np.zeros((2, 3)), kind='soft', mean_states=mean)
        np.testing.assert_array_equal(result['node_weights'][:, 0], [.25, .25, .5])
        np.testing.assert_array_equal(result['node_weights'][:, 1], [.5, 0., .5])
        np.testing.assert_array_equal(result['node_weights'][:, 2], 0.)
        np.testing.assert_array_equal(result['group_states'][..., 2:], mean[..., 2:])
        np.testing.assert_array_equal(result['group_states'][:, 2], mean[:, 2])
        # Reorder pack source rows, retaining their physical node mapping.
        permuted = deepcopy(pack)
        for key in ('features', 'source_states', 'available', 'supported_counts', 'source_group_indices', 'source_member_count'):
            permuted[key] = permuted[key][::-1].copy()
        permuted['node_to_source_row'] = 1-permuted['node_to_source_row']
        swapped = r.recover_source_policy(states=data['states'], support=data['support'], groups=data['groups'],
            pack=permuted, scores=np.zeros((2, 3)), kind='soft', mean_states=mean)
        np.testing.assert_array_equal(result['group_states'], swapped['group_states'])
        singleton = inputs(('a',), q=2)
        pack = r.build_source_features(**singleton)
        out = r.recover_source_policy(states=singleton['states'], support=singleton['support'], groups=[[0]],
            pack=pack, scores=np.zeros((1, 2)), kind='hard', mean_states=singleton['states'].copy())
        np.testing.assert_array_equal(out['group_states'], singleton['states'])

    def test_weighted_fit_hand_solution_and_constant_feature_normalizer(self):
        x = np.zeros((3, 10)); x[:, 0] = [-1., 0., 1.]
        y = 2.+3.*x[:, 0]
        models = r.fit_score_models(x, y, np.array([1., 2., 1.]))
        linear = models['LINEAR']
        np.testing.assert_allclose(linear['mean'], 0.)
        self.assertAlmostEqual(linear['std'][0], np.sqrt(.5))
        np.testing.assert_array_equal(linear['std'][1:], 1.)
        self.assertAlmostEqual(linear['beta'][0], 2.)
        self.assertAlmostEqual(linear['beta'][1], 3.*np.sqrt(.5)/2.)
        np.testing.assert_allclose(r.predict_score(linear, x), [0.5, 2., 3.5])
        self.assertEqual(linear['training_weight_sum'], 4.)
        for name, model in models.items():
            self.assertEqual(len(model['beta']), 11 if name == 'LINEAR' else 66)
            np.testing.assert_array_equal(model['permutation'], np.random.default_rng(20).permutation(3)
                if name == 'SHUFFLED' else np.arange(3))
        scaled = r.fit_score_models(x, y, np.array([10., 20., 10.]))
        for name in models:
            np.testing.assert_allclose(models[name]['beta'], scaled[name]['beta'])

    def test_fit_empty_and_fold_inputs_only(self):
        empty = r.fit_score_models(np.empty((0, 10)), np.empty(0), np.empty(0))
        for model in empty.values():
            self.assertTrue(model['no_labels_fallback'])
            np.testing.assert_array_equal(r.predict_score(model, np.ones((3, 10))*1e9), 0.)
        x = np.zeros((5, 10)); y = np.arange(5, dtype=float); weight = np.ones(5)
        train = np.array([True, True, True, False, False])
        fitted = r.fit_score_models(x[train], y[train], weight[train])
        x[~train] = 1e9; y[~train] = -1e9; weight[~train] = 1e9
        replay = r.fit_score_models(x[train], y[train], weight[train])
        for head in r.HEADS:
            for key in ('mean', 'std', 'beta', 'permutation'):
                np.testing.assert_array_equal(fitted[head][key], replay[head][key])

    def test_labels_unique_first_anchor_multisource_weights_and_prediction_isolation(self):
        data = inputs(('a', 'b', 'a', 'b', 'c', 'a'), q=3,
                      groups=[[0, 1], [2, 3, 4], [5]])
        data['support'][1, 1] = False
        pack = r.build_source_features(**data)
        original = deepcopy(pack)
        identities = np.array(['t0', 't0', 't0', 't1', 't1', 'unknown'])
        truth = np.zeros((3, 3, 4)); truth[..., 0] = 30.; truth[..., 1] = 120.
        valid = np.ones_like(truth, bool); valid[0, 2, :2] = False; truth[0, 2, :2] = np.nan
        labels = r.source_labels(pack, data['groups'], identities, np.array(['t0', 't1', 't2']), truth, valid)
        self.assertEqual(int(labels['labeled'].sum()), 2)
        np.testing.assert_array_equal(labels['labeled'][:2, 0], True)
        np.testing.assert_array_equal(labels['sample_weight'][:2, 0], [.5, .5])
        self.assertEqual(labels['y'][0, 0], 0.)
        self.assertGreater(labels['y'][1, 0], 0.)
        self.assertTrue(np.isnan(labels['y'][~labels['labeled']]).all())
        self.assertEqual(labels['audit']['duplicate_anchor_groups_without_training_labels'], 1)
        self.assertEqual(labels['audit']['unknown_anchor_groups'], 1)
        self.assertEqual(labels['audit']['target_without_group_anchor_count'], 2)
        self.assertEqual(labels['audit']['all_matched_reference_valid_query_count'], 2)
        truth[0, :2, 0] = 40.
        changed = r.source_labels(pack, data['groups'], identities, np.array(['t0', 't1', 't2']), truth, valid)
        self.assertNotEqual(changed['y'][0, 0], labels['y'][0, 0])
        for key in pack:
            if key != 'audit':
                np.testing.assert_array_equal(pack[key], original[key])
        # A fresh three-source target contributes total training weight one.
        three = inputs(q=1); p = r.build_source_features(**three)
        label = r.source_labels(p, [[0, 1, 2]], np.array(['t']*3), np.array(['t']),
            np.zeros((1, 1, 4)), np.ones((1, 1, 4), bool))
        np.testing.assert_allclose(label['sample_weight'][:, 0], [1/3]*3)
        self.assertAlmostEqual(label['sample_weight'].sum(), 1.)

    def test_empty_scene_and_invalid_inputs(self):
        data = inputs((), q=2)
        pack = r.build_source_features(**data)
        self.assertEqual(pack['features'].shape, (0, 2, 10))
        np.testing.assert_array_equal(pack['group_offsets'], [0])
        result = r.recover_source_policy(states=data['states'], support=data['support'], groups=[],
            pack=pack, scores=np.zeros((0, 2)), kind='equal', mean_states=np.empty((0, 2, 4)))
        self.assertEqual(result['group_states'].shape, (0, 2, 4))
        for update in ({'nearest_seconds': np.full((3, 3), -1.)},
                       {'sources': np.array(['a', '', 'b'])}, {'groups': [[0, 1]]},
                       {'native_count': np.array([1., np.nan, 2.])}):
            with self.assertRaises(ValueError):
                r.build_source_features(**(inputs() | update))
        with self.assertRaises(ValueError):
            r.source_weights(np.zeros((3, 2)), np.ones((3, 2), bool), [0, 2], kind='soft')
        with self.assertRaises(ValueError):
            r.fit_score_models(np.zeros((2, 10)), np.zeros(2), [1., 0.])
        with self.assertRaises(ValueError):
            r.source_weights(np.array([[np.nan]]), np.ones((1, 1), bool), [0, 1], kind='soft')


if __name__ == '__main__':
    unittest.main()
