"""Software contracts only: no fixture here is scientific MTAD evidence."""
from copy import deepcopy
import unittest

import numpy as np

from association.evaluation import observation_quality as q
from association.evaluation import source_risk as r
from association.evaluation.query_contribution import recover_group_states
from tests.integration.test_source_risk import inputs


def native_fixture(positions, times):
    values = np.zeros((len(times), 4))
    values[:, :2] = positions
    cache = dict(node_ids=np.array(['node']), sources=np.array(['s']), query_seconds=np.array([0., 1.]),
        event_offsets=np.array([0, len(times)]), event_times_seconds=np.asarray(times, float),
        event_values=values, event_valid=np.ones(values.shape, bool),
        event_node_ids=np.array(['node']*len(times)), event_canonical_indices=np.arange(len(times)))
    motion = {key: cache[key].copy() for key in ('node_ids', 'sources', 'query_seconds')}
    motion['fit_rmse_m'] = np.array([2.])
    return cache, motion


class ObservationQualityTests(unittest.TestCase):
    def test_native_loo_irregular_hand_case_and_constant_translation_blindness(self):
        # North positions [0, 3, 3]m at seconds [0,1,3]: neighbor prediction=1m.
        degree = 180./(np.pi*q.RADIUS)
        cache, motion = native_fixture(np.column_stack((np.array([0., 3., 3.])*degree, np.zeros(3))), [0., 1., 3.])
        result = q.native_quality(native_cache=cache, motion=motion)
        self.assertAlmostEqual(result['loo_proxy_m2'][0], 4./(1.+4./9.+1./9.), places=12)
        np.testing.assert_array_equal(result['ols_proxy_m2'], [4.])
        np.testing.assert_array_equal(result['loo_residual_count'], [1])
        # Translation along latitude preserves this north-only residual, while
        # its absolute reference error can increase greatly. No truth is input.
        shifted = deepcopy(cache)
        shifted['event_values'][:, 0] += .1
        other = q.native_quality(native_cache=shifted, motion=motion)
        np.testing.assert_allclose(other['loo_proxy_m2'], result['loo_proxy_m2'], rtol=1e-10)
        self.assertTrue(other['loo_known'][0])

    def test_native_dates_and_missing_proxy(self):
        cache, motion = native_fixture([[0., 179.], [0., -180.], [0., -179.]], [0., 1., 2.])
        result = q.native_quality(native_cache=cache, motion=motion)
        self.assertLess(result['loo_proxy_m2'][0], 1e-16)
        cache, motion = native_fixture([[1., 2.], [1.1, 2.1]], [0., 1.])
        result = q.native_quality(native_cache=cache, motion=motion)
        self.assertFalse(result['loo_known'][0])
        self.assertEqual(result['loo_residual_count'][0], 0)
        bad = deepcopy(cache)
        bad['identities'] = np.array(['target'])
        with self.assertRaisesRegex(ValueError, 'physical-only'):
            q.native_quality(native_cache=bad, motion=motion)
        bad = deepcopy(cache)
        bad['event_times_seconds'][1] = 0.
        with self.assertRaisesRegex(ValueError, 'time/order'):
            q.native_quality(native_cache=bad, motion=motion)

    def test_proxy_mean_not_independent_sample_variance_and_floors(self):
        data = inputs(('x', 'x', 'y'), q=3)
        data['support'][1, 1] = False
        data['support'][:, 2] = False
        pack = r.build_source_features(**data)
        proxy = q.source_proxy(node_proxy_m2=np.array([1., 9., 20.]), node_known=np.ones(3, bool),
                               support=data['support'], pack=pack)
        np.testing.assert_array_equal(proxy['quality_m2'], [[5., 1., 0.], [20., 20., 0.]])
        one = q.observation_weights(proxy=proxy, pack=pack, nearest_seconds=data['nearest_seconds'],
                                    support=data['support'], kind='inverse', floor_m2=1.)
        hundred = q.observation_weights(proxy=proxy, pack=pack, nearest_seconds=data['nearest_seconds'],
                                        support=data['support'], kind='inverse', floor_m2=100.)
        np.testing.assert_allclose(one['source_weights'][:, 0], [.8, .2])
        np.testing.assert_allclose(hundred['source_weights'][:, 0], [.5, .5])
        np.testing.assert_allclose(one['node_weights'][:, 0], [.4, .4, .2])
        np.testing.assert_array_equal(one['source_weights'][:, 2], 0.)
        with self.assertRaisesRegex(ValueError, 'fixed floor'):
            q.observation_weights(proxy=proxy, pack=pack, nearest_seconds=data['nearest_seconds'],
                                  support=data['support'], kind='inverse', floor_m2=.01)

    def test_unknown_any_available_source_falls_back_without_dropping(self):
        data = inputs(('x', 'x', 'y'), q=3)
        data['support'][1, 1:] = False
        pack = r.build_source_features(**data)
        proxy = q.source_proxy(node_proxy_m2=np.array([1., 0., 100.]), node_known=np.array([True, False, True]),
                               support=data['support'], pack=pack)
        weights = q.observation_weights(proxy=proxy, pack=pack, nearest_seconds=data['nearest_seconds'],
                                       support=data['support'], kind='inverse', floor_m2=1.)
        np.testing.assert_array_equal(weights['uninformative_fallback'], [[True, False, False]])
        np.testing.assert_allclose(weights['source_weights'][:, 0], [.5, .5])
        self.assertGreater(weights['source_weights'][0, 1], .99)

    def test_general_k_source_rename_and_row_permutation(self):
        for k in (1, 2, 3, 4):
            data = inputs(tuple(f's{i}' for i in range(k)), q=4)
            pack = r.build_source_features(**data)
            proxy = q.source_proxy(node_proxy_m2=np.arange(k, dtype=float)**2, node_known=np.ones(k, bool),
                                   support=data['support'], pack=pack)
            weights = q.observation_weights(proxy=proxy, pack=pack, nearest_seconds=data['nearest_seconds'],
                                            support=data['support'], kind='inverse', floor_m2=1.)
            renamed = deepcopy(data)
            renamed['sources'] = np.array([f'reverse{k-i}' for i in range(k)])
            renamed_pack = r.build_source_features(**renamed)
            other = q.observation_weights(proxy=proxy, pack=renamed_pack, nearest_seconds=data['nearest_seconds'],
                                          support=data['support'], kind='inverse', floor_m2=1.)
            np.testing.assert_array_equal(weights['node_weights'], other['node_weights'])
            perm = np.arange(k)[::-1]
            reordered = deepcopy(pack)
            for name in ('available', 'supported_counts'):
                reordered[name] = pack[name][perm]
            reordered['node_to_source_row'] = np.argsort(perm)[pack['node_to_source_row']]
            reordered_proxy = {name: value[perm] for name, value in proxy.items()}
            other = q.observation_weights(proxy=reordered_proxy, pack=reordered, nearest_seconds=data['nearest_seconds'],
                                          support=data['support'], kind='inverse', floor_m2=1.)
            np.testing.assert_array_equal(weights['node_weights'], other['node_weights'])

    def test_nearest_ties_and_direct_node_circular_recovery(self):
        data = inputs(('x', 'x', 'y'), q=2)
        data['states'][:, :, 1] = np.array([0., 120., 120.])[:, None]
        data['nearest_seconds'][:] = 1.
        data['support'][:, 1] = False
        pack = r.build_source_features(**data)
        proxy = q.source_proxy(node_proxy_m2=np.ones(3), node_known=np.ones(3, bool), support=data['support'], pack=pack)
        weights = q.observation_weights(proxy=proxy, pack=pack, nearest_seconds=data['nearest_seconds'],
                                       support=data['support'], kind='nearest')
        np.testing.assert_array_equal(weights['source_weights'][:, 0], [.5, .5])
        mean = data['states'][[0]].copy()
        actual = q.recover_observation_policy(states=data['states'], support=data['support'], groups=data['groups'],
            pack=pack, source_weights=weights['source_weights'], mean_states=mean)
        expected = recover_group_states(data['states'], weights['node_weights'], data['groups'])['group_states']
        np.testing.assert_array_equal(actual['group_states'][..., :2], expected[..., :2])
        # Averaging the normalized source angles first would give 90 degrees,
        # while direct original-node mass retains source1's shorter resultant.
        self.assertGreater(abs(actual['group_states'][0, 0, 1]-90.), 10.)
        np.testing.assert_array_equal(actual['group_states'][..., 2:], mean[..., 2:])
        np.testing.assert_array_equal(actual['group_states'][:, 1], mean[:, 1])

    def test_constant_proxy_is_exact_equal_and_input_validation(self):
        data = inputs(('a', 'b', 'a'), q=3)
        pack = r.build_source_features(**data)
        proxy = q.source_proxy(node_proxy_m2=np.ones(3), node_known=np.ones(3, bool), support=data['support'], pack=pack)
        kws = dict(proxy=proxy, pack=pack, nearest_seconds=data['nearest_seconds'], support=data['support'])
        equal = q.observation_weights(**kws, kind='equal')
        inverse = q.observation_weights(**kws, kind='inverse', floor_m2=1.)
        np.testing.assert_array_equal(equal['node_weights'], inverse['node_weights'])
        bad = deepcopy(pack)
        bad['supported_counts'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'binding'):
            q.source_proxy(node_proxy_m2=np.ones(3), node_known=np.ones(3, bool), support=data['support'], pack=bad)


if __name__ == '__main__':
    unittest.main()
