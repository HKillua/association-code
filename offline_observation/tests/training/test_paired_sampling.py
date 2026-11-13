"""Strict paired draws, small/tied pools and completed-update exposure."""
import copy
import json
import unittest

import numpy as np

from association.training.paired_sampling import PairedNegativeIntervention, validate_sampling_config
from association.training.sampling import BalancedScenePairSampler


def config(mode):
    return dict(mode=mode, intervention_seed=20261023, hard_slots=64, hard_fraction=.1)


def scene(name, negative, positive=1):
    size = negative+positive
    return dict(scene_id=name, split='T_core',
        pairs=np.column_stack((np.zeros(size, np.int64), np.arange(1, size+1))),
        labels=np.r_[np.ones(positive, np.float32), np.zeros(negative, np.float32)],
        base_scores=-np.arange(size, dtype=np.float64)/1000.)


def assert_groups(test, left, right):
    test.assertEqual([index for index, _ in left], [index for index, _ in right])
    for (_, first), (_, second) in zip(left, right):
        np.testing.assert_array_equal(first, second)


class PairedSamplingTests(unittest.TestCase):
    def test_one_negative_can_fill_64_hard_slots_without_changing_any_pair(self):
        train = [scene('one', 1)]
        sampler = BalancedScenePairSampler(train, 1)
        intervention = PairedNegativeIntervention(train, config('hard_negative'))
        raw = sampler.sample(256)
        actual, record = intervention.apply(raw)
        assert_groups(self, raw, actual)
        self.assertFalse(np.shares_memory(raw[0][1], actual[0][1]))
        intervention.record_completed_batch(actual, record)
        audit = intervention.audit()
        self.assertEqual(audit['hard_source_slots'], 64)
        self.assertEqual(audit['changed_slots'], 0)
        self.assertEqual(audit['final_negative_hard_pool_hits'], 128)
        self.assertEqual(audit['uniform_negative_hard_pool_hits'], 64)
        self.assertEqual(audit['hard_batches_with_replacement'], 1)
        self.assertEqual(audit['distinct_negative_pair_count'], 1)
        self.assertEqual(audit['repeated_negative_draw_count'], 127)
        for source in ('negative_uniform', 'negative_hard'):
            self.assertEqual(audit['sources'][source]['draw_count'], 64)
            self.assertEqual(audit['sources'][source]['distinct_pair_count'], 1)
            self.assertEqual(audit['sources'][source]['repeated_draw_count'], 63)
        json.dumps(audit, allow_nan=False)

    def test_exact_ceil_pool_tie_break_and_replacement_boundary(self):
        for negative, expected_pool, replace in ((630, 63, 1), (631, 64, 0)):
            train = [scene(str(negative), negative)]
            sampler = BalancedScenePairSampler(train, 8)
            intervention = PairedNegativeIntervention(train, config('hard_negative'))
            raw = sampler.sample(256)
            actual, record = intervention.apply(raw)
            slots = np.flatnonzero(train[0]['labels'][raw[0][1]] == 0)[:64]
            draws = actual[0][1][slots]
            self.assertTrue(np.isin(draws, np.arange(1, expected_pool+1)).all())
            if replace:
                self.assertLess(len(np.unique(draws)), 64)
            else:
                self.assertEqual(len(np.unique(draws)), 64)
            intervention.record_completed_batch(actual, record)
            self.assertEqual(intervention.audit()['hard_pool_pair_count'], expected_pool)
            self.assertEqual(intervention.audit()['hard_batches_with_replacement'], replace)
        tied = scene('tied', 11)
        tied['base_scores'][1:4] = 0.
        intervention = PairedNegativeIntervention([tied], config('hard_negative'))
        raw = BalancedScenePairSampler([tied], 2).sample(256)
        actual, record = intervention.apply(raw)
        negative_slots = np.flatnonzero(tied['labels'][raw[0][1]] == 0)
        self.assertEqual(set(actual[0][1][negative_slots[:64]]), {1, 2})
        # Highest-scoring positive index zero must not enter the hard pool.
        intervention.record_completed_batch(actual, record)
        self.assertEqual(intervention.audit()['hard_pool_pair_count'], 2)

    def test_overlap_with_retained_uniform_is_allowed_without_deduplication(self):
        train = [scene('overlap', 640)]
        raw = [(0, np.r_[np.zeros(128, np.int64), np.arange(65, 129), np.arange(1, 65)])]
        intervention = PairedNegativeIntervention(train, config('hard_negative'))
        actual, record = intervention.apply(raw)
        np.testing.assert_array_equal(actual[0][1][192:], raw[0][1][192:])
        self.assertEqual(set(actual[0][1][128:192]), set(range(1, 65)))
        intervention.record_completed_batch(actual, record)
        audit = intervention.audit()
        self.assertEqual(audit['changed_slots'], 64)
        self.assertEqual(audit['final_negative_hard_pool_hits'], 128)
        self.assertEqual(audit['distinct_negative_pair_count'], 64)
        self.assertEqual(audit['repeated_negative_draw_count'], 64)
        self.assertEqual(audit['sources']['negative_uniform']['distinct_pair_count'], 64)
        self.assertEqual(audit['sources']['negative_hard']['distinct_pair_count'], 64)
        self.assertAlmostEqual(audit['negative_pool_scores']['mean'], -.3205)
        self.assertAlmostEqual(audit['hard_pool_scores']['mean'], -.0325)
        for source in ('negative_uniform', 'negative_hard'):
            scores = audit['sources'][source]['ols_scores']
            self.assertAlmostEqual(scores['mean'], -.0325)
            self.assertEqual(scores['minimum'], -.064)
            self.assertEqual(scores['maximum'], -.001)
            self.assertEqual(sum(scores['histogram_counts']), 64)

    def test_1000_step_pairing_preserves_original_sampler_and_uniform_content(self):
        train = [scene('large', 710, 5), scene('small', 4, 1), scene('positive', 0, 2), scene('empty', 0, 0)]
        raw_a = BalancedScenePairSampler(train, 20261022)
        raw_b = BalancedScenePairSampler(train, 20261022)
        a = PairedNegativeIntervention(train, config('uniform'))
        b = PairedNegativeIntervention(train, config('hard_negative'))
        for _ in range(1000):
            first, second = raw_a.sample(256), raw_b.sample(256)
            assert_groups(self, first, second)
            before = [(index, indices.copy()) for index, indices in second]
            uniform, record_a = a.apply(first)
            hard, record_b = b.apply(second)
            assert_groups(self, first, uniform)
            assert_groups(self, before, second)
            self.assertEqual([index for index, _ in hard], [index for index, _ in first])
            remaining = 64
            for (index, original), (_, actual) in zip(before, hard):
                positives = train[index]['labels'][original].astype(bool)
                np.testing.assert_array_equal(original[positives], actual[positives])
                slots = np.flatnonzero(~positives)
                altered = min(remaining, len(slots))
                np.testing.assert_array_equal(original[slots[altered:]], actual[slots[altered:]])
                remaining -= altered
            self.assertEqual(remaining, 0)
            self.assertEqual(raw_a._rng.bit_generator.state, raw_b._rng.bit_generator.state)
            a.record_completed_batch(uniform, record_a)
            b.record_completed_batch(hard, record_b)
            raw_a.record_completed_batch(uniform)
            raw_b.record_completed_batch(hard)
        audit_a, audit_b = a.audit(), b.audit()
        self.assertEqual(audit_a['completed_batch_count'], 1000)
        self.assertEqual(audit_b['completed_batch_count'], 1000)
        self.assertEqual(audit_a['hard_source_slots'], 0)
        self.assertEqual(audit_b['hard_source_slots'], 64000)
        self.assertEqual(audit_a['changed_slots'], 0)
        self.assertEqual(audit_b['sampled_pair_count'], 256000)
        self.assertEqual(audit_b['positive_slots'], 128000)
        self.assertEqual(audit_b['negative_slots'], 128000)
        for scene_a, scene_b, completed in zip(audit_a['scenes'], audit_b['scenes'], raw_b.coverage()['scenes']):
            self.assertEqual(scene_a['completed_batch_count'], scene_b['completed_batch_count'])
            self.assertEqual(scene_a['positive_slots'], scene_b['positive_slots'])
            self.assertEqual(scene_a['sources']['positive_uniform'], scene_b['sources']['positive_uniform'])
            self.assertEqual(scene_b['distinct_negative_pair_count'], completed['distinct_negative_count'])
        self.assertEqual(audit_a['sources']['negative_uniform']['draw_count'], 128000)
        self.assertEqual(audit_b['sources']['negative_uniform']['draw_count'], 64000)
        self.assertEqual(audit_b['sources']['negative_hard']['draw_count'], 64000)
        self.assertEqual(audit_b['scenes'][-1]['negative_pool_scores']['count'], 0)
        self.assertIsNone(audit_b['scenes'][-1]['hard_pool_scores']['mean'])
        json.dumps(audit_b, allow_nan=False)

    def test_failed_update_does_not_commit_and_records_cannot_be_reused_or_mismatched(self):
        train = [scene('small', 20)]
        sampler = BalancedScenePairSampler(train, 2)
        intervention = PairedNegativeIntervention(train, config('hard_negative'))
        initial = intervention.audit()
        groups, record = intervention.apply(sampler.sample(256))
        self.assertEqual(intervention.audit(), initial)  # a failed optimizer would stop here
        tampered = [(index, indices.copy()) for index, indices in groups]
        tampered[0][1][0] = (tampered[0][1][0]+1) % 21
        with self.assertRaisesRegex(ValueError, 'differ'):
            intervention.record_completed_batch(tampered, record)
        self.assertEqual(intervention.audit(), initial)
        intervention.record_completed_batch(groups, record)
        self.assertEqual(intervention.audit()['completed_batch_count'], 1)
        with self.assertRaisesRegex(ValueError, 'uncommitted'):
            intervention.record_completed_batch(groups, record)
        other = PairedNegativeIntervention(train, config('hard_negative'))
        _, foreign = other.apply(sampler.sample(256))
        with self.assertRaisesRegex(ValueError, 'uncommitted'):
            intervention.record_completed_batch(groups, foreign)

    def test_rejects_validation_pools_nonfinite_scores_and_unfrozen_design(self):
        train = [scene('small', 1)]
        for key, value in (('mode', 'adaptive'), ('hard_slots', 63), ('hard_slots', True),
                           ('hard_fraction', .2), ('hard_fraction', float('nan')),
                           ('intervention_seed', -1), ('intervention_seed', True)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_sampling_config(dict(config('uniform'), **{key: value}))
        for key, value in (('split', 'V_select'), ('base_scores', np.array([0., np.nan]))):
            changed = copy.deepcopy(train)
            changed[0][key] = value
            with self.assertRaises(ValueError):
                PairedNegativeIntervention(changed, config('uniform'))
        intervention = PairedNegativeIntervention(train, config('uniform'))
        with self.assertRaisesRegex(ValueError, '256'):
            intervention.apply(BalancedScenePairSampler(train, 1).sample(128))


if __name__ == '__main__':
    unittest.main()
