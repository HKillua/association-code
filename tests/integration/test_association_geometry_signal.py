"""128 geometry/cache behavior only; synthetic fixtures are not MTAD results."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration.test_query_contribution_control import cache_fixture
from tools.analysis import association_geometry_signal as p


class AssociationGeometryTests(unittest.TestCase):
    def fixture(self):
        states = np.zeros((4, 64, 4))
        states[..., 0] = np.array([.1, .1001, .1002, .2])[:, None]
        states[..., 1] = np.array([179.999, -179.999, 179.998, 179.])[:, None]
        support = np.ones((4, 64), bool)
        support[1, :3] = False
        pairs = np.stack(np.triu_indices(4, 1), axis=-1)
        return states, support, np.array(['a', 'b', 'a', 'b']), [[0, 1], [2], [3]], pairs, np.zeros(6), 1.

    def test_same_candidates_full_pair_information_and_negative_controls(self):
        private, audit = p.construct_features(*self.fixture())
        offsets, members = private['candidate_offsets'], private['candidate_node_indices']
        candidates = [members[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
        self.assertEqual(candidates, [[0, 1], [2], [3], [0, 1, 2], [0, 1, 3]])
        np.testing.assert_array_equal(private['candidate_is_original'], [True, True, True, False, False])
        self.assertEqual(audit['candidate_node_coverage_count'], 4)
        self.assertGreater(audit['two_member_pair_query_checks'], 0)
        self.assertGreater(audit['euclidean_query_checks'], 0)
        self.assertLessEqual(audit['maximum_two_member_difference_m'], p.TWO_ATOL_M)
        self.assertEqual(private['pair_distance_m'].shape, (6, 64))
        self.assertTrue(np.isnan(private['features'][1, p.FEATURE_NAMES.index('loo_median_m')]))
        evidence = p.evaluate_features(private, np.array(['x', 'x', 'x', 'y']))
        evidence['original'] = private['candidate_is_original']
        report = p.summarize(evidence)
        self.assertEqual(report['all']['candidate_count'], 5)
        self.assertEqual(report['all']['pure_count'], 4)
        self.assertEqual(report['all']['mixed_count'], 1)
        self.assertEqual(report['size1']['scorers']['LOO_MEDIAN']['defined_count'], 0)

    def test_ties_and_missing_support_are_not_failed_or_negative_labels(self):
        self.assertEqual(p.auc([True, False], [0., 0.]), .5)
        self.assertEqual(p.auc([True, False], [1., 0.]), 1.)
        self.assertIsNone(p.auc([True, True], [1., 0.]))
        args = list(self.fixture()); args[1][:] = False
        private, audit = p.construct_features(*args)
        evidence = p.evaluate_features(private, np.array(['x', 'x', '', 'y']))
        report = p.summarize(evidence)
        self.assertEqual(report['all']['scorers']['LOO_MEDIAN']['defined_count'], 0)
        self.assertGreater(report['all']['unknown_label_count'], 0)
        self.assertEqual(audit['two_member_pair_query_checks'], 0)
        with self.assertRaises(TypeError):
            p.construct_features(*args, identities=np.array(['x']*4))
        args[4] = args[4][::-1]
        with self.assertRaises(ValueError):
            p.construct_features(*args)

    def test_cached_runner_freezes_features_before_labels_and_preserves_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                old_args, meta = cache_fixture(root)
                args = SimpleNamespace(source_system=old_args.source_system,
                    partition_run=old_args.partition_run, manifest=old_args.manifest,
                    output=root/'geometry-out', cpu_threads=1)
                if fail:
                    cache = meta['association']/'final-scores.npz'
                    with np.load(cache, allow_pickle=False) as saved:
                        payload = {key: saved[key] for key in saved.files}
                    payload['2_labels'] = 1-payload['2_labels']
                    np.savez_compressed(cache, **payload)
                with patch.object(p.old, 'metadata', return_value=meta):
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'post-freeze identity'):
                            p.run(args)
                        state = p.old.frozen.read(args.output/'run.json')
                        self.assertEqual((state['status'], state['completed_scenes']), ('FAILED', 1))
                        self.assertTrue((args.output/'private/features-2.npz').exists())
                        self.assertFalse((args.output/'private/evaluation-2.npz').exists())
                        self.assertFalse((args.output/'summary.json').exists())
                    else:
                        result = p.run(args)
                        self.assertEqual(result['scene_ids'], ['1', '2'])
                        self.assertTrue((args.output/'private/features-1.npz').exists())
                        self.assertFalse(p.old.frozen.read(args.output/'run.json')['reference_array_fields_loaded'])
                        self.assertEqual(result['model_forwards'], 0)
                    with self.assertRaises(FileExistsError):
                        p.run(args)


if __name__ == '__main__':
    unittest.main()
