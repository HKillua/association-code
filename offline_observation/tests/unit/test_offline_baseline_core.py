import unittest

import numpy as np

from tools.analysis.offline_baseline_core import predict_arms


def _scene(sources, spans, endpoints, *, supports=None, courses=None, query_longitudes=None):
    query = np.arange(64, dtype=float)
    spans = np.asarray(spans, dtype=float)
    positions = np.column_stack((np.zeros(spans.size), np.repeat(endpoints, 2)))
    offsets = np.arange(0, spans.size + 1, 2)
    courses = np.zeros((len(sources), 2)) if courses is None else np.asarray(courses, dtype=float)
    values = np.column_stack((positions, np.repeat(courses, 2, axis=0)))
    native_motion = np.broadcast_to(values[offsets[:-1], None, :], (len(sources), 64, 4)).copy()
    if query_longitudes is not None:
        longitudes = np.asarray(query_longitudes)
        native_motion[:, :longitudes.shape[1], 1] = longitudes
    if supports is None:
        supports = (query >= spans[:, :1]) & (query <= spans[:, 1:])
    return dict(nodes=np.arange(len(sources)), sources=np.asarray(sources), values=values,
                native=(query, spans.ravel(), positions, offsets), support=supports, native_motion=native_motion)


class OfflineBaselineCoreTests(unittest.TestCase):
    def test_itsa_wraps_longitude_and_preserves_chain_and_singletons(self):
        scene = _scene([1, 1, 1, 1], [(8, 9), (0, 1), (4, 5), (3, 4)],
                       [-179.998, 179.999, -179.999, 10.0])
        for output in predict_arms(scene).values():
            self.assertEqual(set(output['itsa_links']), {(1, 2), (2, 0)})
            self.assertEqual(output['mstsa_links'], [])
            np.testing.assert_array_equal(output['membership'], [0, 0, 0, 1])
            self.assertEqual(output['audit']['singleton_groups'], 1)

    def test_cv_changes_ranking_with_same_candidates(self):
        scene = _scene([1, 1, 1], [(0, 1), (101, 120), (101, 120)],
                       [0.0, 0.00425, 0.00925], courses=[(20, 90)] * 3)
        arms = predict_arms(scene)
        self.assertEqual(arms['END_DTW']['itsa_links'], [(0, 1)])
        self.assertEqual(arms['CV_DTW']['itsa_links'], [(0, 2)])
        self.assertEqual(arms['END_DTW']['audit']['itsa_candidates'], 2)
        self.assertEqual(arms['CV_DTW']['audit']['itsa_candidates'], 2)

    def test_banded_dtw_and_sync_share_admissibility(self):
        scene = _scene([1, 2], [(0, 3), (0, 3)], [0.0, 0.0],
                       query_longitudes=[[0.0, 0.0, 0.22, 0.22], [0.0, 0.22, 0.22, 0.22]])
        arms = predict_arms(scene)
        self.assertEqual(arms['END_DTW']['audit']['mstsa_candidates'], 1)
        self.assertEqual(arms['END_SYNC']['audit']['mstsa_candidates'], 1)
        self.assertEqual(arms['END_DTW']['mstsa_links'], [((0,), (1,))])
        self.assertEqual(arms['END_SYNC']['mstsa_links'], [])

    def test_stitch_creates_no_support_for_cross_source_candidates(self):
        scene = _scene([1, 1, 2], [(0, 2), (6, 8), (2, 5)], [0.0] * 3)
        for output in predict_arms(scene).values():
            self.assertEqual(output['itsa_links'], [(0, 1)])
            self.assertEqual(output['audit']['mstsa_time_qualified'], 0)
            self.assertEqual(output['mstsa_links'], [])

    def test_mstsa_preserves_source_uniqueness_across_transitive_merges(self):
        scene = _scene([1, 2, 3, 1], [(0, 5)] * 4, [0.0, 0.001, 0.002, 0.0035])
        for output in predict_arms(scene).values():
            np.testing.assert_array_equal(output['membership'], [0, 0, 0, 1])
            self.assertEqual(len(output['mstsa_links']), 2)
