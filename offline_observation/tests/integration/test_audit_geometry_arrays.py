import unittest
from tools.analysis.audit_geometry_arrays import rank_auc


class RankAuditTest(unittest.TestCase):
    def test_ties_and_direction(self):
        self.assertEqual(rank_auc([0, 1, 0, 1], [1, 1, 2, 3]), .625)
        self.assertEqual(rank_auc([0, 1], [0, 1]), 1.)
        self.assertEqual(rank_auc([0, 1], [1, 0]), 0.)

    def test_single_class_is_undefined(self):
        self.assertIsNone(rank_auc([], []))
        self.assertIsNone(rank_auc([1, 1], [0, 3]))
