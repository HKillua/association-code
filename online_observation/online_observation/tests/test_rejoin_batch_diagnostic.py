import unittest

from online_observation.rejoin_batch_diagnostic import _capture, _report_at


def _report(stamp, actual=None, masked=None):
    return dict(stamp=stamp, geometry={"actual": 1., "masked": -1.},
                shape=dict(actual=actual, masked=masked, available_at=stamp, end=stamp-2,
                           current_used=actual != masked, end_after_start=False, post_start_used=False))


class RejoinBatchTest(unittest.TestCase):
    def test_same_batch_mask_and_strict_cache_use_current_geometry(self):
        earlier = _report(2., actual={"two_block": 3.})
        current = _report(4., actual={"two_block": -1., "net": 2.}, masked={"two_block": 2.})
        result = _capture([earlier, current])
        self.assertEqual(result["cache_stamp"], 2.)
        self.assertEqual(result["decisions"], dict(original=False, geometry=True, geometry_masked=False,
                         net_geometry=True, two_block_geometry=False, masked_shape_geometry=True,
                         cached_shape_geometry=True))

    def test_missing_previous_score_does_not_backfill_and_later_reports_fail(self):
        earlier = _report(2., actual={"two_block": 3.})
        current = _report(4., actual={"two_block": -1., "net": 2.}, masked={"two_block": 2.})
        result = _capture([earlier, _report(3.), current])
        self.assertEqual(result["cache_stamp"], 3.)
        self.assertIsNone(result["decisions"]["cached_shape_geometry"])
        with self.assertRaises(ValueError):
            _report_at([earlier, current], 3.)
