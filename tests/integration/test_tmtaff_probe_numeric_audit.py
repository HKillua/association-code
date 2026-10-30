import unittest
import numpy as np
from tools.analysis.audit_tmtaff_training_probe import errors
from association.evaluation.fusion_controls import attribute_errors

class NumericAuditTests(unittest.TestCase):
    def test_saved_float32_and_independent_double_are_separate(self):
        prediction=np.asarray([[[39.12345,121.98765,3.,359.9],[39.45678,122.12345,5.,.1]]],np.float32)
        target=np.asarray([[[39.12445,121.98465,4.,.1],[39.45578,122.12445,4.,359.9]]],np.float64)
        original=errors(prediction,target,double=False);double=errors(prediction,target,double=True)
        expected=attribute_errors(prediction,target)
        for attr in original:np.testing.assert_array_equal(original[attr],expected[attr])
        self.assertEqual(double['position'].dtype,np.float64)
        self.assertGreater(np.max(np.abs(double['position']-original['position'])),1e-5)

if __name__=='__main__':unittest.main()
