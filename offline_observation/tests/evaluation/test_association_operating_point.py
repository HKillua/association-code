import unittest
import numpy as np
from tools.analysis.association_operating_point_pilot import calibrate,operating_point,THRESHOLD

class OperatingPointTests(unittest.TestCase):
    def test_all_scorers_obey_same_T_budget_and_reject_V(self):
        rows=[dict(split='T_core',labels=np.array([1,1,0,0],bool),WS=np.array([4.,2.,1.,0.]),CTX=np.array([6.,4.,3.,2.])) for _ in range(2)]
        limit=operating_point(rows,'WS',THRESHOLD)['m_pfa']
        own,curve=calibrate(rows,'WS',limit);ctx,other=calibrate(rows,'CTX',limit)
        self.assertLessEqual(ctx['metrics']['m_pfa'],limit);self.assertEqual(ctx['metrics']['m_pca'],own['metrics']['m_pca'])
        chosen=rows[0]['CTX']+(THRESHOLD-ctx['threshold'])
        np.testing.assert_array_equal(chosen>=THRESHOLD,rows[0]['CTX']>=ctx['threshold'])
        rows[0]['split']='V_select'
        with self.assertRaisesRegex(ValueError,'T-only'):calibrate(rows,'CTX',limit)
    def test_undefined_scene_is_never_silently_deleted(self):
        rows=[dict(split='T_core',labels=np.array([1,0],bool),WS=np.array([0.,1.]))]
        best,_=calibrate(rows,'WS',0.)
        self.assertIsNone(best)

if __name__=='__main__':unittest.main()
