import unittest
import numpy as np
from tools.analysis.association_decision_audit import dissect,THRESHOLD

class DecisionAuditTest(unittest.TestCase):
    def test_oracle_separates_search_gap_from_objective_conflict(self):
        pairs=np.column_stack(np.triu_indices(3,1));identity=np.array([0,0,1]);groups=np.zeros(3,int)
        for w,expected in [([2,-3,1],2.),([2,3,-1],-2.)]:
            r=dissect(pairs,THRESHOLD+np.array(w),groups,identity,np.arange(3),np.ones((3,2),bool))
            self.assertAlmostEqual(r['oracle_split_J_change'],expected)
            self.assertEqual(r['counts']['cross']['false_raw_negative_forced'],1)
            self.assertEqual(r['counts']['cross']['true_raw_kept'],1)
    def test_full_partition_of_pairs_and_suppressed_positive(self):
        pairs=np.column_stack(np.triu_indices(3,1));r=dissect(pairs,THRESHOLD+np.array([1,1,-1]),np.arange(3),np.array([0,0,1]),np.array([0,1,1]),np.zeros((3,2),bool))
        z=r['counts']['all'];self.assertEqual(z['true_raw_suppressed'],1)
        self.assertEqual(sum(v for k,v in z.items() if k.startswith(('true_','false_'))),3)
        self.assertEqual(r['counts']['cross_joint']['pairs'],0)

if __name__=='__main__':unittest.main()
