import unittest
import numpy as np
from tools.analysis.decoded_coverage_calibration import coverage,choose,aggregate

class DecodedCoverageTests(unittest.TestCase):
    def test_objective_and_coverage_are_distinct_on_same_eligible_grid(self):
        def row(t,r,d,points,targets):return dict(threshold=t,metrics=dict(raw=dict(pca=r,pfa=.1),decoded=dict(pca=d,pfa=.1),coverage=dict(points=points,targets=targets)))
        curve=[row(1,.9,.6,12,3),row(2,.8,.9,8,2),row(3,.7,.8,12,3)]
        limits=dict(raw=.2,decoded=.2);floor=dict(points=12,targets=3)
        self.assertEqual(choose(curve,limits,floor,'R')['threshold'],1)
        self.assertEqual(choose(curve,limits,floor,'D')['threshold'],2)
        self.assertEqual(choose(curve,limits,floor,'C')['threshold'],3)
        bad=row(4,1.,1.,12,3);bad['metrics']['decoded']['pfa']=None
        self.assertIsNone(choose([bad],limits,floor,'D'))
        bad['metrics']['decoded']['pfa']=.3;self.assertIsNone(choose([bad],limits,floor,'R'))

    def test_coverage_counts_unique_anchor_targets_and_valid_position_only(self):
        member=np.array([0,0,1,2]);ids=np.array(['a','b','a','c']);targets=np.array(['a','b','c','d']);valid=np.ones((4,3,4),bool);valid[2,0,:2]=False
        self.assertEqual(coverage(member,ids,targets,valid),dict(points=5,targets=2))
        rows=[dict(raw=dict(pca=.5,pfa=.2),decoded=dict(pca=.7,pfa=.1),coverage=dict(points=5,targets=2)),dict(raw=dict(pca=.9,pfa=.1),decoded=dict(pca=.6,pfa=None),coverage=dict(points=3,targets=1))]
        self.assertIsNone(aggregate(rows)['decoded']['pfa']);self.assertAlmostEqual(aggregate(rows)['raw']['pca'],.7)

if __name__=='__main__':unittest.main()
