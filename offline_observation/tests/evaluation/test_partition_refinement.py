import unittest
import numpy as np
from association.evaluation.partition_refinement import refine_partition

class RefinementTests(unittest.TestCase):
    def test_split_rejects_negative_members_and_keeps_singletons(self):
        pairs=np.column_stack(np.triu_indices(3,1));scores=np.array([5.,-4.,-4.])
        m,stats,trace=refine_partition(3,pairs,scores,np.zeros(3,int),threshold=0.)
        np.testing.assert_array_equal(m,[0,0,1]);self.assertEqual(stats['final_J']-stats['initial_J'],8.);self.assertTrue(stats['locally_optimal'])
        self.assertEqual(trace[0]['node'],2)
        m,s,_=refine_partition(1,np.empty((0,2),int),[],[0],threshold=0.)
        self.assertEqual(s['remaining_best_gain'],0.);self.assertTrue(s['locally_optimal'])
    def test_move_can_rejoin_another_group_without_dropping_any_node(self):
        p=np.column_stack(np.triu_indices(4,1));w=np.array([2.,4.,3.,-3.,-3.,5.])
        m,s,t=refine_partition(4,p,w,[0,0,1,1],threshold=0.,max_moves=1)
        self.assertEqual(t[0]['node'],0);self.assertEqual(m[0],m[2]);self.assertNotEqual(m[0],m[1]);self.assertGreater(s['final_J'],s['initial_J']);self.assertEqual(len(m),4)

if __name__=='__main__':unittest.main()
