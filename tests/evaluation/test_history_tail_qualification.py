import unittest
import numpy as np
from tests.integration.test_source_risk_pilot import scene_arrays
from tools.analysis.history_tail_qualification import analyze,summarize

class HistoryTailQualificationTests(unittest.TestCase):
    def test_native_age_opportunity_purity_and_duplicate_accounting(self):
        m,ev,_,_=scene_arrays('1');m['support'][:]=False;m['support'][0,1]=True
        # A valid bad selection with a better in-group state; a duplicate is not relabeled.
        member=np.array([0,0,1,2]);base=np.stack([m['states'][0],m['states'][2],m['states'][3]])
        m['states'][1,:,0]+=.1;candidate=base.copy();candidate[0,:,:2]=m['states'][1,:,:2]
        events=dict(event_offsets=np.arange(5)*2,event_times_seconds=np.tile([10.,20.],4),node_ids=m['node_ids'])
        weights=np.zeros((4,64));weights[1]=1;weights[2:]=1
        policy={'HARD__scores':np.zeros((4,64)),'HARD__features':np.zeros((4,64,12)),'HARD__count':np.zeros((3,64),int),'HO__weights':weights};policy['HARD__count'][0,1]=1
        rows,pop=analyze(m,events,ev,member,base,candidate,policy,'HARD',dict(mean=np.zeros(12),std=np.ones(12)))
        self.assertEqual(pop['duplicate_groups'],1);self.assertEqual(len(rows),127)
        bad=next(r for r in rows if r['target_index']==0 and r['query_index']==0)
        self.assertTrue(bad['oracle_avoids_harm']);self.assertTrue(bad['base_has_node']);self.assertTrue(bad['before']);self.assertEqual(bad['extrapolation_seconds'],10)
        after=next(r for r in rows if r['target_index']==0 and r['query_index']==3);self.assertTrue(after['after']);self.assertEqual(after['extrapolation_seconds'],10)
        self.assertTrue(next(r for r in rows if r['target_index']==0 and r['query_index']==1)['pure'])
        s=summarize(rows);self.assertGreater(s['pure_one']['harm']['points'],0)
        # All current candidates can be worse than a retained composite/base state.
        m['states'][0,:,0]+=.2
        rows,_=analyze(m,events,ev,member,base,candidate,policy,'HARD',dict(mean=np.zeros(12),std=np.ones(12)))
        bad=next(r for r in rows if r['target_index']==0 and r['query_index']==0)
        self.assertFalse(bad['oracle_avoids_harm']);self.assertFalse(bad['base_has_node'])

if __name__=='__main__':unittest.main()
