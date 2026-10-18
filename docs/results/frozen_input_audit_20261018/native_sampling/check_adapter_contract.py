"""Synthetic assembly check only. No ZIP or model is opened."""
import importlib.util
from pathlib import Path
import time
import numpy as np

path=Path(__file__).with_name('audit_native_sampling.py')
spec=importlib.util.spec_from_file_location('native_sampling_audit',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from association.data import Scene, Track

tracks=[Track('fixture:9001:0','9001','0','a-b','a',np.array([0.,5.,10.]),
              np.array([[10.,20.,1.,30.],[10.1,20.1,1.,30.],[10.2,20.2,1.,30.]])),
        Track('fixture:9002:1','9002','1','a-b','a',np.array([2.,3.]),
              np.array([[10.,20.,1.,30.],[10.1,20.1,1.,30.]]))]
scene=Scene('fixture',tracks,dict(quarantined_nodes=0,quarantine_reasons={},quarantined=[]))
class FakeArchive:
    def load_scene(self,sid):
        assert sid=='fixture'
        return scene
module.DEADLINE=time.monotonic()+10
acc=module.accumulator()
row=module.audit_scene(FakeArchive(),'T_core','fixture',acc)
assert row['valid_tracks']==2 and row['positive_pairs']==1 and row['negative_pairs']==0
assert [x['original_candidate_index'] for x in row['tracks']]==[0,1]
assert acc['counts']['public_timed_adapter_scenes']==1
assert all(item['counts']['endpoints_lost']==0 for item in acc['caps'].values())
print('PASS: public timed adapter, original intervals, candidate ordering, cap endpoints, time-only FP32 checks; synthetic only')
