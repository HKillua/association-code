import unittest
import numpy as np
from tests.integration.test_source_risk_pilot import scene_arrays
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from tools.analysis.low_support_attribution import arm_details,compare

class LowSupportTests(unittest.TestCase):
    def test_saved_single_source_replay_oracles_and_complete_transition_counts(self):
        from tools.analysis import association_context_pilot as p
        m,ev,_,_=scene_arrays('1');groups=[[0,1],[2,3]]
        # Both fragments in each group have one source; no head can select another.
        m['sources']=np.array(['s','s','t','t']);m['support'][0,0]=False;m['support'][1,0]=False
        from association.evaluation.fusion_controls import ordinary_fusion
        from association.evaluation import source_risk as risk
        mean=ordinary_fusion(m['states'],m['nearest_seconds'],m['support'],groups)['predictions']['mean_supported'][[0,2]]
        pack=risk.build_source_features(**{k:m[k] for k in p.source.MOTION_KEYS},groups=groups)
        raw=risk.recover_source_policy(states=m['states'],support=m['support'],groups=groups,pack=pack,scores=np.arange(pack['available'].size,dtype=float).reshape(pack['available'].shape),kind='hard',mean_states=mean)['group_states']
        reports,art=evaluate_fusion_methods({'A':emit_group_outputs(raw,groups,node_count=4)},prediction_speed_units={'A':'kn'},node_identities=ev['identities'],target_ids=ev['target_ids'],target_states=ev['target_states'],target_valid=ev['target_valid'],support=m['support'],nearest_seconds=m['nearest_seconds'],target_speed_unit='mps',split='T_core')
        a=arm_details(m,ev,art['A']);r=compare(a,a)
        self.assertGreater(a['invariance_queries'],0);self.assertEqual(r['strata']['all']['new_harm_over_1km'],0)
        self.assertEqual(sum(x['points'] for x in r['transitions'].values()),r['strata']['all']['points'])
        self.assertEqual(len(r['examples']),0)
        self.assertEqual(sum(r['strata'][k]['points'] for k in ('new_zero','new_one','new_multi')),r['strata']['all']['points'])

if __name__=='__main__':unittest.main()
