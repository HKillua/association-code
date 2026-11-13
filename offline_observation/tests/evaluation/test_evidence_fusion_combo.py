import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from association.evaluation.evidence_fusion_combo import (
    predict_partition, replay_check)
from association.models.evidence_fusion import EvidenceFusion
from association.training.evidence_fusion import prepare_evidence_scenes, evaluate_evidence_fusion
from tools.experiments.evidence_fusion_fit import synthetic_scenes


class FrozenEvidenceComboTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        _, validation = synthetic_scenes(length=64, max_observations=4)
        self.scene = prepare_evidence_scenes(validation)[0]
        self.scene.update(target_identity_ids=list(dict.fromkeys(self.scene['identities'])),
                          observed=validation[0]['observed'])
        self.model = EvidenceFusion()
        with torch.no_grad():
            self.model.utility[-1].weight.normal_(0., 0.01)
            self.model.utility[-1].bias.fill_(0.5)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        evaluate_evidence_fusion(self.model, [self.scene], device='cpu', prediction_dir=root/'old')
        with np.load(root/'old'/('states-'+self.scene['scene_id']+'.npz'), allow_pickle=False) as saved:
            self.saved = {key: saved[key].copy() for key in saved.files}

    def test_frozen_partition_and_replay_reject_prediction_routing_or_control_drift(self):
        membership = self.saved['predicted_anchor_direct_membership']
        groups = [np.flatnonzero(membership == group).tolist()
                  for group in dict.fromkeys(membership.tolist())]
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        values, routing = predict_partition(self.model, self.scene['motion'], groups, 'cpu')
        self.assertTrue(replay_check(self.saved, values, routing)['passed'])
        self.assertEqual(values['learned'].shape, (4, 64, 4))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        # Shared062 behavior: drift in a learned state, ordinary control,
        # routing weight or membership must not be repaired or tolerated away.
        for field in ('learned', 'mean_supported', 'anchor_weights', 'membership'):
            with self.subTest(field=field):
                changed_values, changed_routing = copy.deepcopy((values, routing))
                target = changed_values if field in changed_values else changed_routing
                target[field].flat[0] += 1 if field == 'membership' else 0.01
                self.assertFalse(replay_check(self.saved, changed_values, changed_routing)['passed'])


if __name__ == '__main__':
    unittest.main()
