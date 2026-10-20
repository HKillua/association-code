"""Research-critical symmetry, visible evidence and ordinary-score baseline."""
import unittest

import numpy as np
import torch

from association.models.evidence_association import (
    EvidenceAssociation, FEATURE_NAMES, features_from_motion)
from tools.experiments.evidence_association_fit import synthetic_scenes
from tools.analysis.motion_association_control import rms_haversine_scores


class EvidenceAssociationTests(unittest.TestCase):
    def test_symmetric_native_only_features_all_pairs_and_exact_initial_score(self):
        _, scenes = synthetic_scenes()
        scene = scenes[0]
        features, base = features_from_motion(scene['motion'], scene['pairs'])
        reverse, reverse_base = features_from_motion(scene['motion'], scene['pairs'][:, ::-1])
        np.testing.assert_array_equal(features, reverse)
        np.testing.assert_array_equal(base, reverse_base)
        self.assertEqual(features.shape, (15, len(FEATURE_NAMES)))
        # Positive pairs without common support are retained and identifiable.
        self.assertEqual(features[0, 6], 0.)
        self.assertEqual(features[0, 5], 0.)
        self.assertTrue(scene['labels'][0])
        expected = -np.log1p(-rms_haversine_scores(scene['motion']['states'][..., :2], scene['pairs'])/1000.)
        np.testing.assert_array_equal(base, expected)
        scene['motion'].update(target=object(), identities=object(), labels=object())
        again, _ = features_from_motion(scene['motion'], scene['pairs'])
        np.testing.assert_array_equal(features, again)
        for variant in ('geometry_only', 'full_evidence'):
            model = EvidenceAssociation(variant)
            with torch.no_grad():
                scores = model(torch.tensor(features, dtype=torch.float32), torch.tensor(base))
            np.testing.assert_array_equal(scores.numpy(), base)

    def test_geometry_masks_after_normalization_and_full_can_use_attributes(self):
        x = torch.zeros((2, len(FEATURE_NAMES)), dtype=torch.float32)
        x[1, 9:] = 100.
        base = torch.tensor([-1., -1.], dtype=torch.float64)
        geometry = EvidenceAssociation('geometry_only')
        full = EvidenceAssociation('full_evidence')
        for model in (geometry, full):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(.01)
        self.assertEqual(*geometry(x, base).tolist())
        self.assertNotEqual(*full(x, base).tolist())


if __name__ == '__main__':
    unittest.main()
