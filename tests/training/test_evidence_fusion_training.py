import json
from pathlib import Path
import tempfile
import unittest

import torch

from association.models.cooperative import CooperativeConfig
from association.models.evidence_fusion import fusion_inputs
from association.training.evidence_fusion import load_checkpoint, prepare_evidence_scenes, train_evidence_fusion
from tools.experiments.cooperative_fit import synthetic_scenes
from tools.experiments.evidence_fusion_fit import configuration


class EvidenceFusionTrainingTests(unittest.TestCase):
    def test_three_modes_update_checkpoint_and_full_population(self):
        torch.set_num_threads(1)
        train, validation = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
        config = configuration(json.loads(Path('configs/evidence-fusion.json').read_text()))
        config['training'].update(updates=3, batch_size=2, evaluate_every=3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fit'
            report = train_evidence_fusion(train, validation, config, output)
            self.assertEqual(report['status'], 'COMPLETED')
            self.assertEqual(report['optimizer_updates'], 3)
            for counts in report['training_samples'].values():
                self.assertEqual(counts['anchors'], 2)
            model, state = load_checkpoint(output/'final.pt')
            initial, _ = load_checkpoint(output/'step-0.pt')
            self.assertEqual(state['step'], 3)
            self.assertTrue(any(not torch.equal(a, b)
                                for a, b in zip(model.parameters(), initial.parameters())))
            scene = prepare_evidence_scenes(validation)[0]
            arguments = fusion_inputs(scene['motion'], [0, 1])
            prediction = model(*arguments)['states']
            clone, _ = load_checkpoint(output/'final.pt')
            self.assertTrue(torch.equal(prediction, clone(*arguments)['states']))
            evaluation = json.loads((output/'evaluation-step-3.json').read_text())
            self.assertEqual(evaluation['fragment_count'], 4)
            self.assertEqual(evaluation['query_count'], 256)
            self.assertEqual(len(evaluation['per_anchor']), 4)
            self.assertFalse(evaluation['test_opened'])
            self.assertTrue((output/'predictions-final/states-synthetic-validation.npz').exists())


if __name__ == '__main__':
    unittest.main()
