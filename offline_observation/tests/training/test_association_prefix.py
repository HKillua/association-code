"""Fresh-fit prefix reproduction without loading weights or changing its RNG stream."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.training.common import _update
from association.training.evidence_association import fit_evidence_association, load_checkpoint
from association.utils.runtime import read_json, write_json
from tools.experiments.evidence_association_fit import synthetic_config, synthetic_scenes


class AssociationPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def prepare(self, root):
        train, validation = synthetic_scenes()
        config = synthetic_config('full_evidence')
        config['training']['updates'] = 2
        source = root/'source'
        fit_evidence_association(config, source, 'cpu', train_scenes=train,
                                 validation_scenes=validation, synthetic=True)
        config['training']['updates'] = 3
        return source, config, dict(train_scenes=train, validation_scenes=validation, synthetic=True)

    def test_fresh_prefix_exact_and_monitoring_does_not_change_final_state_or_selection(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, config, kwargs = self.prepare(root)
            ordinary = fit_evidence_association(config, root/'ordinary', 'cpu', **kwargs)
            monitored = fit_evidence_association(config, root/'monitored', 'cpu',
                                                  prefix_reference=source, **kwargs)
            report = read_json(root/'monitored/prefix-comparison.json')
            self.assertEqual(report['comparison_status'], 'EXACT_MATCH')
            self.assertTrue(report['causal_attribution_valid'])
            self.assertTrue(report['bitwise_state_and_scores_equal'])
            self.assertTrue(report['optimizer']['exact'])
            self.assertEqual(report['own_threshold_changed_decisions'], 0)
            self.assertEqual(monitored['optimizer_updates'], 3)
            self.assertEqual(monitored['selected'], ordinary['selected'])
            self.assertEqual(monitored['final_result'], ordinary['final_result'])
            _, final = load_checkpoint(root/'monitored/final.pt', 'cpu')
            _, plain = load_checkpoint(root/'ordinary/final.pt', 'cpu')
            self.assertEqual(final['update'], 3)
            for key in final['state_dict']:
                torch.testing.assert_close(final['state_dict'][key], plain['state_dict'][key], atol=0, rtol=0)
            prefix = torch.load(root/'monitored/prefix-step-2.pt', weights_only=False)
            reference = torch.load(source/'final.pt', weights_only=False)
            self.assertEqual(prefix['update'], 2)
            self.assertEqual(prefix['config']['training']['updates'], 3)
            for key in prefix['state_dict']:
                torch.testing.assert_close(prefix['state_dict'][key], reference['state_dict'][key], atol=0, rtol=0)
            exposure = read_json(root/'monitored/prefix-step-2-sampling_audit.json')
            self.assertEqual(exposure['actual_optimizer_updates'], 2)
            self.assertEqual(exposure['configured_updates'], 3)
            self.assertEqual(read_json(root/'monitored/prefix-binding.json')['target_updates'], 3)
            self.assertFalse((root/'ordinary/prefix-comparison.json').exists())

    def test_invalid_source_and_protocol_fail_before_any_optimizer_update(self):
        for mutation in ('failed source', 'config', 'normalizer', 'base scores', 'wrong final step',
                         'feature names', 'feature mask', 'precision', 'motion policy'):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temp:
                root = Path(temp)
                source, config, kwargs = self.prepare(root)
                if mutation == 'failed source':
                    value = read_json(source/'run.json')
                    value['status'] = 'FAILED'
                    write_json(source/'run.json', value)
                elif mutation == 'config':
                    config['training']['learning_rate'] *= 2
                elif mutation in ('normalizer', 'wrong final step', 'feature names', 'feature mask', 'precision', 'motion policy'):
                    value = torch.load(source/'final.pt', weights_only=False)
                    if mutation == 'normalizer':
                        value['normalizer']['mean'][0] += 1
                    elif mutation == 'wrong final step':
                        value['update'] = 0
                    elif mutation == 'feature names':
                        value['feature_names'] = value['feature_names'][::-1]
                    elif mutation == 'feature mask':
                        value['state_dict']['feature_mask'][0] = 0.
                    elif mutation == 'precision':
                        value['precision'] = 'different precision policy'
                    else:
                        value['motion_observation_policy'] = 'canonical_all'
                    torch.save(value, source/'final.pt')
                else:
                    path = source/'baseline-scores.npz'
                    with np.load(path, allow_pickle=False) as archive:
                        value = {key: archive[key] for key in archive.files}
                    value['validation_scores'][0] += .01
                    np.savez_compressed(path, **value)
                with self.assertRaises(ValueError):
                    fit_evidence_association(config, root/'failed', 'cpu', prefix_reference=source, **kwargs)
                state = read_json(root/'failed/run.json')
                self.assertEqual(state['status'], 'FAILED')
                self.assertEqual(state['optimizer_updates'], 0)
                self.assertFalse((root/'failed/final.pt').exists())

    def test_numerical_mismatch_is_explicit_and_continues_original_budget(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, config, kwargs = self.prepare(root)
            for offset, status, valid in ((.01, 'MISMATCH', False), (5e-9, 'TOLERANT_MATCH', True)):
                with self.subTest(status=status):
                    output = root/status

                    def perturbed_update(loss, optimizer, parameters, clip):
                        result = _update(loss, optimizer, parameters, clip)
                        with torch.no_grad():
                            optimizer.param_groups[-1]['params'][-1].add_(offset)
                        return result

                    with patch('association.training.evidence_association._update', side_effect=perturbed_update):
                        result = fit_evidence_association(config, output, 'cpu', prefix_reference=source, **kwargs)
                    report = read_json(output/'prefix-comparison.json')
                    self.assertEqual(report['comparison_status'], status)
                    self.assertEqual(report['causal_attribution_valid'], valid)
                    self.assertEqual(report['weights']['within_tolerance'], valid)
                    self.assertFalse(report['scores']['exact'])
                    self.assertFalse(report['bitwise_state_and_scores_equal'])
                    self.assertIn('reference_threshold_changed_decisions', report)
                    self.assertEqual(result['optimizer_updates'], 3)
                    self.assertEqual(read_json(output/'run.json')['status'], 'COMPLETED')
                    self.assertEqual(torch.load(output/'final.pt', weights_only=False)['update'], 3)


if __name__ == '__main__':
    unittest.main()
