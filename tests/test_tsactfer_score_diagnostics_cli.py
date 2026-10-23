"""075 uses real tiny074D score/log artifacts but never models or new decisions."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from association.utils.runtime import read_json, read_json_or_gzip, write_json
from tools.experiments import tsactfer_level1_fit as fit_entry
from tools.analysis import tsactfer_score_diagnostics as entry
from tests.test_tsactfer_level1 import tiny_config, training_fixture, validation_fixture, source_scores


def completed_source(root):
    """Actual optimization/scoring/system output, synthetic inputs only."""
    config = tiny_config()
    train, normalizer = training_fixture(config)
    source056, metadata, validation = validation_fixture(root, config, normalizer)
    manifest = {'splits': {'T_core': ['t1', 't2'], 'V_select': ['3035'], 'T_test': ['sealed']},
                'components': {'3035': 'shared'}}
    frozen = root/'frozen'
    (frozen/'configs/data').mkdir(parents=True)
    write_json(frozen/'configs/tsactfer-level1.json', config)
    write_json(frozen/'configs/data/mtad-level1-scene3000.json', manifest)
    original = fit_entry.evaluate_system
    def evaluate(*args, **kwargs):
        return original(*args, **kwargs, **source_scores(source056, args[0]))
    source = root/entry.SOURCE_TASK
    args = Namespace(source_system=str(source056), manifest=str(frozen/'configs/data/mtad-level1-scene3000.json'),
        config=str(frozen/'configs/tsactfer-level1.json'), archive=['never-read.zip'], output=str(source),
        device='cpu', cpu_threads=4, time_policy='advisory')
    with patch.object(fit_entry, 'production_contract'), patch.object(fit_entry, '_bind_source'), \
            patch.object(fit_entry, '_load_metadata', return_value=metadata), \
            patch.object(fit_entry, 'prepare_native_dataset', return_value=(train, validation, normalizer, {})), \
            patch.object(fit_entry, 'evaluate_system', side_effect=evaluate):
        fit_entry.run(args)
    return source, frozen


class TSA075CLITests(unittest.TestCase):
    def test_real_cached_cli_never_loads_weights_or_selects_threshold(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frozen = completed_source(root)
            original = {p.relative_to(source): p.read_bytes() for p in source.rglob('*')
                        if p.is_file() and p.suffix in ('.json', '.jsonl', '.npz')}
            args = Namespace(source_074d=str(source), output=str(root/'audit'), cpu_threads=4)
            with patch.object(entry, 'ROOT', frozen), \
                    patch('torch.load', side_effect=AssertionError('075 cannot deserialize weights')), \
                    patch('baselines.tsactfer.TSActfer.encode', side_effect=AssertionError('075 cannot forward')), \
                    patch('association.evaluation.association.select_threshold', side_effect=AssertionError('075 cannot select')):
                result = entry.run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['completed_scenes'], 1)
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertEqual(result['model_forwards'], 0)
            self.assertFalse(result['threshold_reselected'])
            self.assertFalse(result['checkpoints_opened'])
            for name, content in original.items():
                self.assertEqual((source/name).read_bytes(), content)
            verified = read_json(root/'audit/source-verification.json')
            self.assertTrue(verified['source_metrics_reproduced'])
            self.assertTrue(verified['source_time_eligibility_reproduced'])
            self.assertEqual(verified['pair_count'], result['pair_count'])
            self.assertIn('074D原成绩不变', (root/'audit/report.md').read_text())
            self.assertTrue(read_json_or_gzip(root/'audit/score-summary.json'))
            self.assertTrue(read_json_or_gzip(root/'audit/scene-summary.json'))
            with self.assertRaises(FileExistsError):
                entry.run(args)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                entry.run(Namespace(source_074d=str(source), output=str(source/'audit'), cpu_threads=4))

    def test_wrong_endpoint_metrics_or_population_preserve_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frozen = completed_source(root)
            target = source/'final-endpoint.json'
            original = read_json(target)
            cases = [
                ('selected', lambda r: r.update(checkpoint=str(source/'fit/selected.pt'))),
                ('updates', lambda r: r.update(actual_updates=2)),
                ('metric', lambda r: r['metrics'].update(fp=r['metrics']['fp']+1)),
                ('population', lambda r: r.update(scene_ids=[])),
                ('normalizer', lambda r: r['normalizer'].update(training_scene_ids=['3035'])),
            ]
            with patch.object(entry, 'ROOT', frozen), patch('torch.load', side_effect=AssertionError('weights forbidden')):
                for name, mutate in cases:
                    wrong = deepcopy(original); mutate(wrong); write_json(target, wrong)
                    output = root/('bad-'+name)
                    with self.assertRaises(ValueError):
                        entry.run(Namespace(source_074d=str(source), output=str(output), cpu_threads=4))
                    self.assertEqual(read_json(output/'run.json')['status'], 'FAILED')
                    self.assertFalse((output/'report.md').exists())
                write_json(target, original)
                cache = source/'final-scores/scene_000000.npz'
                with np.load(cache, allow_pickle=False) as z:
                    arrays = {key:z[key] for key in z.files}
                arrays['labels'][0] = 1-arrays['labels'][0]
                np.savez_compressed(cache, **arrays)
                with self.assertRaisesRegex(ValueError, 'labels disagree'):
                    entry.bind_source(source)

    def test_missing_duplicate_or_reordered_inventory_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frozen = completed_source(root)
            target = source/'final-scores/index.json'
            original = read_json(target)
            with patch.object(entry, 'ROOT', frozen):
                wrong = deepcopy(original); wrong['records'] *= 2; write_json(target, wrong)
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    entry.bind_source(source)
                write_json(target, original)
                cache = source/'final-scores/scene_000000.npz'
                cache.rename(cache.with_suffix('.missing'))
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    entry.bind_source(source)


if __name__ == '__main__':
    unittest.main()
