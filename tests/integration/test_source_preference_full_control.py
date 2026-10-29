"""123 import/extension behavior only; fixtures do not establish scientific value."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.integration.test_shared_component_full_control import cache_fixture
from tools.analysis import source_preference_full_control as p
from tools.analysis import audit_shared_control_arrays as audit


class SourcePreferenceFullTests(unittest.TestCase):
    def test_import_exact_and_only_remaining_scenes_construct_new_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args120, meta = cache_fixture(root)
            args120.output = root/p.SHARED_RUN
            with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)):
                p.shared.run(args120)
            args121 = SimpleNamespace(source_system=args120.source_system,
                partition_run=args120.partition_run, shared_run=args120.pilot_run,
                manifest=args120.manifest, output=root/p.PILOT_RUN, cpu_threads=1)
            pilot_meta = deepcopy(meta)
            with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(pilot_meta)):
                p.p.run(args121)
            args = SimpleNamespace(source_system=args120.source_system,
                partition_run=args120.partition_run, pilot_run=args121.output,
                control_run=args120.output, manifest=args120.manifest,
                output=root/'result123', cpu_threads=1)
            with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)), \
                 patch.object(p.p, 'construct_predictions', wraps=p.p.construct_predictions) as construct:
                result = p.run(args)
            self.assertEqual(construct.call_count, 2)
            self.assertEqual((result['scene_prediction_arms_imported'], result['scene_prediction_arms_computed']), (26, 4))
            self.assertEqual(result['original32']['scene_ids'], ['1'])
            self.assertEqual(result['remaining268_seen_components']['scene_ids'], ['2'])
            self.assertEqual(result['remaining268_unseen_components']['scene_ids'], ['3'])
            self.assertEqual(len(result['pairwise']), 45)
            self.assertFalse((args.output/'private/states-1.npz').exists())
            self.assertTrue((args.output/'private/states-3.npz').exists())
            for name in p.p.METHODS:
                self.assertEqual(result['original32']['methods'][name],
                                 p.old.frozen.read(args121.output/'summary.json')['methods'][name])
            state = p.old.frozen.read(args.output/'run.json')
            self.assertEqual((state['status'], state['exit_code'], state['completed_scenes']), ('COMPLETED', 0, 3))
            checks = audit.Checks()
            verified = audit.audit_run(args.output, checks)
            self.assertEqual((verified['scene_count'], verified['method_count'], verified['pair_count_per_scene']), (3, 10, 45))
            with self.assertRaises(FileExistsError):
                p.run(args)


if __name__ == '__main__':
    unittest.main()
