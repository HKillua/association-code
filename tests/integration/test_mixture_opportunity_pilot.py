"""134 wiring and independent-audit fixtures, never scientific evidence."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from association.utils import runtime
from tests.integration.test_source_risk_pilot import cached_input
from tools.analysis import source_risk_pilot as prior
from tools.analysis import mixture_opportunity_pilot as pilot
from tools.analysis import audit_mixture_opportunity as auditor


class MixtureOpportunityPilotTests(unittest.TestCase):
    def test_boundary_diagnostics_preserve_undefined_and_endpoint_ties(self):
        for key, value, scale, expected in (('selected_native_count', 4., 1., '4..8'),
                                            ('selected_nearest_seconds', 900., 60., '900..1800')):
            decoded = np.expm1(np.log1p(value/scale))*scale
            selected = [name for name, mask in pilot.bin_masks(np.asarray([decoded]), pilot.BINS[key]) if mask[0]]
            self.assertEqual(selected, [expected])
        directions = pilot.direction_cos(np.zeros((1, 2, 2)), np.zeros((1, 2)))
        self.assertTrue(np.isnan(directions[0]))
        moments = np.asarray([[[30., 0., 1.], [30., 0., 1.]]])
        oracle = pilot.mix.grid_mixture_oracle(moments, np.asarray([[30., 0.]]), np.asarray([[29., 0.]]))
        self.assertEqual(oracle['alpha'][0], .5)
        self.assertEqual(oracle['gain_m'][0], 0.)
        self.assertEqual(oracle['tied_grid_count'][0], 4097)
        np.testing.assert_array_equal(oracle['best_single_error_m'], oracle['grid_error_m'])

    def test_complete_T_only_chain_and_independent_oracle_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_args = cached_input(root)
            prior.run(source_args)
            # Real V array directories are unavailable to this T-only run.
            for category in ('native-motion', 'native-evaluation'):
                directory = source_args.native_run/'private'/category/'V_select'
                directory.rename(directory.with_name('unavailable-V'))
            args = SimpleNamespace(source_run=source_args.output, native_run=source_args.native_run,
                                   manifest=source_args.manifest, output=root/'new134', cpu_threads=1,
                                   time_policy='advisory')
            result = pilot.run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['grid_size'], 4097)
            self.assertEqual(result['all_components']['query_count'], 1952)
            self.assertEqual(len(result['by_component']), 12)
            self.assertEqual(set(result['all_components']['heads']), {'LINEAR','QUADRATIC','SHUFFLED'})
            run = runtime.read_json(args.output/'run.json')
            for key in ('new_fits', 'neural_updates', 'teacher_forwards'):
                self.assertEqual(run[key], 0)
            for key in ('V_arrays_opened', 'test_opened', 'archive_read'):
                self.assertFalse(run[key])
            audit = auditor.audit(args.output, args.source_run, args.native_run, args.manifest,
                                  args.output/'independent-audit.json')
            self.assertEqual(audit['status'], 'COMPLETED', audit.get('error'))
            self.assertEqual(audit['query_count'], 1952)
            self.assertTrue(audit['public_bootstrap_compared'])
            self.assertTrue(audit['main_array_metrics_and_strata_compared'])
            saved = auditor.load(args.output/'private/scene-0.npz')
            saved['moments'] = saved['moment_pairs']
            saved['gain_m'] = saved['gain_m']+1.
            with self.assertRaisesRegex(ValueError, 'oracle/gain_m'):
                auditor.inspect_arrays(saved, auditor.core.Checks())
            # Binding error must leave a separate failure, never overwrite133/134.
            path = args.native_run/'private/native-evaluation/T_core/scene-0.npz'
            broken = auditor.load(path); broken['query_seconds'] = broken['query_seconds']+1.
            np.savez_compressed(path, **broken)
            other = SimpleNamespace(**vars(args)); other.output = root/'broken134'
            with self.assertRaises((AssertionError, ValueError)):
                pilot.run(other)
            self.assertEqual(runtime.read_json(other.output/'run.json')['status'], 'FAILED')
            self.assertFalse((other.output/'summary.json').exists())
            self.assertEqual(runtime.read_json(args.output/'run.json')['status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
