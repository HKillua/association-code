"""CPU saved-array diagnosis checks; no model execution or real dataset."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from association import runtime
from association.full_scene_evaluation import evaluate_full_scene
from association.prediction_artifacts import ScenePredictionWriter
from tests.test_complete_evaluation import scene
from tests.test_prediction_artifacts import artifact_for
from tools.analysis.inspect_cooperative015_arrays import ARMS, inspect, select_cases, speed_statistics


class InspectCooperative015Tests(unittest.TestCase):
    def test_speed_statistics_separate_nonfinite_and_fixed_native_cutoffs(self):
        result = speed_statistics([0., 1e-7, -2., 100., 1001., 10001., np.nan, np.inf])
        self.assertEqual(result['valid_mask_count'], 8)
        self.assertEqual(result['finite_count'], 6)
        self.assertEqual(result['nonfinite_count'], 2)
        self.assertEqual(result['negative_count'], 1)
        self.assertEqual(result['near_zero_abs_le_1e_6_count'], 2)
        self.assertEqual(result['greater_than_100_count'], 2)
        self.assertEqual(result['greater_than_1000_count'], 2)
        self.assertEqual(result['greater_than_10000_count'], 1)
        self.assertIsNone(speed_statistics([])['quantiles'])

    def test_case_selection_uses_numeric_ids_ties_lower_median_and_keeps_no_overlap(self):
        def row(identifier, delta):
            return dict(scene_id=identifier, common={
                'left': {'all': {'position': {'mae': 10. if delta is not None else None}}},
                'right': {'all': {'position': {'mae': 10.+delta if delta is not None else None}}}})
        selected = select_cases({'per_scene':[row('10', -2.), row('20', 3.), row('2', None), row('4', 0.)]})
        self.assertEqual(selected['eligible_scene_count'], 3)
        self.assertEqual([x['scene_id'] for x in selected['cases']], ['10', '20', '4', '2'])
        self.assertIsNone(selected['cases'][-1]['utility_minus_identity_mae_metres'])
        tied = select_cases({'per_scene':[row('10', 0.), row('2', 0.)]})
        self.assertEqual([x['scene_id'] for x in tied['cases']], ['2', '10'])
        self.assertEqual(len(tied['cases'][0]['selection_reasons']), 3)

    def test_whole_inspection_compares_six_times_and_copies_original_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            runtime.write_json(source/'run.json', dict(status='COMPLETED', test_opened=False, completed_routes=list(ARMS)))
            ids = ['10', '20', '2', '4']
            for arm in ARMS:
                writer = ScenePredictionWriter(source/(arm+'-predictions'))
                for index, identifier in enumerate(ids):
                    example = scene(include_missing_target=True)
                    example['scene_id'] = identifier
                    example['native_times'] = np.broadcast_to(example['grid'], example['observed'].shape).copy()
                    example['native_observed'] = example['observed'].copy()
                    outputs, kwargs = artifact_for(example)
                    shift = {'plain':.2, 'identity':.1, 'learned':.15,
                             'utility':.1+[-.01, .02, .01, 0.][index]}[arm]
                    outputs = deepcopy(outputs)
                    for item in outputs:
                        if item['states'] is not None:
                            item['states'][..., 0] += shift
                    methods = {}
                    for method in ('predicted', 'oracle'):
                        method_outputs = (outputs if method == 'predicted' else
                            [{**outputs[0], 'node_indices':[0, 1]}, outputs[2]])
                        methods[method] = evaluate_full_scene(method_outputs,
                            **{**kwargs, 'grouping_kind':method}, return_artifact=True)['artifact']
                    writer(index, example, methods)
                runtime.write_json(source/(arm+'-evaluation.json'),
                    dict(per_scene=[{'scene_id':x} for x in ids]))
            original = {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            result = inspect(source, root/'inspection', expected_scenes=4)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['comparison_count'], 6)
            self.assertEqual(len(result['support_times']), 4)
            self.assertIn('native_track_span_seconds', result['support_times'][0])
            self.assertEqual(len(result['copied_cases']), 16)
            self.assertEqual(len(result['case_selection']['cases']), 4)
            for item in result['copied_cases']:
                self.assertEqual(Path(item['source_npz']).read_bytes(),
                                 (root/'inspection'/item['copied_npz']).read_bytes())
                self.assertEqual(Path(item['source_metadata']).read_bytes(),
                                 (root/'inspection'/item['copied_metadata']).read_bytes())
            stat = result['speed']['utility']['predicted']
            self.assertGreater(stat['reference_valid']['valid_mask_count'],
                               stat['prediction_on_reference_intersection']['valid_mask_count'])
            self.assertEqual(original, {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):
                inspect(source, root/'inspection', expected_scenes=4)
            with self.assertRaisesRegex(ValueError, 'outside'):
                inspect(source, source/'bad', expected_scenes=4)
            runtime.write_json(source/'run.json', dict(status='COMPLETED', test_opened=True, completed_routes=list(ARMS)))
            with self.assertRaisesRegex(ValueError, 'development'):
                inspect(source, root/'forbidden', expected_scenes=4)
            self.assertFalse((root/'forbidden').exists())


if __name__ == '__main__':
    unittest.main()
