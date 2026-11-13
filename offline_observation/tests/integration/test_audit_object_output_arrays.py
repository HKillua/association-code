"""Variable-object audit regression fixtures; synthetic values are software only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analysis import audit_object_output_arrays as audit
from tools.analysis import object_hypothesis_pilot as producer
from tests.integration.test_static_partition_position_control import fixture


def variable_scene(sid='1', component='a'):
    record, fragment, _, _, _, _ = fixture(sid)
    record['component_id'] = component
    memberships = dict(WS=np.array([0, 1, 2]), NO_REJECT=np.array([0, 0, 0]), T_REJECT=np.array([0, 0, 1]))
    predictions, groups = producer.construct_predictions(fragment, memberships)
    # Original registry has two input-backed targets and one no-input target.
    targets = fragment['target_identity_ids']
    truth = np.zeros((3, 64, 4)); truth[..., 0] = np.array([.001, 1., 2.])[:, None]
    valid = np.zeros(truth.shape, bool); valid[..., :2] = True
    methods, arrays, saved = {}, {}, {}
    regions = producer.old.fixed_regions(fragment, targets)
    for name in producer.METHODS:
        partition = 'NO_REJECT' if name.startswith('NO_REJECT') else 'T_REJECT' if name.startswith('T_REJECT') else 'WS'
        outputs = producer.emit_group_outputs(predictions[name], groups[partition], node_count=3)
        if partition == 'T_REJECT':
            # A query-level validity change as well as the grouping change.
            outputs[0]['prediction_valid'] = np.ones((64, 4), bool)
            outputs[0]['prediction_valid'][:2] = False
        report = producer.old.evaluate_full_scene(outputs, node_identities=fragment['identities'],
            target_identities=targets, target_states=truth, target_valid=valid,
            node_observed=fragment['support'], split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        artifact = report.pop('artifact')
        methods[name], arrays[name] = producer.qc.describe(report, artifact, fragment, regions)
        saved[name] = artifact
    pairs, pair_arrays = {}, {}
    for base, candidate in producer.COMPARISONS:
        key = candidate+'_vs_'+base
        pairs[key], pair_arrays[key], _ = producer.old.compare_pair(saved[base], saved[candidate], regions)
        pairs[key].update(baseline_method=base, candidate_method=candidate,
                          side_labels=dict(A0=base, candidate=candidate))
    association = dict(RAW=producer.old.frozen.evaluate_scenes([record], producer.old.frozen.THRESHOLD))
    for partition, membership in memberships.items():
        accepted = membership[record['pairs'][:, 0]] == membership[record['pairs'][:, 1]]
        association[partition] = producer.old.frozen.evaluate_scenes([dict(record, scores=accepted.astype(float))], .5)
    row = dict(scene_id=sid, component_id=component, methods=methods, pairwise=pairs, association=association)
    return saved, row, dict(methods=arrays, pairs=pair_arrays), record


class ObjectAuditTests(unittest.TestCase):
    def test_variable_partitions_validity_duplicates_and_true_common_pool(self):
        first, row1, arrays1, record1 = variable_scene('1', 'a')
        second, row2, arrays2, record2 = variable_scene('2', 'b')
        checks = audit.core.Checks()
        results = [audit.inspect_variable_scene(first, row1, checks), audit.inspect_variable_scene(second, row2, checks)]
        self.assertEqual(first['WS_MEAN']['group_states'].shape[0], 3)
        self.assertEqual(first['NO_REJECT_MEAN']['group_states'].shape[0], 1)
        self.assertEqual(row1['methods']['WS_MEAN']['population']['duplicate_prediction_group_count'], 1)
        self.assertEqual(row1['methods']['NO_REJECT_MEAN']['population']['duplicate_prediction_group_count'], 0)
        pair = results[0]['pairs']['NO_REJECT_MEAN_vs_WS_MEAN']
        self.assertEqual(pair['sides']['A0'][0]['intersection_count'], 64)
        self.assertEqual(results[0]['methods']['WS_MEAN']['position']['intersection_count'], 128)
        self.assertEqual(pair['transitions']['lost_target_points'], 64)
        partial = results[0]['pairs']['T_REJECT_MEAN_vs_WS_MEAN']
        self.assertEqual(partial['sides']['A0'][0]['intersection_count'], 126)
        summary = producer.aggregate([row1, row2], [arrays1, arrays2])
        audit.inspect_variable_summary(summary, [0, 1], results, [row1, row2], checks, 'two-scene')
        cache = {str(i)+'_'+key: record[key] for i, record in [(1, record1), (2, record2)]
                 for key in ('pairs', 'labels', 'sources', 'node_ids')}
        decoded = [audit.inspect_decoded_association(first, row1, cache, checks),
                   audit.inspect_decoded_association(second, row2, cache, checks)]
        audit.inspect_decoded_summary(summary, [0, 1], decoded, checks, 'two-scene')
        self.assertGreater(checks.numeric_checks, 0)

    def test_reject_own_population_substitution_and_corrupt_pair_tail(self):
        saved, row, _, _ = variable_scene()
        for corrupt in ('common', 'tail'):
            modified = deepcopy(row)
            pair = modified['pairwise']['NO_REJECT_MEAN_vs_WS_MEAN']
            if corrupt == 'common':
                pair['common_valid_position']['A0'] = deepcopy(row['methods']['WS_MEAN']['position']['all'])
            else:
                pair['strata']['all']['A0']['p99'] += 1.
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                audit.inspect_variable_scene(saved, modified, audit.core.Checks())

    def test_cross_arm_uncovered_reference_and_existing_output_are_protected(self):
        saved, row, _, _ = variable_scene()
        # No arm predicts target2; own errors cannot reveal its corruption.
        modified = deepcopy(saved)
        modified['T_REJECT_MEAN']['aligned_truth'][2, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'cross-arm'):
            audit.inspect_variable_scene(modified, row, audit.core.Checks())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'audit.json'; output.write_text('original evidence')
            with self.assertRaises(FileExistsError):
                audit.audit(Path(temporary)/'missing-run', output)
            self.assertEqual(output.read_text(), 'original evidence')


if __name__ == '__main__':
    unittest.main()
