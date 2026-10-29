"""125 auditor schema/population regressions; fixtures are software evidence only."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analysis import audit_conditional_value_arrays as audit
from tools.analysis import object_hypothesis_pilot as producer
from tests.integration.test_audit_object_output_arrays import variable_scene
from tests.integration.test_static_partition_position_control import fixture


def fixed_scene():
    """Uncovered target + duplicate group, and distinct predictions in11 arms."""
    original, _, _, _ = variable_scene()
    _, fragment, _, _, _, _ = fixture('1')
    base = original['WS_MEAN']; t = len(base['target_ids'])
    groups = [base['group_node_indices'][start:end].tolist()
              for start, end in zip(base['group_offsets'][:-1], base['group_offsets'][1:])]
    regions = producer.old.fixed_regions(fragment, base['target_ids'])
    methods, arrays, saved = {}, {}, {}
    for index, name in enumerate(audit.METHODS):
        states = base['group_states'].copy(); states[..., 0] += index*1e-5
        outputs = producer.emit_group_outputs(states, groups, node_count=len(base['node_group_indices']))
        report = producer.old.evaluate_full_scene(outputs, node_identities=fragment['identities'],
            target_identities=base['target_ids'], target_states=base['aligned_truth'][:t],
            target_valid=base['aligned_truth_valid'][:t], node_observed=fragment['support'], split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        artifact = report.pop('artifact')
        methods[name], arrays[name] = producer.qc.describe(report, artifact, fragment, regions)
        saved[name] = artifact
    pairwise, pairs = {}, {}
    for left, right in audit.PAIRS:
        key = right+'_vs_'+left
        pairwise[key], pairs[key], _ = producer.old.compare_pair(saved[left], saved[right], regions)
        pairwise[key].update(baseline_method=left, candidate_method=right, side_labels=dict(A0=left, candidate=right))
    row = dict(scene_id='1', component_id='a', node_count=len(base['node_group_indices']),
        methods=methods, pairwise=pairwise, fixed_R_population_invariants_verified=True)
    return saved, row, dict(methods=arrays, pairs=pairs)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False)


class ConditionalValueAuditTests(unittest.TestCase):
    def test_all55_actual_fixed_population_and_anchor_boundaries(self):
        saved, row, arrays = fixed_scene(); checks = audit.core.Checks()
        result = audit.inspect_fixed_scene(saved, row, checks)
        self.assertEqual((len(result), len(row['pairwise'])), (11, 55))
        self.assertEqual(result['WS_MEAN']['population']['duplicate_prediction_group_count'], 1)
        self.assertEqual(result['WS_MEAN']['unmatched_target_count'], 1)
        second = deepcopy(row); second.update(scene_id='2', component_id='b')
        summary = producer.qc.aggregate([row, second], [arrays, arrays], method_names=audit.METHODS)
        audit.inspect_fixed_summary(summary, [0, 1], [result, result], [row, second], checks, 'two-scene')
        self.assertGreater(checks.numeric_checks, 0)
        for change in ('anchor', 'population', 'common', 'uncovered_truth'):
            public, artifacts = deepcopy(row), deepcopy(saved)
            if change == 'anchor':
                public['pairwise']['WS_FIXED_MEAN_vs_WS_MEAN']['changed_anchor_target_count'] = 1
            elif change == 'population':
                public['methods']['WS_LINEAR_VALUE_GATE']['population']['target_identity_count'] -= 1
            elif change == 'common':
                public['pairwise']['WS_LINEAR_VALUE_GATE_vs_WS_MEAN']['common_valid_position']['A0']['intersection_count'] -= 1
            else:
                artifacts['WS_LINEAR_VALUE_GATE']['aligned_truth'][2, 0, 0] += .01
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.inspect_fixed_scene(artifacts, public, audit.core.Checks())

    def test_actual125_completion_schema_and_main_array_entrypoint(self):
        saved, row, arrays = fixed_scene()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run_path = root/'run'; source = root/'source'; pilot = root/'pilot'; mix = root/'mix'
            rows = [dict(deepcopy(row), scene_id=str(i), component_id='a' if i <= 19 else 'b')
                    for i in range(1, 33)]
            payload = {name+'_'+key: value for name, art in saved.items() for key, value in art.items()}
            for path in (run_path, pilot, mix):
                (path/'private').mkdir(parents=True)
                for item in rows:
                    np.savez_compressed(path/'private'/f"states-{item['scene_id']}.npz", **payload)
            for item in rows:
                write_json(run_path/'scenes'/f"scene-{item['scene_id']}.json.gz", item)
            run = dict(status='COMPLETED', exit_code=0, idea_id=audit.IDEA, plan_version='1.0',
                methods=list(audit.METHODS), scene_ids=[item['scene_id'] for item in rows],
                completed_V_scenes=32, total_V_scenes=32, completed_T_scenes=32, total_T_scenes=32,
                training_scene_ids=[str(i) for i in range(1000, 1032)],
                source_control_run=str(source), mix_run=str(mix))
            # Match actual123 storage: original32 arrays only exist in121,
            # not123/private. The declared original32/pilot lineage is required.
            write_json(source/'run.json', dict(status='COMPLETED', exit_code=0,
                idea_id='IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029', plan_version='1.0',
                original32_ids=run['scene_ids'], scene_ids=[str(i) for i in range(1, 301)], pilot_run=str(pilot)))
            write_json(pilot/'run.json', dict(status='COMPLETED', exit_code=0,
                idea_id='IDEA-SOURCE-PREFERENCE-CONTROL-20261029', plan_version='1.0',
                scene_ids=run['scene_ids'], completed_scenes=32))
            self.assertFalse((source/'private').exists())
            def aggregate(items):
                return producer.qc.aggregate(items, [arrays]*len(items), method_names=audit.METHODS)
            summary = dict(aggregate(rows), status='COMPLETED', idea_id=audit.IDEA, plan_version='1.0',
                full_T_models_frozen_before_V=True, fixed_R_population_invariants_verified=True,
                components={c: aggregate([item for item in rows if item['component_id'] == c]) for c in ('a', 'b')},
                without_largest_component=aggregate(rows[19:]))
            write_json(run_path/'run.json', run); write_json(run_path/'summary.json', summary)
            write_json(run_path/'scene-summary.json.gz', rows)
            # The formerly missing key is deliberately absent: no alias or
            # patched original run may be used to make the new auditor pass.
            self.assertNotIn('completed_scenes', run)
            result = audit.audit_run(run_path, audit.core.Checks())
            self.assertEqual((result['scene_count'], result['method_count'], result['pair_count_per_scene']), (32, 11, 55))
            self.assertEqual(result['exact_imported_control_scene_arm_count'], 224)
            changed = deepcopy(payload); key = 'WS_MEAN_aligned_prediction'
            changed[key] = changed[key].copy(); changed[key][0, 0, 0] += .01
            np.savez_compressed(pilot/'private/states-1.npz', **changed)
            with self.assertRaisesRegex(ValueError, 'exact imported control'):
                audit.audit_run(run_path, audit.core.Checks())
            write_json(run_path/'run.json', dict(run, completed_V_scenes=31, completed_scenes=32))
            with self.assertRaisesRegex(ValueError, 'completed_V_scenes'):
                audit.metadata(run_path, audit.core.Checks())
            output = root/'old-audit.json'; output.write_text('original failed evidence')
            with self.assertRaises(FileExistsError):
                audit.audit(run_path, output)
            self.assertEqual(output.read_text(), 'original failed evidence')


if __name__ == '__main__':
    unittest.main()
