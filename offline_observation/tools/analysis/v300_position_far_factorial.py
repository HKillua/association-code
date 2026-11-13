"""Compare frozen T128 positions and FAR speed on the same V300 WS groups."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.paper_fusion_metrics import aggregate_paper_fusion
from tools.analysis.held_fusion_population import HELD_IDS, PARENTS, SPLIT, load_full_population, read_evaluation
from tools.analysis.motion_fallback_diagnostic import INVARIANTS, require
from tools.analysis.motion_readout_evaluation import POPULATION_KEYS, _stats, _wrap
from tools.analysis.same_source_swap_pilot import MAX_AS, group_members, peak_rss_bytes, read, write
from tools.analysis.v300_far_paper_bridge import C83, point_errors


C89 = 'LEVEL1-V300-FAR-PAPER-BRIDGE-20261112-attempt-1'
RUN157 = 'LEVEL1-T128-FROZEN-FULL-SCOPE-20261030-157-attempt-1'
POSITION = 'WS_T128_Q_HARD'
ARMS = ('B', 'FAR', 'T128_B', 'T128_FAR')
POINT_FIELDS = ('longitude', 'latitude', 'speed', 'heading', 'position')


def bind(runs, ids):
    c89, position, held = (read(runs/name/'run.json') for name in (C89, RUN157, C83))
    require(all(row['status'] == 'COMPLETED' and row['exit_code'] == 0 for row in (c89, position, held)),
            'completed C89, 157 and C83 required')
    require(c89['scene_ids'] == position['scene_ids'] == ids and c89['completed_scenes'] == 300
            and c89['held_c83_replayed'] == 48 and tuple(held['scene_ids']) == HELD_IDS
            and position['completed_V_scenes'] == 300 and position['evaluation_splits'] == [SPLIT]
            and position['data_run'] == str(runs/PARENTS['full_data'])
            and position['baseline_run'] == str(runs/PARENTS['mean'])
            and POSITION in position['methods'] and c89['test_opened'] is position['test_opened'] is False,
            'frozen 157 and C89 do not share the original V300 registry')


def assemble(far, position):
    base, candidate = far['B'], far['FAR']
    require(base.shape == candidate.shape == position.shape == far['MEAN'].shape,
            'group and Q dimensions changed')
    np.testing.assert_array_equal(base[..., :2], far['MEAN'][..., :2])
    np.testing.assert_array_equal(base[..., [0, 1, 3]], candidate[..., [0, 1, 3]])
    np.testing.assert_array_equal(base[..., 2][~far['far']], candidate[..., 2][~far['far']])
    np.testing.assert_array_equal(far['MEAN'][..., 2:], position[..., 2:])
    output = dict(B=base, FAR=candidate)
    for source, name in (('B', 'T128_B'), ('FAR', 'T128_FAR')):
        output[name] = output[source].copy()
        output[name][..., :2] = position[..., :2]
        np.testing.assert_array_equal(output[name][..., 2:], output[source][..., 2:])
    return output


def freeze_scene(runs, sid):
    with np.load(runs/C89/'private'/f'{sid}.npz', allow_pickle=False) as original, np.load(
            runs/PARENTS['mean']/'private'/f'policy-{sid}.npz', allow_pickle=False) as pack, np.load(
            runs/RUN157/'private'/f'raw-V_select-{sid}.npz', allow_pickle=False) as position:
        far = {key: original[key] for key in original.files}
        for new, old in (('nodes', 'node_ids'), ('query', 'query_seconds'), ('membership', 'membership')):
            np.testing.assert_array_equal(far[new], pack[old], err_msg=f'C89/157 {sid} {new}')
        np.testing.assert_array_equal(far['MEAN'], position['WS_MEAN'], err_msg=f'C89/157 {sid} Mean')
        output = assemble(far, position[POSITION])
    return dict(output, **{key: far[key] for key in ('nodes', 'query', 'membership', 'support', 'nearest', 'far')})


def errors(prediction, ev, mask):
    target = ev['target_states']
    result = point_errors(prediction, target, mask)
    for name, column in (('longitude', 1), ('latitude', 0)):
        valid = mask[..., column]
        result[name] = (prediction[..., column][valid]-target[..., column][valid])*10000.
    return result


def check_157(runs, sid, frozen, artifact):
    with np.load(runs/RUN157/'private'/f'states-V_select-{sid}.npz', allow_pickle=False) as old:
        fields = (*INVARIANTS, 'target_ids', 'aligned_truth', 'group_valid', 'node_group_indices',
                  'correspondence_policy', 'state_speed_unit', 'split')
        for field in fields:
            np.testing.assert_array_equal(artifact[field], old[f'{POSITION}_{field}'],
                                          err_msg=f'157/C94 {sid} {field}')
        np.testing.assert_array_equal(frozen['T128_FAR'][..., :2], old[f'{POSITION}_group_states'][..., :2])
        np.testing.assert_array_equal(artifact['aligned_prediction'][..., :2],
                                      old[f'{POSITION}_aligned_prediction'][..., :2])


def audit_near(runs, sid, frozen, ev, target, q, group):
    if sid not in HELD_IDS:
        return []
    with np.load(runs/C83/'private'/f'{sid}.npz', allow_pickle=False) as near:
        for field in ('nodes', 'query', 'membership', 'support', 'nearest', 'B', 'FAR', 'far'):
            np.testing.assert_array_equal(frozen[field], near[field])
        selected = near['near'][group, q]
        require(not (selected & frozen['far'][group, q]).any(), 'C83 near and FAR overlap')
        rows = []
        for t, g, slot in zip(target[selected], group[selected], q[selected], strict=True):
            reference = float(ev['target_states'][t, slot, 2])
            b = float(frozen['B'][g, slot, 2])
            near_speed = float(near['NEAR'][g, slot, 2])
            outputs = {name: float(frozen[name][g, slot, 2]) for name in ARMS}
            require(all(value == b for value in outputs.values()), 'a production arm changed near speed')
            rows.append(dict(scene=sid, group=int(g), query_index=int(slot), reference=reference,
                             speed=outputs, abs_error=abs(b-reference), near_control_speed=near_speed,
                             near_control_abs_error=abs(near_speed-reference),
                             near_control_error_change=abs(near_speed-reference)-abs(b-reference)))
    return rows


def evaluate_scene(runs, sid, component, frozen, ev, historical):
    groups = [g.tolist() for g in group_members(frozen['membership'])]
    report, artifacts = evaluate_fusion_methods(
        {name: emit_group_outputs(frozen[name], groups, node_count=len(frozen['nodes'])) for name in ARMS},
        prediction_speed_units=dict.fromkeys(ARMS, 'mps'), node_identities=ev['identities'],
        target_ids=ev['target_ids'], target_states=ev['target_states'], target_valid=ev['target_valid'],
        support=frozen['support'], nearest_seconds=frozen['nearest'], target_speed_unit='mps', split=SPLIT)
    n = len(ev['target_ids'])
    base = artifacts['B']
    coverage = {key: report['B'][key] for key in POPULATION_KEYS}
    point, paper = {}, {}
    for name in ARMS:
        a = artifacts[name]
        for field in (*INVARIANTS, 'aligned_truth'):
            np.testing.assert_array_equal(base[field], a[field])
        require(all(report[name][key] == value for key, value in coverage.items()), 'population changed')
        valid = a['aligned_prediction_valid'][:n] & ev['target_valid']
        point[name] = errors(a['aligned_prediction'][:n], ev, valid)
        paper[name] = report[name]['paper_fusion_metrics']
    require(coverage == historical['ws_coverage'] and all(
        paper[name] == historical['own_paper'][name] for name in ('B', 'FAR')), 'C89 published parity failed')
    check_157(runs, sid, frozen, artifacts['T128_FAR'])
    for name in POINT_FIELDS:
        first = ('FAR' if name in ('position', 'longitude', 'latitude', 'heading') else 'T128_B')
        second = ('T128_B' if name in ('position', 'longitude', 'latitude') else 'FAR')
        np.testing.assert_array_equal(point['B'][name], point[first][name])
        np.testing.assert_array_equal(point['T128_FAR'][name], point[second][name])
    target, q = np.nonzero(base['aligned_prediction_valid'][:n, :, 2] & ev['target_valid'][..., 2])
    group = base['aligned_group_indices'][target]
    require((group >= 0).all(), 'valid speed row lacks WS group')
    far = frozen['far'][group, q]
    near = audit_near(runs, sid, frozen, ev, target, q, group)
    return dict(scene=sid, component=component, population=coverage, unmatched_targets=report['B']['target_identity_count']
                - report['B']['anchor_matched_target_count'], paper=paper, near_points=len(near)), point, far, near


def change(before, after, threshold):
    delta = np.abs(after)-np.abs(before)
    return dict(improved=int((delta < 0).sum()), worsened=int((delta > 0).sum()),
                unchanged=int((delta == 0).sum()), harm_above_threshold=int((delta > threshold).sum()),
                threshold=threshold, max_harm=float(delta.max(initial=0.)))


def pooled(rows, points, far, indices):
    selected = [rows[i] for i in indices]
    vectors = {arm: {name: np.concatenate([points[i][arm][name] for i in indices]) for name in POINT_FIELDS}
               for arm in ARMS}
    position = change(vectors['B']['position'], vectors['T128_B']['position'], 1000.)
    speed = change(vectors['B']['speed'], vectors['FAR']['speed'], 1.)
    b_far, far_far = (np.concatenate([points[i][arm]['speed'][far[i]] for i in indices]) for arm in ('B', 'FAR'))
    return dict(scenes=len(indices), components=len({r['component'] for r in selected}),
                population={key: sum(r['population'][key] for r in selected) for key in POPULATION_KEYS},
                unmatched_targets=sum(r['unmatched_targets'] for r in selected),
                point={arm: {name: _stats(vector) for name, vector in values.items()}
                       for arm, values in vectors.items()},
                paper={arm: aggregate_paper_fusion([r['paper'][arm] for r in selected]) for arm in ARMS},
                position_change=position, speed_change=speed,
                far_speed=dict(B=_stats(b_far), FAR=_stats(far_far), change=change(b_far, far_far, 1.)))


def run(runs, output):
    require(output.parent == runs, 'new output directory must be under frozen runs')
    output.mkdir(exist_ok=False)
    private = output/'private'
    private.mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', argv=sys.argv, stages=[], split=SPLIT, raw_archive_reads=0,
                 optimizer_updates=0, gpu_used=False, test_opened=False, time_policy='advisory',
                 affinity=sorted(os.sched_getaffinity(0)), started_at=datetime.now(timezone.utc).isoformat())

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', state)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components, _ = load_full_population(runs)
        bind(runs, ids)
        checkpoint(phase='freeze_composition', scene_ids=ids)
        for i, sid in enumerate(ids, 1):
            np.savez_compressed(private/f'{sid}.npz', **freeze_scene(runs, sid))
            if i % 20 == 0:
                checkpoint(frozen_scenes=i)
        state['stages'].append('all_fixed_component_outputs_saved_before_reference')
        checkpoint(phase='evaluate', frozen_scenes=len(ids))
        prior = read(runs/C89/'scenes.json.gz')
        require([row['scene'] for row in prior] == ids, 'original C89 row order differs')
        rows, points, masks, near = [], [], [], []
        population = set(ids)
        for i, (sid, historical) in enumerate(zip(ids, prior, strict=True), 1):
            with np.load(private/f'{sid}.npz', allow_pickle=False) as saved:
                frozen = {key: saved[key] for key in saved.files}
            ev = read_evaluation(runs, sid, frozen['nodes'], frozen['query'], scene_ids=population)
            row, vector, mask, diagnostic = evaluate_scene(runs, sid, components[sid], frozen, ev, historical)
            rows.append(row)
            points.append(vector)
            masks.append(mask)
            near.extend(diagnostic)
            if i % 20 == 0:
                checkpoint(completed_scenes=i)
        reference = {c: sum(r['paper']['B']['attributes']['latitude']['available_target_points']
                            for r in rows if r['component'] == c) for c in set(components.values())}
        largest = min(reference, key=lambda c: (-reference[c], c))
        result = pooled(rows, points, masks, list(range(len(rows))))
        require((result['scenes'], result['components'], result['population']['target_identity_count'])
                == (300, 38, 9100), 'published population differs')
        require(len(near) == 21 and sum(row['near_control_error_change'] > 0. for row in near) == 5,
                'C83 near population or adverse count differs')
        near.sort(key=lambda r: (-r['abs_error'], r['scene'], r['group'], r['query_index']))
        diagnostics = dict(points=len(near), ordered_by_production_abs_error=near,
                           production_worst=near[0],
                           near_control_worst=max(near, key=lambda r: r['near_control_abs_error']),
                           near_control_largest_harm=max(near, key=lambda r: r['near_control_error_change']))

        def subset(predicate):
            return pooled(rows, points, masks, [i for i, row in enumerate(rows) if predicate(row)])

        for row, vector in zip(rows, points, strict=True):
            row['point'] = {arm: {name: _stats(values) for name, values in fields.items()}
                            for arm, fields in vector.items()}
            row['position_change'] = change(vector['B']['position'], vector['T128_B']['position'], 1000.)
            row['speed_change'] = change(vector['B']['speed'], vector['FAR']['speed'], 1.)
        write(output/'scenes.json.gz', rows)
        write(output/'summary.json', dict(arms=ARMS, primary='T128_FAR', all_scenes=result,
            held_v48=subset(lambda r: r['scene'] in HELD_IDS),
            remaining_v252=subset(lambda r: r['scene'] not in HELD_IDS),
            without_largest=dict(component=largest, result=subset(lambda r: r['component'] != largest)),
            by_component={c: subset(lambda r: r['component'] == c) for c in sorted(reference)}, near=diagnostics))
        state['stages'].append('all_c89_157_reference_group_and_point_coverage_verified')
        checkpoint(status='COMPLETED', phase='done', completed_scenes=len(rows), exit_code=0,
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs.resolve(), args.output.resolve())
