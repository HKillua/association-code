"""C147: fresh TMTAFF-style TCM, TAM, and TFM on the original MTAD V300."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from association.data.physical_units import KNOT_TO_MPS, SPEED_UNIT_POLICY
from association.evaluation.association import _counts, predict
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.tmtaff import predicted_observed_controls
from association.training.tmtaff import load_complete_snapshot
from tools.analysis import held_fusion_population as population
from tools.analysis.offline_baseline_restart import aggregate, digest, residual_arrays, write
from tools.analysis.tmtaff_own_chain import INPUT_ARRAYS, predict_scenes


ARMS = ('OUR_QHARD', 'TMTAFF_OWN', 'TMTAFF_GROUP_MEAN')
BASE = 'LEVEL1-COMPLETION-SYSTEM-20261025-096A-attempt-1'
THRESHOLD = -1.6035346561693586e-05
OUR_THRESHOLD = 1.7320103112387182
PREDICTION_BATCH = 16
MODEL_ARGS = dict(batch_tracks=4, pair_batch_size=1024, fusion_batch_size=16)
PHASES = {
    'tcm': ('LEVEL1-TMTAFF-TCM-20261024-088-attempt-1', 2424),
    'tam': ('LEVEL1-COMPLETION-VALUE-TAM-20261024-094A-attempt-1', 4000),
    'tfm': ('LEVEL1-COMPLETION-VALUE-TFM-20261024-093A-attempt-1', 1000),
}
PREFIX = 'WS_SHARED_Q_HARD_'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def bound_checkpoints(runs, binding, ids, components, device):
    require(binding['normalizer_equal'] and binding['frozen_completion_equal']
            and binding['full_data_contract_equal'] and binding['final_tam_selection_bound']
            and binding['final_tfm_snapshot_bound'] and not binding['test_opened']
            and binding['completion_policy'] == 'learned' and binding['threshold'] == THRESHOLD,
            'original 096A verified modular source binding is required')
    paths = {phase: runs / folder for phase, (folder, _) in PHASES.items()}
    require({p: str(d) for p, d in paths.items()} == binding['source_runs'],
            '096A source run directories differ')
    saved = {}
    for phase, path in paths.items():
        run = population.read(path/'run.json')
        require(run['status'] == 'COMPLETED' and run['phase'] == phase
                and run['final_evaluation_status'] == 'COMPLETED' and not run['test_opened'],
                f'unfinished or incompatible source {phase}')
        ckpt = torch.load(path/'checkpoint.pt', map_location='cpu', weights_only=True)
        require(ckpt['cursor']['completed_updates'] == PHASES[phase][1],
                f'original checkpoint endpoint count differs for {phase}')
        saved[phase] = ckpt
    tcm, tam, tfm = (saved[phase] for phase in ('tcm', 'tam', 'tfm'))
    for key in ('normalizer', 'data_contract', 'model_config', 'fusion_config', 'input_adapter',
                'max_observations', 'split_unit', 'completion_policy'):
        require(tam[key] == tfm[key], f'TAM and TFM differ at {key}')
    require(tam['normalizer'] == tcm['normalizer'] == binding['normalizer'],
            'training normalizers differ')
    for p in ('tam', 'tfm'):
        require(saved[p]['initialization']['source_run'] == str(paths['tcm']), 'TCM source differs')
        require(saved[p]['config']['validation_batches'] == {'tracks': 4, 'pairs': 1024}
                and saved[p]['config']['singleton_policy'] == 'observed',
                'historical configuration differs')
    validation = tam['data_contract'][1]
    require([str(row['scene_id']) for row in validation] == ids
            and [str(row['component_id']) for row in validation] == [components[sid] for sid in ids]
            and all(row['split'] == population.SPLIT for row in validation),
            'checkpoint scene registry differs')
    require(tam['selected']['threshold'] == THRESHOLD and tam['selected']['stage'] == 'TAM_FROZEN'
            and tam['selected']['update'] == PHASES['tam'][1]
            and tam['selected']['score_records'] == binding['original_score_records'],
            'original own-TAM threshold and final-score source differ')
    for ckpt, module in ((tam, 0), (tfm, 1)):
        terminal, selected = ckpt['states'][module], ckpt['selected']['states'][module]
        require(terminal.keys() == selected.keys() and all(torch.equal(terminal[k], selected[k]) for k in terminal),
                'final selected weights disagree with the terminal state')
    common = [k for k in tam['states'][0] if k in ('positions', 'heading_mean_std')
              or k.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.'))]
    require(len(common) == binding['frozen_completion_tensor_count'] and all(
        torch.equal(tam['states'][0][k], tcm['states'][0][k]) and
        torch.equal(tam['states'][0][k], tfm['states'][0][k]) for k in common),
            'shared learned completion parameters differ')
    snapshot = load_complete_snapshot(paths['tam']/'checkpoint.pt', selected=False, device=device)
    snapshot['fusion'].load_state_dict(tfm['states'][1], strict=True)
    return snapshot


def original_input(runs, prior, entry, sid, component):
    meta = population.read(prior/'predictions'/entry['metadata'])
    require(meta['scene_id'] == sid and meta['split'] == population.SPLIT
            and meta['component_id'] == component and not meta['test_opened']
            and meta['input_adapter'] == 'native_time_v1' and meta['npz'] == entry['npz'],
            '096A prediction input metadata differs')
    with np.load(prior/'predictions'/entry['npz'], allow_pickle=False) as archive:
        scene = {key: archive['scene__'+key] for key in (*INPUT_ARRAYS, 'pairs')}
    scene.update(scene_id=sid, split=population.SPLIT)
    motion = runs/population.PARENTS['full_data']/'private/motion'/population.SPLIT/f'scene-{sid}.npz'
    with np.load(motion, allow_pickle=False) as saved:
        for key, name in (('node_ids', 'node_ids'), ('query_seconds', 'grid'),
                          ('sources', 'sources'), ('cap128_support', 'observed')):
            np.testing.assert_array_equal(saved[key], scene[name], err_msg=f'{sid} {key}')
        require(str(saved['state_speed_unit']) == 'mps', '151 measurement state unit differs')
        mask = scene['observed']
        difference = scene['values'][mask].copy()
        difference[:, 2] *= KNOT_TO_MPS
        difference -= saved['cap128_states'][mask]
        difference[:, [1, 3]] = (difference[:, [1, 3]]+180.) % 360. - 180.
        np.testing.assert_allclose(difference, 0., rtol=0., atol=1e-9, err_msg=f'{sid} CAP128 inputs')
        original_points = int(saved['native_count'].sum())
        canonical_support = saved['support']
    left, right = np.triu_indices(len(scene['node_ids']), 1)
    np.testing.assert_array_equal(scene['pairs'], np.column_stack((left, right)))
    counts = dict(cap128_reports=int(scene['native_observed'].sum()), canonical_reports=original_points)
    return scene, canonical_support, counts


def write_prediction(output, runs, sid, scene, canonical_support, tm):
    groups = tm['groups']
    ptr = np.r_[0, np.cumsum([len(group) for group in groups])]
    flat = np.concatenate(groups).astype(np.int64)
    states = tm['states'].copy()
    singleton = np.diff(ptr) == 1
    states[singleton, :, 2] *= KNOT_TO_MPS
    mean = predicted_observed_controls(scene['values'], scene['observed'], groups)['predicted_mean']
    mean_states = np.stack([group['states'] for group in mean])
    mean_states[..., 2] *= KNOT_TO_MPS
    ws_path = runs/population.PARENTS['mean']/'private'/f'states-{sid}.npz'
    with np.load(ws_path, allow_pickle=False) as old:
        require(str(old[PREFIX+'state_speed_unit']) == 'mps' and str(old[PREFIX+'split']) == population.SPLIT,
                'Qhard state unit/role differs')
        ours = {k: old[PREFIX+k] for k in ('group_offsets', 'group_node_indices', 'group_states', 'group_valid')}
    partition = population.read(runs/population.PARENTS['partition']/'private'/f'scene-{sid}.json.gz')
    membership = np.asarray(partition['memberships']['WS'])
    for a, b in zip(ours['group_offsets'][:-1], ours['group_offsets'][1:], strict=True):
        members = ours['group_node_indices'][a:b]
        require(len(np.unique(membership[members])) == 1, 'Qhard grouping differs from WS')
    require(len(np.unique(membership)) == len(ours['group_states']), 'Qhard grouping omits an original WS group')
    np.savez_compressed(output, nodes=scene['node_ids'], sources=scene['sources'], query=scene['grid'],
                        pairs=scene['pairs'], cap_support=scene['observed'], canonical_support=canonical_support,
                        tam_scores=tm['scores'], tm_offsets=ptr, tm_nodes=flat, tm_states=states, tm_valid=tm['valid'],
                        mean_states=mean_states, mean_valid=np.stack([g['prediction_valid'] for g in mean]), **ours)


def group_records(offsets, nodes, states, valid):
    return [dict(node_indices=nodes[a:b], states=value, prediction_valid=mask) for a, b, value, mask in
            zip(offsets[:-1], offsets[1:], states, valid, strict=True)]


def association_rows(groups, pairs, truth, sources, scores, threshold, sid, component):
    membership = np.empty(len(sources), np.int64)
    for g, group in enumerate(groups):
        membership[group['node_indices']] = g
    first, second = pairs.T
    same = sources[first] == sources[second]
    co_grouped = membership[first] == membership[second]
    scores_applied = predict(scores, threshold)
    rows = {}
    for scope, mask in (('ITSA', same), ('MSTSA', ~same)):
        fixed = dict(scene_id=sid, component_id=component, pair_count=int(mask.sum()),
                     positive_count=int(truth[mask].sum()))
        for label, decision in (('', co_grouped), ('_raw', scores_applied)):
            rows[scope+label] = dict(**fixed, **_counts(truth[mask], decision[mask]))
    return rows


def compare_historical(old, prefix, old_tam, new, reference):
    target_ids = old[prefix+'target_ids']
    index = {key: i for i, key in enumerate(target_ids.tolist())}
    require(len(index) == len(target_ids) == len(reference['target_ids']), 'historical target count differs')
    order = [index[key] for key in reference['target_ids'].tolist()]
    valid = reference['target_valid']
    np.testing.assert_array_equal(old[prefix+'aligned_truth_valid'][order], valid)
    np.testing.assert_array_equal(old[prefix+'aligned_truth'][order][valid], reference['target_states'][valid])
    np.testing.assert_array_equal(old['scene__identities'], reference['identities'])
    np.testing.assert_array_equal(old_tam['pairs'], new['pairs'])
    np.testing.assert_array_equal(old_tam['sources'], new['sources'])
    np.testing.assert_array_equal(old_tam['identities'], reference['identities'])
    current = new['tam_scores']
    previous = old_tam['scores']
    different = predict(current, THRESHOLD) != predict(previous, THRESHOLD)
    unchanged = (np.array_equal(old[prefix+'group_offsets'], new['tm_offsets']) and
                 np.array_equal(old[prefix+'group_node_indices'], new['tm_nodes']))
    row = dict(edges=len(previous), threshold_drift=int(different.sum()),
               threshold_old_boundary_max=float(np.abs(previous[different]-THRESHOLD).max(initial=0)),
               score_abs_max=float(np.abs(current-previous).max(initial=0)),
               unchanged_groups=int(unchanged), historical_groups=len(old[prefix+'group_states']))
    if unchanged:
        previous_states = old[prefix+'group_states'].copy()
        previous_states[np.diff(new['tm_offsets']) == 1, :, 2] *= KNOT_TO_MPS
        support = new['tm_valid']
        np.testing.assert_array_equal(old[prefix+'group_valid'], support)
        err = attribute_errors(new['tm_states'], previous_states)
        row.update(state_replay_max_position_m=float(err['position'][support[..., :2].all(-1)].max(initial=0)),
                   state_replay_max_speed_mps=float(err['speed'][support[..., 2]].max(initial=0)),
                   state_replay_max_heading_degrees=float(err['heading'][support[..., 3]].max(initial=0)))
    return row


def score_scene(saved, reference, sid, component, ours, expected_labels):
    left, right = saved['pairs'].T
    truth = reference['identities'][left] == reference['identities'][right]
    np.testing.assert_array_equal(truth, expected_labels)
    target_count = len(reference['target_ids'])
    options = dict(node_identities=reference['identities'], target_identities=reference['target_ids'],
                   target_states=reference['target_states'], target_valid=reference['target_valid'],
                   split=population.SPLIT, grouping_kind='predicted', correspondence_policy='anchor_identity_first',
                   return_artifact=True, speed_unit_policy=SPEED_UNIT_POLICY)
    rows = {}
    for name in ARMS:
        is_ours = name == 'OUR_QHARD'
        arrays = ('group_offsets', 'group_node_indices', 'group_states', 'group_valid') if is_ours else (
            'tm_offsets', 'tm_nodes', 'tm_states', 'tm_valid') if name == 'TMTAFF_OWN' else (
                'tm_offsets', 'tm_nodes', 'mean_states', 'mean_valid')
        groups = group_records(*(saved[k] for k in arrays))
        result = evaluate_full_scene(groups, node_observed=saved['canonical_support'], **options)
        errors, available = residual_arrays(result['artifact'])
        rows[name] = dict(errors=errors[:target_count], reference_mask=available[:target_count], groups=len(groups),
                          duplicates=result['duplicate_prediction_group_count'],
                          unmatched_targets=len(result['unmatched_target_ids']), paper=result['paper_fusion_metrics'],
                          association=association_rows(
                              groups, saved['pairs'], truth, saved['sources'], ours if is_ours else saved['tam_scores'],
                              OUR_THRESHOLD if is_ours else THRESHOLD, sid, component))
    reference_mask = rows[ARMS[0]]['reference_mask']
    for row in rows.values():
        np.testing.assert_array_equal(row.pop('reference_mask'), reference_mask)
    common = reference_mask & np.logical_and.reduce([np.isfinite(row['errors']) for row in rows.values()])
    for row in rows.values():
        row['common'] = np.where(common, row['errors'], np.nan)
        row['own'] = np.where(reference_mask, row.pop('errors'), np.nan)
        row['available'] = reference_mask.sum(axis=(0, 1))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True, choices=['cuda:0'])
    parser.add_argument('--time-policy', required=True, choices=['advisory'])
    args = parser.parse_args()
    wall = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    runs, output = args.runs.resolve(), args.output.resolve()
    private = output/'private'
    output.mkdir()
    private.mkdir(mode=0o700)
    errors = private/'errors'
    errors.mkdir()
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    state = dict(status='LOADING', code=code, started=started, device=args.device, time_policy=args.time_policy)
    write(output/'run.json', state)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    prior = runs/BASE
    ids, components, _ = population.load_full_population(runs)
    historical = population.read(prior/'run.json')
    binding = population.read(prior/'source-binding.json')
    pred_index = population.read(prior/'predictions/index.json')
    entries = pred_index['scenes']
    require(historical['status'] == 'COMPLETED' and not historical['test_opened'] and not pred_index['test_opened']
            and binding['validation_scene_ids'] == ids and [row['scene_id'] for row in entries] == ids,
            'original 096A V300 source required')
    snapshot = bound_checkpoints(runs, binding, ids, components, args.device)
    state.update(status='PREDICTING', scenes=len(ids), device_name=torch.cuda.get_device_name(args.device))
    write(output/'run.json', state)
    diagnostics, timings, hashes = {}, [], {}
    for offset in range(0, len(ids), PREDICTION_BATCH):
        chunk = list(range(offset, min(offset+PREDICTION_BATCH, len(ids))))
        pack = [original_input(runs, prior, entries[i], ids[i], components[ids[i]]) for i in chunk]
        arms, timing = predict_scenes(snapshot, [s for s, _, _ in pack], threshold=THRESHOLD,
                                      device=args.device, **MODEL_ARGS)
        timings.append(timing)
        for i, (scene, support, stats), prediction in zip(chunk, pack, arms, strict=True):
            sid = ids[i]
            artifact = private/f'scene-{sid}.npz'
            write_prediction(artifact, runs, sid, scene, support, prediction)
            hashes[sid] = digest(artifact)
            diagnostics[sid] = dict(**stats, logical_pair_fusions=prediction['steps'],
                                    terminal_group_decodes=sum(len(g) > 1 for g in prediction['groups']),
                                    **prediction['timing_seconds'])
        print(f'prediction {chunk[-1]+1}/{len(ids)}', flush=True)
    write(output/'freeze.json', dict(code=code, scene_ids=ids, components=components, sha256=hashes,
          split=population.SPLIT, source=BASE, source_runs=binding['source_runs'], threshold=THRESHOLD,
          own_threshold=OUR_THRESHOLD, model_args=MODEL_ARGS, test_opened=False, cached_tam_used_for_prediction=False))
    state.update(status='SCORING', peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device))
    write(output/'run.json', state)
    tm_cache = Path(binding['original_score_records'])
    tm_index = population.read(tm_cache/'index.json')
    require([row['scene_id'] for row in tm_index['records']] == ids and
            tm_index['binding']['threshold'] == THRESHOLD, 'the original 094A score registry is required for parity')
    ours = runs/population.PARENTS['association']/'association'
    require(population.read(ours/'summary.json')['final_selection']['threshold'] == OUR_THRESHOLD,
            'original 056 threshold differs')
    results, scenes = [], []
    with np.load(ours/'final-scores.npz', allow_pickle=False) as qhard_tam:
        np.testing.assert_array_equal(qhard_tam['scene_ids'], ids)
        for i, (sid, tm_record) in enumerate(zip(ids, tm_index['records'], strict=True)):
            path = private/f'scene-{sid}.npz'
            require(digest(path) == hashes[sid], 'prediction changed after freeze')
            metadata = population.read(prior/'predictions'/entries[i]['metadata'])
            prefix = metadata['methods']['predicted']['prefix']
            with (np.load(path, allow_pickle=False) as saved,
                  np.load(prior/'predictions'/entries[i]['npz'], allow_pickle=False) as old,
                  np.load(tm_cache/tm_record['file'], allow_pickle=False) as old_tam):
                reference = population.read_evaluation(runs, sid, saved['nodes'], saved['query'], scene_ids=ids)
                np.testing.assert_array_equal(tm_record['node_ids'], saved['nodes'])
                for key, k in (('node_ids', 'nodes'), ('pairs', 'pairs'), ('sources', 'sources')):
                    np.testing.assert_array_equal(qhard_tam[f'{sid}_{key}'], saved[k])
                parity = compare_historical(old, prefix, old_tam, saved, reference)
                rows = score_scene(saved, reference, sid, components[sid],
                                   qhard_tam[f'{sid}_scores'], qhard_tam[f'{sid}_labels'])
            results.append(rows)
            scenes.append(dict(scene_id=sid, component_id=components[sid], historical_parity=parity,
                               diagnostics=diagnostics[sid], arms=aggregate([rows], paper=False, arms=ARMS)))
            np.savez_compressed(errors/f'scene-{sid}.npz', **{
                f'{arm}_{scope}': rows[arm][scope] for arm in ARMS for scope in ('own', 'common')})
            if (i+1) % 25 == 0:
                print(f'evaluation {i+1}/{len(ids)}', flush=True)
    component_sizes = Counter(components.values())
    largest = min(component_sizes, key=lambda key: (-component_sizes[key], key))
    by_component = {key: [row for row, sid in zip(results, ids, strict=True) if components[sid] == key]
                    for key in sorted(component_sizes)}
    write(output/'scenes.json.gz', scenes)
    write(output/'summary.json', dict(arms=aggregate(results, arms=ARMS), by_component={
              key: aggregate(rows, paper=False, arms=ARMS) for key, rows in by_component.items()},
          without_largest=aggregate([row for row, sid in zip(results, ids, strict=True)
                                     if components[sid] != largest], arms=ARMS),
          component_sizes=dict(component_sizes), excluded_largest_component=largest,
          timing=dict(fusion_batches=timings), test_opened=False, optimizer_updates=0))
    state.update(status='COMPLETED', exit_code=0, split=population.SPLIT, elapsed_seconds=time.monotonic()-wall,
                 ended=datetime.now(timezone.utc).isoformat(), optimizer_updates=0,
                 test_opened=False, original_raw_archive_opened=False, new_tam_scoring=True,
                 new_tcm_forward=True, new_tfm_forward=True)
    write(output/'run.json', state)


if __name__ == '__main__':
    main()
