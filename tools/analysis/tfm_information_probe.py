"""Zero-update 049 TFM activation/conditional-response probe on frozen TCM embeddings.

Export 2--8 deterministic observation-only pairs from at most four explicit
049 validation scenes, or inspect a pre-exported left/right latent NPZ.
Use the original 049 step-0/100/300 snapshots and, for archive export, the 033
source checkpoint. No training, reference reads, or test-scene reads occur.
External NPZ provenance remains unknown; it cannot establish a T_test claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.training.tmtaff import load_complete_snapshot


def _dispersion(array):
    """Absolute RMS variability over samples and over time (not % information lost)."""
    array = np.asarray(array, np.float64)
    if not np.isfinite(array).all():
        raise ValueError('nonfinite internal activation')
    row = array.reshape(array.shape[0], -1)
    cross = float(np.sqrt(np.mean((row - row.mean(axis=0))**2)))
    result = dict(shape=list(array.shape), cross_sample_centered_rms=cross,
                  total_rms=float(np.sqrt(np.mean(array**2))),
                  pair_0_1_rms=float(np.sqrt(np.mean((row[0]-row[1])**2))))
    if array.ndim >= 3:
        centered = array - array.mean(axis=1, keepdims=True)
        result['within_sample_time_centered_rms'] = float(np.sqrt(np.mean(centered**2)))
        result['time_0_last_rms'] = float(np.sqrt(np.mean((array[:, 0]-array[:, -1])**2)))
    return result


def _circular_difference_deg(a, b):
    return (np.asarray(a, np.float64) - np.asarray(b, np.float64) + 180.) % 360. - 180.


def _heading_dispersion(heading):
    """Circular heading in degrees; 359 deg and 1 deg are 2 deg apart."""
    values = np.asarray(heading, np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError('heading dispersion expects [K,Q] finite, K>=2')
    angle = np.deg2rad(values)
    cross_r = np.hypot(np.mean(np.cos(angle), axis=0), np.mean(np.sin(angle), axis=0))
    time_r = np.hypot(np.mean(np.cos(angle), axis=1), np.mean(np.sin(angle), axis=1))
    # Circular standard deviation approximates linear standard deviation at
    # small angles and is bounded here to avoid undefined numerical infinity.
    def spread(resultant):
        return float(np.mean(np.rad2deg(np.sqrt(-2.*np.log(np.clip(resultant, 1e-12, 1.))))))
    return dict(unit='degrees',
        cross_sample_mean_circular_std_deg=spread(cross_r),
        within_sample_time_mean_circular_std_deg=spread(time_r),
        pair_0_1_mean_absolute_short_arc_deg=float(np.abs(_circular_difference_deg(values[0], values[1])).mean()),
        pair_0_1_max_absolute_short_arc_deg=float(np.abs(_circular_difference_deg(values[0], values[1])).max()))


def _position_metres(states):
    """Local north/east diagnostic centred at a fixed geographical origin.

    Longitude uses shortest arc to the fixed origin. This is not a global
    physical-error metric or uncertainty estimate; avoid projecting polar tracks.
    """
    values = np.asarray(states, np.float64)
    reference_lat = float(values[0, 0, 0]); reference_lon = float(values[0, 0, 1])
    latitude = values[..., 0]
    north = 6371000. * np.deg2rad(latitude - reference_lat)
    east = 6371000. * np.cos(np.deg2rad(reference_lat)) * np.deg2rad(
        _circular_difference_deg(values[..., 1], reference_lon))
    return np.stack((north, east), axis=-1)


def _physical_output_dispersion(states):
    """Never let variable speed/heading conceal a collapsed position output."""
    states = np.asarray(states, np.float64)
    if states.ndim != 3 or states.shape[-1] != 4 or states.shape[0] < 2:
        raise ValueError('physical output must be [K,Q,4] with K>=2')
    if not np.isfinite(states).all():
        raise ValueError('nonfinite physical output')
    pos = _position_metres(states)
    return dict(
        position_metres={**_dispersion(pos), 'unit':'local_east_north_metres',
            'max_pair_0_1_separation_metres':float(np.linalg.norm(pos[0]-pos[1], axis=-1).max()),
            'max_cross_sample_range_metres':float(np.linalg.norm(pos-pos[:1], axis=-1).max())},
        speed_native={**_dispersion(states[..., 2:3]), 'unit':'native_speed'},
        heading_circular=_heading_dispersion(states[..., 3]),
        diagnostic='Physical attributes assessed separately; position uses local tangent, heading uses circular differences')


def _collect(model, left, right, *, return_arrays=False):
    captured = {}
    decoder_last = []
    hooks = []

    def hook(name):
        def fn(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if name == 'decoder':
                decoder_last.append(value[:, -1].detach().cpu().numpy().copy())
            else:
                captured[name] = value.detach().cpu().numpy().copy()
        return fn

    for name in ('left_encoder', 'right_encoder', 'fc1', 'fc2', 'decoder', 'state_projection'):
        hooks.append(getattr(model, name).register_forward_hook(hook(name)))
    try:
        with torch.inference_mode():
            result = model(left, right)
    finally:
        for h in hooks:
            h.remove()
    if len(decoder_last) != model.config.length:
        raise RuntimeError('decoder hook did not observe exactly one query for each generated step')
    captured['decoder_generated_latents'] = np.stack(decoder_last, axis=1)
    captured['output_physical_states'] = result['states'].detach().cpu().numpy()
    captured['fused_memory'] = result['fused_features'].detach().cpu().numpy()
    result = {key: _dispersion(value) for key, value in captured.items()}
    result['output_physical_by_attribute'] = _physical_output_dispersion(captured['output_physical_states'])
    return (result, captured) if return_arrays else result


def _conditional_response(baseline, changed):
    """Matched output differences under a one-sided latent replacement."""
    stages = {}
    for key in baseline:
        if key == 'output_physical_states':
            continue  # Physical attributes have their own units below.
        delta = np.asarray(changed[key], np.float64) - np.asarray(baseline[key], np.float64)
        stages[key] = dict(rms=float(np.sqrt(np.mean(delta**2))), max_absolute=float(np.abs(delta).max()))
    old, new = baseline['output_physical_states'], changed['output_physical_states']
    north = 6371000. * np.deg2rad(new[..., 0].astype(np.float64)-old[..., 0])
    east = 6371000. * np.cos(np.deg2rad(old[..., 0])) * np.deg2rad(
        _circular_difference_deg(new[..., 1], old[..., 1]))
    distance = np.hypot(north, east)
    speed = new[..., 2].astype(np.float64)-old[..., 2]
    heading = _circular_difference_deg(new[..., 3], old[..., 3])
    stages['physical'] = dict(
        position_local_metres=dict(rms=float(np.sqrt(np.mean(distance**2))), maximum=float(distance.max())),
        speed_native=dict(rms=float(np.sqrt(np.mean(speed**2))), maximum=float(np.abs(speed).max())),
        heading_short_arc_degrees=dict(rms=float(np.sqrt(np.mean(heading**2))), maximum=float(np.abs(heading).max())))
    return stages


def _snapshot(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    cursor = saved.get('cursor', {})
    update = cursor.get('update')
    if (saved.get('format_version') != 1 or saved.get('input_adapter') != 'native_time_v1'
            or saved.get('phase') != 'tfm' or cursor.get('stage') != 2
            or update not in (0, 100, 300) or cursor.get('completed_updates') != update
            or saved.get('model_config', {}).get('length') != 64):
        raise ValueError('expected original native-time 049 TFM snapshots at update 0/100/300')
    updates = saved['config']['updates']
    if (updates != dict(TCM=0, TAM_FROZEN=0, TFM_FROZEN=300, JOINT=0)
            or [(row.get('stage'), row.get('update')) for row in saved.get('history', [])]
               != [('TFM_FROZEN', i) for i in range(1, update+1)]):
        raise ValueError('snapshot history/configuration does not describe the 049 frozen TFM run')
    return saved


def _same_completion(first, other):
    from association.models.tmtaff import ModelConfig
    # 033 predates explicit heading/dropout/loss-scale default fields. Compare
    # their model meaning just as the production initialization/loader does;
    # retain exact parameter, normalization and population checks below.
    if asdict(ModelConfig(**first['model_config'])) != asdict(ModelConfig(**other['model_config'])):
        raise ValueError('TFM snapshots disagree about frozen TCM input semantics: model_config')
    for key in ('normalizer', 'max_observations', 'input_adapter', 'data_contract'):
        if first.get(key) != other.get(key):
            raise ValueError('TFM snapshots disagree about frozen TCM input semantics: '+key)
    a, b = first['states'][0], other['states'][0]
    if set(a) != set(b) or any(not torch.equal(a[key], b[key]) for key in a):
        raise ValueError('TCM weights differ between frozen TFM snapshots/source')


def _export_pairs(args, snapshot, model):
    """Use the production native adapter without loading reference targets."""
    from association.data.archive import MTADArchive, validate_manifest
    from association.data.legacy_dataset import validate_archive
    from association.data.native import _record
    from association.data.timed import prepare_timed_scene
    from association.training.common import normalize_scene

    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    ids = args.scene_ids
    if (not ids or not 1 <= len(ids) <= 4 or len(set(ids)) != len(ids)
            or any(sid not in manifest['splits']['V_select'] for sid in ids)):
        raise ValueError('choose 1..4 distinct explicit original V_select scenes; T_test is sealed')
    if args.pairs_per_scene not in (1, 2) or not 2 <= len(ids)*args.pairs_per_scene <= 8:
        raise ValueError('request 2..8 pairs, with one or two fixed pairs per scene')
    contract = snapshot['data_contract']
    if ([row['scene_id'] for row in contract[1]] != manifest['splits']['V_select']
            or [row['scene_id'] for row in contract[0]] != snapshot['normalizer']['training_scene_ids']
            or snapshot['normalizer']['training_scene_ids'] != manifest['splits']['T_core']):
        raise ValueError('manifest does not match the original 049 training/validation population')
    data = {key: snapshot['normalizer']['adapter_config'][key]
            for key in ('length', 'max_observations', 'time_unit_seconds')}
    pairs_out, left, right = [], [], []
    model.eval().requires_grad_(False)
    with MTADArchive(args.archive) as archive:
        validate_archive(archive, manifest)
        for sid in ids:
            native = _record(archive.load_scene(sid), manifest['components'][sid], 'V_select',
                             data['max_observations'], split_unit=manifest.get('split_unit'))
            scene = prepare_timed_scene(native, **data)
            candidates = np.asarray(scene['pairs'])
            if len(candidates) < args.pairs_per_scene:
                raise ValueError(f'scene {sid}: insufficient pairs; do not silently resample scenes')
            indices = np.linspace(0, len(candidates)-1, args.pairs_per_scene, dtype=int)
            chosen = candidates[indices]
            nodes, inverse = np.unique(chosen, return_inverse=True)
            values, observed = normalize_scene(scene, snapshot['normalizer'])
            with torch.inference_mode():
                completed = model.complete(torch.as_tensor(values[nodes], dtype=torch.float32),
                                           torch.as_tensor(observed[nodes])).cpu().numpy()
            local_pairs = inverse.reshape(-1, 2)
            for (i, j), (a, b), candidate in zip(chosen, local_pairs, indices):
                left.append(completed[a]); right.append(completed[b])
                pairs_out.append(dict(scene_id=sid, candidate_index=int(candidate),
                    node_indices=[int(i), int(j)], node_ids=[scene['node_ids'][i], scene['node_ids'][j]],
                    sources=[scene['sources'][i], scene['sources'][j]],
                    query_start_seconds=float(scene['grid'][0]), query_stop_seconds=float(scene['grid'][-1])))
        members = list(archive.members_read)
    return np.stack(left), np.stack(right), dict(
        provenance='exported_from_original_frozen_TCM', split='V_select', scene_ids=ids,
        pairs=pairs_out, selection='evenly spaced candidate indices, one/two per fixed scene; no label or error selection',
        data=data, export_snapshot_cursor=snapshot['cursor'],
        checkpoint=str(Path(args.checkpoint[0]).resolve()), manifest=str(Path(args.manifest).resolve()),
        archive=[str(Path(p).resolve()) for p in args.archive], members_read=members,
        reference_read=False, test_opened=False, optimizer_updates=0, normalizer_refit=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--pairs-npz', help='external latent-only input; provenance is reported as unknown')
    source.add_argument('--archive', nargs='+', help='original 049 Level2 archive, used only for frozen diagnosis')
    p.add_argument('--manifest', help='original 049 manifest; required with --archive')
    p.add_argument('--scene-ids', nargs='+', help='one to four fixed original V_select scenes')
    p.add_argument('--pairs-per-scene', type=int, default=2)
    p.add_argument('--tcm-checkpoint', help='original 033 checkpoint; required with --archive')
    p.add_argument('--checkpoint', action='append', required=True, help='repeat in step 0,100,300 order')
    p.add_argument('--output', required=True)
    p.add_argument('--cpu-threads', type=int, default=2)
    args = p.parse_args(argv)
    if args.cpu_threads < 1 or len(args.checkpoint) != 3:
        raise ValueError('positive CPU threads and exactly three snapshots required')
    if args.archive and (not args.manifest or not args.scene_ids or not args.tcm_checkpoint):
        raise ValueError('--archive requires --manifest, --scene-ids and --tcm-checkpoint')
    torch.set_num_threads(args.cpu_threads)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='RUNNING', scope='049-frozen-conditional-response',
                  test_opened=False if args.archive else None, reference_read=False if args.archive else None,
                  optimizer_updates=0, checkpoints=[],
                  interpretation='RMS is a response diagnostic, not information lost; limited fixed samples are not full-V metrics')
    try:
        snapshots = [_snapshot(path) for path in args.checkpoint]
        if [s['cursor']['update'] for s in snapshots] != [0, 100, 300]:
            raise ValueError('provide distinct actual snapshots in 0,100,300 order')
        base = snapshots[0]
        for saved in snapshots[1:]:
            _same_completion(base, saved)
        if args.tcm_checkpoint:
            tcm = torch.load(args.tcm_checkpoint, map_location='cpu', weights_only=True)
            if tcm.get('phase') != 'tcm' or tcm.get('input_adapter') != 'native_time_v1':
                raise ValueError('original TCM checkpoint required')
            _same_completion(base, tcm)
            report['source_tcm'] = dict(path=str(Path(args.tcm_checkpoint).resolve()), cursor=tcm['cursor'],
                                        frozen_weights_identical=True)
            del tcm
        if args.archive:
            modules = load_complete_snapshot(args.checkpoint[0], device='cpu', selected=False)
            left, right, provenance = _export_pairs(args, base, modules['model'])
            np.savez_compressed(destination/'pairs.npz', left=left, right=right)
            (destination/'pairs-metadata.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
            del modules
        else:
            with np.load(args.pairs_npz, allow_pickle=False) as data:
                if set(data.files) != {'left', 'right'}:
                    raise ValueError('input NPZ must contain only left/right latent arrays')
                left, right = data['left'].copy(), data['right'].copy()
            provenance = dict(provenance='external_NPZ_unverified', path=str(Path(args.pairs_npz).resolve()))
        if (left.ndim != 3 or left.shape != right.shape or left.shape[1] != 64
                or not 2 <= len(left) <= 8 or not np.isfinite(left).all() or not np.isfinite(right).all()):
            raise ValueError('2..8 finite completed pair embeddings [K,64,D] required')
        report.update(input_pairs=len(left), input_provenance=provenance,
            first_two_inputs_nearly_identical=bool(np.allclose(left[0],left[1]) and np.allclose(right[0],right[1])),
            input_left=_dispersion(left), input_right=_dispersion(right))
        for checkpoint, saved in zip(args.checkpoint, snapshots):
            modules = load_complete_snapshot(checkpoint, device='cpu', selected=False)
            fusion = modules['fusion'].eval()
            if left.shape[2] != fusion.config.d_model:
                raise ValueError('TCM latent width does not match snapshot PairFusion')
            left_t, right_t = torch.as_tensor(left,dtype=torch.float32), torch.as_tensor(right,dtype=torch.float32)
            stats, baseline = _collect(fusion, left_t, right_t, return_arrays=True)
            responses = {}
            for name, changed_left, changed_right in (
                    ('fixed_left_replace_right',left_t,torch.roll(right_t,1,0)),
                    ('fixed_right_replace_left',torch.roll(left_t,1,0),right_t)):
                _, changed = _collect(fusion, changed_left, changed_right, return_arrays=True)
                responses[name] = dict(
                    replacement='one-row cyclic permutation; fixed opposite input, same model and eval mode',
                    input_left_rms=float(torch.sqrt(torch.mean((changed_left-left_t)**2))),
                    input_right_rms=float(torch.sqrt(torch.mean((changed_right-right_t)**2))),
                    stages=_conditional_response(baseline,changed))
            report['checkpoints'].append(dict(path=str(Path(checkpoint).resolve()),
                phase=saved['phase'], cursor=saved['cursor'], configured_updates=saved['config']['updates'],
                output_parameterization=saved['fusion_config'].get('output_parameterization','raw'),
                frozen_tcm_identical=True, stages=stats, controlled_responses=responses))
        report.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        report.update(status='FAILED', error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        (destination/'report.json').write_text(json.dumps(report, ensure_ascii=False,
                              indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('COMPLETED:', destination/'report.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
