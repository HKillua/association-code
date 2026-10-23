"""071 independent-arm audits; pairing is established after both runs finish.

No training, archive reads, threshold selection, or model forward occurs here.
The two arms never depend on one another while running.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from association.evaluation.support_gate import compare_endpoints
from association.evaluation.support_gate_linear import _component_split, _verify_route
from association.evaluation.support_gate_objectives import load_scene, summarize, verify_source_summaries
from association.training.evidence_fusion import TRAIN_MODES
from association.training.support_gate import configuration, load_checkpoint, loss_semantics
from association.utils.runtime import read_json, write_json, write_json_gzip

SCOPE = '071-independent-seed-position-gate-replication'
SAMPLE_KEYS = {'scene_id', 'anchor_index', 'node_id', 'members', 'eligible_query_points',
               'eligible_valid_position_points', 'valid_position_points'}
EXPOSURE_KEYS = ('successful_updates', 'sampled_anchors', 'eligible_query_points',
                 'eligible_valid_position_points', 'valid_position_points', 'scenes', 'anchors', 'modes')


def _sampling_exposure(exposure):
    result = {key: deepcopy(exposure[key]) for key in EXPOSURE_KEYS}
    for counters in result['modes'].values():
        for key in ('nonzero_gradient_updates', 'zero_eligible_updates'):
            counters.pop(key, None)
    return result


def _fit(path, expected_loss):
    """Validate each arm's own actual successful updates and checkpoint records."""
    path = Path(path).resolve()
    record = read_json(path/'run.json')
    config = configuration(record['configuration'])
    steps, size = config['training']['updates'], config['training']['batch_size']
    if (config.get('position_loss', 'log1p') != expected_loss
            or record.get('status') != 'COMPLETED' or record.get('optimizer_updates') != steps
            or record.get('loss_semantics') != loss_semantics(config)
            or record.get('test_opened') is not False or record.get('final_checkpoint') != 'final.pt'
            or record.get('exposure', {}).get('successful_updates') != steps):
        raise ValueError('pairing requires both completed fits at their configured final updates')
    scene_ids = record.get('training_scene_ids', [])
    if not scene_ids or len(set(scene_ids)) != len(scene_ids):
        raise ValueError('fit must retain distinct ordered training scene IDs')
    rows = [json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
    if len(rows) != steps:
        raise ValueError('every successful update must have an actual batch log')
    exposure = dict(successful_updates=0, sampled_anchors=0, eligible_query_points=0,
        eligible_valid_position_points=0, valid_position_points=0, scenes={}, anchors={},
        modes={mode: dict(updates=0, anchors=0, eligible_query_points=0,
                         eligible_valid_position_points=0, valid_position_points=0) for mode in TRAIN_MODES})
    for step, row in enumerate(rows, 1):
        mode = TRAIN_MODES[(step-1) % len(TRAIN_MODES)]
        if (row.get('step') != step or row.get('mode') != mode or len(row.get('batch', [])) != size
                or not np.isfinite(row.get('loss', np.nan)) or not np.isfinite(row.get('gradient_norm', np.nan))):
            raise ValueError('successful update order/mode/batch/loss is invalid')
        exposure['successful_updates'] += 1
        exposure['sampled_anchors'] += size
        exposure['modes'][mode]['updates'] += 1
        exposure['modes'][mode]['anchors'] += size
        for sample in row['batch']:
            if (set(sample) != SAMPLE_KEYS or sample['scene_id'] not in scene_ids
                    or type(sample['anchor_index']) is not int or sample['anchor_index'] < 0
                    or not isinstance(sample['node_id'], str) or not sample['node_id']
                    or not sample['members'] or sample['members'][0] != sample['anchor_index']
                    or len(set(sample['members'])) != len(sample['members'])
                    or any(type(x) is not int or x < 0 for x in sample['members'])):
                raise ValueError('actual sampled scene/anchor/member record is invalid')
            for key in ('eligible_query_points', 'eligible_valid_position_points', 'valid_position_points'):
                if type(sample[key]) is not int or not 0 <= sample[key] <= config['data']['length']:
                    raise ValueError('sample counts exceed the fixed query population')
                exposure[key] += sample[key]
                exposure['modes'][mode][key] += sample[key]
            if sample['eligible_valid_position_points'] > min(sample['eligible_query_points'], sample['valid_position_points']):
                raise ValueError('eligible valid points exceed available sampled points')
            for key, name in (('scenes', sample['scene_id']), ('anchors', sample['node_id'])):
                exposure[key][name] = exposure[key].get(name, 0) + 1
    if exposure != _sampling_exposure(record['exposure']):
        raise ValueError('actual successful batches do not reproduce recorded sampling exposure')
    with torch.random.fork_rng(devices=[]):
        _, initial = load_checkpoint(path/'step-0.pt', device='cpu')
        _, final = load_checkpoint(path/'final.pt', device='cpu')
    if any(snapshot['configuration'] != config or snapshot['step'] != expected
           for snapshot, expected in ((initial, 0), (final, steps))):
        raise ValueError('fit checkpoint endpoint/configuration mismatch')
    return record, rows, initial, final, exposure


def verify_pairing(log_fit, linear_fit):
    """Fail closed: equal seed alone is never evidence of paired actual exposure."""
    a, ar, ai, af, ae = _fit(log_fit, 'log1p')
    b, br, bi, bf, be = _fit(linear_fit, 'linear')
    ca, cb = deepcopy(a['configuration']), deepcopy(b['configuration'])
    ca.pop('position_loss', None); cb.pop('position_loss', None)
    if ca != cb or a['training_scene_ids'] != b['training_scene_ids']:
        raise ValueError('paired configs may differ only in loss; ordered training scenes must agree')
    if (set(ai['model']) != set(bi['model']) or any(
            ai['model'][key].shape != bi['model'][key].shape
            or ai['model'][key].dtype != bi['model'][key].dtype
            or not torch.equal(ai['model'][key], bi['model'][key]) for key in ai['model'])):
        raise ValueError('independent arms did not start with exactly equal model parameters')
    for left, right in zip(ar, br):
        if any(left[key] != right[key] for key in ('step', 'mode', 'batch')):
            raise ValueError(f"actual successful batch pairing differs at update {left['step']}")
    if ae != be or ai['numpy_rng'] != bi['numpy_rng'] or af['numpy_rng'] != bf['numpy_rng']:
        raise ValueError('paired sampling exposure or saved sampler RNG states differ')
    return dict(status='PAIR_VERIFIED', initial_parameters_exact=True,
        ordered_training_scene_ids_exact=True, actual_successful_batches_exact=True,
        sampling_exposure_exact=True, sampler_rng_exact=True, matched_updates=len(ar),
        matched_anchors=ae['sampled_anchors'], seed=ca['training']['seed'],
        log_fit=str(Path(log_fit).resolve()), linear_fit=str(Path(linear_fit).resolve()),
        model_forwards=0, test_opened=False,
        interpretation='Loss transform and its optimizer dynamics differ; gradient counts/weights need not match.')


def compare_cached_endpoints(initial, final, components, output=None, *, check=None):
    """Existing full-population physical and four-weight objective comparisons."""
    initial, final = Path(initial).resolve(), Path(final).resolve()
    old, new = read_json(initial/'summary.json'), read_json(final/'summary.json')
    ids = old['scene_ids']
    if new['scene_ids'] != ids or set(components) != set(ids):
        raise ValueError('endpoint comparison population/component registry differs')
    for directory in (initial, final):
        expected = {f'states-{sid}.npz' for sid in ids}
        for sub in ('predictions', 'system-predictions'):
            if {p.name for p in (directory/sub).glob('*.npz')} != expected:
                raise ValueError('endpoint inventory must contain all and only the declared V scenes')
    rows = []
    for sid in ids:
        if check:
            check()
        _verify_route(initial, sid)
        _verify_route(final, sid)
        rows.append(load_scene(initial, final, sid))
    verified = verify_source_summaries(rows, old, new)
    objective = summarize(rows, components)
    component_split = _component_split(rows, objective, components)
    scene_rows = objective.pop('per_scene')
    result = dict(status='COMPLETED', control=str(initial), candidate=str(final),
        model_forwards=0, test_opened=False, population_and_masks_exact=True,
        objectives=objective, largest_component_vs_remaining=component_split,
        physical=compare_endpoints(initial, final), **verified)
    if output is not None:
        output = Path(output).resolve()
        if any(output == p or output in p.parents or p in output.parents for p in (initial, final)):
            raise ValueError('comparison output must not overlap either endpoint')
        output.mkdir(parents=True, exist_ok=False)
        write_json_gzip(output/'objective-summary.json.gz', result)
        write_json_gzip(output/'scene-summary.json.gz', scene_rows)
    else:
        result['per_scene'] = scene_rows
    return result


def _completed_arm(root):
    root = Path(root).resolve()
    run, summary, fit = (read_json(root/name) for name in ('run.json', 'summary.json', 'fit/run.json'))
    config = configuration(run['configuration'])
    if any(record.get('status') != 'COMPLETED' or record.get('configuration') != config
           or record.get('test_opened') is not False
           or record.get('optimizer_updates') != config['training']['updates'] for record in (run, summary, fit)):
        raise ValueError('both full arm jobs must be completed, not only training')
    if (run.get('scope') != SCOPE or summary.get('scope') != SCOPE
            or run.get('source_system') != summary.get('source_system')
            or run.get('completed_endpoints') != ['initial', 'final']
            or run.get('validation_selection') is not False):
        raise ValueError('replication arm provenance/endpoint status differs')
    for label, step in (('initial', 0), ('final', config['training']['updates'])):
        for name in ('run.json', 'summary.json'):
            record = read_json(root/label/name)
            checkpoint = record.get('checkpoint', {})
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                    or record.get('source_system') != run['source_system'] or record.get('device') != 'cpu'
                    or record.get('test_opened') is not False or record.get('optimizer_updates') != 0
                    or checkpoint.get('step') != step or checkpoint.get('actual_updates') != step
                    or checkpoint.get('configured_final_updates') != config['training']['updates']
                    or checkpoint.get('loss_semantics') != loss_semantics(config)
                    or Path(checkpoint.get('path', '')).resolve() != root/'fit'/('step-0.pt' if label == 'initial' else 'final.pt')
                    or any(record.get(key) is not False for key in ('groups_recomputed', 'scores_recomputed', 'threshold_reselected'))):
                raise ValueError('CPU endpoint must bind this arm, checkpoint and frozen source')
    return run


def compare_replica_runs(log_run, linear_run, output, *, check=None):
    """Post-hoc CPU task; never claim pairing after incomplete or failed work."""
    log_run, linear_run, output = map(lambda x: Path(x).resolve(), (log_run, linear_run, output))
    if output.exists():
        raise FileExistsError('pair output already exists; no overwrite or automatic retry')
    sources = [Path(read_json(path/'run.json')['source_system']).resolve().parent
               for path in (log_run, linear_run)]
    for protected in (log_run, linear_run, *sources):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('pair output must be separate from immutable arm and056 runs')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='RUNNING', paired=False, log_run=str(log_run), linear_run=str(linear_run),
                 model_forwards=0, test_opened=False)
    write_json(output/'run.json', state)
    try:
        a, b = _completed_arm(log_run), _completed_arm(linear_run)
        if a['source_system'] != b['source_system']:
            raise ValueError('arms must use the same immutable056 source')
        pairing = verify_pairing(log_run/'fit', linear_run/'fit')
        first = read_json(log_run/'initial/summary.json')
        components = {sid: component for component, row in first['components'].items() for sid in row['scene_ids']}
        compare_cached_endpoints(log_run/'initial', linear_run/'initial', components, check=check)
        for sid in first['scene_ids']:
            with np.load(log_run/'initial/predictions'/f'states-{sid}.npz', allow_pickle=False) as x, \
                 np.load(linear_run/'initial/predictions'/f'states-{sid}.npz', allow_pickle=False) as y:
                for key in ('group_logits', 'group_gate', 'learned_gate_group_states'):
                    if not np.array_equal(x[key], y[key]):
                        raise ValueError('initial CPU model outputs must be exactly paired')
        final = compare_cached_endpoints(log_run/'final', linear_run/'final', components,
                                         output/'comparison', check=check)
        result = dict(status='COMPLETED', pairing=pairing, initial_cpu_outputs_exact=True,
            final_comparison=str(output/'comparison'), model_forwards=0, test_opened=False,
            scientific_status='One additional paired seed on reused V; not independent generalization.',
            final_population=final['objectives']['population'])
        write_json(output/'summary.json', result)
        state.update(status='COMPLETED', paired=True)
        write_json(output/'run.json', state)
        return result
    except BaseException as error:
        state.update(status='FAILED', paired=False, error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise
