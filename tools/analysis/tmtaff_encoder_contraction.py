"""Frozen TFM layer trace and a same-snapshot paired contrast intervention."""

import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from tools.analysis import tmtaff_feedback_gradient_qualification as feedback
from tools.analysis import tmtaff_memory_prefix_diagnostic as previous
from tools.analysis import tmtaff_true_pair_qualification as fixed


IDEA = 'IDEA-TMTAFF-ENCODER-CONTRACTION-20261112'
C96 = 'LEVEL1-TMTAFF-MEMORY-PREFIX-20261112-attempt-1'
SOURCES = ('ORIGINAL-u000', 'ORIGINAL-u032', 'DETACHED_HISTORY-u032')
TRANSITIONS = (('input', 'attn_residual'), ('attn_residual', 'norm1'),
               ('norm1', 'ffn_residual'), ('ffn_residual', 'norm2'))


def trace_forward(model, left, right):
    stages = {}
    hooks = []

    def capture(module, key, *, incoming=False):
        def output(_module, args, value):
            stages[key] = (args[0] if incoming else value).detach()
        hooks.append(module.register_forward_hook(output))

    for side in ('left', 'right'):
        for index, layer in enumerate(getattr(model, f'{side}_encoder').layers):
            prefix = f'{side}.{index}.'
            for module, name, incoming in (
                (layer, 'input', True), (layer.dropout1, 'attn_branch', False),
                (layer.norm1, 'attn_residual', True), (layer.norm1, 'norm1', False),
                (layer.linear1, 'ffn_expand', False), (layer.linear2, 'ffn_hidden', True),
                (layer.dropout2, 'ffn_branch', False), (layer.norm2, 'ffn_residual', True),
                (layer.norm2, 'norm2', False),
            ):
                capture(module, prefix + name, incoming=incoming)
    capture(model.fc1, 'fc_input', incoming=True)
    capture(model.fc1, 'fc1')
    capture(model.fc_activation, 'fc_activation')
    capture(model.fc2, 'fc2')
    try:
        native = previous.capture(model, left, right)
    finally:
        for hook in hooks:
            hook.remove()
    for side in ('left', 'right'):
        for index in range(model.config.num_layers):
            prefix = f'{side}.{index}.'
            for entry, branch, added in (('input', 'attn_branch', 'attn_residual'),
                                         ('norm1', 'ffn_branch', 'ffn_residual')):
                torch.testing.assert_close(stages[prefix + entry] + stages[prefix + branch],
                                           stages[prefix + added], atol=0, rtol=0)
    for key in ('left_input', 'right_input', 'left_encoder', 'right_encoder', 'memory'):
        stages[key] = native[key]
    for key in ('fc_input', 'fc1', 'fc_activation', 'fc2'):
        stages[key] = stages[key][:, None]
    torch.testing.assert_close(stages['fc2'].reshape_as(stages['memory']), stages['memory'], atol=0, rtol=0)
    return stages, native


def difference(first, second):
    first, second = first.astype(np.float64), second.astype(np.float64)
    absolute = previous.rms(first - second)
    energy = np.sqrt((previous.rms(first) ** 2 + previous.rms(second) ** 2) / 2)
    return {'absolute': absolute, 'energy': energy, 'relative': previous.ratio(absolute, energy)}


def trace_values(stages):
    return {f'{stage}/{kind}': data for stage, tensor in stages.items()
            for kind, data in difference(tensor, tensor[np.arange(len(tensor)) ^ 1]).items()}


def group_masks(scenes):
    distinct = np.repeat([row['distinct_targets'] for row in scenes], 2)
    return {'distinct': distinct, 'same': ~distinct}


def trace_summary(values, scenes):
    return {group: {name: fixed.statistics(array[mask].reshape(-1)) for name, array in values.items()}
            for group, mask in group_masks(scenes).items()}


def select_boundary(trace, layers):
    distinct = trace['distinct']
    rows = []
    for side in ('left', 'right'):
        for layer in range(layers):
            for entry, exit_name in TRANSITIONS:
                prefix = f'{side}.{layer}.'
                before = distinct[prefix + entry + '/relative']['mae']
                after = distinct[prefix + exit_name + '/relative']['mae']
                rows.append(dict(side=side, layer=layer, entry=entry, exit=exit_name,
                                 ratio=after / before if before else None))
    candidates = [row for row in rows if row['ratio'] is not None and row['ratio'] < 1]
    return (min(candidates, key=lambda row: row['ratio']) if candidates else None), rows


def preserve_midpoint(entry, exit_value):
    entry64, exit64 = entry.double(), exit_value.double()
    donor = torch.arange(len(entry64), device=entry64.device) ^ 1
    entry_energy = ((entry64.square() + entry64[donor].square()).mean(-1, keepdim=True) / 2).sqrt()
    exit_energy = ((exit64.square() + exit64[donor].square()).mean(-1, keepdim=True) / 2).sqrt()
    scale = exit_energy / entry_energy.clamp_min(torch.finfo(entry64.dtype).tiny)
    centered = (entry64 - entry64[donor]) / 2
    midpoint = (exit64 + exit64[donor]) / 2
    restored = (midpoint + centered * scale).to(exit_value)
    new_midpoint = (restored.double() + restored[donor].double()) / 2
    return restored, float((new_midpoint - midpoint).abs().max())


def replay(model, left, right, boundary, value):
    layer = getattr(model, f"{boundary['side']}_encoder").layers[boundary['layer']]
    name = boundary['exit']
    if name in ('norm1', 'norm2'):
        hook = getattr(layer, name).register_forward_hook(lambda _module, _args, _out: value)
    else:
        target = layer.norm1 if name == 'attn_residual' else layer.norm2
        hook = target.register_forward_pre_hook(lambda _module, _args: (value,))
    try:
        return previous.capture(model, left, right)
    finally:
        hook.remove()


def compare_replay(model, left, right, stages, native, boundary):
    prefix = f"{boundary['side']}.{boundary['layer']}."
    entry, exit_value = (stages[prefix + boundary[key]] for key in ('entry', 'exit'))
    unchanged = replay(model, left, right, boundary, exit_value)
    for key in ('left_encoder', 'right_encoder', 'memory', 'native', 'states'):
        torch.testing.assert_close(unchanged[key], native[key], rtol=0, atol=0)
    patched, midpoint = preserve_midpoint(entry, exit_value)
    patched_output = replay(model, left, right, boundary, patched)
    arrays = {name: tensor.cpu().numpy() for name, tensor in (
        ('native_boundary', exit_value), ('patched_boundary', patched),
        ('native_states', native['states']), ('patched_states', patched_output['states']),
        ('native_memory', native['memory']), ('patched_memory', patched_output['memory']),
        ('native_latent', native['native']), ('patched_latent', patched_output['native']),
        ('native_standardized', previous.output_positions(model, native['native'])[0]),
        ('patched_standardized', previous.output_positions(model, patched_output['native'])[0]),
    )}
    fixed.require(all(np.isfinite(array).all() for array in arrays.values()), 'nonfinite frozen replay')
    return arrays, midpoint


def response_values(arrays):
    donor = np.arange(len(arrays['native_states'])) ^ 1
    measures = {}
    for arm in ('native', 'patched'):
        for name in ('boundary', 'memory', 'latent', 'standardized'):
            array = arrays[f'{arm}_{name}']
            measures[f'{arm}_{name}_pair_relative'] = difference(array, array[donor])['relative']
        state = arrays[f'{arm}_states']
        measures[f'{arm}_pair_m'] = previous.distance(state, state[donor])
    for name in ('boundary', 'memory', 'latent', 'standardized'):
        delta = difference(arrays[f'patched_{name}'], arrays[f'native_{name}'])
        measures[f'{name}_response_rms'] = delta['absolute']
        measures[f'{name}_response_relative'] = delta['relative']
    measures['physical_response_m'] = previous.distance(arrays['patched_states'], arrays['native_states'])
    return measures


def run(args, progress):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    fixed.require(device.type == 'cuda' and torch.cuda.is_available(), 'assigned CUDA device required')
    private = args.output / 'private'
    private.mkdir(mode=0o700)
    roots = {name: args.runs / value for name, value in fixed.RUNS.items()}
    fixed.bind_raw_audit(args.runs / feedback.RAW_AUDIT / 'summary.json')
    model, normalizer, config, contracts, _, _ = fixed.source_checkpoints(roots['checkpoint'], device)
    population, scenes = fixed.load_population(roots, model, normalizer, contracts, device)
    del model
    fixed.bind_reference_population(args.runs, population)
    feedback.bind_c91(args.runs / feedback.C91, population)
    np.testing.assert_array_equal(population['query'][::2], population['query'][1::2])
    fixed.require(config.output_parameterization == 'train_position_affine', 'unexpected physical projection')
    c93 = args.runs / previous.C93 / 'private'
    c96 = args.runs / C96
    c96_run = json.loads((c96 / 'run.json').read_text())
    fixed.require(c96_run['status'] == 'COMPLETED' and c96_run['optimizer_updates'] == 0, 'C96 receipt differs')
    np.savez_compressed(private / 'population.npz', valid=population['valid'], support=population['support'],
                        truth=population['truth'], **{key: population[key] for key in fixed.CONTROLS})
    snapshots, boundary, candidates = {}, None, []
    order = (1, 0, 2)
    for index in order:
        snapshot, source = previous.SNAPSHOTS[index], SOURCES[index]
        progress('trace_snapshot', snapshot=snapshot)
        torch.manual_seed(fixed.SEED)
        fusion = previous.PairFusion(config, normalizer).to(device).eval()
        if index:
            payload = torch.load(c93 / f'{source}.pt', map_location='cpu', weights_only=True)
            fusion.load_state_dict(payload['fusion'])
            del payload
        chunks, traces, states = [], [], []
        for start in (0, fixed.MICROBATCH):
            part = slice(start, start + fixed.MICROBATCH)
            left, right = (population[key][part] for key in ('left', 'right'))
            stages, native = trace_forward(fusion, left, right)
            traces.append(trace_values({key: tensor.cpu().numpy() for key, tensor in stages.items()}))
            states.append(native['states'].cpu().numpy())
            chunks.append((left, right, stages, native))
        states = np.concatenate(states)
        np.testing.assert_array_equal(states, fixed.read_npz(c93 / f'{source}.npz')['states'],
                                      err_msg=f'{snapshot} original C93 parity')
        values = {key: np.concatenate([row[key] for row in traces]) for key in traces[0]}
        fixed.require(all(np.isfinite(array).all() for array in values.values()), 'nonfinite layer trace')
        c96_values = fixed.read_npz(c96 / 'private' / f'{snapshot}.npz')
        for key in ('left_input', 'right_input', 'left_encoder', 'right_encoder', 'memory'):
            np.testing.assert_allclose(values[key + '/relative'], c96_values[key + '_relative_rms'],
                                       atol=1e-7, rtol=1e-6, err_msg=key)
        trace = trace_summary(values, scenes)
        np.savez_compressed(private / f'{snapshot}-trace.npz', **values)
        if index == order[0]:
            boundary, candidates = select_boundary(trace, config.num_layers)
            fixed.write_json(args.output / 'selection.json', dict(boundary=boundary, candidates=candidates))
        row = dict(trace=trace, native_metrics=fixed.metrics(states, population, scenes, support_detail=True))
        if boundary is not None:
            progress('replay_snapshot', snapshot=snapshot, boundary=boundary)
            replays = [compare_replay(fusion, *chunk, boundary) for chunk in chunks]
            arrays = {key: np.concatenate([result[key] for result, _ in replays]) for key in replays[0][0]}
            np.savez_compressed(private / f'{snapshot}-replay.npz', **arrays)
            row.update(replay=previous.aggregate(response_values(arrays), population, scenes),
                       patched_metrics=fixed.metrics(arrays['patched_states'], population, scenes, support_detail=True),
                       midpoint_max_abs=max(error for _, error in replays))
        snapshots[snapshot] = row
        del fusion, chunks, stages, native
    controls = {key: fixed.metrics(population[key], population, scenes, support_detail=True) for key in fixed.CONTROLS}
    fixed.write_json(args.output / 'input_audit.json', dict(scenes=scenes, pair_count=16, pair_queries=1024,
                     same_query=True, c91_exact=True, c93_native_outputs_exact=True,
                     c96_input_encoder_memory_verified=True, both_residual_additions_exact=True,
                     native_replays_exact=boundary is not None, optimizer_updates=0))
    fixed.write_json(args.output / 'summary.json', dict(scenes=scenes, boundary=boundary,
                     snapshots={key: snapshots[key] for key in previous.SNAPSHOTS}, controls=controls))
    progress('completed', status='COMPLETED', exit_code=0,
             peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))


def self_check():
    torch.set_num_threads(2)
    config = fixed.PairFusionConfig(fc_hidden=4, length=4, d_model=4, nhead=2, num_layers=2,
                                   output_parameterization='train_position_affine')
    torch.manual_seed(20)
    model = previous.PairFusion(config, dict(mean=[0., 0., 0., 0.], std=[1., 1., 1., 1.])).eval()
    left = torch.rand(2, 4, 4)
    right = torch.rand(2, 4, 4)
    stages, native = trace_forward(model, left, right)
    for entry, exit_name in TRANSITIONS:
        boundary = dict(side='left', layer=0, entry=entry, exit=exit_name)
        arrays, midpoint = compare_replay(model, left, right, stages, native, boundary)
        assert midpoint < 1e-6
        assert not np.array_equal(arrays['native_boundary'], arrays['patched_boundary'])
        np.testing.assert_allclose(
            arrays['patched_boundary'].astype(np.float64).mean(axis=0),
            arrays['native_boundary'].astype(np.float64).mean(axis=0), atol=1e-6, rtol=0)
    print('Encoder residual closure, unmodified replay and midpoint preservation passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.runs is None or args.output is None:
        parser.error('--runs and --output are required')
    args.output.mkdir(mode=0o700, exist_ok=False)
    start = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, optimizer_updates=0, raw_archive_read=False,
                 validation_opened=False, test_opened=False, time_policy=args.time_policy,
                 commit=os.environ.get('C99_EXPECTED_COMMIT'))

    def progress(phase, **fields):
        state.update(fields, phase=phase, elapsed_seconds=time.monotonic() - start,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        fixed.write_json(args.output / 'run.json', state)
        if phase != 'failed' and state['peak_rss_bytes'] > 8 * 1024**3:
            raise MemoryError('C99 resident budget exceeded')

    try:
        progress('bind_inputs')
        run(args, progress)
    except Exception as error:
        progress('failed', status='FAILED', exit_code=1, error=repr(error))
        raise


if __name__ == '__main__':
    main()
