"""C96 frozen TMTAFF conditioning and generated-prefix interventions."""

import argparse
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from association.models.fusion import PairFusion, _causal_mask
from tools.analysis import tmtaff_feedback_gradient_qualification as feedback
from tools.analysis import tmtaff_true_pair_qualification as fixed


C93 = 'LEVEL1-TMTAFF-FEEDBACK-GRADIENT-20261112-attempt-1'
SNAPSHOTS = ('u0', 'original_u32', 'detached_u32')
INTERVENTIONS = ('self', 'memory', 'prefix', 'both')


def capture(model, left, right):
    layers = {}
    decoder_inputs, generated = [], []

    def save_output(name):
        def save(_module, _args, output):
            layers[name] = output.detach()
        return save

    def save_decoder(_module, args, output):
        decoder_inputs.append(tuple(item.detach() for item in args))
        generated.append(output[:, -1:].detach())

    handles = [model.left_encoder.register_forward_hook(save_output('left_encoder')),
               model.right_encoder.register_forward_hook(save_output('right_encoder')),
               model.decoder.register_forward_hook(save_decoder)]
    try:
        result = model(left.detach().requires_grad_(), right.detach().requires_grad_())
    finally:
        for handle in handles:
            handle.remove()
    native = torch.cat(generated, dim=1)
    memory = result['fused_features'].detach()
    for q, (history, saved_memory) in enumerate(decoder_inputs):
        rebuilt = torch.cat((model.bos.detach().expand(len(left), -1, -1), native[:, :q]), dim=1)
        torch.testing.assert_close(history, rebuilt, atol=0, rtol=0)
        torch.testing.assert_close(memory, saved_memory, atol=0, rtol=0)
    return dict(**layers, left_input=left.detach(), right_input=right.detach(), memory=memory,
                native=native, states=result['states'].detach(), histories=[x[0] for x in decoder_inputs])


def output_positions(model, latent):
    projected = model.state_projection(model.output_dropout(latent))
    position = projected[..., :2]
    physical = position * model.position_std + model.position_mean
    state = torch.cat((physical, F.softplus(projected[..., 2:3]), projected[..., 3:]), dim=-1)
    return position.detach(), state.detach()


def intervene(model, captured):
    memory, native = captured['memory'], captured['native']
    donor = torch.arange(len(native), device=native.device) ^ 1
    swapped_memory = memory[donor]
    per_arm = {name: [] for name in INTERVENTIONS}
    for history in captured['histories']:
        mask = _causal_mask(history.shape[1], history.device)
        swapped_history = history[donor]
        for name, current_history, current_memory in (
            ('self', history, memory), ('memory', history, swapped_memory),
            ('prefix', swapped_history, memory), ('both', swapped_history, swapped_memory),
        ):
            value = model.decoder(current_history, current_memory, tgt_mask=mask)
            per_arm[name].append(value[:, -1:].detach())
    result = {}
    for name, latents in per_arm.items():
        latent = torch.cat(latents, dim=1)
        projected, physical = output_positions(model, latent)
        result[name] = dict(latent=latent, standardized=projected, physical=physical)
    torch.testing.assert_close(result['self']['latent'], native, atol=0, rtol=0)
    torch.testing.assert_close(result['self']['physical'], captured['states'], atol=0, rtol=0)
    for key, values in result['self'].items():
        torch.testing.assert_close(result['both'][key], values[donor], atol=2e-5, rtol=1e-6)
        torch.testing.assert_close(result['prefix'][key][:, 0], values[:, 0], atol=0, rtol=0)
    return result


def rms(a):
    return np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=-1))


def ratio(value, scale):
    return np.divide(value, scale, out=np.zeros_like(value), where=scale > 0)


def distance(a, b):
    return fixed.attribute_errors(a.astype(np.float64), b.astype(np.float64))['position']


def measure(captured, outputs):
    donor = np.arange(len(captured['states'])) ^ 1
    result = {}
    for key in ('left_input', 'right_input', 'left_encoder', 'right_encoder', 'memory'):
        value = captured[key]
        diff = rms(value - value[donor])
        energy = np.sqrt((rms(value) ** 2 + rms(value[donor]) ** 2) / 2)
        result[f'{key}_rms'] = diff
        result[f'{key}_relative_rms'] = ratio(diff, energy)
    own = outputs['self']
    for name in ('memory', 'prefix', 'both'):
        changed = outputs[name]
        for key in ('latent', 'standardized'):
            diff = rms(changed[key] - own[key])
            result[f'{name}_{key}_rms'] = diff
            energy = np.sqrt((rms(changed[key]) ** 2 + rms(own[key]) ** 2) / 2)
            result[f'{name}_{key}_relative_rms'] = ratio(diff, energy)
        result[f'{name}_physical_m'] = distance(changed['physical'], own['physical'])
        result[f'{name}_latitude_abs_deg'] = np.abs(changed['physical'][..., 0] - own['physical'][..., 0])
        lon = changed['physical'][..., 1].astype(np.float64) - own['physical'][..., 1]
        result[f'{name}_longitude_abs_deg'] = np.abs((lon + 180) % 360 - 180)
    result['donor_physical_m'] = distance(own['physical'], own['physical'][donor])
    return result


def aggregate(values, population, scenes):
    valid = population['valid'][..., :2].all(-1)
    donor = np.arange(len(valid)) ^ 1
    support = population['support'] == 2
    subsets = {'blind': np.ones_like(valid), 'valid_both': valid & valid[donor],
               'valid_dual_both': valid & valid[donor] & support & support[donor]}
    query = np.arange(valid.shape[1])
    periods = {'all': query >= 0, 'bos': query == 0, 'early': (query >= 1) & (query < 32),
               'late': query >= 32, 'last': query == 63}
    groups = {row['scene']: np.repeat(np.arange(len(scenes)) == i, 2)
              for i, row in enumerate(scenes)}
    groups['distinct'] = np.repeat([row['distinct_targets'] for row in scenes], 2)
    groups['same'] = ~groups['distinct']
    return {group: {period: {subset: {
        key: fixed.statistics(value[(pairs[:, None] & queries[None, :]) & selected])
        for key, value in values.items()
    } for subset, selected in subsets.items()} for period, queries in periods.items()}
        for group, pairs in groups.items()}


def run(args, start):
    private = args.output / 'private'
    private.mkdir(mode=0o700)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    roots = {name: args.runs / value for name, value in fixed.RUNS.items()}
    fixed.bind_raw_audit(args.runs / feedback.RAW_AUDIT / 'summary.json')
    model, normalizer, config, contracts, _, _ = fixed.source_checkpoints(roots['checkpoint'], device)
    population, scenes = fixed.load_population(roots, model, normalizer, contracts, device)
    del model
    fixed.bind_reference_population(args.runs, population)
    feedback.bind_c91(args.runs / feedback.C91, population)
    np.testing.assert_array_equal(population['query'][::2], population['query'][1::2])
    fixed.require(config.output_parameterization == 'train_position_affine', 'unexpected physical projection')
    np.savez_compressed(private / 'masks.npz', valid=population['valid'], support=population['support'])
    c93 = args.runs / C93 / 'private'
    summary, closed = {}, {}
    for snapshot, original in zip(SNAPSHOTS, ('ORIGINAL-u000', 'ORIGINAL-u032', 'DETACHED_HISTORY-u032'), strict=True):
        torch.manual_seed(fixed.SEED)
        fusion = PairFusion(config, normalizer).to(device).eval()
        if snapshot != 'u0':
            fusion.load_state_dict(torch.load(c93 / f'{original}.pt', map_location=device,
                                             weights_only=True)['fusion'])
        chunks = []
        for i in (0, fixed.MICROBATCH):
            part = slice(i, i + fixed.MICROBATCH)
            native = capture(fusion, population['left'][part], population['right'][part])
            arms = intervene(fusion, native)
            arrays = {name: {key: tensor.cpu().numpy() for key, tensor in output.items()}
                      for name, output in arms.items()}
            captured = {key: native[key].cpu().numpy() for key in
                        ('states', 'left_input', 'right_input', 'left_encoder', 'right_encoder', 'memory')}
            chunks.append((captured, arrays, measure(captured, arrays)))
        states = np.concatenate([chunk[0]['states'] for chunk in chunks])
        stored = fixed.read_npz(c93 / f'{original}.npz')['states']
        np.testing.assert_array_equal(states, stored, err_msg=f'C93 {snapshot} native output')
        values = {key: np.concatenate([chunk[2][key] for chunk in chunks]) for key in chunks[0][2]}
        closure = {key: float(max(np.abs(arms['both'][key] - arms['self'][key][np.arange(8) ^ 1]).max()
                                 for _, arms, _ in chunks)) for key in ('latent', 'standardized', 'physical')}
        np.savez_compressed(private / f'{snapshot}.npz', **values)
        closed[snapshot] = closure
        summary[snapshot] = dict(response=aggregate(values, population, scenes),
                                 original_metrics=fixed.metrics(states, population, scenes, support_detail=True))
        fixed.write_json(args.output / 'run.json', dict(status='RUNNING', last_snapshot=snapshot,
                         elapsed_seconds=time.monotonic() - start))
    fixed.write_json(args.output / 'input_audit.json', dict(scenes=scenes, same_query=True,
                     c91_exact=True, all_c93_native_outputs_exact=True, double_swap_max_abs=closed,
                     self_and_bos_prefix_exact=True))
    ratios = {}
    initial = summary['u0']['response']['distinct']
    for snapshot in SNAPSHOTS[1:]:
        changed = summary[snapshot]['response']['distinct']
        ratios[snapshot] = {
            period: {subset: {
                metric: value['mae'] / initial[period][subset][metric]['mae']
                if initial[period][subset][metric]['mae'] else None
                for metric, value in metrics.items()
            } for subset, metrics in subsets.items()}
            for period, subsets in changed.items()
        }
    fixed.write_json(args.output / 'summary.json', dict(scenes=scenes, snapshots=summary,
                     distinct_response_ratios_to_u0=ratios))
    fixed.write_json(args.output / 'run.json', dict(status='COMPLETED', elapsed_seconds=time.monotonic() - start,
                     optimizer_updates=0, validation_opened=False, test_opened=False,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    start = time.monotonic()
    fixed.write_json(args.output / 'run.json', dict(status='RUNNING', phase='bind'))
    try:
        run(args, start)
    except Exception as error:
        fixed.write_json(args.output / 'run.json', dict(status='FAILED', error=repr(error),
                         elapsed_seconds=time.monotonic() - start))
        raise


if __name__ == '__main__':
    main()
