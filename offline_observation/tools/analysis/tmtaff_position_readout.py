"""Scene-held linear position readouts of frozen TFM interfaces."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from tools.analysis import tmtaff_encoder_contraction as contraction
from tools.analysis import tmtaff_feedback_gradient_qualification as feedback
from tools.analysis import tmtaff_memory_prefix_diagnostic as memory_trace
from tools.analysis import tmtaff_true_pair_qualification as fixed


RIDGES = (0.01, 0.0001)
TAPS = ('fc_input', 'fc1', 'fc_activation', 'fc2')


def capture(model, left, right):
    stages = {}

    def save(name, incoming=False):
        def hook(_module, args, output):
            stages[name] = (args[0] if incoming else output).detach()
        return hook

    hooks = [model.fc1.register_forward_hook(save('fc_input', True)),
             model.fc1.register_forward_hook(save('fc1')),
             model.fc_activation.register_forward_hook(save('fc_activation')),
             model.fc2.register_forward_hook(save('fc2'))]
    try:
        native = memory_trace.capture(model, left, right)
    finally:
        for hook in hooks:
            hook.remove()
    encoded = torch.cat((native['left_encoder'], native['right_encoder']), dim=1)
    torch.testing.assert_close(encoded.flatten(start_dim=1), stages['fc_input'], atol=0, rtol=0)
    torch.testing.assert_close(stages['fc2'].reshape_as(native['memory']), native['memory'], atol=0, rtol=0)
    stages.update(
        tcm_local=torch.cat((native['left_input'], native['right_input']), dim=-1),
        encoder_local=torch.cat((native['left_encoder'], native['right_encoder']), dim=-1),
        memory_local=native['memory'], history_last=torch.cat([x[:, -1:] for x in native['histories']], dim=1),
        generated=native['native'], states=native['states'])
    return {key: value.cpu().numpy() for key, value in stages.items()}


def scene_identity_counts(native_root, scenes):
    previous = set()
    counts = {}
    for row in scenes:
        name = f"scene-{row['scene']}.npz"
        events = fixed.read_npz(native_root / 'private/native-events/T_core' / name)
        with np.load(native_root / 'private/native-evaluation/T_core' / name, allow_pickle=False) as values:
            identities = values['identities']
        pairs = fixed.make_pairs(identities, events['sources'])
        selected = set(map(str, identities[pairs[:, 0]]))
        fixed.require(not selected & previous, 'target identity crosses probe folds')
        counts[row['scene']] = len(selected)
        previous.update(selected)
    return counts


def linear_kernel(values, paired):
    values = values.astype(np.float64)
    if paired:
        values = values[::2] - values[1::2]
    else:
        # Translation improves Gram precision and cancels in fold centering.
        values = values - values[0, 0]
    rows = values.reshape(-1, values.shape[-1])
    return rows @ rows.T


def full_kernel(values, steps, paired):
    values = values.astype(np.float64)
    values = values[::2] - values[1::2] if paired else values - values[0]
    return np.repeat(np.repeat(values @ values.T, steps, axis=0), steps, axis=1)


def prefix_kernel(history, paired):
    history = history.astype(np.float64)
    if paired:
        history = history[::2] - history[1::2]
    # Each history is exactly BOS plus earlier latent tokens, padded with zeros.
    tokens = history.transpose(1, 0, 2)
    per_token = tokens @ tokens.transpose(0, 2, 1)
    cumulative = per_token.cumsum(axis=0)
    pairs, steps = history.shape[:2]
    pair = np.repeat(np.arange(pairs), steps)
    query = np.tile(np.arange(steps), pairs)
    return cumulative[np.minimum.outer(query, query), pair[:, None], pair[None, :]]


def feature_kernels(arrays, paired):
    steps = arrays['states'].shape[1]
    kernels = {name: (linear_kernel(arrays[name], paired),) for name in
               ('tcm_local', 'encoder_local', 'memory_local', 'history_last', 'generated')}
    kernels.update({name: (full_kernel(arrays[name], steps, paired),) for name in TAPS})
    kernels['history_full'] = (prefix_kernel(arrays['history_last'], paired),)
    kernels['decoder_local'] = kernels['memory_local'] + kernels['history_last']
    kernels['decoder_full'] = kernels['fc2'] + kernels['history_full']
    return {'query_only': (), **kernels}


def center(kernel, train):
    mean = kernel[:, train].mean(axis=1)
    return kernel - mean[:, None] - mean[None, :] + mean[train].mean()


def design(kernels, query, train, paired):
    query_kernel = np.outer(query, query)
    combined = np.zeros_like(query_kernel) if paired else query_kernel + query_kernel**2
    for kernel in kernels:
        values = kernel if paired else center(kernel, train)
        scale = float(values.diagonal()[train].mean())
        if scale > np.finfo(np.float64).eps:
            combined += values / scale * (1 + query_kernel)
    return combined if paired else center(combined, train)


def fit_fold(kernel, labels, train, test, ridge, paired):
    intercept = 0 if paired else labels[train].mean(axis=0)
    target = labels[train] - intercept
    covariance = kernel[np.ix_(train, train)]
    covariance.flat[::len(train) + 1] += len(train) * ridge
    coefficients = np.linalg.solve(covariance, target)
    residual = float(np.max(np.abs(covariance @ coefficients - target)))
    fixed.require(residual < 1e-7, 'ridge normal equation residual too large')
    return intercept + kernel[np.ix_(test, train)] @ coefficients, residual


def fit_readouts(kernels, labels, mask, scene_index, query, paired):
    predictions = {str(ridge): np.full_like(labels, np.nan) for ridge in RIDGES}
    checks = []
    for scene in np.unique(scene_index[mask]):
        train = np.flatnonzero(mask & (scene_index != scene))
        test = np.flatnonzero((mask if paired else np.ones_like(mask)) & (scene_index == scene))
        fixed.require(len(train) > 0 and len(test) > 0, 'empty readout fold')
        kernel = design(kernels, query, train, paired)
        for ridge in RIDGES:
            predictions[str(ridge)][test], residual = fit_fold(kernel, labels, train, test, ridge, paired)
            checks.append(residual)
    fixed.require(all(np.isfinite(value[mask]).all() for value in predictions.values()), 'missing prediction')
    return predictions, max(checks)


def distance(first, second):
    def pad(values):
        return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 2)])
    return fixed.attribute_errors(pad(first), pad(second))['position']


def difference(positions):
    delta = positions[::2] - positions[1::2]
    delta[..., 1] = (delta[..., 1] + 180.) % 360. - 180.
    return delta


def summarize(position, delta, truth, valid, support, distinct, scenes):
    fixed.require(np.isfinite(position[valid]).all() and (np.abs(position[..., 0][valid]) <= 90).all(),
                  'readout left valid physical latitude range')
    error = distance(position, truth)
    swapped = distance(position[np.arange(len(position)) ^ 1], truth)
    common = valid[::2] & valid[1::2] & distinct[:, None]
    dual_common = common & (support[::2] == 2) & (support[1::2] == 2)
    scale = np.stack((np.ones_like(truth[::2, :, 0]),
                      np.cos(np.deg2rad((truth[::2, :, 0] + truth[1::2, :, 0]) / 2))), axis=-1)
    scale *= 6371000. * np.pi / 180.
    delta_m = delta * scale
    true_m = difference(truth) * scale
    vector_error = np.linalg.norm(delta_m - true_m, axis=-1)
    length = np.linalg.norm(delta_m, axis=-1)
    direction = np.sum(delta_m * true_m, axis=-1) > 0
    assignment_gain = (swapped[::2] + swapped[1::2] - error[::2] - error[1::2]) / 2
    def contrast(keep):
        return dict(error_m=fixed.statistics(vector_error[keep]), length_m=fixed.statistics(length[keep]),
                    direction_positive=int(direction[keep].sum()), assignment_correct=int((assignment_gain[keep] > 1e-6).sum()),
                    assignment_tie=int((np.abs(assignment_gain[keep]) <= 1e-6).sum()),
                    assignment_mean_gain_m=float(assignment_gain[keep].mean()) if keep.any() else None)
    rows = {row['scene']: dict(position_m=fixed.statistics(error[2*i:2*i+2][valid[2*i:2*i+2]]),
                               contrast=contrast(common & (np.arange(len(scenes)) == i)[:, None]))
            for i, row in enumerate(scenes)}
    result = dict(position_m=fixed.statistics(error[valid]), dual_m=fixed.statistics(error[valid & (support == 2)]),
                  bos_m=fixed.statistics(error[:, 0][valid[:, 0]]), later_m=fixed.statistics(error[:, 1:][valid[:, 1:]]),
                  contrast=contrast(common), dual_contrast=contrast(dual_common), scenes=rows)
    return result, dict(position_error_m=error[valid], contrast_error_m=vector_error[common],
                        contrast_length_m=length[common], contrast_direction=direction[common],
                        assignment_gain_m=assignment_gain[common])


def fc_inversion(model, arrays):
    layer = model.fc2
    weight, bias = (tensor.detach().cpu().numpy().astype(np.float64) for tensor in (layer.weight, layer.bias))
    recovered, _, rank, singular = np.linalg.lstsq(weight, (arrays['fc2'] - bias).T, rcond=None)
    native = arrays['fc_activation'].astype(np.float64)
    diff = recovered.T - native
    paired_error = (diff[::2] - diff[1::2])[None, ...]
    paired = native[::2] - native[1::2]
    return dict(columns=weight.shape[1], rank=int(rank), condition=float(singular[0] / singular[-1]),
                activation_max_abs=float(np.max(np.abs(diff))),
                paired_activation_relative_rmse=float(np.linalg.norm(paired_error) / np.linalg.norm(paired)))


def probe_snapshot(arrays, population, scenes, normalizer):
    truth = population['truth'][..., :2].astype(np.float64)
    valid = population['valid'][..., :2].all(-1)
    support = population['support']
    distinct = np.array([row['distinct_targets'] for row in scenes])
    common = valid[::2] & valid[1::2] & distinct[:, None]
    count, steps = valid.shape
    scene_index = np.repeat(np.arange(len(scenes)), 2 * steps)
    q = np.linspace(-1, 1, steps)
    means, scales = (np.asarray(normalizer[key][:2], dtype=np.float64) for key in ('mean', 'std'))
    position_label = ((truth - means) / scales).reshape(-1, 2)
    contrast_label = (difference(truth) / scales).reshape(-1, 2)
    absolute = feature_kernels(arrays, paired=False)
    contrasts = feature_kernels(arrays, paired=True)
    controls = {'native': arrays['states'][..., :2].astype(np.float64),
                'CAP128_MEAN': population['CAP128_MEAN'][..., :2].astype(np.float64)}
    details, metrics, raw = {}, {}, {}
    for name, position in controls.items():
        metrics[name], raw[name] = summarize(position, difference(position), truth, valid, support, distinct, scenes)
    for name in absolute:
        position, error_a = fit_readouts(absolute[name], position_label, valid.ravel(), scene_index, np.tile(q, count), False)
        delta, error_d = fit_readouts(contrasts[name], contrast_label, common.ravel(), scene_index.reshape(count, steps)[::2].ravel(),
                                    np.tile(q, count // 2), True)
        details[name] = max(error_a, error_d)
        for ridge in map(str, RIDGES):
            prediction = position[ridge].reshape(count, steps, 2) * scales + means
            differential = delta[ridge].reshape(count // 2, steps, 2) * scales
            row = f'{name}/{ridge}'
            metrics[row], raw[row] = summarize(prediction, differential, truth, valid, support, distinct, scenes)
            metrics[row]['absolute_readout_contrast'] = summarize(prediction, difference(prediction), truth, valid,
                                                                  support, distinct, scenes)[0]['contrast']
    return metrics, raw, details


def run(args, start):
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
    identity_counts = scene_identity_counts(roots['native'], scenes)
    metrics, audits, raw = {}, {}, {}
    c93 = args.runs / memory_trace.C93 / 'private'
    for snapshot, source in zip(memory_trace.SNAPSHOTS, contraction.SOURCES, strict=True):
        fixed.write_json(args.output / 'run.json', dict(status='RUNNING', snapshot=snapshot))
        torch.manual_seed(fixed.SEED)
        model = memory_trace.PairFusion(config, normalizer).to(device).eval()
        if snapshot != 'u0':
            model.load_state_dict(torch.load(c93 / f'{source}.pt', map_location='cpu', weights_only=True)['fusion'])
        chunks = [capture(model, population['left'][part], population['right'][part]) for part in
                  (slice(0, fixed.MICROBATCH), slice(fixed.MICROBATCH, 2 * fixed.MICROBATCH))]
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
        np.testing.assert_array_equal(arrays['states'], fixed.read_npz(c93 / f'{source}.npz')['states'])
        fixed.require(all(np.isfinite(value).all() for value in arrays.values()), 'nonfinite interface')
        np.savez_compressed(private / f'{snapshot}-representations.npz', **arrays)
        metrics[snapshot], raw[snapshot], residual = probe_snapshot(arrays, population, scenes, normalizer)
        audits[snapshot] = dict(original_output_exact=True, encoder_fc_input_exact=True,
                                fc2_memory_exact=True, generated_history_exact=True,
                                fc2_inversion=fc_inversion(model, arrays), ridge_normal_equation_max_abs=residual)
        del model
    valid = population['valid'][..., :2].all(-1)
    common = valid[::2] & valid[1::2] & np.array([row['distinct_targets'] for row in scenes])[:, None]
    fixed.write_json(args.output / 'summary.json', dict(scenes=scenes, identity_counts=identity_counts,
                     ridges=RIDGES, snapshots=metrics, interfaces=audits))
    np.savez_compressed(args.output / 'public-errors.npz', valid=valid, common=common, support=population['support'],
                        **{f'{snapshot}/{arm}/{key}': value for snapshot, arms in raw.items()
                           for arm, values in arms.items() for key, value in values.items()})
    fixed.write_json(args.output / 'run.json', dict(status='COMPLETED', elapsed_seconds=time.monotonic()-start,
                     optimizer_updates=0, validation_opened=False, test_opened=False,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    start = time.monotonic()
    try:
        run(args, start)
    except Exception as error:
        fixed.write_json(args.output / 'run.json', dict(status='FAILED', error=repr(error),
                         elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
