"""C90. Qualify the fixed TMTAFF T8 true-pair experiment before group evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from association.data.motion_evidence import DATA, scene_motion_evidence
from association.data.physical_units import KNOT_TO_MPS, SPEED_UNIT_POLICY, normalizer_speed_policy
from association.data.timed import prepare_timed_scene
from association.evaluation.fusion_controls import attribute_errors, ordinary_fusion
from association.models.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.training.common import _optimizer, normalize_scene
from association.training.tmtaff import load_complete_snapshot

IDEA = 'IDEA-TMTAFF-COMPONENT-QUALIFICATION-20261111'
PLAN_VERSION = '1.1'
LEARNING_RATE = .002
GRADIENT_CLIP = 1.
LOSS_FIELDS = dict(position_loss_mode='standard_smooth_l1', longitude_loss_mode='shortest_arc')
SCENES = ('0', '3', '7', '9', '10', '11', '14', '17')
FEATURES = ('lat', 'lon', 'vel', 'cou')
CONTROLS = ('CAP128_MEAN', 'CANONICAL_MEAN', 'OLD_TFM')
SNAPSHOTS = (0, 16, 32)
SEED = 20
MICROBATCH = 8
COUNTS = (732, 732, 729, 732)
RUNS = {
    'native': 'LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1',
    'probe': 'LEVEL1-TMTAFF-T-PHYSICAL-PROBE-20261030-153-attempt-1',
    'checkpoint': 'LEVEL1-TMTAFF-UNIT-BASELINE-A-20261027-107-attempt-1',
}
COMPLETION_PREFIXES = ('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')
COMPLETION_BUFFERS = ('positions', 'speed_unit_contract')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, allow_nan=False, indent=2) + '\n')
    temp.replace(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


class PositionOnlyLoss(FusionLoss):
    def forward(self, prediction, target, valid, *, normalization_counts=None,
                regularizer_weight=1.0):
        result = super().forward(prediction, target, valid,
                                 normalization_counts=normalization_counts,
                                 regularizer_weight=regularizer_weight)
        loss = torch.stack([result['losses'][name] for name in FEATURES[:2]])
        variance = self.log_variance[:2]
        result['total'] = (0.5 * (-variance).exp() * loss
                           + regularizer_weight * F.softplus(variance)).sum()
        return result


ARMS = {'ALL_ATTRIBUTES': FusionLoss, 'POSITION_ONLY': PositionOnlyLoss}


def source_checkpoints(root, device):
    tcm_path, tfm_path = (root / phase / 'checkpoint.pt' for phase in ('tcm', 'tfm'))
    tcm = torch.load(tcm_path, map_location='cpu', weights_only=True)
    tfm = torch.load(tfm_path, map_location='cpu', weights_only=True)
    for payload, phase, updates in ((tcm, 'tcm', 2424), (tfm, 'tfm', 1000)):
        require(payload['phase'] == phase and payload['cursor']['completed_updates'] == updates,
                f'wrong 107 {phase} checkpoint')
        require(payload['input_adapter'] == 'native_time_v1' and payload['max_observations'] == 128
                and normalizer_speed_policy(payload['normalizer']) == SPEED_UNIT_POLICY,
                f'wrong 107 {phase} native-time policy')
    require(tcm['data_contract'] == tfm['data_contract']
            and tcm['normalizer'] == tfm['normalizer'], '107 phase input metadata differs')
    completion = {
        name for name in tcm['states'][0]
        if name in COMPLETION_BUFFERS or name.startswith(COMPLETION_PREFIXES)
    }
    require(tuple(tfm['initialization']['transferred_buffers']) == COMPLETION_BUFFERS
            and tuple(tfm['initialization']['transferred_modules'])
            == tuple(prefix[:-1] for prefix in COMPLETION_PREFIXES)
            and set(COMPLETION_BUFFERS) <= completion
            and all(any(name.startswith(prefix) for name in completion)
                    for prefix in COMPLETION_PREFIXES)
            and all(torch.equal(tcm['states'][0][name], tfm['states'][0][name])
                    for name in completion), '107 frozen TCM weights differ')
    expected = dict(seed=SEED, pair_microbatch=MICROBATCH, gradient_clip=GRADIENT_CLIP,
                    lr_schedule='constant', **LOSS_FIELDS)
    require(all(tfm['config'][key] == value for key, value in expected.items())
            and tfm['config']['learning_rates']['TFM_FROZEN'] == LEARNING_RATE
            and {row['learning_rate'] for row in tfm['history']} == {LEARNING_RATE},
            '107 optimizer/loss contract differs')
    groups = tfm['optimizer']['param_groups']
    require(len(groups) == 1 and len(groups[0]['params']) == 260,
            '107 TFM optimizer group contract differs')
    optimizer_reference = group_settings(groups[0])
    require(optimizer_reference['lr'] == LEARNING_RATE
            and optimizer_reference['betas'] == (.85, .99)
            and optimizer_reference['eps'] == 1e-6
            and optimizer_reference['weight_decay'] == 0., '107 Adam settings differ')
    fusion = PairFusionConfig(fc_hidden=128, output_parameterization='train_position_affine')
    require(tfm['fusion_config'] == vars(fusion), '107 TFM architecture differs')
    rows = {str(row['scene_id']): row for row in tcm['data_contract'][0]
            if str(row['scene_id']) in SCENES}
    require(set(rows) == set(SCENES), '107 T scene contract differs')
    normalizer = tcm['normalizer']
    del tcm, tfm
    model = load_complete_snapshot(tcm_path, device=device, selected=False)['model']
    return model.requires_grad_(False), normalizer, fusion, rows, len(completion), optimizer_reference


def timed_scene(cache, identities, scene_id):
    tracks, intervals = [], []
    offsets = cache['event_offsets']
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        times = cache['event_times_seconds'][start:stop]
        tracks.append(dict(node_id=str(cache['node_ids'][index]), source=str(cache['sources'][index]),
                           identity_id=str(identities[index]), canonical_times=times,
                           canonical_values=cache['event_values'][start:stop]))
        intervals.append([times[0], times[-1]])
    native = dict(scene_id=scene_id, split='T_core', tracks=tracks, node_ids=cache['node_ids'],
                  sources=cache['sources'], identities=identities, intervals=np.asarray(intervals))
    return prepare_timed_scene(native, **DATA, query_times=cache['query_seconds'],
                               speed_unit_policy=SPEED_UNIT_POLICY)


def make_pairs(identities, sources):
    pairs = []
    for left, identity in enumerate(identities):
        for right in range(left + 1, len(identities)):
            if identity == identities[right] and sources[left] != sources[right]:
                pairs.append((left, right))
                if len(pairs) == 2:
                    return np.asarray(pairs)
    raise ValueError('T8 scene has fewer than two eligible true cross-source pairs')


def ordinary_pair(motion, pairs):
    rows = []
    for pair in pairs:
        result = ordinary_fusion(motion['states'][pair], motion['nearest_seconds'][pair],
                                 motion['support'][pair], [[0, 1]])
        rows.append(result['predictions']['mean_supported'][0])
    return np.asarray(rows)


def load_population(roots, model, normalizer, contracts, device):
    native = roots['native']
    index = json.loads((native / 'native-cache.json').read_text())
    require(tuple(map(str, index['scene_ids']['T_core'][:8])) == SCENES, '127 T8 scene order differs')
    arrays = {name: [] for name in ('left', 'right', 'truth', 'valid', 'support', 'query', *CONTROLS)}
    scenes = []
    for scene_id in SCENES:
        filename = f'scene-{scene_id}.npz'
        events = read_npz(native / 'private/native-events/T_core' / filename)
        with np.load(native / 'private/native-evaluation/T_core' / filename, allow_pickle=False) as data:
            identities = data['identities']
        previous = read_npz(roots['probe'] / 'private' / f'predictions-{scene_id}.npz')
        pairs = make_pairs(identities, events['sources'])
        require(np.array_equal(pairs, previous['pairs']), '153 fixed true-pair order differs')
        scene = timed_scene(events, identities, scene_id)
        row = contracts[scene_id]
        require(row['split'] == 'T_core'
                and row['reference_alignment']['max_gap_seconds'] == 600
                and row['reference_alignment']['longitude_policy'] == 'shortest_arc',
                '107 scene/reference contract differs')
        for key, actual in {
            'grid': scene['grid'], 'node_ids': scene['node_ids'], 'identities': identities,
            'sources': scene['sources'], 'intervals': scene['intervals'],
            'observed_counts': scene['observed'].sum(1),
            'native_observed_counts': scene['native_observed'].sum(1),
        }.items():
            require(np.array_equal(actual, row[key]), f'107 T input parity failed for {key}')
        packed, observed = normalize_scene(scene, normalizer)
        nodes, lookup = np.unique(pairs, return_inverse=True)
        lookup = lookup.reshape(pairs.shape)
        with torch.no_grad():
            x = torch.as_tensor(packed[nodes], dtype=next(model.parameters()).dtype, device=device)
            mask = torch.as_tensor(observed[nodes], device=device)
            completed = model.complete(x, mask)
        arrays['left'].append(completed[lookup[:, 0].tolist()])
        arrays['right'].append(completed[lookup[:, 1].tolist()])

        cap = scene_motion_evidence(scene, motion_observation_policy='selected_cap128')
        canonical = read_npz(native / 'private/native-motion/T_core' / filename)
        canonical['states'][..., 2] *= KNOT_TO_MPS
        for name, motion in (('CAP128_MEAN', cap), ('CANONICAL_MEAN', canonical)):
            output = ordinary_pair(motion, pairs)
            require(np.array_equal(output, previous[name]), f'153 {name} predictions differ')
            arrays[name].append(output)
        arrays['OLD_TFM'].append(previous['TFM'])
        evaluation = read_npz(native / 'private/native-evaluation/T_core' / filename)
        for name, expected in (('node_ids', events['node_ids']), ('identities', identities),
                               ('query_seconds', scene['grid']), ('target_ids', row['target_ids'])):
            require(np.array_equal(evaluation[name], expected), f'T reference parity failed for {name}')
        targets = list(evaluation['target_ids'])
        chosen = [targets.index(identities[i]) for i in pairs[:, 0]]
        arrays['truth'].append(evaluation['target_states'][chosen])
        arrays['valid'].append(evaluation['target_valid'][chosen])
        arrays['support'].append(cap['support'][pairs].sum(1))
        arrays['query'].append(np.broadcast_to(scene['grid'], (len(pairs), 64)))
        scenes.append(dict(scene=scene_id, pair_count=len(pairs), nodes=len(nodes),
                           distinct_targets=chosen[0] != chosen[1]))
    population = {name: torch.cat(value) if name in ('left', 'right') else np.concatenate(value)
                  for name, value in arrays.items()}
    require(population['valid'].shape == (16, 64, 4)
            and tuple(population['valid'].sum((0, 1))) == COUNTS
            and sum(row['nodes'] for row in scenes) == 27, '153 T8 population differs')
    position = population['valid'][..., :2].all(-1)
    require(tuple(np.bincount(population['support'][position], minlength=3)) == (132, 299, 301),
            '153 T8 position support population differs')
    require(np.isfinite(population['truth'][population['valid']]).all(), 'T supervision is nonfinite')
    return population, scenes


def statistics(values):
    if not len(values):
        return dict(count=0, mae=None, rmse=None, p95=None, p99=None, max=None)
    return dict(count=len(values), mae=float(values.mean()), rmse=float(np.sqrt(np.mean(values**2))),
                p95=float(np.quantile(values, .95)), p99=float(np.quantile(values, .99)),
                max=float(values.max()))


def metrics(prediction, population, scenes):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth, valid, support = (population[key] for key in ('truth', 'valid', 'support'))
    errors = attribute_errors(prediction, truth)
    errors['latitude'] = np.abs(prediction[..., 0] - truth[..., 0])
    errors['longitude'] = np.abs((prediction[..., 1] - truth[..., 1] + 180.) % 360. - 180.)
    masks = dict(position=valid[..., :2].all(-1), speed=valid[..., 2], heading=valid[..., 3],
                 latitude=valid[..., 0], longitude=valid[..., 1])
    regions = {'all': np.ones(support.shape, dtype=bool)}
    regions.update({name: support == index for index, name in
                    enumerate(('zero_supported', 'one_supported', 'both_supported'))})
    result = {region: {name: statistics(values[masks[name] & keep])
                       for name, values in errors.items()} for region, keep in regions.items()}
    scene_stats = {}
    displacement = []
    for index, row in enumerate(scenes):
        part = slice(2 * index, 2 * index + 2)
        scene_stats[row['scene']] = {name: statistics(values[part][masks[name][part]])
                                     for name, values in errors.items()}
        if row['distinct_targets']:
            keep = masks['position'][part].all(0)
            first, second = prediction[part]
            displacement.extend(attribute_errors(first, second)['position'][keep])
    result['scenes'] = scene_stats
    result['distinct_position_values'] = int(len(np.unique(prediction[..., :2].reshape(-1, 2), axis=0)))
    result['same_query_distinct_target_separation_m'] = statistics(np.asarray(displacement))
    return result


def group_settings(group):
    return {key: value for key, value in group.items() if key != 'params'}


def make_arms(config, normalizer, device, optimizer_reference):
    models = {}
    for name, loss_class in ARMS.items():
        torch.manual_seed(SEED)
        fusion = PairFusion(config, normalizer).to(device)
        objective = loss_class(**LOSS_FIELDS).to(device)
        optimizer = _optimizer([*fusion.parameters(), *objective.parameters()],
                               {'learning_rate': LEARNING_RATE})
        groups = optimizer.param_groups
        require(len(groups) == 1 and group_settings(groups[0]) == optimizer_reference
                and not optimizer.state, f'{name} initial Adam does not match 107')
        models[name] = fusion, objective, optimizer
    initial_a, initial_b = (model[0].state_dict() for model in models.values())
    require(initial_a.keys() == initial_b.keys()
            and all(torch.equal(initial_a[key], initial_b[key]) for key in initial_a)
            and all(not bool(values[1].log_variance.any()) for values in models.values()),
            'A/B initialization mismatch')
    return models


def checkpoint(fusion, objective, population, targets, valid, counts, path):
    """Save forward predictions before opening the quantitative readout."""
    fusion.eval()
    chunks, gradient_sq = [], [0., 0.]
    for start in (0, MICROBATCH):
        part = slice(start, start + MICROBATCH)
        inputs = [population[name][part].detach().requires_grad_() for name in ('left', 'right')]
        prediction = fusion(*inputs)['states']
        chunks.append(prediction.detach().cpu().numpy())
        result = objective(prediction, targets[part], valid[part], normalization_counts=counts,
                           regularizer_weight=0.)
        position_loss = sum(result['losses'][name] for name in FEATURES[:2])
        gradients = torch.autograd.grad(position_loss, inputs)
        for index, gradient in enumerate(gradients):
            gradient_sq[index] += float(gradient.detach().double().square().sum())
    output = np.concatenate(chunks)
    np.savez_compressed(path, states=output)
    return output, dict(left_l2=float(np.sqrt(gradient_sq[0])), right_l2=float(np.sqrt(gradient_sq[1])))


def update(fusion, objective, optimizer, population, targets, valid, counts):
    started = time.monotonic()
    fusion.train()
    optimizer.zero_grad(set_to_none=True)
    components, total = np.zeros(4), 0.
    for start in (0, MICROBATCH):
        part = slice(start, start + MICROBATCH)
        prediction = fusion(population['left'][part], population['right'][part])['states']
        result = objective(prediction, targets[part], valid[part], normalization_counts=counts,
                           regularizer_weight=float(start == 0))
        result['total'].backward()
        components += torch.stack(tuple(result['losses'].values())).detach().cpu().numpy()
        total += float(result['total'].detach())
    if isinstance(objective, PositionOnlyLoss):
        require(not bool(objective.log_variance.grad[2:].any())
                and not bool(fusion.state_projection.weight.grad[2:].any())
                and not bool(fusion.state_projection.bias.grad[2:].any()),
                'position-only branch sent speed/heading gradients')
    gradient = torch.nn.utils.clip_grad_norm_(
        [*fusion.parameters(), *objective.parameters()], GRADIENT_CLIP, error_if_nonfinite=True)
    optimizer.step()
    variance = objective.log_variance.detach().exp().cpu().tolist()
    return dict(total=total, components=dict(zip(FEATURES, components.tolist())),
                gradient_norm=float(gradient), variance=variance, seconds=time.monotonic() - started)


def gate(final, initial, cap):
    position, reference = final['all']['position'], cap['all']['position']
    comparable = final['same_query_distinct_target_separation_m']
    control_separation = cap['same_query_distinct_target_separation_m']['mae']
    decisions = dict(
        position_beats_initial=position['mae'] < initial['all']['position']['mae'],
        position_within_control=all(position[key] <= reference[key] for key in ('mae', 'p99')),
        improves_two_scenes=sum(final['scenes'][scene]['position']['mae']
                                < initial['scenes'][scene]['position']['mae'] for scene in SCENES) >= 2,
        has_conditioned_output=(comparable['count'] > 0 and control_separation is not None
                                and comparable['mae'] > .01 * control_separation
                                and final['distinct_position_values'] > 1),
        motion_within_control=all(final['all'][feature][key] <= cap['all'][feature][key]
                                  for feature in ('speed', 'heading') for key in ('mae', 'p99')),
    )
    return dict(**decisions, position_pass=all(value for key, value in decisions.items()
                                              if key != 'motion_within_control'),
                four_attribute_pass=all(decisions.values()))


def self_check():
    source, position = FusionLoss(**LOSS_FIELDS), PositionOnlyLoss(**LOSS_FIELDS)
    prediction = torch.tensor([[[.2, -179.9, 5., 359.]], [[.4, 45., 3., 1.]]], requires_grad=True)
    target = torch.tensor([[[.1, 179.9, 7., 1.]], [[.5, 44., 2., 359.]]])
    valid = torch.tensor([[[True, True, False, True]], [[True, True, True, True]]])
    original = source(prediction, target, valid)
    result = position(prediction, target, valid)
    expected = 0.5 * (original['losses']['lat'] + original['losses']['lon']) + 2 * np.log(2.)
    torch.testing.assert_close(result['total'], expected)
    parts = [position(prediction[i:i + 1], target[i:i + 1], valid[i:i + 1],
                      normalization_counts=valid.sum((0, 1)), regularizer_weight=float(i == 0))['total']
             for i in range(2)]
    torch.testing.assert_close(result['total'], sum(parts))
    result['total'].backward()
    require(bool(prediction.grad[..., :2].abs().all()) and not bool(prediction.grad[..., 2:].any())
            and not bool(position.log_variance.grad[2:].any()), 'loss ablation self-check failed')
    original_adam = _optimizer([torch.nn.Parameter(torch.tensor(1.))],
                               {'learning_rate': LEARNING_RATE})
    reference = group_settings(original_adam.param_groups[0])
    require(reference['betas'] == (.85, .99) and reference['eps'] == 1e-6
            and reference['weight_decay'] == 0., 'reference Adam fixture differs')
    small = PairFusionConfig(fc_hidden=4, length=2, d_model=4, nhead=2, num_layers=1)
    models = make_arms(small, None, 'cpu', reference)
    with torch.no_grad():
        left = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        right = left.flip(1)
        values = [fusion.eval()(left, right)['states'] for fusion, _, _ in models.values()]
    require(torch.equal(*values), 'same-initialization forward fixture differs')
    print('107 Adam, same-init forward, position loss wrap, two-chunk parity and isolation passed')


def run(args):
    output = args.output.resolve()
    output.mkdir(mode=0o700, exist_ok=False)
    private = output / 'private'
    private.mkdir(mode=0o700)
    start = time.monotonic()
    roots = {name: args.runs.resolve() / value for name, value in RUNS.items()}
    counts = dict.fromkeys(ARMS, 0)
    state = dict(status='RUNNING', idea_id=IDEA, plan_version=PLAN_VERSION, sources={k: str(v) for k, v in roots.items()},
                 optimizer_updates=counts, raw_archive_read=False, validation_opened=False, test_opened=False,
                 tcm_updates=0, tam_updates=0, joint_updates=0, time_policy=args.time_policy,
                 device=args.device, commit=os.environ.get('C90_EXPECTED_COMMIT'))

    def progress(phase, **fields):
        state.update(fields, phase=phase, elapsed_seconds=time.monotonic() - start,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write_json(output / 'run.json', state)
        if phase != 'failed' and state['peak_rss_bytes'] > 8 * 1024**3:
            raise MemoryError('C90 resident budget exceeded')

    try:
        torch.set_num_threads(4)
        device = torch.device(args.device)
        require(device.type == 'cuda' and torch.cuda.is_available(), 'C90 requires assigned CUDA device')
        progress('bind_inputs', gpu=torch.cuda.get_device_name(device))
        model, normalizer, config, contracts, bound_tensors, optimizer_reference = source_checkpoints(
            roots['checkpoint'], device)
        population, scenes = load_population(roots, model, normalizer, contracts, device)
        del model
        controls = {name: metrics(population[name], population, scenes) for name in CONTROLS}
        np.savez_compressed(private / 'contract.npz', query_seconds=population['query'],
                            target_states=population['truth'], target_valid=population['valid'],
                            selected_support=population['support'],
                            **{name: population[name] for name in CONTROLS})
        models = make_arms(config, normalizer, device, optimizer_reference)
        require(all(len(values[2].param_groups[0]['params']) == 260 for values in models.values()),
                'current optimizer parameter count differs from 107')
        audit = dict(status='PASSED', scenes=scenes, pair_count=16, pair_queries=1024,
                     feature_valid=dict(zip(FEATURES, COUNTS)), support_position_counts=[132, 299, 301],
                     source_completion_tensors=bound_tensors, reference_speed_unit='mps',
                     adam=optimizer_reference, same_initial_state=True, lr_schedule='constant',
                     gradient_clip=GRADIENT_CLIP,
                     controls=controls, read_validation=False, read_test=False, read_raw_archive=False)
        write_json(output / 'input_audit.json', audit)
        if args.preflight:
            progress('completed', status='COMPLETED', exit_code=0)
            return
        targets = torch.as_tensor(population['truth'], dtype=torch.float32, device=device)
        valid = torch.as_tensor(population['valid'], device=device)
        full_counts = valid.sum((0, 1))
        torch.manual_seed(SEED)
        cuda_rng = {name: torch.cuda.get_rng_state(device).clone() for name in ARMS}
        snapshots, histories = ({name: {} for name in ARMS}, {name: [] for name in ARMS})
        initial_predictions = {}
        progress('training', fixed_inputs_bound=True)
        for step in range(SNAPSHOTS[-1] + 1):
            for name, (fusion, objective, optimizer) in models.items():
                if step:
                    torch.cuda.set_rng_state(cuda_rng[name], device)
                    measurement = update(fusion, objective, optimizer, population, targets, valid, full_counts)
                    cuda_rng[name] = torch.cuda.get_rng_state(device)
                    histories[name].append(dict(update=step, **measurement))
                    counts[name] = step
                if step in SNAPSHOTS:
                    prediction, gradients = checkpoint(fusion, objective, population, targets, valid, full_counts,
                                                       private / f'{name}-u{step:03d}.npz')
                    snapshots[name][str(step)] = dict(metrics=metrics(prediction, population, scenes),
                                                      position_input_gradients=gradients)
                    if step == 0:
                        initial_predictions[name] = prediction
                    if step == SNAPSHOTS[-1]:
                        torch.save(dict(fusion=fusion.state_dict(), loss=objective.state_dict()),
                                   private / f'{name}-u{step:03d}.pt')
            if step == 0:
                require(np.array_equal(*initial_predictions.values()), 'A/B u0 predictions differ')
            write_json(output / 'training.json', dict(history=histories, snapshots=snapshots))
            progress('training', matched_cuda_rng=torch.equal(*cuda_rng.values()),
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
            require(state['matched_cuda_rng'], 'A/B CUDA dropout RNG streams differ')
        results = {name: gate(by_step[str(SNAPSHOTS[-1])]['metrics'], by_step['0']['metrics'], controls['CAP128_MEAN'])
                   for name, by_step in snapshots.items()}
        write_json(output / 'summary.json', dict(
            status='COMPLETED', idea_id=IDEA, plan_version=PLAN_VERSION, training_population='fixed original 153 T8',
            generalization_measured=False, counts=audit['feature_valid'], scenes=scenes, controls=controls,
            snapshots=snapshots, gates=results, ready_for_separate_true_group_qualification=
            results['ALL_ATTRIBUTES']['four_attribute_pass'], updates=counts,
            pair_presentations={name: count * 16 for name, count in counts.items()}))
        progress('completed', status='COMPLETED', exit_code=0)
    except BaseException as error:
        progress('failed', status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if args.runs is None or args.output is None:
            parser.error('--runs and --output are required')
        run(args)


if __name__ == '__main__':
    main()
