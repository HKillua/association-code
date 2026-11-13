"""Isolate generated-history gradient feedback on the fixed C91 true pairs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from tools.analysis import tmtaff_true_pair_qualification as qualification


IDEA = 'IDEA-TMTAFF-FEEDBACK-GRADIENT-20261112'
C91 = 'LEVEL1-TMTAFF-TRUE-PAIR-SUPERVISION-20261112-attempt-1'
RAW_AUDIT = 'LEVEL1-TMTAFF-TRUE-PAIR-RAW-AUDIT-20261112-attempt-1'
ARMS = dict(ORIGINAL=qualification.PositionOnlyLoss, DETACHED_HISTORY=qualification.PositionOnlyLoss)
STEPS = qualification.SUPPORT_SNAPSHOTS


def detach_generated_history(_decoder, inputs):
    history, memory = inputs
    if history.shape[1] > 1:
        history = torch.cat((history[:, :1], history[:, 1:].detach()), dim=1)
    return history, memory


def path_gradients(fusion, left, right):
    """Read the cross-step, local output, and common BOS/memory derivatives."""
    outputs = []
    hook = fusion.decoder.register_forward_hook(lambda _, __, value: outputs.append(value))
    try:
        result = fusion.eval()(left, right)
    finally:
        hook.remove()
    early, penultimate, last = outputs[0], outputs[-2], outputs[-1]
    states = result['states']
    early_local, = torch.autograd.grad(states[0, 0, 0], early, retain_graph=True)
    names = ('last_to_first_l2', 'last_to_penultimate_l2', 'last_local_l2',
             'last_to_bos_l2', 'last_to_memory_l2')
    gradients = torch.autograd.grad(states[0, -1, 0],
                                    (early, penultimate, last, fusion.bos, result['fused_features']),
                                    allow_unused=True)
    norms = [0. if item is None else float(item.double().norm()) for item in gradients]
    return dict(first_local_l2=float(early_local.double().norm()), **dict(zip(names, norms)))


def verify_paths(models, left, right):
    diagnostics = {name: path_gradients(fusion, left, right) for name, (fusion, _, _) in models.items()}
    original, detached = diagnostics.values()
    crosses = ('last_to_first_l2', 'last_to_penultimate_l2')
    local = ('first_local_l2', 'last_local_l2', 'last_to_bos_l2', 'last_to_memory_l2')
    qualification.require(all(np.isfinite(value) for row in diagnostics.values() for value in row.values())
                          and all(original[key] > 0. and detached[key] == 0. for key in crosses)
                          and all(min(original[key], detached[key]) > 0. for key in local),
                          'cross-step gradient isolation or retained local gradient failed')
    device = left.device
    devices = [device.index] if device.type == 'cuda' else []
    get_rng = (lambda: torch.cuda.get_rng_state(device)) if devices else torch.get_rng_state
    results, rng = [], []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        for fusion, _, _ in models.values():
            torch.manual_seed(qualification.SEED)
            results.append(fusion.train()(left, right))
            rng.append(get_rng())
    qualification.require(torch.equal(*rng)
                          and all(torch.equal(results[0][name], results[1][name]) for name in results[0]),
                          'training-mode initial forward or dropout RNG differs')
    return dict(training_forward_exact=True, training_rng_exact=True, gradient_paths=diagnostics)


def bind_c91(path, population):
    previous = json.loads((path / 'summary.json').read_text())
    qualification.require(previous['status'] == 'COMPLETED'
                          and previous['idea_id'] == qualification.SUPPORT_IDEA
                          and previous['updates']['FULL_VALID'] == STEPS[-1], 'C91 final receipt differs')
    contract = qualification.read_npz(path / 'private/contract.npz')
    for original, current in (('query_seconds', 'query'), ('target_states', 'truth'),
                              ('target_valid', 'valid'), ('selected_support', 'support'),
                              *((name, name) for name in qualification.CONTROLS)):
        np.testing.assert_array_equal(population[current], contract[original], err_msg=original)


def gates(original, detached, control):
    position = detached['all']['position']
    better = all(position[key] < original['all']['position'][key] for key in ('mae', 'p99'))
    scenes = sum(detached['scenes'][scene]['position']['mae'] < row['position']['mae']
                 for scene, row in original['scenes'].items())
    separation = 'same_query_distinct_target_separation_m'
    separated = (detached[separation]['mae'] > original[separation]['mae']
                 and detached[separation]['mae'] > .01 * control[separation]['mae'])
    control_matched = all(detached[scope]['position'][key] <= control[scope]['position'][key]
                          for scope in ('all', 'both_supported') for key in ('mae', 'p99'))
    return dict(full_mae_and_p99_improved=better, full_scenes_improved=scenes,
                conditioned_separation_rescued=separated,
                feedback_hypothesis_supported=better and scenes >= 2 and separated,
                ordinary_control_matched=control_matched)


def self_check():
    torch.set_num_threads(2)
    optimizer = qualification._optimizer([torch.nn.Parameter(torch.tensor(1.))],
                                         {'learning_rate': qualification.LEARNING_RATE})
    config = qualification.PairFusionConfig(fc_hidden=8, length=4, d_model=8, nhead=2, num_layers=1)
    models = qualification.make_arms(config, None, 'cpu',
                                     qualification.group_settings(optimizer.param_groups[0]), ARMS)
    models['DETACHED_HISTORY'][0].decoder.register_forward_pre_hook(detach_generated_history)
    left = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 32.
    print(json.dumps(verify_paths(models, left, left.flip(1)), allow_nan=False))
    with torch.no_grad():
        original, detached = (model.eval()(left, left.flip(1)) for model, _, _ in models.values())
    for key in original:
        torch.testing.assert_close(original[key], detached[key], atol=0, rtol=0)


def run(args):
    output = args.output.resolve()
    output.mkdir(mode=0o700, exist_ok=False)
    private = output / 'private'
    private.mkdir(mode=0o700)
    start = time.monotonic()
    runs = args.runs.resolve()
    roots = {name: runs / value for name, value in qualification.RUNS.items()}
    c91 = runs / C91
    raw_audit = runs / RAW_AUDIT / 'summary.json'
    updates = dict.fromkeys(ARMS, 0)
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
                 sources={**{key: str(value) for key, value in roots.items()},
                          'c91': str(c91), 'raw_audit': str(raw_audit)},
                 optimizer_updates=updates, raw_archive_read=False, validation_opened=False,
                 test_opened=False, tcm_updates=0, tam_updates=0, joint_updates=0,
                 time_policy=args.time_policy, device=args.device, commit=os.environ.get('C93_EXPECTED_COMMIT'))

    def progress(phase, **fields):
        state.update(fields, phase=phase, elapsed_seconds=time.monotonic() - start,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        qualification.write_json(output / 'run.json', state)
        if phase != 'failed' and state['peak_rss_bytes'] > 8 * 1024**3:
            raise MemoryError('TMTAFF feedback qualification resident budget exceeded')

    try:
        progress('bind_raw_audit')
        qualification.bind_raw_audit(raw_audit)
        torch.set_num_threads(4)
        device = torch.device(args.device)
        qualification.require(device.type == 'cuda' and torch.cuda.is_available(),
                              'feedback qualification requires the assigned CUDA device')
        progress('bind_inputs', gpu=torch.cuda.get_device_name(device))
        model, normalizer, config, contracts, bound_tensors, optimizer_reference = qualification.source_checkpoints(
            roots['checkpoint'], device)
        population, scenes = qualification.load_population(roots, model, normalizer, contracts, device)
        del model
        qualification.bind_reference_population(runs, population)
        bind_c91(c91, population)
        controls = {name: qualification.metrics(population[name], population, scenes, support_detail=True)
                    for name in qualification.CONTROLS}
        models = qualification.make_arms(config, normalizer, device, optimizer_reference, ARMS)
        qualification.require(all(len(model[2].param_groups[0]['params']) == 260 for model in models.values()),
                              'current optimizer parameter count differs from 107')
        models['DETACHED_HISTORY'][0].decoder.register_forward_pre_hook(detach_generated_history)
        targets = torch.as_tensor(population['truth'], dtype=torch.float32, device=device)
        valid = torch.as_tensor(population['valid'], device=device)
        counts = valid.sum((0, 1))
        qualification.require(tuple(counts.tolist()) == qualification.COUNTS, 'supervision counts differ from C91')
        progress('verify_initial_gradient_paths')
        probe = verify_paths(models, population['left'][:1], population['right'][:1])
        qualification.write_json(output / 'input_audit.json', dict(
            status='PASSED', scenes=scenes, pair_count=16, pair_queries=1024,
            feature_valid=dict(zip(qualification.FEATURES, counts.tolist())),
            training_position_counts={name: counts[:2].tolist() for name in ARMS},
            support_position_counts=[132, 299, 301], source_completion_tensors=bound_tensors,
            reference_speed_unit='mps', c90_contract_exact=True, c91_contract_exact=True,
            same_initial_state=True, adam=optimizer_reference, lr_schedule='constant',
            gradient_clip=qualification.GRADIENT_CLIP, read_validation=False, read_test=False,
            read_raw_archive=False, **probe))

        torch.manual_seed(qualification.SEED)
        rng = {name: torch.cuda.get_rng_state(device).clone() for name in ARMS}
        history, snapshots = ({name: [] for name in ARMS}, {name: {} for name in ARMS})
        original_parity = {}
        progress('training', fixed_inputs_bound=True)
        for step in range(STEPS[-1] + 1):
            initial = []
            for name, (fusion, objective, optimizer) in models.items():
                if step:
                    torch.cuda.set_rng_state(rng[name], device)
                    measurement = qualification.update(fusion, objective, optimizer, population, targets, valid, counts)
                    rng[name] = torch.cuda.get_rng_state(device)
                    history[name].append(dict(update=step, **measurement))
                    updates[name] = step
                if step in STEPS:
                    prediction, gradients = qualification.checkpoint(
                        fusion, objective, population, targets, valid, counts, private / f'{name}-u{step:03d}.npz')
                    snapshots[name][str(step)] = dict(
                        metrics=qualification.metrics(prediction, population, scenes, support_detail=True),
                        position_input_gradients=gradients)
                    if name == 'ORIGINAL':
                        old = qualification.read_npz(c91 / 'private' / f'FULL_VALID-u{step:03d}.npz')['states']
                        np.testing.assert_array_equal(prediction, old, err_msg=f'C91 original u{step}')
                        original_parity[str(step)] = True
                    if step == 0:
                        initial.append(prediction)
                    if step == STEPS[-1]:
                        torch.save(dict(fusion=fusion.state_dict(), loss=objective.state_dict()),
                                   private / f'{name}-u{step:03d}.pt')
            if step == 0:
                qualification.require(np.array_equal(*initial), 'A/B complete u0 predictions differ')
            qualification.write_json(output / 'training.json', dict(history=history, snapshots=snapshots))
            matched_rng = torch.equal(*rng.values())
            progress('training', matched_cuda_rng=matched_rng, c91_original_parity=original_parity,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
            qualification.require(matched_rng, 'A/B CUDA dropout RNG streams differ')

        original, detached = (by_step[str(STEPS[-1])]['metrics'] for by_step in snapshots.values())
        qualification.write_json(output / 'summary.json', dict(
            status='COMPLETED', idea_id=IDEA, plan_version='1.0', training_population='fixed original 153 T8',
            generalization_measured=False, scenes=scenes, controls=controls, snapshots=snapshots,
            gates=gates(original, detached, controls['CAP128_MEAN']), updates=updates,
            pair_presentations={name: count * 16 for name, count in updates.items()},
            c91_original_parity=original_parity, u0_exact=True, matched_cuda_rng=True))
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
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.runs is None or args.output is None:
        parser.error('--runs and --output are required')
    else:
        run(args)


if __name__ == '__main__':
    main()
