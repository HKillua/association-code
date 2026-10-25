"""Finite, native-style DeepAF core qualification; not an author-result reproduction.

Two independently generated splits contain CV/CA/CT Cartesian trajectories.
Six correctly associated noisy observations are GIVEN to start each track;
birth/death and dynamic association gating are deliberately outside this probe.
Every subsequent observation/clutter candidate enters the same P/V core, and
no-detection/no-clutter frames contain only the previous-V virtual candidate.

Paper settings: dt1, 30 seconds, sigma50, Pd.9, Poisson8 clutter in +/-500m,
initial speed100..250, speed bounds50..300, 6+6/d32/head4/FF64 P/V core.
Explicit choices where the paper is incomplete: 31 inclusive samples; noisy
six-point OLS initialization; per-axis CV acceleration N(0,10^2), per-axis
constant CA acceleration N(10,10), CT omega in rad/s, trapezoidal CV/CA
position integration after speed projection; no gating; dropout0/scale10;
last-token readout and detached-feedback TBPTT1. No original table is matched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.models.deepaf_peer import DeepAFPeer
from association.utils.runtime import write_json

SCOPE = 'deepaf-native-core-probe-v1'
QUALIFICATION_SCOPE = 'deepaf-native-adapted-qualification-v1'
MODES = ('CV', 'CA', 'CT')
ARMS = ('native_labels', 'adapted_labels')
SEED = 20261025
INITIAL_COUNT, SAMPLE_COUNT, SCALE = 6, 31, 10.
VISIBLE_KEYS = {'position', 'velocity', 'history', 'measurements', 'valid'}


def _bounded_velocity(value):
    speed = np.linalg.norm(value)
    if speed == 0.:
        raise ValueError('zero speed has no defined projection direction')
    return value * (np.clip(speed, 50., 300.) / speed)


def generate_scene(rng, mode, scene_id):
    """Truth affects ONLY simulated sensor generation and stored supervision."""
    if mode not in MODES:
        raise ValueError('unknown motion mode')
    position = [rng.uniform(-200000., 200000., 2)]
    angle, speed = rng.uniform(-np.pi, np.pi), rng.uniform(100., 250.)
    velocity = [speed * np.array([np.cos(angle), np.sin(angle)])]
    acceleration = rng.normal(10., np.sqrt(10.), 2)
    omega = rng.uniform(-np.pi / 60., np.pi / 15.)
    for _ in range(1, SAMPLE_COUNT):
        old = velocity[-1]
        if mode == 'CT':
            cosine, sine = np.cos(omega), np.sin(omega)
            current = np.array([[cosine, -sine], [sine, cosine]]) @ old
            # Exact integral of the rotating velocity over one second.
            turn = np.array([-old[1], old[0]])
            shift = (np.sinc(omega / np.pi) * old +
                     (0.5 * omega * np.sinc(omega / (2. * np.pi)) ** 2) * turn)
        else:
            change = rng.normal(0., 10., 2) if mode == 'CV' else acceleration
            current = _bounded_velocity(old + change)
            shift = .5 * (old + current)
        position.append(position[-1] + shift)
        velocity.append(current)
    position, velocity = np.asarray(position), np.asarray(velocity)
    initial = position[:INITIAL_COUNT] + rng.normal(0., 50., (INITIAL_COUNT, 2))
    measurements, detected, clutter_counts = [], [], []
    for point in position[INITIAL_COUNT:]:
        detection, clutter = bool(rng.random() < .9), int(rng.poisson(8.))
        candidates = point + rng.uniform(-500., 500., (clutter, 2))
        if detection:
            candidates = np.vstack((candidates, point + rng.normal(0., 50., 2)))
        rng.shuffle(candidates, axis=0)
        measurements.append(candidates)
        detected.append(detection)
        clutter_counts.append(clutter)
    return dict(scene_id=scene_id, motion=mode, initial_observations=initial,
                measurements=measurements, truth_position=position, truth_velocity=velocity,
                detected=np.asarray(detected), clutter_counts=np.asarray(clutter_counts))


def generate_split(per_motion, seed, prefix):
    if type(per_motion) is not int or per_motion < 1:
        raise ValueError('positive integer scenes per motion required')
    rng = np.random.default_rng(seed)
    return [generate_scene(rng, mode, f'{prefix}-{mode}-{i:03d}')
            for mode in MODES for i in range(per_motion)]


def observable_batch(scenes):
    """The exact forward interface excludes truth, motion labels and identities."""
    if not scenes:
        raise ValueError('nonempty observable batch required')
    initial = np.stack([scene['initial_observations'] for scene in scenes])
    if initial.shape != (len(scenes), INITIAL_COUNT, 2) or not np.isfinite(initial).all():
        raise ValueError('six finite given noisy initial observations required')
    x = np.arange(INITIAL_COUNT, dtype=np.float64)
    centered = x - x.mean()
    v = (initial * centered[None, :, None]).sum(1) / np.square(centered).sum()
    p = initial.mean(1) + (x[-1] - x.mean()) * v
    q = SAMPLE_COUNT - INITIAL_COUNT
    if any(len(scene['measurements']) != q for scene in scenes):
        raise ValueError('each scene must preserve all 25 subsequent scans')
    maximum = max(1, max(len(z) for scene in scenes for z in scene['measurements']))
    z, valid = np.zeros((len(scenes), q, maximum, 2)), np.zeros((len(scenes), q, maximum), bool)
    for i, scene in enumerate(scenes):
        for k, candidates in enumerate(scene['measurements']):
            candidates = np.asarray(candidates)
            if candidates.shape != (len(candidates), 2) or not np.isfinite(candidates).all():
                raise ValueError('finite Cartesian observation candidates required')
            z[i, k, :len(candidates)] = candidates
            valid[i, k, :len(candidates)] = True
    return dict(position=torch.tensor(p, dtype=torch.float64),
                velocity=torch.tensor(v, dtype=torch.float64),
                history=torch.tensor(np.repeat(v[:, None, :], 5, axis=1) / SCALE, dtype=torch.float32),
                measurements=torch.tensor(z, dtype=torch.float64), valid=torch.tensor(valid))


@torch.no_grad()
def rollout(model, visible, *, cache=False):
    """Only the previous OWN P/V state and current sensor candidates are used."""
    if set(visible) != VISIBLE_KEYS:
        raise ValueError('native forward accepts visible fields only')
    p, v = visible['position'].clone(), visible['velocity'].clone()
    ph, vh = visible['history'].clone(), visible['history'].clone()
    b, q = visible['valid'].shape[:2]
    hv = torch.ones((b, 5), dtype=torch.bool)
    output = {key: [] for key in ('position', 'velocity', 'previous_position')}
    saved = {key: [] for key in ('candidate', 'valid', 'ph', 'vh')} if cache else None
    for k in range(q):
        live = torch.cat((torch.ones((b, 1), dtype=torch.bool), visible['valid'][:, k]), 1)
        difference = visible['measurements'][:, k] - p[:, None]  # dt=1s
        candidate = torch.cat((v[:, None], difference), 1)
        candidate = torch.where(live[..., None], candidate, torch.zeros_like(candidate)).float() / SCALE
        if cache:
            for key, value in (('candidate', candidate), ('valid', live), ('ph', ph), ('vh', vh)):
                saved[key].append(value.clone())
        predicted_p = model.position_network(candidate, live, ph, hv)
        predicted_v = model.velocity_network(candidate, live, vh, hv)
        output['previous_position'].append(p)
        p, v = p + SCALE * predicted_p.double(), SCALE * predicted_v.double()
        output['position'].append(p)
        output['velocity'].append(v)
        ph, vh = torch.cat((ph[:, 1:], predicted_p[:, None]), 1), torch.cat((vh[:, 1:], predicted_v[:, None]), 1)
    result = {key: torch.stack(value, 1) for key, value in output.items()}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('nonfinite native closed-loop output')
    if cache:
        result['cache'] = {key: torch.stack(value, 1).detach() for key, value in saved.items()}
    return result


def supervision(scenes, result, arm):
    """Both P interpretations are explicit; only native V uses instantaneous truth."""
    truth = torch.tensor(np.stack([s['truth_position'] for s in scenes]), dtype=torch.float64)
    difference = truth[:, INITIAL_COUNT:] - truth[:, INITIAL_COUNT - 1:-1]
    if arm == 'native_labels':
        p = difference
        v = torch.tensor(np.stack([s['truth_velocity'][INITIAL_COUNT:] for s in scenes]), dtype=torch.float64)
    elif arm == 'adapted_labels':
        p, v = truth[:, INITIAL_COUNT:] - result['previous_position'].detach(), difference
    else:
        raise ValueError('unknown predeclared label interpretation')
    return p.flatten(0, 1) / SCALE, v.flatten(0, 1) / SCALE


def train_update(model, optimizer, scenes, arm):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = rollout(model, observable_batch(scenes), cache=True)
    pt, vt = supervision(scenes, result, arm)
    cache = {key: value.flatten(0, 1) for key, value in result['cache'].items()}
    losses = np.zeros(2)
    for start in range(0, len(pt), 64):
        end = min(start + 64, len(pt))
        c, valid, ph, vh = [cache[key][start:end] for key in ('candidate', 'valid', 'ph', 'vh')]
        hv = torch.ones(ph.shape[:2], dtype=torch.bool)
        predicted = (model.position_network(c, valid, ph, hv), model.velocity_network(c, valid, vh, hv))
        values = [((value.double() - target[start:end]).square().sum() / len(pt))
                  for value, target in zip(predicted, (pt, vt))]
        if any(not torch.isfinite(value) for value in values):
            raise FloatingPointError('nonfinite native probe loss')
        sum(values).backward()
        losses += [float(value.detach()) for value in values]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(p_mse=float(losses[0]), v_mse=float(losses[1]))


def _metrics(position, velocity):
    return dict(position_mae_m=float(position.mean()), position_rmse_m=float(np.sqrt(np.square(position).mean())),
                velocity_rmse_mps=float(np.sqrt(np.square(velocity).mean())))


@torch.no_grad()
def truth_reset_shadow(model, visible, closed_loop, truth_position):
    """Oracle diagnostic: reset only previous P, never the real feedback path.

    Each shadow query keeps the ACTUAL closed-loop previous V and P/V histories.
    k=0 has no preceding truth within the evaluated sequence and retains the
    given noisy OLS initialization. It is reported separately, not as a reset.
    """
    b, q = visible['valid'].shape[:2]
    truth = torch.as_tensor(truth_position, dtype=torch.float64)
    if truth.shape != (b, q, 2) or not torch.isfinite(truth).all():
        raise ValueError('finite evaluated truth positions [B,Q,2] required for oracle diagnostic')
    previous_p = torch.cat((visible['position'][:, None], truth[:, :-1]), 1)
    previous_v = torch.cat((visible['velocity'][:, None], closed_loop['velocity'][:, :-1]), 1)
    live = closed_loop['cache']['valid']
    difference = visible['measurements'] - previous_p[:, :, None]
    candidates = torch.cat((previous_v[:, :, None], difference), 2)
    candidates = torch.where(live[..., None], candidates, torch.zeros_like(candidates)).float() / SCALE
    ph, vh = closed_loop['cache']['ph'], closed_loop['cache']['vh']
    outputs = [[], []]
    # Match the actual rollout's batch shape for each query; no shadow feedback.
    for k in range(q):
        hv = torch.ones(ph[:, k].shape[:2], dtype=torch.bool)
        outputs[0].append(model.position_network(candidates[:, k], live[:, k], ph[:, k], hv))
        outputs[1].append(model.velocity_network(candidates[:, k], live[:, k], vh[:, k], hv))
    p, v = (torch.stack(values, 1).double() * SCALE for values in outputs)
    result = dict(position=previous_p + p, velocity=v, previous_position=previous_p)
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('nonfinite truth-reset diagnostic output')
    return result


@torch.no_grad()
def evaluate(model, scenes, *, truth_reset=False):
    model.eval()
    rows, shadow_rows = [], []
    for start in range(0, len(scenes), 8):
        chosen = scenes[start:start + 8]
        visible = observable_batch(chosen)
        result = rollout(model, visible, cache=truth_reset)
        shadow = (truth_reset_shadow(model, visible, result,
                  np.stack([s['truth_position'][INITIAL_COUNT:] for s in chosen]))
                  if truth_reset else None)
        q = result['position'].shape[1]
        extrapolated = visible['position'][:, None] + torch.arange(1, q + 1)[None, :, None] * visible['velocity'][:, None]
        for i, scene in enumerate(chosen):
            truth, vel = scene['truth_position'][INITIAL_COUNT:], scene['truth_velocity'][INITIAL_COUNT:]
            pe = np.linalg.norm(result['position'][i].numpy() - truth, axis=-1)
            ve = np.linalg.norm(result['velocity'][i].numpy() - vel, axis=-1)
            ce = np.linalg.norm(extrapolated[i].numpy() - truth, axis=-1)
            cv = np.linalg.norm(visible['velocity'][i].numpy()[None] - vel, axis=-1)
            centroid = np.asarray([z.mean(0) if len(z) else extrapolated[i, k].numpy()
                                   for k, z in enumerate(scene['measurements'])])
            centroid_v = np.diff(np.vstack((visible['position'][i].numpy(), centroid)), axis=0)
            ae, av = np.linalg.norm(centroid - truth, axis=-1), np.linalg.norm(centroid_v - vel, axis=-1)
            rows.append(dict(scene_id=scene['scene_id'], motion=scene['motion'], deepaf=_metrics(pe, ve),
                visible_cv=_metrics(ce, cv), visible_centroid=_metrics(ae, av),
                queries=q, position_errors_m=pe.tolist(), velocity_errors_mps=ve.tolist(),
                control_position_errors_m=ce.tolist(), control_velocity_errors_mps=cv.tolist(),
                centroid_position_errors_m=ae.tolist(), centroid_velocity_errors_mps=av.tolist()))
            if shadow is not None:
                spe = np.linalg.norm(shadow['position'][i].numpy() - truth, axis=-1)
                sve = np.linalg.norm(shadow['velocity'][i].numpy() - vel, axis=-1)
                shadow_rows.append(dict(scene_id=scene['scene_id'], motion=scene['motion'],
                    position_errors_m=spe.tolist(), velocity_errors_mps=sve.tolist(),
                    first_query_without_reset=_metrics(spe[:1], sve[:1]),
                    reset_queries_only=_metrics(spe[1:], sve[1:])))
    groups = {}
    for mode in ('all', *MODES):
        selected = [row for row in rows if mode == 'all' or row['motion'] == mode]
        groups[mode] = dict(scene_count=len(selected), deepaf=_metrics(
            np.concatenate([row['position_errors_m'] for row in selected]),
            np.concatenate([row['velocity_errors_mps'] for row in selected])), visible_cv=_metrics(
            np.concatenate([row['control_position_errors_m'] for row in selected]),
            np.concatenate([row['control_velocity_errors_mps'] for row in selected])), visible_centroid=_metrics(
            np.concatenate([row['centroid_position_errors_m'] for row in selected]),
            np.concatenate([row['centroid_velocity_errors_mps'] for row in selected])))
    result = dict(groups=groups, scenes=rows)
    if truth_reset:
        shadow_groups = {}
        for mode in ('all', *MODES):
            selected = [row for row in shadow_rows if mode == 'all' or row['motion'] == mode]
            shadow_groups[mode] = dict(scene_count=len(selected),
                first_query_without_reset=_metrics(
                    np.asarray([row['position_errors_m'][0] for row in selected]),
                    np.asarray([row['velocity_errors_mps'][0] for row in selected])),
                reset_queries_only=_metrics(
                    np.concatenate([row['position_errors_m'][1:] for row in selected]),
                    np.concatenate([row['velocity_errors_mps'][1:] for row in selected])))
        result['truth_reset_previous_position_only'] = dict(oracle_diagnostic=True, peer_score=False,
            feedback_used=False, reset_query_indices=list(range(1, q)),
            first_query_policy='original noisy OLS initialization; no reset',
            retained='actual closed-loop previous V, P/V histories and visible measurements',
            groups=shadow_groups, scenes=shadow_rows)
    return result


def split_counts(scenes):
    return dict(scenes=len(scenes), future_scans=sum(len(s['measurements']) for s in scenes),
                target_detections=sum(int(s['detected'].sum()) for s in scenes),
                missed_target_detections=sum(int((~s['detected']).sum()) for s in scenes),
                clutter_candidates=sum(int(s['clutter_counts'].sum()) for s in scenes),
                empty_real_candidate_frames=sum(not len(z) for s in scenes for z in s['measurements']))


def run_probe(output, *, updates=256, train_per_motion=64, test_per_motion=32, seed=SEED,
              qualification=False, midpoint=256):
    # Small direct-call settings are for CPU tests; the qualification CLI below
    # fixes population, seed, 1024 updates and the 256-update intermediate endpoint.
    limit = 1024 if qualification else 256
    if type(updates) is not int or not 1 <= updates <= limit or type(seed) is not int:
        raise ValueError(f'fixed probe allows 1..{limit} updates and an integer seed')
    if qualification and (type(midpoint) is not int or not 0 < midpoint < updates):
        raise ValueError('qualification requires a positive intermediate endpoint before final')
    arms = ('adapted_labels',) if qualification else ARMS
    output = Path(output)
    if output.exists():
        raise FileExistsError('native probe output exists; no retry/overwrite')
    torch.set_num_threads(1)
    train = generate_split(train_per_motion, seed, 'train')
    test = generate_split(test_per_motion, seed + 1, 'heldout')
    plan = np.random.default_rng(seed + 2).integers(len(train), size=(updates, 8))
    torch.manual_seed(seed)
    model = DeepAFPeer()
    initial_state = deepcopy(model.state_dict())
    output.mkdir(parents=True)
    start = time.monotonic()
    state = dict(scope=QUALIFICATION_SCOPE if qualification else SCOPE,
                 status='RUNNING', device='cpu', cpu_threads=1, seed=seed,
                 configuration=dict(updates=updates, batch_size=8, learning_rate=.001, gradient_clip=1.,
                     train_per_motion=train_per_motion, test_per_motion=test_per_motion),
                 train_population=split_counts(train), heldout_population=split_counts(test),
                 completed_arms=[], updates_by_arm={}, test_used_for_selection=False,
                 original_paper_reproduction=False, production_level1=False)
    if qualification:
        state.update(arms=list(arms), endpoint_updates=[0, midpoint, updates],
                     oracle_shadow_is_peer_score=False, validation_checkpoint_selection=False,
                     fresh_training_not_resume=True)
    state.update(model_configuration=model.configuration,
        parameter_count=sum(p.numel() for p in model.parameters()),
        split_seeds=dict(train=seed, heldout=seed+1, sample_order=seed+2),
        simulation_contract=dict(dt_seconds=1., samples=31, duration_seconds=30.,
            initial_observations=6, initial_segment_correctly_associated_and_given=True,
            position_noise_sigma_m=50., detection_probability=.9, clutter_poisson_mean=8.,
            clutter_halfwidth_m=500., initial_speed_range_mps=[100., 250.], speed_bounds_mps=[50., 300.],
            cv_acceleration_sigma_each_axis_mps2=10., ca_acceleration_mean_each_axis_mps2=10.,
            ca_acceleration_variance_each_axis=10., ct_omega_radians_per_second=[-float(np.pi)/60., float(np.pi)/15.],
            cv_ca_position_rule='trapezoid after velocity norm projection', ct_position_rule='exact circular integral',
            history_initialization='OLS fit to the six given noisy observations, five fitted differences',
            association_gate=False, automatic_birth_death=False, all_real_candidates_used=True),
        control_contract='OLS-CV extrapolation and all-candidate centroid; empty centroid uses OLS-CV, centroid velocity is its consecutive position difference')
    write_json(output/'run.json', state)
    try:
        initial = evaluate(model, test, truth_reset=qualification)
        write_json(output/'initial-heldout.json', initial)
        finals, training = {}, {}
        for arm in arms:
            model.load_state_dict(initial_state)
            if any(not torch.equal(value, initial_state[key]) for key, value in model.state_dict().items()):
                raise ValueError('native probe arms must share exactly the initial parameters')
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            journal = []
            endpoints = {}
            def save_endpoint(step):
                path = output/f'{arm}-step{step}.pt'
                torch.save(dict(scope=state['scope'], arm=arm, completed_updates=step,
                    configuration=deepcopy(state['configuration']), model_configuration=model.configuration,
                    seed=seed, split_seeds=state['split_seeds'], model=deepcopy(model.state_dict()),
                    optimizer=deepcopy(optimizer.state_dict()), torch_rng_state=torch.get_rng_state(),
                    planned_scene_indices=plan.copy(), completed_sampling_records=deepcopy(journal)), path)
                evaluation = initial if step == 0 else evaluate(model, test, truth_reset=True)
                evaluation_path = output/f'{arm}-step{step}-heldout.json'
                write_json(evaluation_path, evaluation)
                endpoints[str(step)] = dict(checkpoint=path.name, evaluation=evaluation_path.name,
                    completed_updates=step, closed_loop=evaluation['groups'],
                    oracle_previous_position_only=evaluation['truth_reset_previous_position_only']['groups'])
                state.update(endpoints=endpoints, active_arm=arm, elapsed_seconds=time.monotonic()-start)
                write_json(output/'run.json', state)
                return evaluation
            if qualification:
                save_endpoint(0)
            for step, indices in enumerate(plan, 1):
                loss = train_update(model, optimizer, [train[i] for i in indices], arm)
                row = dict(step=step, scene_indices=indices.tolist(), **loss)
                if qualification:
                    row['scene_ids'] = [train[i]['scene_id'] for i in indices]
                journal.append(row)
                state['updates_by_arm'][arm] = step
                if qualification:
                    with (output/(arm+'-sampling.jsonl')).open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                    if step in (midpoint, updates):
                        endpoint_evaluation = save_endpoint(step)
                if step % 32 == 0:
                    state.update(active_arm=arm, elapsed_seconds=time.monotonic() - start)
                    write_json(output/'run.json', state)
                    print(f'{arm}: {step}/{updates}', flush=True)
            training[arm], finals[arm] = journal, (endpoint_evaluation if qualification else evaluate(model, test))
            write_json(output/(arm+'-training.json'), journal)
            write_json(output/(arm+'-heldout.json'), finals[arm])
            state['completed_arms'].append(arm)
            if qualification:
                state['endpoints'] = endpoints
        summary = dict(state, status='COMPLETED', initial=initial['groups'],
            final={arm: value['groups'] for arm, value in finals.items()},
            same_initial_parameters=True, same_sampled_scene_order=True,
            actual_updates={arm: len(rows) for arm, rows in training.items()},
            elapsed_seconds=time.monotonic()-start,
            interpretation='Native-style core diagnostic with given correct initialization; not full tracking reproduction or Level1 peer qualification.',
            label_selection_performed=False, method_promoted=False)
        write_json(output/'summary.json', summary)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-start)
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-start)
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('labels', 'qualification'), default='labels')
    parser.add_argument('--updates', type=int)
    parser.add_argument('--train-per-motion', type=int, default=64)
    parser.add_argument('--test-per-motion', type=int, default=32)
    args = parser.parse_args(argv)
    options = vars(args).copy()
    mode = options.pop('mode')
    if mode == 'qualification':
        if (args.updates not in (None, 1024) or args.train_per_motion != 64 or args.test_per_motion != 32):
            parser.error('qualification is fixed: adapted_labels, 1024 updates, train192/heldout96, endpoints0/256/1024')
        options.update(updates=1024, qualification=True)
    elif args.updates is None:
        options['updates'] = 256
    result = run_probe(**options)
    print(json.dumps(dict(status=result['status'], output=str(Path(args.output).resolve()),
        actual_updates=result['actual_updates'], elapsed_seconds=result['elapsed_seconds']),
        ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
