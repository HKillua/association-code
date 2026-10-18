"""Run an explicit CPU/MPS/CUDA candidate check or separately approved native fit.

The synthetic check is software verification, never an MTAD performance result.
Native fit requires all data/config paths and a new output directory. No server
launch, retry, automatic accelerator fallback or test split is implemented.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association import runtime
from association.cooperative import CooperativeConfig, CooperativeModel
from association.cooperative_training import resolve_device, train_cooperative, training_configuration
from association.reference import fit_normalizer, normalize_scene
from association.timed_inputs import prepare_timed_scene


def synthetic_scenes(model_config):
    """Independent tiny training/development identities, original native format."""
    scenes = []
    for name, split in (('synthetic-train', 'T_core'), ('synthetic-validation', 'V_select')):
        tracks = []
        clocks = ([12., 13.], np.linspace(0., 30., model_config.max_observations),
                  np.linspace(2., 27., max(2, model_config.max_observations//2)),
                  np.linspace(3., 29., max(2, model_config.max_observations//3)))
        identities = [name+'-A', name+'-A', name+'-B', name+'-B']
        sources = ['9001', '9002', '9001', '9002']
        for index, clock in enumerate(clocks):
            clock = np.asarray(clock, dtype=np.float64)
            states = np.column_stack((30.+clock*.0001+(index//2)*.01, 120.+clock*.0002,
                                      5.+clock*.01+index*.1, 40.+clock*.1+index))
            tracks.append(dict(node_id=name+':'+str(index), identity_id=identities[index], source=sources[index],
                               canonical_times=clock, canonical_values=states))
        pairs = np.column_stack(np.triu_indices(4, 1)).astype(np.int64)
        native = dict(scene_id=name, component_id=name+'-component', split=split, tracks=tracks,
                      node_ids=[x['node_id'] for x in tracks], identities=np.array(identities), sources=sources,
                      intervals=np.array([[x['canonical_times'][0], x['canonical_times'][-1]] for x in tracks]),
                      pairs=pairs, labels=np.array([identities[a] == identities[b] for a, b in pairs], dtype=np.float32))
        prepared = prepare_timed_scene(native, length=model_config.length,
            max_observations=model_config.max_observations, time_unit_seconds=20.)
        targets = {}
        for index, identity in enumerate(dict.fromkeys(identities)):
            t = prepared['grid']
            values = np.column_stack((30.+t*.0001+index*.01, 120.+t*.0002, 5.+t*.01, 40.+t*.1))
            targets[identity] = dict(states=values, valid=np.ones_like(values, dtype=bool), query_times=t.copy())
        prepared['fusion_targets'] = targets
        scenes.append(prepared)
    return [scenes[0]], [scenes[1]]


def synthetic_configuration():
    return dict(seed=20, batch_size=2, teacher_updates=1, utility_updates=1, adapt_updates=1,
                learning_rate=1e-4, gradient_clip=1., endpoint_fraction=.25, utility_epsilon=1e-3,
                error_scale_policy='training_native_std', routes=['plain', 'identity', 'utility', 'learned'],
                max_seconds=300., batch_tracks=2, pair_batch_size=3)


def production_shape_check(device, output):
    """One full-shape finite forward/backward, no optimization or dataset access."""
    cfg = CooperativeConfig()
    training, _ = synthetic_scenes(cfg)
    normalizer = fit_normalizer(training)
    torch.manual_seed(20)
    model = CooperativeModel(cfg, normalizer).to(device).train()
    packed, mask = normalize_scene(training[0], normalizer)
    values = torch.as_tensor(packed[:2], device=device)
    observed = torch.as_tensor(mask[:2], device=device)
    started = time.monotonic()
    encoded = model.encode(values, observed)
    pair = torch.tensor([[0, 1]], device=device)
    utility = model.utility(encoded, pair)
    logits = model.score_pairs(encoded, pair)
    states = model.decode_group(encoded, [0, 1], routing='utility', identity_logits=logits,
                                utility=utility[:, 0])
    loss = states.square().mean() + utility.square().mean() + logits.square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('nonfinite full-shape gradients')
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)
    report = dict(status='CHECKED', scope='synthetic full-shape forward/backward only', device=str(device),
                  model_config=asdict(cfg), parameters=sum(p.numel() for p in model.parameters()),
                  visible_counts=observed[:, :cfg.max_observations].sum(1).cpu().tolist(),
                  output_shape=list(states.shape), elapsed_seconds=time.monotonic()-started,
                  actual_optimizer_updates=0, native_MTAD_opened=False, test_opened=False)
    output.mkdir(parents=True, exist_ok=False)
    runtime.write_json(output/'shape-check.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'shape-check', 'fit'))
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    for name in ('archive', 'manifest', 'alignment', 'config'):
        parser.add_argument('--'+name)
    args = parser.parse_args(argv)
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise InterruptedError('SIGTERM; candidate did not complete')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.cpu_threads < 1:
            raise ValueError('cpu threads must be positive')
        torch.set_num_threads(args.cpu_threads)
        device, output = resolve_device(args.device), Path(args.output)
        if output.exists():
            raise FileExistsError(f'Output already exists: {output}')
        if args.mode != 'fit' and any(getattr(args, key) for key in ('archive', 'manifest', 'alignment', 'config')):
            raise ValueError('synthetic checks do not accept native dataset/config paths')
        if args.mode == 'shape-check':
            report = production_shape_check(device, output)
        else:
            preparation = None
            if args.mode == 'check':
                model = CooperativeConfig(length=4, max_observations=4, d_model=8, nhead=2,
                                          num_layers=1, dim_feedforward=16, head_hidden=8, dropout=.1)
                training = synthetic_configuration()
                train, validation = synthetic_scenes(model)
            else:
                if not all(getattr(args, key) for key in ('archive', 'manifest', 'alignment', 'config')):
                    raise ValueError('fit requires explicit archive, manifest, alignment and config')
                config = runtime.read_json(args.config)
                if (set(config) != {'scope', 'model', 'training', 'data', 'train_splits'}
                        or config['scope'] != 'cooperative-physical-v1'
                        or config['train_splits'] != ['T_core']
                        or set(config['data']) != {'time_unit_seconds'}):
                    raise ValueError('explicit cooperative-physical-v1 T_core configuration required')
                model = CooperativeConfig(**config['model'])
                training = training_configuration(config['training'])
                def check():
                    if time.monotonic()-started >= training['max_seconds']:
                        raise TimeoutError('candidate budget exhausted during preparation')
                from association.timed_dataset import load_timed_dataset
                dataset = load_timed_dataset(args.archive, runtime.read_json(args.manifest),
                    train_splits=['T_core'], data_config=dict(length=model.length,
                        max_observations=model.max_observations, **config['data']),
                    alignment_by_scene=runtime.read_json(args.alignment), check=check)
                train, validation, preparation = dataset['train'], dataset['validation'], dataset['report']
            try:
                report = train_cooperative(train, validation, model, training, output, device=device,
                                           initial_elapsed_seconds=time.monotonic()-started)
            finally:
                if output.is_dir():
                    runtime.write_json(output/'execution.json', dict(mode=args.mode, synthetic=args.mode=='check',
                        configuration=dict(model=asdict(model), training=training), preparation=preparation,
                        cpu_threads=args.cpu_threads, native_MTAD_opened=args.mode=='fit'))
        print(f"{report['status']}: {output.resolve()}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
