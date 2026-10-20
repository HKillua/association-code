"""One approved training-only CL calibration, not a margin search interface."""
from __future__ import annotations

import time

import numpy as np
import torch

from association.training.sampling import BalancedScenePairSampler
from association.training.batches import _pair_batch

MARGIN_POLICY = 'train_negative_median_v1'
CALIBRATION_SEED = 21
CALIBRATION_BATCHES = 64
CALIBRATION_PAIRS = 64


def _quantiles(values):
    return dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'),
                    map(float, np.quantile(values, [0, .05, .25, .5, .75, .95, 1]))))


def calibrate_training_margin(model, scenes, normalized, cache, *, device,
                              batch_tracks, check, report):
    """Return m without changing parameters, modes, or training RNG state.

    Caller has frozen the completion modules for TAM_FROZEN. The same live
    completion cache is deliberately warmed; this is the only retained mutable
    training state. ``report`` retains partial evidence on error, never a valid
    fallback margin. This function does not read validation scenes.
    """
    started = time.monotonic()
    report.update(policy=MARGIN_POLICY, status='RUNNING', seed=CALIBRATION_SEED,
                  batch_count=CALIBRATION_BATCHES, pair_batch_size=CALIBRATION_PAIRS,
                  completed_batches=0, samples=[], seconds=None,
                  scope='T_core after TCM, before any TAM update; train TAM with configured dropout, eval frozen TCM; no_grad',
                  tam_dropout=model.config.effective_tam_dropout,
                  cache_before=cache.stats())
    device = torch.device(device)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        check()
        if (model.branch != 'contrastive' or model.config.contrastive_positive_power != 1
                or not scenes or any(scene['split'] != 'T_core' for scene in scenes)):
            raise ValueError('margin calibration requires CL p=1 and only the T_core training pool')
        if (not model._completion_frozen
                or any(p.requires_grad for p in model.completion_parameters(include_embedding=True))):
            raise ValueError('margin calibration requires frozen completion parameters')
        sampler = BalancedScenePairSampler(scenes, CALIBRATION_SEED)
        report['sampling'] = sampler.describe()
        # Local generators only supply seeded states. manual_seed on torch's
        # global entry point would also reset other CUDA devices.
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(CALIBRATION_SEED).get_state())
        if device.type == 'cuda':
            state = torch.Generator(device=device).manual_seed(CALIBRATION_SEED).get_state()
            torch.cuda.set_rng_state(state, device)
        model.train()
        all_distances, all_labels, presentations = [], [], []
        with torch.no_grad():
            for batch in range(CALIBRATION_BATCHES):
                check()
                groups = sampler.sample(CALIBRATION_PAIRS)
                metadata = sampler.batch_metadata(groups)
                if (metadata['pair_count'] != 64 or metadata['positive_count'] != 32
                        or metadata['negative_count'] != 32):
                    raise ValueError('margin calibration batch must contain exactly 32 positive and 32 negative pairs')
                keys, _, _, pairs, labels, _ = _pair_batch(scenes, normalized, groups, device)
                complete = cache.get(keys)
                encoded = []
                for start in range(0, len(keys), batch_tracks):
                    check()
                    encoded.append(model.tam_encoder(complete[start:start + batch_tracks]).flatten(1))
                distances = -model.score_pairs(torch.cat(encoded), pairs)
                if (distances.shape != (64,) or labels.shape != (64,)
                        or not torch.isfinite(distances).all() or (distances < 0).any()):
                    raise ValueError('margin calibration requires exactly 64 finite nonnegative distances')
                distance_dtype = distances.dtype
                distance_values = distances.detach().cpu().numpy().astype(np.float64)
                label_values = labels.detach().cpu().numpy()
                if (not np.isin(label_values, (0, 1)).all()
                        or int((label_values == 1).sum()) != 32):
                    raise ValueError('margin calibration labels must contain 32 positive and 32 negative pairs')
                sample_groups = [dict(scene_index=int(s), scene_id=scenes[s]['scene_id'],
                                      pair_indices=indices.tolist()) for s, indices in groups]
                presentations.extend((int(s), int(index)) for s, indices in groups for index in indices)
                all_distances.extend(distance_values.tolist())
                all_labels.extend(label_values.astype(int).tolist())
                report['samples'].append(dict(batch=batch + 1, groups=sample_groups,
                    labels=label_values.astype(int).tolist(), distances=distance_values.tolist(),
                    duplicate_pair_count=metadata['duplicate_pair_count']))
                report['completed_batches'] = batch + 1
                check()
        distances, labels = np.asarray(all_distances), np.asarray(all_labels)
        positive, negative = distances[labels == 1], distances[labels == 0]
        if len(distances) != 4096 or len(positive) != 2048 or len(negative) != 2048:
            raise ValueError('margin calibration requires 4096 presentations, 2048 per class')
        margin = float(np.median(negative))  # even sample: mean of the central two
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError('calibrated margin must be finite and positive; no fallback')
        # Keep the exact float64 median of the observed distance values, while
        # reporting the actual scalar conversion/subtraction used by training.
        # An adjacent-float midpoint can round onto one of the negative values.
        training_margin = float(torch.tensor(margin, dtype=distance_dtype).item())
        hinge = torch.relu(margin - torch.tensor(negative, dtype=distance_dtype)).double().numpy()
        unique = len(set(presentations))
        # l_i=.5[y*d+(1-y)*relu(m-d)^2]. Derivatives below are dl_i/dd,
        # followed by the actual 64-pair batch-mean derivatives (divide by 64).
        derivative = {'positive': np.full(len(positive), .5), 'negative': -hinge}
        report.update(effective_margin=margin, median_rule='numpy median; mean of two central negative presentations',
            training_dtype_margin=training_margin, distance_dtype=str(distance_dtype),
            pair_presentations=len(distances), positive_presentations=len(positive), negative_presentations=len(negative),
            unique_pair_count=unique, duplicate_pair_count=len(presentations) - unique,
            unique_scene_count=len({s for s, _ in presentations}),
            distance_quantiles={'positive': _quantiles(positive), 'negative': _quantiles(negative)},
            negative_hinge=dict(strict_active_count=int((negative < training_margin).sum()),
                                strict_active_fraction=float((negative < training_margin).mean()),
                                ties_count=int((negative == training_margin).sum())),
            mathematical_median_hinge=dict(strict_active_count=int((negative < margin).sum()),
                strict_active_fraction=float((negative < margin).mean()), ties_count=int((negative == margin).sum())),
            loss=dict(positive_class_mean=float(.5 * positive.mean()),
                      negative_class_mean=float(.5 * np.square(hinge).mean()),
                      positive_contribution=float(.25 * positive.mean()),
                      negative_contribution=float(.25 * np.square(hinge).mean()),
                      total=float(.25 * (positive.mean() + np.square(hinge).mean())),
                      reduction='float64 diagnostic mean over 4096 presentations, using the training-dtype hinge; equal class weighting, no parameter gradient'),
            distance_derivative={name: dict(per_example_mean=float(values.mean()),
                per_example_quantiles=_quantiles(values), batch_mean_mean=float(values.mean() / 64),
                batch_mean_quantiles=_quantiles(values / 64)) for name, values in derivative.items()},
            derivative_scope='d loss / d distance only, not parameter gradient; per_example l_i and batch_mean l_i/64')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        check()
        report['status'] = 'COMPLETED'
        return margin
    except BaseException as error:
        report.update(status='FAILED', failure_type=type(error).__name__, failure=str(error))
        # No completed calibration may be inferred from a computed-but-rejected m.
        report.pop('effective_margin', None)
        raise
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        for module, mode in modes:
            module.training = mode
        report.update(seconds=time.monotonic() - started, cache_after=cache.stats())
