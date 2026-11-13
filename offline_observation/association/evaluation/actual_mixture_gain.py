"""144 continuous source-symmetric mixture family learned on actual output gain."""
from __future__ import annotations

import numpy as np
from association.evaluation import source_risk as risk
from association.evaluation.source_contribution_readout import embedding_km, recover_contributions

ALPHAS = np.linspace(0., 1., 9)
NEW = ('WS_GLOBAL_ACTUAL_MIX', 'WS_CONDITIONAL_ACTUAL_MIX')


def context(pack, scores):
    scores = np.asarray(scores, float)
    available, features, offsets = pack['available'], pack['features'], pack['group_offsets']
    hard = risk.source_weights(scores, available, offsets, kind='hard')
    equal = risk.source_weights(scores, available, offsets, kind='equal')
    points = embedding_km(pack['source_states'])
    values = np.zeros((len(offsets)-1, scores.shape[1], 24))
    for gid, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        u, h, x = equal[a:b], hard[a:b], features[a:b]
        k = available[a:b].sum(0)
        centre = np.sum(points[a:b]*u[..., None], axis=0)
        spread = np.sqrt(np.maximum(0., 2*np.sum(np.sum((points[a:b]-centre)**2, axis=-1)*u, axis=0)))
        ordered = np.sort(np.where(available[a:b], scores[a:b], np.inf), axis=0)
        margin = np.zeros_like(k, float)
        if b-a >= 2:
            margin[k >= 2] = ordered[1, k >= 2]-ordered[0, k >= 2]
        values[gid] = np.concatenate((np.sum(x*u[..., None], axis=0), np.sum(x*h[..., None], axis=0),
            np.column_stack((np.log1p(k), np.log1p(spread), margin, np.sum(scores[a:b]*u, axis=0)))), axis=1)
    if not np.isfinite(values).all():
        raise FloatingPointError('nonfinite mixture context')
    return values, hard, equal


def recover(motion, groups, mean, pack, hard, equal, alpha):
    alpha = np.asarray(alpha, float)
    shape = (len(groups), hard.shape[1])
    if alpha.shape != shape or not np.isfinite(alpha).all() or ((alpha < 0) | (alpha > 1)).any():
        raise ValueError('one legal alpha per group/query required')
    expanded = alpha[pack['source_group_indices']]
    weights = hard*(1-expanded)+equal*expanded
    return recover_contributions(states=motion['states'], support=motion['support'], groups=groups,
        pack=pack, source_weights=weights, mean_states=mean)


def _design(x, mean, std):
    return np.column_stack((np.ones(len(x)), np.clip((x-mean)/std, -8., 8.)))


def fit_gain_model(x, gains):
    x, gains = np.asarray(x, float), np.asarray(gains, float)
    if x.ndim != 2 or x.shape[1] != 24 or gains.shape != (len(x), 9) or not len(x) or not np.isfinite(x).all() or not np.isfinite(gains).all():
        raise ValueError('finite context[N,24] and actual gain[N,9] required')
    if not np.array_equal(gains[:, 0], np.zeros(len(x))):
        raise ValueError('alpha0 gain must be exactly zero')
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.
    z = _design(x, mean, std)
    alpha = ALPHAS[1:]
    design = np.concatenate((z[:, None, :]*alpha[None, :, None], z[:, None, :]*(alpha**2)[None, :, None]), axis=2).reshape(-1, 50)
    target = gains[:, 1:].ravel()
    penalty = np.eye(50); penalty[0, 0] = 0.; penalty[25, 25] = 0.
    beta = np.linalg.solve(design.T@design/len(target)+penalty, design.T@target/len(target))
    if not np.isfinite(beta).all():
        raise FloatingPointError('nonfinite actual-gain fit')
    global_losses = gains.mean(0)
    global_alpha = float(ALPHAS[np.argmin(global_losses)])
    return dict(mean=mean, std=std, beta=beta, global_alpha=global_alpha,
        training_queries=len(x), training_candidates=len(target),
        training_mse_km2=float(np.mean((design@beta-target)**2)), global_mean_gain_km=global_losses)


def predict_alpha(model, x):
    x = np.asarray(x, float)
    shape = x.shape[:-1]
    z = _design(x.reshape(-1, 24), model['mean'], model['std'])
    a, b = z@model['beta'][:25], z@model['beta'][25:]
    alpha = (a+b < 0).astype(float)
    use = b > 0
    alpha[use] = np.clip(-a[use]/(2*b[use]), 0., 1.)
    return alpha.reshape(shape), (a*alpha+b*alpha**2).reshape(shape), b.reshape(shape)
