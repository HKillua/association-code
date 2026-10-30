"""143 shared source potential from relative, optionally regret-weighted labels."""
from __future__ import annotations

import numpy as np
from association.evaluation import source_risk as risk

NEW = ('WS_PAIR_LOGIT_HARD', 'WS_REGRET_LOGIT_HARD')


def pair_data(packs, mean, std):
    differences, targets, weights, costs = [], [], [], []
    ties = 0
    for pack in packs:
        phi = risk._design(pack['features'].reshape(-1, 10), mean, std, True).reshape(pack['features'].shape[:2]+(66,))
        for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:]):
            counts = pack['labeled'][a:b].sum(0)
            pairs = np.maximum(counts*(counts-1)/2, 1)
            for i in range(a, b):
                for j in range(i+1, b):
                    use = pack['labeled'][i] & pack['labeled'][j]
                    delta = (pack['errors_m'][i]-pack['errors_m'][j])/1000.
                    ties += int((use & (delta == 0)).sum())
                    use &= delta != 0
                    differences.append((phi[i, use]-phi[j, use])[:, 1:])
                    targets.append(np.sign(delta[use])); weights.append(1./pairs[use]); costs.append(np.abs(delta[use]))
    if not differences:
        raise ValueError('no labelled source pair blocks')
    return np.concatenate(differences), np.concatenate(targets), np.concatenate(weights), np.concatenate(costs), ties


def fit_pair_model(packs, mean, std, *, regret_weighted):
    x, y, weight, cost, ties = pair_data(packs, mean, std)
    if regret_weighted:
        weight = weight*cost
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(weight).all() or weight.sum() <= 0:
        raise ValueError('finite nonempty strict preference pairs required')
    w = weight/weight.sum()
    beta = np.zeros(65)
    def objective(b):
        return float(w@np.logaddexp(0., -y*(x@b))+.5*(b@b))
    initial = objective(beta)
    for step in range(31):
        margin = y*(x@beta)
        probability = np.exp(-np.logaddexp(0., margin))
        gradient = x.T@(-w*y*probability)+beta
        norm = float(np.max(np.abs(gradient)))
        if norm <= 1e-8:
            break
        if step == 30:
            raise FloatingPointError('pair logistic did not converge in30 updates')
        hessian = x.T@(x*(w*probability*(1-probability))[:, None])+np.eye(65)
        direction = np.linalg.solve(hessian, gradient)
        current, slope, scale = objective(beta), float(gradient@direction), 1.
        for backtrack in range(20):
            candidate = beta-scale*direction
            if objective(candidate) <= current-1e-4*scale*slope:
                beta = candidate
                break
            scale *= .5
        else:
            raise FloatingPointError('pair logistic line search failed')
    return dict(head='QUADRATIC', mean=np.asarray(mean).copy(), std=np.asarray(std).copy(),
        beta=np.r_[0., beta], newton_updates=step, gradient_infinity_norm=norm,
        initial_objective=initial, objective=objective(beta), strict_training_pairs=len(y),
        tied_training_pairs=ties, raw_training_weight_sum=float(weight.sum()), regret_weighted=regret_weighted)


def predict(model, features):
    return risk.predict_score(model, np.asarray(features).reshape(-1, 10)).reshape(features.shape[:2])
