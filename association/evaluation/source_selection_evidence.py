"""139: one-factor radial-meter target with unchanged Q feature basis/readout.

Scores are finite regression outputs, not nonnegative calibrated risk or Cov.
All label/fold permissions belong to the runner; this module has no file I/O.
"""
from __future__ import annotations

import numpy as np

from association.evaluation import source_risk as risk

TARGET = 'radial_error_km'


def fit_quadratic_target(x, y, sample_weight):
    """Original133 weighted ridge for a caller-declared target; exactly one fit."""
    x, y, weight = [np.asarray(v, np.float64) for v in (x, y, sample_weight)]
    if (x.ndim != 2 or x.shape[1] != 10 or y.shape != (len(x),)
            or weight.shape != y.shape or not all(np.isfinite(v).all() for v in (x, y, weight))
            or (weight <= 0).any()):
        raise ValueError('finite x[L,10], y[L], positive weights[L] required')
    total = float(weight.sum())
    if not np.isfinite(total):
        raise FloatingPointError('nonfinite weight sum')
    normalized = weight/total if len(x) else weight
    mean = normalized @ x if len(x) else np.zeros(10)
    std = np.sqrt(normalized @ ((x-mean)**2)) if len(x) else np.ones(10)
    std[std < 1e-6] = 1.
    z = np.clip((x-mean)/std, -8., 8.)
    i, j = np.triu_indices(10)
    design = np.concatenate((np.ones((len(x), 1)), z, z[:, i]*z[:, j]), axis=1)
    beta = np.zeros(66)
    if len(x):
        penalty = np.eye(66)
        penalty[0, 0] = 0.
        beta = np.linalg.solve(design.T @ (design*normalized[:, None])+penalty,
                               design.T @ (normalized*y))
    if not all(np.isfinite(v).all() for v in (mean, std, beta)):
        raise FloatingPointError('nonfinite source target fit')
    return dict(head='QUADRATIC', mean=mean, std=std, beta=beta,
        training_label_count=len(y), training_weight_sum=total,
        no_labels_fallback=not len(y), permutation=np.arange(len(y)))


def fit_meter_model(x, errors_m, sample_weight):
    errors = np.asarray(errors_m, np.float64)
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('finite nonnegative radial error labels required')
    return dict(fit_quadratic_target(x, errors/1000., sample_weight), target_kind=TARGET)


def score_stats(scores, pack, weights, model=None):
    available = pack['available']
    values = scores[available]
    selected = (weights > 0) & available
    group_ties = 0
    for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:]):
        group_ties += int(((weights[a:b] > 0).sum(0) > 1).sum())
    out = dict(available_source_queries=len(values), negative_score_count=int((values < 0).sum()),
        negative_score_fraction=float((values < 0).mean()) if len(values) else None,
        available_score_quantiles=np.quantile(values, [0, .01, .5, .99, 1]).tolist() if len(values) else [],
        selected_source_queries=int(selected.sum()),
        selected_negative_count=int(((scores < 0) & selected).sum()),
        selected_negative_fraction=float((scores[selected] < 0).mean()) if selected.any() else None,
        exact_tied_group_queries=group_ties)
    if model is not None:
        raw = (pack['features']-model['mean'])/model['std']
        out['available_feature_value_count'] = int(available.sum()*10)
        out['available_feature_clipped_count'] = int((np.abs(raw[available]) > 8).sum())
    return out


def describe(values):
    v = np.asarray(values, np.float64)
    if not np.isfinite(v).all():
        raise ValueError('finite diagnostic samples required')
    return dict(count=len(v), sum=float(v.sum()), mean=float(v.mean()) if len(v) else None,
        rmse=float(np.sqrt(np.mean(v*v))) if len(v) else None,
        median=float(np.median(v)) if len(v) else None,
        p95=float(np.quantile(v, .95)) if len(v) else None,
        p99=float(np.quantile(v, .99)) if len(v) else None,
        maximum=float(v.max()) if len(v) else None)


def diagnostic_block(data, mask=None):
    n = len(data['group_indices'])
    use = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    out = dict(query_count=int(use.sum()), heads={})
    for head in ('LOG', 'METER'):
        regret = data[head+'__selection_regret_m'][use]
        credit = data[head+'__selection_credit'][use]
        count = int(np.ceil(.01*len(regret)))
        harm = np.sort(np.maximum(regret, 0))[::-1]
        out['heads'][head] = dict(actual_error_m=describe(data[head+'__actual_error_m'][use]),
            selection_cost_m=describe(data[head+'__selection_cost_m'][use]),
            selection_regret_m=describe(regret), selection_credit=describe(credit),
            strict_correct_count=int((data[head+'__strict_correct'][use]).sum()),
            strict_wrong_count=int((data[head+'__strict_wrong'][use]).sum()),
            tie_query_count=int(data[head+'__tie'][use].sum()),
            harm_gt_1000m_count=int((regret > 1000.).sum()),
            harm_up_to_1000m_count=int(((regret > 0) & (regret <= 1000.)).sum()),
            harm_gt_1000m_sum=float(regret[regret > 1000.].sum()),
            harm_up_to_1000m_sum=float(regret[(regret > 0) & (regret <= 1000.)].sum()),
            top_one_percent_query_count=count,
            top_one_percent_regret_share=float(harm[:count].sum()/harm.sum()) if harm.sum() else None)
    delta = data['METER__actual_error_m'][use]-data['LOG__actual_error_m'][use]
    dr = data['METER__selection_regret_m'][use]-data['LOG__selection_regret_m'][use]
    out['meter_minus_log_actual_m'] = describe(delta)
    out['positive_harm_m'] = describe(delta[delta > 0])
    out['benefit_m'] = describe(-delta[delta < 0])
    out['meter_minus_log_regret_m'] = describe(dr)
    out['strict_transition'] = {}
    for old_label in ('correct', 'wrong'):
        for new_label in ('correct', 'wrong'):
            take = use & data['LOG__strict_'+old_label] & data['METER__strict_'+new_label]
            out['strict_transition'][old_label+'_to_'+new_label] = dict(count=int(take.sum()),
                log_regret_sum_m=float(data['LOG__selection_regret_m'][take].sum()),
                meter_regret_sum_m=float(data['METER__selection_regret_m'][take].sum()))
    out['at_least_one_score_tie_count'] = int((use & (data['LOG__tie'] | data['METER__tie'])).sum())
    return out


def summarize_diagnostic(data):
    return dict(overall=diagnostic_block(data),
        by_role={name: diagnostic_block(data, data['role_best'] == i)
                 for i, name in enumerate(('role_0_best', 'role_1_best', 'reference_tie'))},
        by_purity={name: diagnostic_block(data, data['group_pure'] == pure)
                   for name, pure in (('pure', True), ('mixed', False))})


def build_diagnostic(original, pack, scores, policies, motion, ev, membership, groups):
    """Bind original134 D2 before computing new-head descriptive quantities."""
    for key, expected in (('node_ids', motion['node_ids']), ('query_seconds', motion['query_seconds']),
                          ('membership', membership)):
        if not np.array_equal(original[key], expected):
            raise ValueError('134 D2 input binding: '+key)
    gi, qi, sr = (original[k] for k in ('group_indices', 'query_indices', 'source_rows'))
    n = len(gi)
    if sr.shape != (n, 2) or qi.shape != gi.shape:
        raise ValueError('fixed K2 D2 keys required')
    if len(set(zip(gi.tolist(), qi.tolist()))) != n:
        raise ValueError('duplicate D2 query keys')
    for i in range(n):
        a, b = pack['group_offsets'][gi[i]:gi[i]+2]
        if (sr[i] < a).any() or (sr[i] >= b).any() or sr[i, 0] == sr[i, 1]:
            raise ValueError('D2 rows must belong to declared group')
    expected = np.zeros_like(pack['labeled'])
    expected[sr, qi[:, None]] = True
    if (not np.array_equal(expected, pack['labeled'])
            or not np.all(pack['available'][sr, qi[:, None]])):
        raise ValueError('134 complete labeled source-query binding')
    registry = {str(value): i for i, value in enumerate(ev['target_ids'])}
    target_rows = np.asarray([registry[str(ev['identities'][groups[int(gid)][0]])] for gid in gi], np.int64)
    if (not ev['target_valid'][target_rows, qi, :2].all()
            or not np.array_equal(original['truth'], ev['target_states'][target_rows, qi, :2])):
        raise ValueError('134 truth differs from original native first-anchor target')
    endpoint = original['endpoint_errors_m']
    if not np.allclose(endpoint, pack['errors_m'][sr, qi[:, None]], rtol=1e-9, atol=1e-5):
        raise ValueError('134 source error binding')
    best = endpoint.min(1)
    if not np.allclose(best, original['best_single_error_m'], rtol=1e-9, atol=1e-5):
        raise ValueError('134 best endpoint binding')
    data = {k: original[k] for k in ('group_indices', 'query_indices', 'source_rows', 'role_indices',
                                     'group_pure', 'truth', 'endpoint_errors_m', 'best_single_error_m')}
    roles = original['role_indices']
    if set(roles.ravel().tolist())-set((0, 1)):
        raise ValueError('original anonymous two-role diagnostic required')
    role_best = np.where(endpoint[:, 0] < endpoint[:, 1], roles[:, 0], roles[:, 1])
    role_best[endpoint[:, 0] == endpoint[:, 1]] = 2
    data['role_best'] = role_best
    for head in ('LOG', 'METER'):
        weight = policies[head]['source_weights'][sr, qi[:, None]]
        if not np.allclose(weight.sum(1), 1., atol=1e-12, rtol=0):
            raise ValueError('D2 complete normalized source weights')
        position = policies[head]['group_states'][gi, qi, :2]
        actual = risk._position_distance(position, original['truth'])
        cost = (weight*endpoint).sum(1)
        credit = (weight*(endpoint == best[:, None])).sum(1)
        tie = (weight > 0).sum(1) > 1
        data.update({head+'__scores': scores[head][sr, qi[:, None]],
            head+'__source_weights': weight, head+'__actual_position': position,
            head+'__actual_error_m': actual, head+'__selection_cost_m': cost,
            head+'__selection_regret_m': cost-best, head+'__selection_credit': credit,
            head+'__tie': tie, head+'__strict_correct': (~tie) & (credit == 1),
            head+'__strict_wrong': (~tie) & (credit == 0)})
    if (not np.array_equal(data['LOG__scores'], original['oof_QUADRATIC_scores'])
            or not np.array_equal(data['LOG__actual_position'], original['QUADRATIC_hard_position'])
            or not np.allclose(data['LOG__actual_error_m'], original['QUADRATIC_hard_error_m'], rtol=1e-9, atol=1e-5)):
        raise ValueError('old LOG actual hard differs from134')
    return data
