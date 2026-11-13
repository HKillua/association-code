"""075 fixed-score diagnostics; no fitting, score changes or physical inference.

``analyze_scores`` accepts frozen all-pair scene records. It reports pooled,
per-scene, component-pooled and fixed scene-size-stratum views for seven scopes.
AUC treats tied positive/negative scores as half a win; AP is the noninterpolated
precision-recall step integral with all score ties processed together. No working
point is selected. Scene/component macros include only defined values and expose
their denominator; they are descriptive summaries, not independent-sample tests.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.tsactfer_level1 import paper_masks

SCOPES = ('all', 'itsa_all', 'itsa_eligible', 'itsa_ineligible',
          'mstsa_all', 'mstsa_eligible', 'mstsa_ineligible')
SIZE_STRATA = ('lt32', '32_to128', 'ge129')
QUANTILES = (('p01', .01), ('p05', .05), ('p25', .25), ('p50', .5),
             ('p75', .75), ('p95', .95), ('p99', .99))
MACRO_FIELDS = ('roc_auc', 'average_precision', 'ap_over_prevalence',
                'precision', 'recall', 'f1', 'false_positive_rate',
                'false_discovery_rate')
EXAMPLE_LIMIT = 20


def _quantiles(values):
    values = np.asarray(values)
    result = {'count': int(values.size), 'min': None, 'max': None,
              **{key: None for key, _ in QUANTILES}}
    if values.size:
        result.update(min=float(values.min()), max=float(values.max()))
        result.update({key: float(value) for (key, _), value in
                       zip(QUANTILES, np.quantile(values, [p for _, p in QUANTILES]))})
    return result


def _ranking(scores, labels):
    """One grouped stable sort, exact tie handling, no threshold optimization."""
    positive = int(labels.sum())
    negative = len(labels)-positive
    if not positive:
        return None, None
    if not negative:
        return None, 1.0
    order = np.argsort(-scores, kind='stable')
    descending, sorted_labels = scores[order], labels[order]
    starts = np.r_[0, np.flatnonzero(descending[1:] != descending[:-1])+1]
    ends = np.r_[starts[1:], len(scores)]
    positive_by_tie = np.add.reduceat(sorted_labels.astype(np.int64), starts)
    negative_by_tie = ends-starts-positive_by_tie
    cumulative_positive = np.cumsum(positive_by_tie)
    negatives_below = negative-np.cumsum(negative_by_tie)
    concordant = np.sum(positive_by_tie.astype(np.float64)*
                        (negatives_below+.5*negative_by_tie))
    auc = float(concordant/(positive*negative))
    ap = float(np.sum((positive_by_tie/positive)*(cumulative_positive/ends)))
    return auc, ap


def _metrics(scores, labels, threshold):
    n, positive = len(scores), int(labels.sum())
    negative = n-positive
    predicted = np.zeros(n, bool) if threshold is None else scores >= threshold
    tp = int(np.count_nonzero(labels & predicted))
    fp = int(np.count_nonzero(~labels & predicted))
    fn, tn = positive-tp, negative-fp
    prevalence = positive/n if n else None
    auc, ap = _ranking(scores, labels)
    denominator = 2*tp+fp+fn
    # Match the project's negative-scene F1 convention, but an empty scope is
    # not an observed successful negative scene and never enters a macro.
    f1 = 2*tp/denominator if denominator else (1.0 if n else None)
    return dict(pair_count=n, positive_count=positive, negative_count=negative,
        prevalence=prevalence, tp=tp, fp=fp, fn=fn, tn=tn,
        predicted_positive_count=tp+fp, predicted_negative_count=fn+tn,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/positive if positive else None, f1=f1,
        false_positive_rate=fp/negative if negative else None,
        false_discovery_rate=fp/(tp+fp) if tp+fp else None,
        roc_auc=auc, average_precision=ap,
        ap_over_prevalence=ap/prevalence if prevalence else None,
        scores=dict(positive=_quantiles(scores[labels]), negative=_quantiles(scores[~labels])))


def _validate(row, seen):
    sid, component = str(row['scene_id']), str(row['component_id'])
    if sid in seen:
        raise ValueError('duplicate diagnostic scene_id')
    seen.add(sid)
    scores = np.asarray(row['scores'], dtype=np.float64)
    labels = np.asarray(row['labels'])
    pairs = np.asarray(row['pairs'])
    sources, identities = np.asarray(row['sources']), np.asarray(row['identities'])
    intervals, node_ids = np.asarray(row['intervals'], dtype=np.float64), np.asarray(row['node_ids'])
    n = len(node_ids)
    if (scores.ndim != 1 or labels.shape != scores.shape or pairs.shape != (len(scores), 2)
            or not np.issubdtype(pairs.dtype, np.integer)):
        raise ValueError('need scores[E], binary labels[E], integer pairs[E,2]')
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('diagnostic scores must be finite and labels binary')
    if (node_ids.ndim != 1 or sources.shape != (n,) or identities.shape != (n,)
            or intervals.shape != (n, 2) or not np.isfinite(intervals).all()
            or (intervals[:, 1] < intervals[:, 0]).any()):
        raise ValueError('diagnostic node metadata or intervals differ')
    if len(pairs) and ((pairs < 0).any() or (pairs >= n).any()
                       or (pairs[:, 0] >= pairs[:, 1]).any()):
        raise ValueError('diagnostic pairs must satisfy 0 <= i < j < node count')
    # CLI verifies the complete candidate population once. This linear check
    # catches label corruption without sorting/uniquing a million-pair graph.
    truth = identities[pairs[:, 0]] == identities[pairs[:, 1]]
    if not np.array_equal(labels, truth):
        raise ValueError('diagnostic labels differ from frozen identities')
    return dict(scene_id=sid, component_id=component, scores=scores,
        labels=labels.astype(bool), pairs=pairs, sources=sources, identities=identities,
        node_ids=node_ids, intervals=intervals)


def _masks(row):
    pairs = row['pairs']
    same = row['sources'][pairs[:, 0]] == row['sources'][pairs[:, 1]]
    eligible = paper_masks(row)
    result = dict(all=np.ones(len(pairs), bool), itsa_all=same,
        itsa_eligible=eligible['itsa'], itsa_ineligible=same & ~eligible['itsa'],
        mstsa_all=~same, mstsa_eligible=eligible['mstsa'],
        mstsa_ineligible=~same & ~eligible['mstsa'])
    if (not np.array_equal(result['itsa_eligible'] | result['itsa_ineligible'], same)
            or np.any(result['itsa_eligible'] & result['itsa_ineligible'])
            or not np.array_equal(result['mstsa_eligible'] | result['mstsa_ineligible'], ~same)
            or np.any(result['mstsa_eligible'] & result['mstsa_ineligible'])):
        raise ValueError('paper eligibility masks fail candidate partition')
    return result


def _size(n):
    return 'lt32' if n < 32 else '32_to128' if n < 129 else 'ge129'


def _pooled(rows, masks, threshold):
    result = {}
    for scope in SCOPES:
        scores = [row['scores'][mask[scope]] for row, mask in zip(rows, masks)]
        labels = [row['labels'][mask[scope]] for row, mask in zip(rows, masks)]
        result[scope] = _metrics(np.concatenate(scores) if scores else np.empty(0),
            np.concatenate(labels) if labels else np.empty(0, bool), threshold)
    return result


def _macro(units):
    result = {}
    for scope in SCOPES:
        measures = [unit['scopes'][scope] for unit in units]
        result[scope] = dict(unit_count=len(units),
            nonempty_unit_count=sum(row['pair_count'] > 0 for row in measures),
            two_class_unit_count=sum(row['positive_count'] > 0 and row['negative_count'] > 0
                                     for row in measures),
            positive_unit_count=sum(row['positive_count'] > 0 for row in measures),
            **{key: dict(mean=float(np.mean(values)) if values else None,
                         defined_unit_count=len(values))
               for key in MACRO_FIELDS
               for values in [[row[key] for row in measures if row[key] is not None]]})
    return result


def _examples(row, kind, threshold):
    labels, scores = row['labels'], row['scores']
    predicted = np.zeros(len(scores), bool) if threshold is None else scores >= threshold
    chosen = np.flatnonzero((~labels & predicted) if kind == 'false_positive' else (labels & ~predicted))
    # Highest-scoring FP and lowest-scoring FN; pair index breaks all score ties.
    order = np.lexsort((chosen, -scores[chosen] if kind == 'false_positive' else scores[chosen]))
    eligible = paper_masks(row)
    output = []
    for index in chosen[order[:EXAMPLE_LIMIT]]:
        a, b = row['pairs'][index]
        same = row['sources'][a] == row['sources'][b]
        task = 'itsa' if same else 'mstsa'
        output.append(dict(scene_id=row['scene_id'], component_id=row['component_id'],
            pair_index=int(index), node_indices=[int(a), int(b)],
            node_ids=[str(row['node_ids'][a]), str(row['node_ids'][b])],
            identities=[str(row['identities'][a]), str(row['identities'][b])],
            sources=[str(row['sources'][a]), str(row['sources'][b])],
            intervals=row['intervals'][[a, b]].tolist(), score=float(scores[index]),
            label=bool(labels[index]), task=task, paper_eligible=bool(eligible[task][index])))
    return output


def analyze_scores(records, threshold, check=None):
    """Summarize frozen scores without changing any decision or fitting a rule.

    Required record keys: scene_id, component_id, scores, labels, pairs, sources,
    identities, intervals, node_ids. CLI binds these to the original run and
    verifies complete candidates. Return values contain JSON-native types only.
    ``check`` is an optional no-argument budget callback, invoked per scene and
    component; it must not mutate records or thresholds.
    """
    if threshold is not None and (isinstance(threshold, bool)
            or not isinstance(threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(threshold)):
        raise ValueError('frozen threshold must be finite or None')
    threshold = float(threshold) if threshold is not None else None
    rows, masks, scenes, seen = [], [], [], set()
    examples = dict(false_positive=[], false_negative=[])
    for record in records:
        if check:
            check()
        row = _validate(record, seen)
        mask = _masks(row)
        rows.append(row); masks.append(mask)
        scopes = {scope: _metrics(row['scores'][mask[scope]], row['labels'][mask[scope]], threshold)
                  for scope in SCOPES}
        scenes.append(dict(scene_id=row['scene_id'], component_id=row['component_id'],
            node_count=len(row['node_ids']), scene_size_stratum=_size(len(row['node_ids'])), scopes=scopes))
        for kind in examples:
            examples[kind].extend(_examples(row, kind, threshold))
    pooled = _pooled(rows, masks, threshold)
    components = []
    for component in sorted({row['component_id'] for row in rows}):
        if check:
            check()
        indexes = [i for i, row in enumerate(rows) if row['component_id'] == component]
        components.append(dict(component_id=component, scene_count=len(indexes),
            scene_ids=[rows[i]['scene_id'] for i in indexes],
            scopes=_pooled([rows[i] for i in indexes], [masks[i] for i in indexes], threshold)))
    strata = {}
    for name in SIZE_STRATA:
        indexes = [i for i, scene in enumerate(scenes) if scene['scene_size_stratum'] == name]
        strata[name] = dict(scene_count=len(indexes), scene_ids=[rows[i]['scene_id'] for i in indexes],
            scopes=_pooled([rows[i] for i in indexes], [masks[i] for i in indexes], threshold),
            scene_macro=_macro([scenes[i] for i in indexes]))
    partitions = {}
    for parent, children in (('all', ('itsa_all', 'mstsa_all')),
                             ('itsa_all', ('itsa_eligible', 'itsa_ineligible')),
                             ('mstsa_all', ('mstsa_eligible', 'mstsa_ineligible'))):
        keys = ('pair_count', 'positive_count', 'negative_count', 'tp', 'fp', 'fn', 'tn')
        counts = {key: dict(parent=pooled[parent][key], children_sum=sum(pooled[c][key] for c in children))
                  for key in keys}
        if any(value['parent'] != value['children_sum'] for value in counts.values()):
            raise ValueError('pooled scope partition counts differ')
        partitions[parent] = dict(children=list(children), counts=counts, passed=True)
    total_fp = pooled['all']['fp']
    top_scenes = sorted(scenes, key=lambda row: (-row['scopes']['all']['fp'], row['scene_id']))[:20]
    for kind, values in examples.items():
        values.sort(key=lambda row: (-row['score'] if kind == 'false_positive' else row['score'],
                                     row['scene_id'], row['pair_index']))
        examples[kind] = values[:EXAMPLE_LIMIT]
    return dict(protocol=dict(scope='075-frozen-tsa-score-diagnostics-v1', threshold=threshold,
        decision='score >= frozen threshold; None rejects all', threshold_reselected=False,
        model_forwards=0, optimizer_updates=0, grouping_changed=False,
        ranking='exact tie-group ROC-AUC and noninterpolated average precision; higher scores predict positive',
        ap_baseline='positive prevalence; AP/prevalence is descriptive, not significance',
        undefined='AUC requires both classes; AP requires positives; ratio metrics with zero denominators are null',
        f1_convention='negative-only nonempty scope with no acceptance: 1; empty scope: null',
        paper_scope='ITSA: same source, gap in [0,1200), both spans >120s; MSTSA: different source, overlap >120s',
        size_strata={'lt32': '[0,32)', '32_to128': '[32,129)', 'ge129': '[129,infinity)'},
        quantiles='NumPy linear sample quantiles; no interpolated precision-recall curve',
        warning='V_select development diagnosis; correlated scenes/identities; no pointwise independence or causal attribution'),
        scene_count=len(rows), component_count=len(components),
        node_count=sum(len(row['node_ids']) for row in rows),
        pooled=pooled, scenes=scenes, components=components,
        scene_macro=_macro(scenes), component_macro=_macro(components),
        scene_size_strata=strata, partition_checks=partitions,
        top_fp_scenes=[dict(scene_id=row['scene_id'], component_id=row['component_id'],
            node_count=row['node_count'], fp=row['scopes']['all']['fp'],
            fp_share=row['scopes']['all']['fp']/total_fp if total_fp else None,
            scopes=row['scopes']) for row in top_scenes if row['scopes']['all']['fp'] > 0],
        examples=examples)
