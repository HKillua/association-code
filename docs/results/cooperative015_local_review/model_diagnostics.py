"""015 head-scale and gate diagnostics; saved scores plus untrained CPU heads only.

No archive, checkpoint, training update, inference on actual MTAD inputs, GPU,
or test split is used. Synthetic time features isolate scaling, not actual weights.
"""
from pathlib import Path
import gzip
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from association.models.cooperative import CooperativeModel, CooperativeConfig

SOURCE = ROOT / 'HandOver/results/COOPERATIVE-COST-20261018-015/files'


def saturation(scores):
    values = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
    probabilities = values.sigmoid()
    weights = .001 + .999 * probabilities
    floor = torch.tensor(.001, dtype=weights.dtype)
    return {
        'count': len(values),
        'logit_quantiles': dict(zip(['min', 'p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99', 'max'],
            map(float, np.quantile(values.numpy(), [0, .01, .05, .25, .5, .75, .95, .99, 1])))),
        'logit_le_minus88': int((values <= -88).sum()),
        'logit_ge88': int((values >= 88).sum()),
        'sigmoid_exact_zero_fp32': int((probabilities == 0).sum()),
        'sigmoid_exact_one_fp32': int((probabilities == 1).sum()),
        'identity_weight_exact_floor_fp32': int((weights == floor).sum()),
        'identity_weight_exact_one_fp32': int((weights == 1).sum()),
    }


def main():
    torch.set_num_threads(1)
    evaluation = json.load(gzip.open(SOURCE / 'identity-evaluation.json.gz'))
    cache = np.load(SOURCE / 'identity-scores.npz', allow_pickle=False)
    all_scores, grouped = [], {'predicted': [], 'oracle': []}
    rows = []
    for i, scene in enumerate(evaluation['per_scene']):
        scores, pairs = cache[f'scores_{i}'], cache[f'pairs_{i}']
        all_scores.append(scores)
        lookup = {tuple(map(int, p)): float(s) for p, s in zip(pairs, scores)}
        for kind in grouped:
            for members in scene[kind + '_groups']:
                anchor_scores = [lookup[(members[0], j)] for j in members[1:]]
                grouped[kind].extend(anchor_scores)
                if anchor_scores:
                    probabilities = torch.tensor(anchor_scores, dtype=torch.float32).sigmoid()
                    weights = .001 + .999 * probabilities
                    rows.append(dict(scene_id=scene['scene_id'], grouping=kind, size=len(members),
                        helper_prior_mass=float(weights.sum()), anchor_prior_fraction=float(1/(1+weights.sum())),
                        helpers_at_floor=int((weights == torch.tensor(.001)).sum())))
    scores = np.concatenate(all_scores)
    history = json.loads((SOURCE / 'history.json').read_text())
    normalizer = dict(mean=[19.996895463358154,130.99004349262427,109.00519634850676,177.95173703514496],
                      std=[.49210509487166887,.5883019770781764,415.2732000413263,104.69449368349127])
    torch.manual_seed(20)
    model = CooperativeModel(CooperativeConfig(), normalizer).eval()
    diagnostics = []
    # Identical zero pooled embeddings; only relative time span changes. First
    # synthetic track spans [0, T], second spans [.05 T, .85 T]. Count stays 2.
    with torch.inference_mode():
        for span in (1., 10., 100., 350., 1000.):
            times = torch.tensor([[0., span], [.05*span, .85*span]])
            encoded = dict(memory=torch.zeros(2,2,128), observed=torch.ones(2,2,dtype=torch.bool),
                times=times, pooled=torch.zeros(2,128), queries=torch.linspace(0,span,64).expand(2,-1))
            pairs = torch.tensor([[0,1]])
            s = model.score_pairs(encoded,pairs)
            first,last,count=model._support(encoded)
            support=torch.stack((first,last,last-first,count.log1p()),dim=-1)
            products=support[0]*support[1]
            u=model.utility(encoded,pairs,summaries=torch.zeros(2,64,128))
            diagnostics.append(dict(span_tau=span,span_seconds=span*20,
                max_support_product=float(products.abs().max()), untrained_identity_logit=float(s[0]),
                untrained_identity_probability=float(s.sigmoid()[0]),
                untrained_utility_min=float(u.min()), untrained_utility_max=float(u.max()),
                note='Synthetic zero semantic embeddings, no model training or actual MTAD inference'))
    result = dict(
        scope='015 saved identity scores; CPU initial-head scale isolation. No actual physical states available locally.',
        threshold=evaluation['threshold'], all_pair_gates=saturation(scores),
        accepted_pair_gates=saturation(scores[scores>=evaluation['threshold']]),
        anchor_helper_gates={k:saturation(v) for k,v in grouped.items()},
        anchor_prior_summary={k:{'group_count':len(x:=[r for r in rows if r['grouping']==k]),
            'anchor_prior_fraction_quantiles':dict(zip(['min','p25','p50','p75','max'], map(float,np.quantile([r['anchor_prior_fraction'] for r in x],[0,.25,.5,.75,1])))),
            'scope':'Prior mass before learned attention logits; not measured final attention weights.'} for k in grouped},
        training_rows=[{k:r[k] for k in ('stage','routing','update','loss','gradient_norm','sampled_pairs','used_pairs','removed_observations','seconds')} | {'sampling':r['sampling']} for r in history],
        synthetic_initial_head_time_scaling=diagnostics,
        limitations=['No post-training checkpoint weights or per-attribute loss/teacher utility labels/gate predictions were returned.',
            'Actual native temporal feature ranges have not been extracted here; synthetic inputs isolate the code scaling mechanism only.',
            'Large speed MAE alone cannot distinguish softplus negative saturation from large positive speed predictions.',
            'Threshold group membership and sigmoid fusion priors are distinct operations; no calibration or threshold was changed.'])
    destination=Path(__file__).with_suffix('.json')
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('all_pair_gates','anchor_helper_gates','synthetic_initial_head_time_scaling')},indent=2))

if __name__ == '__main__':
    main()
