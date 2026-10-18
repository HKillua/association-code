"""Read existing 018 history only; no training, threshold search or data opening."""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'HandOver/results/COOPERATIVE-DEVELOPMENT-20261018-018/files/history.json'


def main():
    history = json.loads(SOURCE.read_text())
    result = {'source': str(SOURCE.relative_to(ROOT)), 'updates': len(history),
              'method': 'Non-overlapping 20-update arithmetic means; batches vary, not a fixed-set convergence test',
              'gradient_norm_scope': 'before clipping, not actual post-clipping norm', 'stages': {}}
    groups = [('teacher', 'plain'), ('utility', 'plain')] + [('adapt', r) for r in ['plain', 'identity', 'utility', 'learned']]
    for stage, route in groups:
        rows = [r for r in history if r['stage'] == stage and r['routing'] == route]
        assert len(rows) == 100 and [r['update'] for r in rows] == list(range(1, 101))
        def windows(values):
            return [float(np.mean(values[k:k+20])) for k in range(0, 100, 20)]
        losses = [r['loss'] for r in rows]
        gradients = [r['gradient_norm'] for r in rows]
        assert all(math.isfinite(x) for x in losses + gradients)
        result['stages'][stage + ':' + route] = {
            'count': len(rows), 'first_loss': losses[0], 'last_loss': losses[-1],
            'mean20_loss': windows(losses), 'median20_loss': [float(np.median(losses[k:k+20])) for k in range(0, 100, 20)],
            'mean20_components': {key: windows([r['loss_components'][key] for r in rows]) for key in rows[0]['loss_components']},
            'mean20_gradient_norm': windows(gradients), 'maximum_preclip_gradient_norm': max(gradients),
            'history_seconds': sum(r['seconds'] for r in rows),
            'sampled_pairs': sum(r['sampled_pairs'] for r in rows),
            'used_pairs': sum(r['used_pairs'] for r in rows),
            'unique_scene_ids_in_sampling': len({s for r in rows for s in r['sampling']['scene_ids']}),
        }
    adapt = [[r for r in history if r['stage'] == 'adapt' and r['routing'] == route] for route in ['plain', 'identity', 'utility', 'learned']]
    result['four_adapt_sampling_metadata_equal'] = all(
        [r['sampling'] for r in rows] == [r['sampling'] for r in adapt[0]] for rows in adapt[1:])
    result['four_adapt_removed_observation_counts_equal'] = all(
        [r['removed_observations'] for r in rows] == [r['removed_observations'] for r in adapt[0]] for rows in adapt[1:])
    # Metadata equality supports matching recorded sampling; it does not prove
    # equality of unreturned tensors, model outputs or sampled row identities.
    result['interpretation'] = 'Teacher physical loss decreases; identity BCE remains near ln(2). Utility training objective decreases, not proof of held-out utility usefulness. Three adapt window means fall and plain is almost flat; first/last batch rise is not evidence all four diverged.'
    (OUT / 'training_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'updates': len(history), 'sampling_metadata_equal': result['four_adapt_sampling_metadata_equal'],
                      'output': 'training_audit.json'}))


if __name__ == '__main__':
    main()
