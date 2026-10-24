# 089 T-only physical-prior replay

Status: COMPLETED

Full and dropped refer to physical fixed.95 starting states, not trained network outputs.

| Population/weight | Full | Dropped |
|---|---:|---:|
| Repeated valid-query MAE m | 3491.132797050519 | 2990.0526527745046 |
| Repeated valid-query RMSE m | 10960.131950788205 | 9027.795190208406 |
| Original sample-equal linear loss (km) | 3.50249489718914 | 2.9581874548878457 |

32000 successful sample exposures, 4 zero-supervision exposures retained in the denominator; coefficient mass 0.9998750000005038.

The joint mode × original-support-region × original/retained helper-count table is in summary.json. Every cell uses the global sample denominator; conditional MAEs must not be added.

Per-scene source/denominator records are in scene-summary.json.gz. No new V comparison is performed.

## Limits

- Physical T starting-state changes do not apportion the final083 validation regression causally.
- Point frequencies, sample-equal loss coefficients and scene-equal descriptions have different denominators.
- Loss coefficients are averaging weights, not gradients, Adam changes or clipping contributions.
- Support means a time interval, not accurate measurements; labels only rebuild historical T corruption and truth diagnostics.
- Original initial tensor equality is inherited from the prior paired checker; no checkpoint is opened here.
- Logs retain mask counts, not complete original per-query masks: reconstructed masks are checked against counts and alignment, not an absent saved mask array.
