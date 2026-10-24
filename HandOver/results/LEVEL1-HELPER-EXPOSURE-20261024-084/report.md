# 084 helper exposure audit

Status: COMPLETED

Read-only accounting of completed successful updates. This is not a model result or a causal diagnosis.

Buckets: zero / one / two-or-more supported helpers at each query; support is an interval, not an exact observation.

## Training seed 20261023

4000 successful updates; 32000 samples; 4 zero-supervision samples retained in the batch denominator. 2100/2100 legal scenes observed.

| Weighting/population | Zero / one / two-or-more |
|---|---|
| All query point exposure | 32.4743% / 45.0862% / 22.4395% |
| Valid query point exposure | 19.3555% / 51.3596% / 29.2848% |
| Actual sample-equal loss coefficients | 19.7983% / 51.4609% / 28.7283% |
| Scene-equal mean of sample coefficients | 19.7934% / 51.4501% / 28.7433% |
| Unique scene-anchor-mode sample coefficients | 19.7796% / 51.4730% / 28.7337% |

Actual coefficient mass = 0.99987500; a value below one reflects zero-supervision samples, not missing log entries.
29029 distinct scene-anchor-mode combinations; 2971 repeated exposures. No deduplication across modes.

| Corruption mode | Actual samples | Valid points | Sample-equal coefficients |
|---|---:|---:|---|
| clean | 10672 | 520861 | 11.9801% / 63.3736% / 24.6369% |
| clean_plus_wrong | 10664 | 525975 | 5.1358% / 33.3039% / 61.5509% |
| wrong_only | 10664 | 519760 | 42.2849% / 57.6964% / 0.0000% |

## Training seed 20261021

4000 successful updates; 32000 samples; 6 zero-supervision samples retained in the batch denominator. 2100/2100 legal scenes observed.

| Weighting/population | Zero / one / two-or-more |
|---|---|
| All query point exposure | 32.6173% / 44.9917% / 22.3910% |
| Valid query point exposure | 19.5030% / 51.1885% / 29.3085% |
| Actual sample-equal loss coefficients | 19.7659% / 51.2835% / 28.9318% |
| Scene-equal mean of sample coefficients | 19.7733% / 51.3257% / 28.8813% |
| Unique scene-anchor-mode sample coefficients | 19.7982% / 51.3086% / 28.8760% |

Actual coefficient mass = 0.99981250; a value below one reflects zero-supervision samples, not missing log entries.
29164 distinct scene-anchor-mode combinations; 2836 repeated exposures. No deduplication across modes.

| Corruption mode | Actual samples | Valid points | Sample-equal coefficients |
|---|---:|---:|---|
| clean | 10672 | 521892 | 11.5758% / 63.3362% / 25.0599% |
| clean_plus_wrong | 10664 | 519210 | 4.8426% / 33.4001% / 61.7385% |
| wrong_only | 10664 | 520692 | 42.8855% / 57.1051% / 0.0000% |

## V_select descriptions only

| Population | Query count | Zero / one / two-or-more |
|---|---:|---|
| all_output_group_queries | 598592 | 55.5270% / 30.1748% / 14.2982% |
| all_output_group_valid_position_queries | 472814 | 46.5145% / 36.4037% / 17.0818% |
| unique_selected_valid_position_queries | 442263 | 43.5250% / 38.2813% / 18.1937% |

The four saved V error-weighting metrics are retained in summary.json. They are physical-error summaries, not recoverable per-anchor training-loss coefficients.

## Interpretation boundaries

- Loss coefficients describe sample/query averaging, not gradient contribution, Adam updates or clipping effects.
- Scene-anchor-mode deduplication keeps distinct corruption modes separate; it is descriptive, not a new training objective.
- Training and validation populations differ in scenes, anchors, groups and repeated exposure; differences are not causal attribution.
- Validation aggregate counts cannot recover per-anchor loss coefficients or identity-by-support joint counts.
- Anchor/node mappings and repeated memberships are checked within saved logs; raw node rosters and per-query arrays are not read.
- No original archives, checkpoints, model forwards, training, threshold selection or T_test are accessed.
