# 096 complete-system comparison

Fixed-final native-time systems; repeated V development, one seed, one-pass 088 source. No promotion.

| System | Own MAE m | Own RMSE m | Own P95 m | Own P99 m | Position coverage | Common MAE m |
|---|---:|---:|---:|---:|---:|---:|
| learned | 58587.019870 | 78610.672717 | 160927.781662 | 289946.717974 | 0.033557 | 59898.417474 |
| state_hold | 57215.079639 | 77090.577922 | 158137.557612 | 284813.430431 | 0.035678 | 59916.821889 |

Own-population errors have different availability. The main paired error comparison uses common points;
both original full-population coverage and unmatched/duplicate outputs are retained.

Ordinary controls use the same predicted groups, observed inputs and explicit masks. Their paired errors
use each within-system intersection. Oracle comparisons are diagnostics, not deployable system rankings
or a causal error decomposition. No 072C cross-schema ranking is performed by this comparator.

Exact all-region P95/P99 are recomputed from pooled valid error arrays. Scene quantiles are not averaged.

Scene wins: {"all_units": 300, "evaluable_units": 300, "learned_better": 179, "state_hold_better": 121, "tied": 0, "equal_unit_mean_delta": -17.552984457701015}
Component wins: {"all_units": 38, "evaluable_units": 38, "learned_better": 24, "state_hold_better": 14, "tied": 0, "equal_unit_mean_delta": -10.280879105724269}
