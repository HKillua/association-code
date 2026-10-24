# 083 dropout vs original linear Transformer

| Method | MAE m | RMSE m | P95 m | P99 m |
|---|---:|---:|---:|---:|
| anchor | 885.5788 | 6307.4864 | 3946.7038 | 11661.2586 |
| mean_supported | 1224.6043 | 3470.0738 | 3081.8287 | 8188.8643 |
| attribute_hybrid | 885.5788 | 6307.4864 | 3946.7038 | 11661.2586 |
| support_switch | 729.9808 | 3332.5364 | 2859.8532 | 8138.7656 |
| fixed_prior095 | 721.1745 | 3326.2427 | 2774.3789 | 8060.9263 |
| linear | 590.7436 | 3289.2144 | 1966.4010 | 7825.6849 |
| dropout | 651.2099 | 3334.0951 | 2165.9167 | 7903.6213 |

{'worth_matched_replication': False, 'relative_mae_reduction': -0.10235624839290436, 'required_mae_reduction': 0.03, 'deltas': {'mae': 60.466296946756415, 'rmse': 44.88062573820844, 'p95': 199.5156732184439, 'p99': 77.9363628014371}, 'nonpoint_mae_nonworse': False, 'outside_largest_component_nonworse': False, 'improved_scenes': 40, 'total_scenes': 300, 'improved_components': 6, 'total_components': 38, 'numerical_guard_m': 1e-06, 'interpretation': 'Predeclared resource screen, not statistical significance or method promotion. Failed recipe closes; no probability sweep or automatic repeat.'}

Full groups/masks, support/helper strata, H/I and four weightings retained. No automatic replication or method promotion.
