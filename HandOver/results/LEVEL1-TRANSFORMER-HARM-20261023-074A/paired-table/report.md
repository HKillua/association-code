# 074 相对初值退化惩罚：同seed固定终点

| 方法 | MAE m | RMSE m | P95 m | P99 m |
|---|---:|---:|---:|---:|
| anchor | 885.5788 | 6307.4864 | 3946.7038 | 11661.2586 |
| mean_supported | 1224.6043 | 3470.0738 | 3081.8287 | 8188.8643 |
| attribute_hybrid | 885.5788 | 6307.4864 | 3946.7038 | 11661.2586 |
| support_switch | 729.9808 | 3332.5364 | 2859.8532 | 8138.7656 |
| fixed_prior095 | 721.1745 | 3326.2427 | 2774.3789 | 8060.9263 |
| linear | 590.7436 | 3289.2144 | 1966.4010 | 7825.6849 |
| harm | 607.4753 | 3274.7779 | 2102.1872 | 7896.3043 |

数值方向筛查：{'direction_screen_passed': False, 'deltas': {'mae': 16.731673742947237, 'rmse': -14.436512273535755, 'p95': 135.78619911948203, 'p99': 70.61938893833576, 'cumulative_harm_m': -9189358.963460356}, 'numerical_guard_m': 1e-06, 'cumulative_guard_m': 0.44226299999999996, 'interpretation': 'Both seeds must pass; this numerical screen does not establish a practically meaningful or statistically independent gain.'}
两seed必须全部报告。H/I、100/1000/10000m额外伤害、逐场/组件与高SSE既定场景见JSON。
