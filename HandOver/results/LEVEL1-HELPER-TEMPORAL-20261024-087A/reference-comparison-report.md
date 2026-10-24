#087 temporal helper comparison

| Method | MAE m | RMSE m | P95 m | P99 m |
|---|---:|---:|---:|---:|
| anchor | 885.5788 | 6307.4864 | 3946.7038 | 11661.2586 |
| mean_supported | 1224.6043 | 3470.0738 | 3081.8287 | 8188.8643 |
| attribute_hybrid | 885.5788 | 6307.4864 | 3946.7038 | 11661.2586 |
| support_switch | 729.9808 | 3332.5364 | 2859.8532 | 8138.7656 |
| fixed_prior095 | 721.1745 | 3326.2427 | 2774.3789 | 8060.9263 |
| linear | 590.7436 | 3289.2144 | 1966.4010 | 7825.6849 |
| pre_pool | 591.0375 | 3328.5083 | 1972.1610 | 7930.8367 |
| shuffled | 592.5792 | 3293.9847 | 1966.4421 | 7886.9324 |

All outputs, duplicate groups and valid target masks retained. H/I and fixed100/1000/10000m harm counts, four weightings, support/helper strata and components are in JSON.
This table contains one declared seed/profile only. No automatic promotion or retry.
