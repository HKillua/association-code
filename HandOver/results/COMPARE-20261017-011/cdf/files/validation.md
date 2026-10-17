# CDF1 exploratory validation comparison

All rows use the same V_select scenes and candidates. Thresholds and head epochs were selected on this set; these are not held-out test results.

| Arm | Macro F1 | Micro F1 | Precision | Recall | Same-source F1 | Cross-source F1 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.426237 | 0.260579 | 0.158408 | 0.734001 | 0.452744 | 0.399644 |
| A_prime | 0.425907 | 0.260429 | 0.158354 | 0.732787 | 0.452743 | 0.398658 |
| B | 0.354768 | 0.242026 | 0.139540 | 0.911435 | 0.347105 | 0.361424 |
| EndpointMarginal | 0.353901 | 0.282499 | 0.170546 | 0.822263 | 0.345180 | 0.361518 |
| PairMLP | 0.425922 | 0.261152 | 0.158902 | 0.732484 | 0.452768 | 0.398663 |
| ScenePool | 0.377824 | 0.211771 | 0.119377 | 0.936912 | 0.372493 | 0.380837 |
| FixedRule | 0.408113 | 0.228769 | 0.132108 | 0.852593 | 0.420109 | 0.395610 |
| B_shuffle | 0.355199 | 0.266993 | 0.157980 | 0.861389 | 0.346805 | 0.362593 |

| Comparison | Macro F1 difference |
| --- | --- |
| B_minus_A0 | -0.071469 |
| B_minus_A_prime | -0.071139 |
| B_minus_EndpointMarginal | +0.000867 |
| B_minus_PairMLP | -0.071153 |
| B_minus_ScenePool | -0.023056 |
| B_minus_FixedRule | -0.053345 |
| B_minus_B_shuffle | -0.000431 |

| Baseline → CDF1 | Wrong → right | Right → wrong | Net corrections |
| --- | ---: | ---: | ---: |
| A0 → B | 3583 | 8671 | -5088 |
| A_prime → B | 3581 | 8681 | -5100 |
| EndpointMarginal → B | 413 | 5464 | -5051 |
| PairMLP → B | 3523 | 8680 | -5157 |
| ScenePool → B | 5958 | 1785 | 4173 |
| FixedRule → B | 4530 | 4399 | 131 |
| B_shuffle → B | 188 | 3416 | -3228 |

A_full and independent literature baselines (including TSA-cTFER) are not included by this entry point. They remain separate comparisons. These numbers alone do not establish novelty, a successful innovation, or superiority to TMTAFF.
