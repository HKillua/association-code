# 077 固定残差减半诊断

两seed全部报告；零训练／前向，未重开原权重验证。

| seed | method | MAE m | RMSE m | P95 m | P99 m | I m | H m |
|---|---|---:|---:|---:|---:|---:|---:|
| 20261023 | fixed_prior095 | 721.1745 | 3326.2427 | 2774.3789 | 8060.9263 | -0.0000 | 0.0000 |
| 20261023 | linear | 590.7436 | 3289.2144 | 1966.4010 | 7825.6849 | 79302569.2306 | 21617808.7166 |
| 20261023 | harm | 607.4753 | 3274.7779 | 2102.1872 | 7896.3043 | 62713410.0425 | 12428449.7531 |
| 20261023 | half_linear | 640.2470 | 3290.8921 | 2246.7246 | 7861.5342 | 45033291.9104 | 9242058.8072 |
| 20261021 | fixed_prior095 | 721.1745 | 3326.2427 | 2774.3789 | 8060.9263 | -0.0000 | 0.0000 |
| 20261021 | linear | 606.6241 | 3313.8028 | 2046.7395 | 8599.7562 | 84729692.6599 | 34068305.0842 |
| 20261021 | harm | 575.7954 | 3265.0919 | 1904.1197 | 7823.4348 | 78684317.3852 | 14388513.7466 |
| 20261021 | half_linear | 624.9927 | 3276.8892 | 2138.2480 | 7875.6858 | 57114419.7956 | 14576777.6015 |

20261023 I/H两维方向：tradeoff；half−harm ΔI=-17680118.1321 m，ΔH=-3186390.9459 m；仅数值方向，不自动晋级。
20261021 I/H两维方向：harm_dominates_two_dimensions；half−harm ΔI=-21569897.5896 m，ΔH=188263.8549 m；仅数值方向，不自动晋级。
重复组全部参与输出变换和几何统计；未人为补真值计算其误差。定义为零的向量方向报null。
One prespecified half-residual control; no V alpha search. Failure does not exclude other shrink values. Success is not a learned-risk contribution. Geometry strata use truth only after predictions; reused V and shared identities are not independent generalization.
