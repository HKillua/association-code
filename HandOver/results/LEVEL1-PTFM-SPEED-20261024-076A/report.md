# 076A：冻结074C速度诊断

仅原缓存读数；零速为预先固定诊断控制，未训练／前向／选择阈值。完整人口、代表与mask不变。

场景 300，速度共同有效点 442476；单位保持原生，不能直接解释成节或米/秒。

| 方法 | MAE | RMSE | P95 | P99 | Pearson | 目标内中心化RMSE | 相邻增量MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor | 62.47326 | 214.4927 | 111.4238 | 145.0504 | 0.02305475 | 189.1869 | 26.68214 |
| mean_supported | 53.85101 | 195.0805 | 98.11107 | 133.875 | 0.01829784 | 179.1113 | 26.18131 |
| ptfm_adapted | 1.460905 | 2.048804 | 3.777299 | 5.686249 | 0.5671285 | 1.459039 | 0.5468464 |
| zero_speed | 3.522706 | 4.250468 | 7.202222 | 9.699762 | null | 0.9948025 | 0.1176214 |

相关系数只有非零双方方差且至少两点时定义；总体、逐目标和逐场定义数分别写入summary。
中心化RMSE使用每个目标自己的均值，仅用于事后识别动态，不生成新预测。增量只用相邻query且两端有效；速度变化率除以实际秒差。

| 真值原生速度分层 | 点数 | PTFM MAE | 零速 MAE |
|---|---:|---:|---:|
| exactly_zero | 37472 | 1.980744 | 0 |
| gt0_le1 | 40088 | 1.901481 | 0.2240646 |
| gt1_le5 | 246046 | 1.057092 | 3.209957 |
| gt5_le20 | 118787 | 1.961821 | 6.371947 |
| gt20 | 83 | 34.14691 | 36.49518 |

这些边界预先固定，仅描述原生数值；不声称对应停航或航行。

## 如何判断下一步

- 若PTFM仅接近零速控制、目标内变化及相关性很弱，则当前低MAE主要支持压制异常值／低速先验解释，不能宣称恢复了运动。
- 若在非零真值分层、目标内中心化和相邻增量上均改善，则支持真实动态恢复线索；仍不能从有无helper两个不同人口推导多源因果收益。
- 无论结果如何，本轮不拟合V常数、不屏蔽helper再前向、不自动启动训练；位置修复由独立实验验证。

## 边界

- V_select reused; shared-identity components are not independent experiments.
- Native speeds retain original units; fixed bins are descriptive, not physical stopped/moving labels.
- Helper strata have different populations and do not establish causal benefit of helper input.
- Correlation/low error is insufficient to prove physical dynamics; all constants use only fixed zero, never fitted V statistics.
- Within-target centering uses truth only for diagnosis, never model input; adjacent differences never bridge invalid queries.
- Original measured observations and training-target statistics are unavailable in this cache; no T mean/median control is fabricated.

完整可复算聚合见summary.json.gz，逐场见scene-summary.json.gz；原始074C缓存保留服务器不回传。
