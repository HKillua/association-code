# 143：相对代价选源 — Record→Squeeze

2026-10-30。科学及限定独立输出审计均COMPLETED/exit0；当前固定配方不扩大。代价加权相对普通成对学习有作用，但没有超过原Qhard。

## Record：设计、输入和实际执行

- EXPERIMENT_RECORD：`LEVEL1-RELATIVE-SOURCE-DECISION-20261030-143-attempt-1`；source_idea_id `IDEA-RELATIVE-SOURCE-DECISION-20261030`、plan_version1.0。事前[Idea/spec](../proposals/relative_source_decision_20261030.md)、[L2S143](../../HandOver/L2S-20261030-143.md)。来源MC-SELECTION-READOUT-v1.2/SR-C1、F240/M108/E139及F241/M109/E140。
- 固定原T32/12组件、V32/9组件、Q64/WS组，完整位置T37635/V71071；T6/V13臂、15/78比较完整，原T4/V11控制exact，0/1源输出不变。0新神经/teacher/状态/组/T_test，无科学重试。
- code6bcdcac、checkout `association-code-run143`，主仓runs下同名143 science/launcher。UTC06:25:14.107680→06:28:15.086428（北京时间14:25→14:28）；程序180.978801932s、launcher185.181021463s、RSS1171845120B。CPU4/taskset12–15、RSS6/AS12GiB、1800s advisory未超建议。
- unit `mtad-749377ede74e.service`，真实inactive/dead success、MainPID0、Restart=no、Linger=yes。原件：[run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/143/run.json)、[完整命令/环境](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/143/launcher/request.json)、[退出与服务](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/143/observed-services-terminal.json)。

干预：在同10维/二次基、原133各fold normalizer和hard物理恢复下，用源对特征差学习共享潜势；对比普通偏好与乘真实误差差绝对值的代价偏好。65可学系数、λ1；12LOCO/fullT各两头共26个严格凸logistic fit，104个Newton更新、每头4步，最大梯度1.717e−13。它是有监督迭代凸拟合，既不是闭式ridge，也不是神经网络训练。T标签来自原133参考，未增加独立健康通道。

计划与实际一致：全T模型在V前冻结，新增评分/贡献/物理状态先存再开V reference；原输入127/125/114/133/136/140/134未改，原坐标/身份/β/大权重预测留服务器。T标签/原评分直接复用已核133/134缓存，本次审计没有独立重拟合。阶段T标签.518s、fit8.230s、T恢复评价40.205s、V129.526s，阶段外管理开销计总时间。

## 全部位置对照

单位m，越低越好。表中历史控制不是可信同行认证；LOW/FixedLOW/Qgate没有被采用为新策略。

| T完整臂 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| SOURCE_EQUAL | 1311.223 | 2878.589 | 2544.797 | 10480.821 |
| SOURCE_NEAREST | 987.325 | 2883.289 | 4116.754 | 10406.995 |
| Qsoft | 1175.045 | 2810.466 | 2434.711 | 10406.995 |
| Qhard | 619.476 | 2666.191 | 2436.283 | 10500.827 |
| PAIR_LOGIT | 650.375 | 2686.336 | 2662.515 | 10500.827 |
| REGRET_LOGIT | 646.968 | 2681.865 | 2553.321 | 10500.827 |

| V完整臂 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| MEAN | 1276.328 | 2168.266 | 3103.795 | 7194.825 |
| NEAREST_SUPPORTED | 927.435 | 2173.150 | 3873.502 | 7301.504 |
| LOW历史控制 | 527.446 | 1813.304 | 2942.040 | 7260.394 |
| FixedLOW历史控制 | 447.060 | 1740.814 | 1012.563 | 7526.008 |
| Qgate历史控制 | 435.061 | 1739.320 | 1014.478 | 7526.008 |
| SOURCE_EQUAL | 1276.441 | 2168.137 | 3097.435 | 7194.825 |
| Qsoft | 1128.755 | 2060.605 | 2990.606 | 7194.825 |
| Qhard | 547.721 | 1829.356 | 3125.980 | 7260.394 |
| LINEAR_SOFT | 1162.880 | 2085.495 | 3011.683 | 7194.825 |
| SHUFFLED_SOFT | 1276.493 | 2168.187 | 3101.807 | 7194.825 |
| SOURCE_NEAREST | 926.659 | 2172.333 | 3872.747 | 7301.504 |
| PAIR_LOGIT | 585.366 | 1865.660 | 3371.388 | 7260.394 |
| REGRET_LOGIT | 579.466 | 1856.256 | 3307.362 | 7260.394 |

## 主比较、尾部与误选

差为candidate−baseline。2000次seed20组件描述bootstrap，不是独立确认性CI；最大组件按场景数定义。好/差/平按MAE符号，不设事后epsilon。

| 分区/比较 | MAE差m [95%] | 场景好/差/平 | 组件好/差/平 | 去最大差m |
|---|---|---|---|---:|
| T REGRET−Qhard | +27.492 [22.881,38.986] | 3/26/3 | 0/11/1 | +33.821 |
| T REGRET−PAIR | −3.408 [−13.791,−1.034] | 14/13/5 | 7/2/3 | −4.695 |
| V REGRET−Qhard | +31.745 [15.215,34.167] | 1/29/2 | 1/7/1 | +18.006 |
| V REGRET−PAIR | −5.900 [−15.585,−.754] | 15/13/4 | 5/2/2 | −7.592 |

V相对Qhard：63点改善、1063点增害、69945点相同；益90402.471m、害2346554.234m。917点增害>1km，最大6541.348m。T相应53/514/37068点，497点增害>1km、最大7749.909m。全P99相同不等决策尾部安全：V多源39847点的P99由2773.957→3568.119m，P95462.997→1084.713m，RMSE588.742→723.496m。0/1源保持不变，其分布未用于掩盖多源损害。

T D2 22627点完整误选迁移：旧Qhard602错；PAIR1010错，救回83/新增491；REGRET1063错，救回53/新增514。REGRET救回原代价126311.891m，新增1160979.564m，>1km选择regret471→926。普通与代价偏好均无精确score tie；两者最终错误次数与错误代价并不按同一方向变化。

匿名反转154点：REGRET正确8→10，救回3/新增1，>1km误选90→89；局部减少2524.659m选择代价，但不抵消完整损害。PAIR只5点正确，仍有局部误差益。纯组MAE266.786→307.380m、混组590.298→663.900m，两层都更差；true purity/role仅作评价，不能作部署规则。完整原件：[T诊断](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/143/T-selection-diagnostics.json)、[T汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/143/T-summary.json)、[V汇总及全部场组件/去最大/harm](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/143/summary.json)。实际输出误差差与相对best endpoint regret分开。

## 限定独立审计

code cfcb687、checkout `association-code-audit143-outputs`、unit `mtad-5928e00eb1d4.service`，COMPLETED/exit0、inactive/dead success、Restart=no。CPU2/600s advisory，程序9.021294292s/RSS217362432B；43119209数组/计数元素、4387数值标量检查通过，不是独立样本数。

[审计原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/143/independent-output-audit.json)及同目录audit-launcher保存完整命令/退出。独立NumPy核native reference/valid、首anchor对应、重复无truth、所有臂完整主位置/覆盖、场/组件/去最大、主actual harm/bootstrap及0/1源不变。**未独立重拟26头或重审全部训练标签/评分、T选择诊断与附加支持层**；后两者为生产侧完整分析，不把审计范围扩大。物理truth/噪声、真实K≥3、外部泛化均未认证。

## Squeeze与Mode C输入

`SQ-RELATIVE-SOURCE-DECISION-20261030-143`消费本Record。DE：代价权比普通成对学习降低总代价，但当前二者都差于原log Qhard；收敛完成不等决策改善，局部角色救回未解决整体问题。AI/UN：共同误差水平、目标/正则、稀少模式、有限十维表达及错组参考的作用仍未分离，不能据一次失败断言相对学习整体无效或信息绝对不可辨。

当前固定配方不扩大、不追V调λ或加Newton步、不退固定source。后继须有新增合法时序/组信息或能区分竞争解释的依据；相似工作不取消发展空间。拟回流F243/E143、限定M108，精读104不增。T/V历史共享4组件、95.37%V点，LOCO只隔离新头；没有可信同行、真实异构K≥3、在线/端到端成绩。下一skill deep-read；无自动后继。

回流已完成：[Mode C1.61](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-161)、[SR v1.3](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md#relative-actual-feedback)登记F243/E143并限定M108。执行回执[S2L143](../../HandOver/S2L-20261030-143.md)。
