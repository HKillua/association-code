# 144：实际坐标集合损失 — Record→Squeeze

2026-10-30。科学及限定独立输出审计均COMPLETED/exit0。条件读出的V平均收益仅.038579m、区间跨0，当前配方不扩大；不能称已经实现有效连续多源融合。

## Record：问题、设计与真实执行

- EXPERIMENT_RECORD：`LEVEL1-ACTUAL-MIXTURE-GAIN-20261030-144-attempt-1`，source_idea_id `IDEA-ACTUAL-MIXTURE-GAIN-20261030`、plan_version1.0；[事前Idea/spec](../proposals/actual_mixture_gain_20261030.md)、[L2S144](../../HandOver/L2S-20261030-144.md)。MC-SELECTION-READOUT-v1.2/SR-C2、F241/M109/E140。
- 固定原133 log评分/OOF/fullT、T32/12组件、V32/9组件、Q64/WS组。9个T实际物理loss训练alpha，13个50系数gain fit、13个全局alpha选择；T32×9=288次曲线恢复，fullT22627合法query/181016非零alpha训练候选。无V oracle/新选参、神经/teacher/新原生CV/组/T_test。
- code6bcdcac，checkout `association-code-run144`、原run/launcher独立一次；UTC06:25:14.514622→06:28:13.937408（北京时间14:25→14:28）。程序179.422835529s/launcher183.531676207s、RSS1172348928B；CPU4/taskset16–19、RSS6/AS12GiB、1800s advisory未超建议。
- unit `mtad-53f6fee6a413.service`，真实inactive/dead success、MainPID0、Restart=no、Linger=yes。[run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/144/run.json)、[完整argv/env](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/144/launcher/request.json)、[服务](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/144/observed-services-terminal.json)。阶段T曲线7.191s、fit10.385s、T评价31.258s、V130.383s，全部计成本。

A旧Qhard/Qsoft/equal；A′为相同真实T集合损失下的GLOBAL强控；B为context条件连续alpha。来源复用Chroma原p5/AppD实际集合损失机制，MeanTraj约束几何/存在域；作者核心代码未得，不称作者复现。不新增精读。MTAD的目标来自T reference监督，与源内自残差不同。

家族为任意K下w=(1−alpha)h+alpha*u；h旧Qhard质量、u可用源等权，alpha∈[0,1]，不使用source名称/编号。24维对称context不含绝对经纬度/身份/role；25维归一基拟合gain=alpha*a+alpha²*b，50系数，λ1、两个alpha截距不罚，gain0严格0。连续动作解析求最小值。有限家族不覆盖全部K单纯形。实际数据K2，软件换序/泛K不代表K≥3效果。

计划与实际一致：T6/V13臂、15/78比较、原T4/V11控制exact，0/1源不变。新模型/fullT冻结在V前，新alpha/质量/物理状态先存再读V reference；全部人口T37635/V71071、原coverage/重复保留，原坐标/身份/β/大预测留服务器。T/V共享组件和历史V开发限制沿旧协议。

## 完整位置对照

单位m，越低越好；LOW/FixedLOW/Qgate仅历史强控，无可信同行认证。

| T完整臂 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| SOURCE_EQUAL | 1311.223 | 2878.589 | 2544.797 | 10480.821 |
| SOURCE_NEAREST | 987.325 | 2883.289 | 4116.754 | 10406.995 |
| Qsoft | 1175.045 | 2810.466 | 2434.711 | 10406.995 |
| Qhard | 619.476 | 2666.191 | 2436.283 | 10500.827 |
| GLOBAL | 619.476 | 2666.191 | 2436.283 | 10500.827 |
| CONDITIONAL | 619.477 | 2666.191 | 2436.283 | 10500.827 |

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
| GLOBAL | 547.721 | 1829.356 | 3125.980 | 7260.394 |
| CONDITIONAL | 547.683 | 1829.326 | 3125.980 | 7260.394 |

## 主差、动作与反例

13fold/fullT的GLOBAL均alpha0，所有T/V输出逐点等于Qhard，三主比较中的GLOBAL−Qhard为0/区间[0,0]。CONDITIONAL−GLOBAL与−Qhard因此相同，但两个对照角色仍分别保留。

| 比较 | MAE差m [2000次组件描述95%] | 场景好/差/平 | 组件好/差/平 | 去最大差m |
|---|---|---|---|---:|
| T CONDITIONAL−Qhard | +.001916 [−.033817,.161775] | 3/3/26 | 1/2/9 | +.065284 |
| V CONDITIONAL−Qhard | −.038579 [−.048688,.046277] | 4/3/25 | 2/1/6 | +.015059 |

V164点改善/154点增害/70753平，益17888.642m、害15146.821m，最大增害242.708m、0个>1km；T19/24/37592，益3017.466m、害3089.594m，最大268.550m、0个>1km。完整P95/P99相同，但多源39847点P99稍差2773.957→2774.825m；多源MAE261.105→261.036m。去最大后反向及T均值无改善不被V小益掩盖。

**真实动作未体现连续内部混合：** T全部23603个多源组query中48个alpha1，其余alpha0；V41518中353个alpha1，其余0；两分区内部alpha均0个。接口允许连续不等真实学出连续动作，本次实际为少量hard/equal切换。组query分母不同于合法首代表多源评价T22627/V39847，未混用。

全T平均gain(km)，alpha0至1九点依次为0,.050895,.180403,.333197,.492951,.655531,.819610,.984755,1.150569；fullT拟合MSE.0876965566km²。**9点平均不证明连续区间没有细小内部收益**，也不否定134已测reference条件空间。原始fold诊断及完整尾部/作用域/所有场组件与去最大见[T-fold](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/144/T-fold-index.json)、[T汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/144/T-summary.json)、[V汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/144/summary.json)及同目录逐场gzip。

## 限定独立审计

code cfcb687、checkout `association-code-audit144-outputs`、unit `mtad-b0174235d5fb.service`，COMPLETED/exit0、inactive/dead success、Restart=no；CPU2/600s advisory，程序8.838298145s、RSS226816000B。43119334数组/计数元素和4493数值标量检查通过，不是独立样本数。

[audit](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_decision_followups/144/independent-output-audit.json)及同目录audit-launcher保存命令/退出。独立NumPy核native参考/valid、首anchor/重复、所有完整主位置/覆盖、逐场/组件/去最大、主actual harm/bootstrap和0/1源不变。**没有独立重拟13头/全局训练loss或重审标签/alpha构造**；动作分布和gain拟合为生产侧分析，附加支持层并非全部独审。物理真值、真实多传感器K≥3、外部泛化与线上效应未认证。

## Squeeze与Mode C输入

`SQ-ACTUAL-MIXTURE-GAIN-20261030-144`消费本Record。DE：真实损失监督的GLOBAL退回Qhard；条件化只改变很少query、未获得稳定有意义增量，真实动作全部在边界。它规避了140将绝对代理风险当实际输出风险的直接解释，却未自动解决贡献可学性。

AI/UN：9点训练分辨率、二次gain表达/收缩、稀少收益及旧评分家族边界仍竞争。尤其最小正训练alpha=.125、推理连续，可能漏掉窄小内部最优；没有单独干预，不认定根因，也不据V改网格/λ或提早称“没有混合空间”。后续可研究真实损失的动作分辨率/条件信息，已有成熟集成先例不取消适配价值；当前50系数配方不扩大。

拟回流F244/E144并限定M109，精读104不增，下一skill deep-read。没有新可信同行、跨传感器/真实K≥3、在线/端到端成绩；与143新头或146状态尚未联合评价，不能拼冠军。无自动后继。

回流已完成：[Mode C1.61](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-161)、[SR v1.3](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md#relative-actual-feedback)登记F244/E144并限定M109。执行回执[S2L144](../../HandOver/S2L-20261030-144.md)。
