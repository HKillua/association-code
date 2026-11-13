# C142 · 对称位置锚下的同源相对运动融合

`IDEA_PACKAGE IDEA-OFFLINE-SOURCE-OFFSET-MOTION-20261114`，v1.0，候选诊断，待资格清点。来自 Mode C2.58 的 P02／P04／P14、F441–F443、M199、E555／E557–E559。 [证据矩阵](</Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-258>) · [C140 完整记录](../reports/2026-11-14-fusion-interval-smoothing-review.md)

## 问题与可检验预测

**DE。** C140 固定 MTAD 原177场、2,923个 REG_MIX 组与原生位置，等权平滑在同123,051个位置点把 MAE／P99 从1,772.182／17,985.623米降至819.088／16,275.023米，约90%的净减误来自多源支撑。1,683个真实组内缺口的 P99 反升1,175.980米，超过250米增害508点。按来源×时间箱等质量改权又使77,043点相对等权增害超过250米。同177场事后只在非精确时刻的多源支撑点使用等权，MAE／P99 为914.768／16,788.377米，四个组件两项均输全等权。该事后拼接只定位作用域。 [事后误差数组诊断](</Users/a1/Projects/biye-ronghe/docs/evidence/20261114_fusion_interval_smoothing/posthoc_multisource_exact_preservation.json>)

**AI。** 某来源的位置有近似恒定偏移时，同源连续区间内的位移仍可能提供有用的运动约束。问题是在同一对称绝对锚、同一原生输入及同一状态模型下，去掉同源区间残差的常量分量，能否保留运动信息，同时减少位置尾部和逐点增害。预期在存在可重复原生共运动的组件上同时观察到共同位置 MAE 和 P99 下降，且原非尾大增害减少。时钟偏移、真实转向、错误分组、轨迹内漂移及发布参考的共源关系都可产生反例。

**近邻。** 181／187 在真零支撑的同源双侧报文上已比较最近点、线性／圆弧桥接、源等权、端点速度和形状限制。C108 已比较双源支撑门和 Mean 回退。118 和120 用其他预测组估相对平移，C113 已用不共用括点的前后半时段互验平移，并出现正确新增同船观测仍使38个启用点全部增害的反例。C142 直接比较状态估计中同源区间绝对残差与去均值残差，使用同一预声明对称位置锚。独立区间仅确定资格。Holistic Fusion 为分开检验局部运动和全局位置提供先例，其绝对坐标约束和参考独立性条件仍待 MTAD 自身证据。ICML Wu等的非平稳对照提示谱一致可掩盖变点失真，需把锚附近、跨机动和绝对尾部单列。 [181](../reports/2026-10-31-native-history-bridge-review.md) · [187](../reports/2026-10-31-endpoint-kinematics-review.md) · [C108](../reports/2026-11-12-tmtaff-dual-position-fallback-review.md) · [120](../reports/2026-10-29-shared-component-full-review.md) · [C113](../reports/2026-11-12-association-two-anchor-time-validation-review.md) · [来源正文分析](</Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/offline_c139_c140_sources_20261114.md>)

## 输入资格与对称绝对锚

仅用 C140 的 MTAD Level1 T_core 原177场／4组件、C121 REG_MIX、canonical_all 原生位置和真实秒、原 Q64、首认领及目标对应。按原生片段把同源报告切成连续段，相邻报文间隔超过120秒时断开。同源重叠片段保留原身份，完全相同的原生行按来源、原生时间和位置归一为一个谱系键。用连续段起止时刻切分候选窗，合并相邻且活跃连续段集合相同的区间。候选窗内至少两个来源，每个参与连续段在窗内至少三个不同时刻。相邻窗共用边界时归较早窗，保证每个 Q 只归一个窗。

先按候选共同支撑窗中点把原生行划成前后两块，各块在自身行中确定共同时段。两块分别至少有20秒共同时间，同一参与来源在每块均至少有三个不同时刻。任何合法插值端点均须在各自块内，最大括距120秒，前后端点及原生行零交集。各块只清点参与来源、谱系和源间同区间位移分歧，报告候选数、拒绝原因、场及组件。多个 Q 复用同一原生区间时按组和重叠区间合并为同一资格单元。两块观测及其分歧不估计校正量，也不作为额外损失或选模分数。若覆盖集中于单一组件，或去最大组件后没有可比原生单元，停止。

每个合资格窗只用一个绝对位置锚。锚时刻取最接近共同窗中点、且所有该窗参与来源都有合法原生括点的原 Q64，平局取较早 Q。沿同一米制短弧坐标，每源先对其合法片段插值位置取均值，再对各源均值等权平均，作为固定约束 `x(t*) = a`。参与来源、括点、场景覆盖和锚时刻在看参考前冻结；无合法锚回到原 EQUAL。锚括点仍属于共同原始输入，硬锚不额外计作一条独立观测。各源编号和成员顺序仅用于分组，不设优先级，软件检查覆盖源换序、K≥3接口和锚位置不变性。

在同一米制坐标图、同一资格和括点下，若所有来源共同平移 `c`，锚与输出应同平移。只平移一个参与来源时，去均值运动残差应保持不变，锚和相对臂应平移 `c/K`，其中 K 为参与来源数。后者记录为锚的响应，不能解释为已知的真实偏差或全局校准。两源自身都可能带偏，公共平移不可识别。

## 最小对照

四臂均保留原组、原时窗、Q64、首认领、速航和参考有效掩码。采用 C140 的坐标图、分段线性状态、约60秒节点、速度差平方项、250米径向 Huber 和固定五次求解，不拟合参考尺度。两实验臂只用各资格窗内的同一批原生行，采用同一个绝对锚和相同原生行基权。作用域锁定为同一有资格窗中的多源原生支撑，组内真缺口、组窗外及其余点均沿 EQUAL。真实缺口作为回退一致性的检查。精确原生时刻单列并仍按同一资格规则计算。

|臂|位置读出|
|---|---|
|支持 Mean|重现 C140 的原始位置对照。|
|原 EQUAL|重现 C140 的等权稳健平滑。|
|同锚绝对残差|同一固定锚和候选窗，对原生行位置残差应用相同稳健损失。|
|同锚相对残差|对每个来源的连续原生段，先计算 `r_i = x(t_i) − y_i`，再以段内原生行等权均值 `mean(r)` 为中心，稳健损失只作用于 `r_i − mean(r)`。|

相对臂中的 `[x(t_i) − mean(x)] − [y_i − mean(y)]` 只携带段内运动。损失以原生行计算，组装时先固定段归属和等权中心，再进行稳健重加权。源偏移量不作为未知量输出，也不注入参考帧。窗口资格与作用域固定，两个实验臂使用相同方程维数、节点和时间标度。必要的软件核验包括两臂分别复算、平移响应、短弧经度、秩与锚约束，以及所有回退点和四属性有效人口一致。若任一臂求解不满秩或数值异常，两新臂在对应窗均回 EQUAL，报共同回退和原因。

## 估计量与停止线

主比较为同锚相对对同锚绝对，以及同锚相对对原 EQUAL。对相同可评位置点报告全域和匹配启用域的 MAE／P99。冻结支持 Mean 的全域 P99 作为旧非尾门槛，各臂分别计相对 Mean 增害超过250米的旧非尾点，使三臂的增害总量可直接比较。另对每项主比较逐点计超过250米的改善及增害、原尾救回与新尾。辅助项包括 RMSE／P95、精确时刻、真实组内缺口、窗口边界，以及同场同 Q 真值间距不超过1公里的两船有向误差和指派反转。原纬经、速航、有符号及绝对指标、覆盖、重复与未匹配均完整计分。留一组件仅从完整预测重聚合。

相对两项主对照均须降低启用域 P99、全域 MAE／P99，以及相对 Mean 的旧非尾大增害点数。各主比较的逐点大改善数须多于大增害数。至少两个组件净减误，去原最大组件后两项主比较仍须达到全域和尾部要求。已启用的近距双船有向误差或指派反转恶化时不晋级，合法近距样本为零时记录该项尚无证据。四个组件仅作描述性异质性检查，多个 Q 及相邻报文共享原生区间。

资格不足、尾部与增害未共同改善、同锚绝对解释全部收益，或改善主要来自单一组件时结束这一配方。若只看到段内运动稳定且绝对位置无净收益，保留相对运动证据并标记对称绝对锚的限制。原177场是 C140 已见开发人口，即使过门也只产生开发证据，作者 TMTAFF 同协议比较仍沿独立资格推进。

候选仅拟定一次 CPU 资格清点及后续零训练的解析对照。私有原生位置、原生谱系与目标身份留服务器，公开仅计数、误差、掩码和脱敏组件统计。两新臂、原始对照及资格在评价前固定，届时再接 C151 缓存。方案状态为 diagnostic，尚无执行合同或远程运行。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-OFFLINE-SOURCE-OFFSET-MOTION-20261114
plan_version: '1.0'
source_matrix_id: MODE-C-OFFLINE-PARTITION-FIT-INTERVAL-20261114
selected_C_id: C142
source_P_ids: [P02, P04, P14]
source_F_ids: [F441, F442, F443]
source_M_ids: [M199]
source_E_ids: [E555, E557, E558, E559]
research_question: Does source-local translation-invariant motion reduce position tail risk with a fixed symmetric absolute anchor?
hypothesis_and_mechanism: Source-local centering removes constant translation from native displacement constraints while the shared anchor fixes the gauge.
testable_predictions: Common-domain P99 and large harm improve against both matched absolute residuals and C140 EQUAL, beyond the largest component.
design_and_controls: C140 MEAN and EQUAL plus matched anchored absolute and source-local centered residuals, with identical reports and support.
sample_materials_or_experimental_units: Original MTAD T_core 177 scenes, C121 REG_MIX, nonoverlapping native block qualification and fixed Q64.
primary_outcome_and_estimand: Full and matched-active paired position MAE and P99, with a common Mean-referenced non-tail harm count and paired harm against each strong control.
secondary_outcomes: Four attributes and coverage, gains and losses, old-tail transitions, exact, true-gap, nearby-target and boundary strata.
analysis_plan_and_uncertainty: Descriptive component directions and leave-one-component checks on the same C140 development population.
resources_time_cost_and_risk: One bounded CPU qualification before any freeze, zero neural training, common solver fallbacks disclosed.
ethics_safety_privacy_or_compliance: MTAD only, sealed split closed, private rows stay on server and reference cache joins after freeze.
stopping_and_interpretation_rules: Stop for insufficient distinct components, no matched tail and harm benefit, or anchoring-only effects.
status: diagnostic
next_skill: record-experiment
```
