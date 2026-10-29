# 121：相对共享校正是否超出经验低源优先

2026-10-29，执行前Idea1.0。消费[融合矩阵§6](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/fusion_action_matrix_20261029.md#6-118回流后的下一层问题相对校正绝对锚定与真正多源增量)的SC-C3；用户授权成熟方案并行验证。本文件只规定一次小验证，是否领取只见HandOver。118旧运行及其方案均不改，不把本次结果倒填为118的事前分析。

## 问题、来源和竞争解释

118在同71071点上FixedMean MAE656.041/RMSE1864.654m，优于原Mean1276.328/2168.266和Anchor/NEAR；但对称gauge恶化，P99对Mean+346.763m，支撑外仍输Anchor。原Anchor只是group[0]，并非按query低source支持成员优先。因此尚未区分：校正后多源带来额外信息，或主要由经验低源锚定解释。

SC-P1/P2、SC-F7–9、SC-M3/4、SC-E7–9均在上游完整六格矩阵；来源包括118真实Record/Squeeze与代码、Homography原文AP、Dualβ参数/状态分工及ESCI条件。局部切平面下`y_s=x+b_s`在`x→x+a,b_s→b_s−a`下不变，相对差不能单独确定共同绝对零点；这不是新定理，亦不否定成熟配准的适配价值。无参考选source是合法透明控制，不等source排序已具物理可靠性。当前比较不宣可信同行胜负或学习融合贡献。

## 输入、干预与公平对照

- 固定原118的V32/9组件和WS节点分组/首代表/全部Q64/输出valid；输入为056system原fragment、114保存WS membership、118完整八臂artifact及保存`fold_center_ne_m/source_order/group_fold`，同一manifest；不访问ZIP/CSV/T_test。
- 原118八臂的**保存预测直接导入**，不重新拟合中心、不运行Huber或重算旧预测。可从保存group_states重建评价artifact，须与原全部artifact字段exact绑定；这只是核绑定和统一评价，不是旧预测重跑。
- 新`WS_LOW_SOURCE_SUPPORTED`：每query有低source支持成员则只平均这些成员的位置；无低源但有高源则平均高源支持成员；全无support复制原group[0]位置。低/高source使用118保存并与当前输入排序核对的顺序，不读identity/reference/目标支持来门控。单source照常平均其支持成员，无support回退。
- 新`WS_FIXED_LOW_SOURCE_SUPPORTED`：先复用118每组所属fold的冻结中心，按118完全相同`spherical_shift`仅修正高source；再走相同源优先规则。全无support复制**相同首节点的校正后位置**，与118 FixedMean保持同gauge/同回退。实施前已澄清“原anchor”指原首节点，数值跟随相应物理输入；不额外引入未校正回退的混淆配方。
- 新臂只改位置，速度/航向复制原WS_MEAN；源内位置等权，圆周经度零resultant回退相应输入首anchor对应角；单个支持成员exact复制。没有学习权重、训练、V调参或重新选择gauge。
- 十臂：原118八臂＋上述两臂；45个无序配对完整输出。主2×2为Mean/FixedMean/LOW/FixedLOW；原Anchor/NEAR与两gauge/Huber仍保留，不只挑有利比较。

## 指标、否证与成本

主估计量为固定完整共同位置人口上FixedMean−FixedLOW的径向MAE差，检验校正后高源是否在低源优先之上有额外价值；同时保留LOW−Mean、FixedMean−Mean、FixedLOW−LOW及FixedLOW−FixedMean的方向和交互。报告RMSE/P95/P99、全部人口/coverage/重复/首代表/支撑内外及场景/组件；预测组双源/仅低/仅高/空支撑query数作为作用率，真目标支撑只作事后评价，不成为选择信号。

- FixedMean优于FixedLOW且整体/尾部可接受：支持该开发条件下校正后高源有额外贡献；仍不证明物理bias或跨source语义泛化。
- LOW已解释大部增益、FixedMean未优于FixedLOW：缩小为经验锚定/源优先效果，保留有价值的简单方法，不用复杂化包装理论。
- 共享校正对Mean有益、对LOW无增量：提示作用与贡献结构有交互；分别核高源回退点/空支撑，不宣普遍校准有效。
- 尾部恶化、部分场景/组件或空支撑损失：全部保留，不按reference选择门控/混合，也不删除难例提高MAE。
- artifact、reference、人口、membership或源顺序绑定失配：失败保全，不弱化核验或自动重试。

一次121 attempt-1，原32终点，CPU≤4/RSS4GiB/AS8GiB/600s advisory，0GPU/模型/前向/训练；失败不重试、不改配方、不扩量、不建自动后继。0新scores/分组/normalizer/阈值；源码、真实坐标/身份/中心与大预测留服务器，只回传脱敏原件/全部逐场和命令/退出/成本。Record→Squeeze→Mode C后再发展后继，不因小CPU代价宣在线。

## 实现规格与验收

| 行为 | 可观察验收 |
|---|---|
| 新预测仅依赖合法物理输入及保存中心/顺序/fold | 预测函数不接受fragment/truth/identity；原始/单source/空支撑/跨日期线检查 |
| 原八臂预测不重算 | 导入保存group_states；构造新臂时不调用heldout_centers/reweighted_mean/construct_predictions；重建artifact全部exact |
| 完整人口不动 | 十臂组/首代表/valid/coverage/重复一致，45配对gain/lost=0 |
| 失败与资源保护 | 固定新输出目录；错误保留先前场景和FAILED run，禁止覆盖；4GiB/8GiB边界沿既有runner |

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-SOURCE-PREFERENCE-CONTROL-20261029
idea_id: IDEA-SOURCE-PREFERENCE-CONTROL-20261029
plan_version: '1.0'
source_matrix_id: MC-FUSION-SOURCE-PREFERENCE-20261029
selected_C_id: SC-C3
source_P_ids: [SC-P1, SC-P2]
source_F_ids: [SC-F7, SC-F8, SC-F9]
source_M_ids: [SC-M3, SC-M4]
source_E_ids: [SC-E7, SC-E8, SC-E9]
research_question: 相对共享校正后的双源贡献是否超出低source支持优先
hypothesis_and_mechanism: 同冻结校正下保留高源应较低源优先有可归因净收益；经验锚定为竞争解释
testable_predictions: FixedMean相对FixedLOW的完整人口MAE与尾部及场景组件效应；2x2交互和支持作用率
design_and_controls: 本文十臂45配对，旧八臂artifact导入，仅新增两臂
sample_materials_or_experimental_units: 原118 V32/9组件开发材料，056/114/118同绑定；非独立测试
primary_outcome_and_estimand: 固定共同人口FixedMean相对FixedLOW的径向MAE差
secondary_outcomes: 全RMSE/P95/P99/coverage/重复/支撑/逐场组件/作用率/成本
analysis_plan_and_uncertainty: 同场确定性配对；完整组件异质性；不把query当IID，不择reference臂
resources_time_cost_and_risk: 单次CPU4/RSS4GiB/AS8GiB/600s advisory/32终点，0训练
ethics_safety_privacy_or_compliance: 私有中心坐标身份留服务器；T_test封存，reference仅评价
stopping_and_interpretation_rules: 本文出口；失败保全无自动重试扩量或替换
status: diagnostic
evidence_status: partial
missing_evidence: [source_preference_attribution, lawful_absolute_anchor, matched_credible_peer, causal_input]
next_skill: record-experiment_after_actual_execution
```
