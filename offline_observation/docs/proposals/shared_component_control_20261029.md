# 留出共享相对分量与局部鲁棒：绝对效果的最小否证

2026-10-29，Idea1.0，执行前方案。用户授权全流程从具体问题推进方案、成熟条件并行验证；精度/真实价值优先，轻量/在线非硬门槛。本项在来源矩阵筛选之后设计，不把相对结构先认定为sensor bias，不选定最终网络。执行状态只见HandOver。

## 问题、证据和选取依据

上游[融合局部Mode C](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/fusion_action_matrix_20261029.md)完整2P×6F，消费单个SC-C1、SC-P1、SC-F1/2/3/4/5/6、SC-M1与SC-E1–6。

DE：C10三场预测关系中的相对源差可跨组留出，但只测相对残差，绝对恢复未知；3035有恶化组，1043留出仅两组。117确认同组的不同恢复规则有实际MAE/尾部取舍。AI：可预测的共享分量与当前局部扰动若被混用，可能误校正或过度拒绝；尚未确定这是当前误差主因。UN：物理bias、绝对锚点、未知时差/机动可分性、全局效果、reference生成。

完整来源包括Dualβ、GPTR、噪声辨识、ESCI，版本、公式/消融/反例、已有同域bias解法与迁移C缺口见上游。Dualβ真实组合有益而仿真不优，提醒不能强行加入参数分支；MDM的G=E和GPTR错prior反例限制共享项解释。B为“可估共享分量/局部扰动分工”，不是腿静力学或原β滤波器，Huber只作透明限影响控制。已有近邻不否定目标适配价值；本项不宣首创、定理或作者复现。

## 一次固定V32的具体干预

原115/116事前固定的32个V场景/9共享身份组件，不再按C10、117误差或尾部挑场。只读056system原Q64 fragment、114冻结WS分区、115完整reference/评价artifact和同一manifest；0ZIP/CSV/T_test/新模型/关联分数/分区/阈值/normalizer/训练。

每场只用可见输入的两个source，稳定排序确定方向高源−低源。按WS组序rank%3固定三fold；每组的共享项只从其它fold的组估计：双方有support的跨源pair-query，先每组等权平均，再对组均向量逐坐标中位数。训练组少于2则零校正并报告，不换组补齐。这里只是组/节点留出，不称真实身份独立或统计独立。

主gauge固定低源，仅高源减去相对中心；对称gauge低源加半中心、高源减半中心。gauge不由reference或效果选择。中心是C10式输入局部米制近似，球面回映仅实现相对修正，不提供绝对可辨保证。所有输入、组、支撑和query保留，先锁定全部预测，再读identity/reference评价。

八臂：WS_MEAN、WS_HUBER、WS_FIXED_MEAN、WS_FIXED_HUBER、WS_SYMMETRIC_MEAN、WS_SYMMETRIC_HUBER、WS_ANCHOR、WS_NEAREST_SUPPORTED。主2×2是共享校正有无×局部限影响有无；两对称臂是gauge敏感性，Anchor/NEAR是强控制。原WS_MEAN及普通强控制exact重放对应历史输入/输出；不能把只胜Mean称整体新方法胜出。

Huber-distance重加权：δ=1000m、固定5轮，以当前球面位置距离得到min(1,δ/max(r,ε))权重，按同一物理均值规则更新位置；不声称是精确球面Huber最优解。δ与轮数事前固定，不在V调整。只有位置改变；速度/航向保留同输入普通Mean。无support按原首anchor回退，单成员复制。具体行为与核验见新增规格；几何映射与鲁棒权重不可混称理论保证。

## 指标、条件和出口

主估计量是各干预在固定WS共同有效位置点的径向MAE差；同时报合并MSE/RMSE、P95/P99、完整目标/缺参考/组数/重复/未匹配、覆盖、同首代表、固定支撑分层、逐场/组件及成本。共享项作用率、中心不足回退、留出相对残差与绝对位置同时保留。相同R输出人口/首代表/valid/coverage应一致，变动视为实现/输出偏差停止，不当收益。

主比较保留完整2×2四条边（Huber−Mean、FIXED_MEAN−Mean、FIXED_HUBER−FIXED_MEAN、FIXED_HUBER−Huber），组合与强Anchor/NEAR及gauge比较全部报告。逐场/组件效应描述，不把query当IID，不从V挑最好gauge/臂作为最终模型。

- 相对残差及绝对误差都改善，且不是单场/gauge/弱控制解释：支持继续发展该条件下分工；仍需可信同行和新材料，不自动扩量。
- 只相对一致、绝对风险未改善：否定当前恢复配方的价值主张，回查锚定/共同误差，不叠加学习头救分。
- Huber无增量：保留共享项独立价值或负面；不把组合更复杂视为更优。
- 只某gauge有利：限定为该经验锚定输出，不宣绝对bias识别。
- 只胜Mean却输强控制、或MAE与尾部取舍：如实保留，不能宣布完整净胜。

单次118 attempt-1，原32终点，CPU≤4/RSS4GiB/AS8GiB/600s advisory。无自动重试、改δ、扩场、增seed/训练；失败保全现场。当前为离线缓存验证，不叫在线/端到端；完整坐标、身份、中心及预测留服务器，仅脱敏汇总/逐场与真实命令/退出回传。真实Record→Squeeze→Mode C后另议后继。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-SHARED-COMPONENT-CONTROL-20261029
idea_id: IDEA-SHARED-COMPONENT-CONTROL-20261029
plan_version: '1.0'
source_matrix_id: MC-FUSION-ACTION-LOCAL-20261029
selected_C_id: SC-C1
source_P_ids: [SC-P1]
source_F_ids: [SC-F1, SC-F2, SC-F3, SC-F4, SC-F5, SC-F6]
source_M_ids: [SC-M1]
source_E_ids: [SC-E1, SC-E2, SC-E3, SC-E4, SC-E5, SC-E6]
research_question: 留出预测组可估的相对共享分量，与局部限影响分工能否改善绝对恢复
hypothesis_and_mechanism: 本文问题/矩阵；相对结构DE，绝对价值与物理原因UN
testable_predictions: 留出相对残差与绝对MAE/尾部共同改善；分工增量须超过各单项/强控制
design_and_controls: 固定WS与原V32，2x2四臂/两gauge敏感性/Anchor及NEAR，共八臂
sample_materials_or_experimental_units: 原115/116 V32/9组件，056fragment/114WS/115truth及同manifest；开发材料
primary_outcome_and_estimand: 固定完整输出同共同有效点径向MAE差及四条2x2比较
secondary_outcomes: 全部RMSE/P95/P99/人口/coverage/duplicate/条件/逐场组件/成本
analysis_plan_and_uncertainty: 同场确定性配对，组件异质性，非IID/非独立测试，无V择gauge
resources_time_cost_and_risk: 一次CPU4/RSS4GiB/AS8GiB/600s advisory/32终点/0训练
ethics_safety_privacy_or_compliance: 私有坐标身份权重留服务器；T_test封存，reference仅评价
stopping_and_interpretation_rules: 本文出口；失败保全，无自动重试扩量或替换
status: exploratory-minimal-validation
evidence_status: partial
evidence_locations: 上游局部矩阵及C10/117真实Record
missing_evidence: [absolute_benefit, physical_bias_identification, absolute_anchor, matched_credible_peer, causal_input]
next_skill: record-experiment_after_actual_execution
```
