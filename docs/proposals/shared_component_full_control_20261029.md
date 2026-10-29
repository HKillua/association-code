# 共享校正：固定配方的其余268场开发验证

2026-10-29，Idea1.0，执行前方案。来源是 Mode C1.39 的 C10/C51、P02/P11/P14、F216/M87/E114 与完整融合局部矩阵 SC-C1。118已在原V32获得主要绝对效果，但gauge、尾部和更广场景未知。本项独立编号120，原118不重开；用户本轮授权继续有效点和成熟方案验证，不新增学习训练。

## 问题与可否证预测

固定低source的组留出相对校正，在未为该配方计算过的其余268个开发场景中，是否仍改善共同人口的MAE/RMSE、超过Anchor/NEAR，并保留可接受的尾部与组件异质性？这是重复开发V上的范围扩展，不是独立测试泛化。原32含9个共享身份组件，全V有38个；其余268再按组件是否出现在原32分层，避免把已见组件误称独立泛化。

假设H1：可跨组估计的共享相对分量在固定经验锚定下仍有可用绝对效果。预测：新268及未见组件子集的FIXED_MEAN相对MEAN/ANCHOR/NEAREST_SUPPORTED主要误差下降，人口不变。竞争解释：原32选择、已见组件、source顺序与reference的关系、局部拟合污染、几何映射及支撑外误差。两源共同偏移不可由相对残差识别；不据好分数宣布物理bias或无偏锚点。

## 固定干预、来源与范围

- 原118八臂22比较全部保留：MEAN/HUBER、FIXED_MEAN/FIXED_HUBER、SYMMETRIC_MEAN/SYMMETRIC_HUBER、ANCHOR/NEAREST_SUPPORTED，前缀均WS。三fold、组等权中心、低source/symmetric gauge、最少2训练组、δ1000m/5轮均不改。不是按V挑赢家后仅扩赢家。
- 原32的全部八臂直接导入118保存artifact及共享项审计，不调用中心拟合、校正或Huber；只重建评价和聚合。其余268按同配方新计算八臂。固定056完整Q64、114 WS及原manifest；reference与普通强控制取已完成117全V产物，预测锁定后才绑定身份/reference。
- 全部300场的WS Mean/Anchor/NEAR按117血缘exact绑定：原32实体保存在116、由117引用，其余268实体保存在117；118原32全部保存artifact逐字段exact复核。原始分组、输出首代表、valid、coverage、重复均保持；不删困难点/组，不读ZIP/CSV/T_test，不重算score、不改normalizer/阈值。
- 上游完整来源、原公式/消融与跨域限制继承[118 Idea](shared_component_control_20261029.md)及其源矩阵，不把扩量另算新论文或新机制。成熟相对校正有先例并不否定适配新增价值；当前与可信同行的同任务比较尚缺。

## 主要估计量与解释出口

主要估计量：**新268固定WS共同有效位置点的径向MAE差**，FIXED_MEAN分别对MEAN、ANCHOR、NEAR。次要报所有22比较的MSE/RMSE/P95/P99、完整目标/缺参考/覆盖/重复、固定支撑分层、逐场/38组件、原32/新268/全300、新268已见组件/未见组件、去最大组件。不能把query当IID，不用显著性或样本点数制造独立重复；完整结果保留。

- 新268及未见组件主要误差好：支持扩大该有条件解法的开发范围；尾部/gauge/最坏组件仍限制价值陈述，后续发展锚定与强baseline。
- 仅原32或已见组件好：收窄可迁移范围，不能全V汇总掩盖新场景失败。
- 只胜Mean、输Anchor/NEAR：保留部分作用，但不晋级完整恢复方案。Huber无增量继续不晋级，不加步救分。
- 相对一致而绝对误差差、或只固定gauge好：分别记相对/绝对及锚定条件，不倒推物理根因。
- 任一输入/普通控制/输出人口不一致、非有限数值、OOM或程序错误：保全现场，结束本次；不修改配方自动重试。

## 行为规格、验证与预算

实现新增 `tools/analysis/shared_component_full_control.py`；旧118只扩展评价helper支持导入artifact及真实来源标签，原默认数值不变。输入参数明确为source-system、partition-run、pilot-run（118）、control-run（117）、manifest、output、cpu-threads。

| 接受条件 | 核验证据 |
|---|---|
| 原32不重算预测、导入八臂完整 | 软件检查将构造器设为禁止调用；真实run记录256导入场景臂 |
| 新268仍是原八臂 | 真实run记录2144新计算场景臂与固定参数 |
| 预测不使用reference | 复用独立物理构造器，无identity/reference参数；评价只在其后 |
| 旧118不变、117三普通控制exact | 原7项必要检查及导入分支检查，真实逐场/逐字段核验 |
| 全部层级/成本/退出可核 | summary、scene-summary、run、launcher、private原数组 |

一次120 attempt-1，0训练/前向/GPU，CPU≤4/RSS4GiB/AS8GiB，1800s advisory，固定300评价终点。原32只导入评价；不增加seed/步数/搜索、无自动扩量/重试/训练或另建跟进。复用117全V成本约7分钟、118原32约92秒，估计本次约10–20分钟，实际全部记录；非在线延迟。完整身份、坐标、中心、权重、预测保留服务器，回传必要脱敏汇总及全部逐场证据。终态后Record→Squeeze→Mode C，不预填结果。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-SHARED-COMPONENT-FULL-CONTROL-20261029
idea_id: IDEA-SHARED-COMPONENT-FULL-CONTROL-20261029
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.39
selected_C_id: C51
source_P_ids: [P02, P11, P14]
source_F_ids: [F216]
source_M_ids: [M87]
source_E_ids: [E114]
research_question: 固定118八臂在其余268开发场景和新组件的绝对效果是否延续
hypothesis_and_mechanism: 可跨组相对分量在经验低source锚定下有条件改善，绝对bias不可识别
testable_predictions: 新268及其未见组件的主要误差改善，完整输出人口不变
design_and_controls: 本文原八臂22比较及全部强控制，原32直接导入
sample_materials_or_experimental_units: 固定V300/38组件；原32与新268及组件已见性分别报告
primary_outcome_and_estimand: 新268同WS共同人口径向MAE差，固定校正对Mean/Anchor/NEAR
secondary_outcomes: 全尾部/覆盖/重复/支撑/逐场组件及成本，不把点当IID
analysis_plan_and_uncertainty: 描述性同场配对和组件异质性；非独立测试，无V择参
resources_time_cost_and_risk: 一次CPU4/RSS4GiB/AS8GiB/1800s advisory/0训练，固定终点
ethics_safety_privacy_or_compliance: 身份坐标大数组留服务器；T_test封存
stopping_and_interpretation_rules: 本文出口；失败保全，无自动重试或后继
status: exploratory-development-extension
evidence_status: partial
evidence_locations: 118真实Record/Squeeze与Mode C1.39及完整融合局部源矩阵
missing_evidence: [remaining268_effect, lawful_absolute_anchor, matched_credible_peer, independent_generalization]
next_skill: record-experiment_after_actual_execution
```
