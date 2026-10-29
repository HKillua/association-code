# 固定轨迹状态，分离声明输出域的覆盖与风险

2026-10-29，Idea1.0，实际执行之前的设计。用户授权全流程找问题/方案并在条件成熟时并行验证；精度/真实价值优先，轻量/在线为软偏好。执行只看HandOver，不从本页旧命令重复领取。

## 从问题到干预

消费[关联/输出局部矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/association_output_action_matrix_20261029.md)中的单个OA-C2：OA-P2、OA-F3/4/5/8、OA-M2、OA-E1/3/7/8。目标DE是114/117的关系改善与全输出覆盖/重复取舍；当前每组全Q64声明位置。AI：没有区分报告域、可预测域与物理存在，可能混合“位置错”与“该时刻没有足够证据声明位置”；UN：是否真是主要风险来源、合法存在语义及怎样自动学出域。

MeanTraj Eq7/15/16将轨迹存在时间与状态质心耦合，TPMBM将存在/死亡/状态分开；二者均有成熟先例及条件/反例，本项不宣领域首创。B候选是域—状态联合推断；本次只建立A′的透明输出域对照，不称完整MeanTraj或学习B已成功。未知availability/FoV不能由support伪造，support也不等真实物理存在。来源、原文回核、迁移C未知及条件见上游。

## 原V32、六臂、完整分母

只用原115/116事前V32/9组件，056物理support、114 A0/WS分区、116保存的A0/WS NEAREST_SUPPORTED完整预测及117已有对应逐场结果。116预测与117相同场景输出的绑定须核，不重新训练、前向、恢复位置、构组或选阈值。原组数、组成员、顺序、首代表和每个位置值exact保持。

每个R=A0/WS的NEAREST_SUPPORTED比较3个位置输出域，共6臂：

- FULL：原全Q64位置有效声明。
- UNION：仅组内任一成员support为真的query声明位置。
- SPAN：该组最早至最晚有support的query区间声明位置，保留中间缺口；从未有support则位置全false。

位置两维mask同步变化；速度/航向mask与状态值保持原FULL。域函数只接受support与groups，不接身份、reference或target有效mask。全部域在评价前冻结；每组都保留，不靠reference删组，不宣去掉duplicate组。当前support来自全窗插值，分组亦含未来，所以六臂都是离线。

主结果为固定全部参考有效target点下的coverage、保留/丢失点、共同点误差和丢失点原FULL风险。每臂自身MAE/RMSE/P95/P99、完整无参考/未匹配/重复/无预测、固定支撑分层、逐场/组件与实际成本并报。缩域后的自身MAE不是同人口改善：共同保留点误差应exact相同，gained应为0，target/ref有效分母不变；不符合则为实现/对齐故障，停止。

FULL必须重放原保存artifact和已有全指标绑定；所有臂raw/decoded关系不变，引用原114结果，不再发布关联收益。拒绝输出的点仍在参考人口，报告其FULL原误差与丢失比例，不把它们从评价分母删除。

## 解释与终点

- 自身MAE下降但coverage显著损失：记录覆盖—风险取舍，不称完整精度提升。
- SPAN比UNION保留缺口而风险代价有限，且跨场/组件成立：支持后续可预测域/联合存在方案，仍不把SPAN当真实存在。
- 所有域均无有意义前沿，或优势只在一场/一个组件：当前透明域配方不晋级，保留条件关联与集合估计路线。
- FULL不符、域读取reference、组/状态/参考分母改变、非有限或资源失败：保全现场并结束，不换配方/删场/重试。

119 attempt-1，一次固定32场，CPU≤4/RSS4GiB/AS8GiB/600s advisory，0GPU/训练/新预测恢复/score/normalizer/ZIP/CSV/test。新run/checkout与历史目录隔离；坐标、身份、大预测留服务器；真实命令、退出、人口、逐场及完整六臂汇总回传。没有自动扩量、拟合域阈值、存在网络训练或后继。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-OA-OUTPUT-DOMAIN-20261029
idea_id: IDEA-OA-OUTPUT-DOMAIN-20261029
plan_version: '1.0'
source_matrix_id: MC-ASSOCIATION-OUTPUT-ACTION-20261029
selected_C_id: OA-C2
source_P_ids: [OA-P2]
source_F_ids: [OA-F3, OA-F4, OA-F5, OA-F8]
source_M_ids: [OA-M2]
source_E_ids: [OA-E1, OA-E3, OA-E7, OA-E8]
research_question: 固定身份与位置状态时，全域/报告并集/首末区间的声明如何改变完整覆盖与风险
hypothesis_and_mechanism: 全域声明混合报告域与可预测域；作用与真实存在语义待验，不能只看缩域MAE
testable_predictions: 共同点误差exact不变/参考分母不变/gained0；丢失点风险与coverage分开
design_and_controls: 原V32，A0/WS的NEAREST_SUPPORTED各FULL/UNION/SPAN共六臂；全部组和状态保持
sample_materials_or_experimental_units: 原115/116固定V32/9组件；056support/114partition/116保存prediction及117同场结果
primary_outcome_and_estimand: 完整target参考人口的coverage/丢失点及其原FULL误差，共同点配对误差
secondary_outcomes: 自身MAE/RMSE/P95/P99/全人口/重复/缺参考/场景组件/支撑层/成本
analysis_plan_and_uncertainty: 开发材料确定性描述；非IID显著性/独立测试，不V选最优域
resources_time_cost_and_risk: 一次CPU4/RSS4GiB/AS8GiB/600s advisory/32终点/0训练
ethics_safety_privacy_or_compliance: reference仅绑定与评价，prediction/domain无ID或ref，private留服务器，测试封存
stopping_and_interpretation_rules: 本文三个结果分支与故障保全；无自动重试/扩量/换阈值
status: exploratory-minimal-validation
evidence_status: partial
evidence_locations: 上游完整局部矩阵及114/117真实Record
missing_evidence: [domain_risk_coverage_effect, physical_existence_semantics, credible_peer, causal_input]
next_skill: record-experiment_after_actual_execution
```
