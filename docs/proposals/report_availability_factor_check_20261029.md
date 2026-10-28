# C52-Pa：报告资格条件的数值核验

2026-10-29，plan1.0；来源[Mode C1.28 §33](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-128)、[C52源公式/解析](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/c52_stepwise_problem_matrix_20261029.md)。用户本轮明确允许四问题并行小范围验证、初步成功后服务器继续；并行是本轮范围，不把不同连接混成一个成功结论。

**问题**：在确知一个来源无报告资格、且该信息不依赖目标状态的条件下，把它仍乘作每簇`1−pD`，是否改变相同两份输入报告的合/拆相对评分？**只核运算/适配条件，非MTAD实证或Wolf作者复现。** 当前056/107没有此miss乘积，不能用本结果解释它们的FP/极值。MTAD availability/FOV生成规则未知，不以无轨迹反推不可报告。

Wolf Eq4–7及作者Likelihood.py63–77/162–193提供逐簇检测/缺报告乘积，代码还可能按传感器数重估pD。本核固定p1=p2=.5及空间likelihood合/拆比.2；比较解释H1={两个异源报告同簇}与H2={分别单簇}。这是条件likelihood的确定性数值计算，不采样模拟数据、没有真值/训练。额外来源3没有报告，对p3=.1/.25/.5/.75/.9逐一计算：

- A：只有两个合法来源；
- A-bad：来源3已知不可报告，却被误计为每簇miss；
- A′：按资格条件排除该无信息来源的miss因子；
- 阴性/反对照：来源3确实可报告，没报告是漏检，则保留miss才是正确模型；不得把它也“修正”为1。

预测：A的H1/H2=.8；A-bad为`.8/(1−p3)`，p3=.5时1.6翻转；A′恢复.8。可报告反对照必须保留`.8/(1−p3)`。unknown资格没有足够模型，不据本核强设unity。固定pD与空间项，分离num_sensors自动重估及几何改变。以逐簇log因子独立计算，再与解析比值核1e−12相对/绝对容差；全5种p3和所有对照保留，不能挑翻转一例称普遍有害。

最小资源：本地一次标准库CPU、约数十标量运算、0数据/权重/GPU/模型前向；新输出目录，约秒级，不重试覆盖。输出所有log比/倍率/解释、真实开始结束/退出与代码定位；非有限或解析不一致则失败保全。正常终点全部5组完成。不以容差通过认证科研效果。

初步数值一致只确认该条件适配有必要，下一步须找到真实可报告语义及真的使用miss模型的系统，才能测身份/数量/物理风险。当前MTAD不满足这项目标条件，因此**不自动上服务器扫描数据或训练**。负结果查推导/实现，不能改pD/空间比到翻转为止。成熟模型已有报告条件不削弱其价值；本轮价值是防止错适配，并明确值得发展的事件接口。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-REPORT-AVAILABILITY-FACTOR-20261029
package_id: IDEA-REPORT-AVAILABILITY-FACTOR-20261029
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.28
selected_C_id: C52
source_P_ids: [P13]
source_F_ids: [C52L-F7, C52L-F8]
source_M_ids: [C52L-Ma]
source_E_ids: [C52L-E5]
research_question: '已知不可报告但无状态信息的来源被误计为每簇miss，是否偏置合拆相对评分'
hypothesis_and_mechanism: '逐簇额外(1-p3)导致K簇分数乘(1-p3)^K；条件适配去掉它'
testable_predictions: '固定空间比.2和p1=p2=.5，A=.8、bad=.8/(1-p3)、适配=.8；可报告反对照保留miss'
design_and_controls: '固定两报告/两指定假设/空间与pD；5个事前p3；正常、错误资格、条件适配及可报告反对照'
sample_materials_or_experimental_units: '5组确定性公式输入，0MTAD/模拟采样/作者原运行'
primary_outcome_and_estimand: '合拆log相对评分及相对于原比值的倍率；不是身份正确率'
secondary_outcomes: '翻转位置、对照一致性、耗时/退出'
analysis_plan_and_uncertainty: '全部精确条件数值与解析比较1e-12；无抽样CI，无跨数据泛化'
resources_time_cost_and_risk: '一次标准库CPU，秒级，0GPU/数据/模型/训练，新目录保全'
ethics_safety_privacy_or_compliance: '不读取数据/身份/reference/test，不改服务器任务'
stopping_and_interpretation_rules: '全部5组完成即终点；数值不符失败保全；通过不认证MTAD效果，不自动扩范围'
status: diagnostic
next_skill: record-experiment_after_actual_execution
evidence_status: partial
missing_evidence: [MTAD_report_availability_semantics, actual_miss_model_target, target_effect]
```
