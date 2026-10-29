# C53：验证透明query控制的全人口效果，并分离支持筛选

2026-10-29；`IDEA-QUERY-CONTRIBUTION-FULL-CONTROL-20261029`，plan1.0。用户本轮明确要求继续严格流程推进有效点、候选和模块拆分，形成独立117；不是116的自动后继。消费[Mode C1.35](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-135)单个C53/P14/P11/P02，F214/M85/E112；116 Record→Squeeze已完成。沿discover-innovation现象驱动与系统审查设计。

## 问题、来源和可否证预测

116固定原V32显示NEAREST_SUPPORTED相对普通均值在两R下MAE均32场改善，但对首ANCHOR仍有MAE/尾部取舍；WS来源平衡几乎没有增量。NEAR与历史全成员nearest在支持query的选择仅84次不同，尚未测该差异的物理影响，不能把收益归支持门控。原V32/开发集选择、偏差/机动/错误组及reference仍是竞争解释。

来源复用[116完整Record/Squeeze](../reports/2026-10-29-query-contribution-review.md)、[39–40节矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-135)、MeanTraj完整AP Eq7/15–16/存在域条件、ASF完整AP的共同query/source接口及消融反例；原文完整来源已核，本项不新增深读数。它们提供R、W与输出分解的机制背景，不证明本项nearest规则已优于可信同行。

假设AI：透明时间线索的收益可能在新增268场保留；支持筛选可能有额外作用，也可能只是与已有全成员nearest几乎等效。强anchor和尾部必须同时看，关系与W分别归因。若新增人口无收益或尾部代价扩大，如实限制当前配方；不调门槛/删场/加网络。

迁移归因：A=A0+MEAN，A′=WS+MEAN为已测静态聚合适配；B_W为固定R下透明query控制。新增全成员nearest保持同信息、同R、同输出，分离最近选择和支持门控；不是ASF/Gelato或学习质量的完整迁移效果。精确独立跨域C仍缺，不无限检索、不以缺C否决。新颖性判断限当前可归因控制和后续适配空间，已有nearest/均值成熟先例不关闭有实质价值的发展。

## 固定八臂、人口和复用规则

R=A0/WS；每R四W为`MEAN`、`ANCHOR`、`NEAREST_ALL`、`NEAREST_SUPPORTED`。MEAN与ANCHOR、NEAREST_ALL直接保留原ordinary_fusion生产计算；NEAREST_ALL可选无support但nearest_seconds更小的成员，NEAREST_SUPPORTED仅在support内选、空集首anchor回退，tie按原index。所有规则输出原partition每组一次，预测不接identity/reference。

人口固定原Level1 V300，原T_test封存；不按成绩选场或新增seed。分开原32、新268、全300及原38个共享identity组件，不称268独立测试（其关联/其他分析已用于开发）。0训练/模型/新score/threshold/normalizer/partition/ZIP/CSV。

- 原116固定V32的六个已有场景臂（两个R各MEAN/ANCHOR/NEAREST_SUPPORTED）**直接读取原116 private artifact和public报告，不重算这些预测**。本次仅新算两R的NEAREST_ALL。原116状态/场景/输入/版本/分区与报告绑定，六臂数组保持原件。
- 其余268场：只读同056片段物理Q64插值/OLS缓存（不是原生CSV点）、原114两R，八臂实际重算。A0_MEAN与原056完整artifact逐值exact，完整registry/target/valid对应每片段并和同源system绑定。原115/116不冒充拥有这268场的新组结果。
- 完整300再汇总；已有192场景臂复用，新算2208场景臂（32×2＋268×8），模型前向仍为0。重聚合/共同集计算不叫重跑模型。源输入/旧run只读，新117拒绝覆盖。
- 同R四W的组、首代表、全人口、valid/覆盖/重复exact一致；差异只在状态。score/decoded关系由原114 membership保持，不新造raw关联成绩。

输入包含cached_reference_read=true，只用于绑定和评价。原身份/坐标/权重/全artifact留服务器private，公开回传计数/误差/匿名场景组件。单独软件合成数据只核实现，不能当MTAD效果。

## 指标、预先对照和出口

主比较固定：WS_NEAREST_SUPPORTED−WS_MEAN、WS_NEAREST_SUPPORTED−WS_ANCHOR；机制比较WS_NEAREST_SUPPORTED−WS_NEAREST_ALL。A0同样控制；相同W的WS−A0分开关系效应。全部16个事前比较全报（旧型14个加两R的NEAR−ALL），不按赢家选择规则/人口。

原32、新268、全300/38组件：各自与逐对共同有效位置MAE/RMSE/P95/P99、scene macro、完整目标/组/重复/未匹配/point及trajectory coverage、prediction_without_target、gained/lost、固定support/远距及same/changed anchor。汇总按原point池计算，不平均RMSE/分位数。论文lat/lon有符号ME、坐标MAE、标准residual_SD和印式Eq29诊断单列，未知作者ESD实现不伪称已知；不解释旧speed单位，不把mME当米MAE。

NEAR−ALL还包含无support时的回退差异，不能把整差归支持内筛选。推理输入先固定三类group-query：有support且两法选不同成员、全无support且ALL选非首anchor、其余同选择。公开三类计数；评价时沿固定首组对应映射到目标共同点，在NEAR−ALL中分列三类误差/尾部/分母。它们是实际预测组的支撑，不混为真目标support并集；missing target保持全分母，不用该mask选输出。

原32只是解释/一致性，新增268为本轮主要效果估计人口；全300用于完整开发人口及稳定性。保留所有坏场景/组件/null，报告去最大组件的描述性稳健性（不删原主表）。不以pair/query作IID CI，38组件不是38独立训练重复，不宣未见身份泛化或论文原表复现。

- NEAR比MEAN在新268和全300改善但不胜ANCHOR：保留透明控制价值与loss取舍，后续质量/有界借用需胜强简单控制，不能只报相对弱均值涨幅。
- NEAR和ALL几乎同效：收益限最近选择整体，不能归新支持门控；保留发生不同选择的场景/群体的代价，不以次数小推出无效。
- 新268不支持、尾部/遗漏代价更大：关闭当前配方的自动后续；不改规则追分，不否定整类R/W或其他候选。
- 即使全人口改善，仍是内部透明控制，不冒称胜可信同行；可信baseline同任务比较需另外真实完成。

## 实现、资源及停止

使用新增预测/组输出接口隔离R/W/恢复/输出包装；旧ordinary_fusion、原116默认与所有冻结run数值不改。仅对确有需要的指标聚合增加可选方法名参数，不搭建插件框架。

一次117、CPU≤4/RSS4GiB/AS8GiB、1200s advisory，固定300终点，0GPU/模型/前向/训练。逐场处理/记录进度，完整错误和峰值保留。缺源输入、artifact/原32绑定不符、A0 exact失败、非法状态/分区或资源失败：保全FAILED及部分结果，结束，不缩规模/换配方/放宽容差自动重试。systemd Linger=yes/Restart=no托管，不SSH值守、不建新自动跟进。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-QUERY-CONTRIBUTION-FULL-CONTROL-20261029
idea_id: IDEA-QUERY-CONTRIBUTION-FULL-CONTROL-20261029
created_at: '2026-10-29'
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.35
selected_C_id: C53
source_ids: [E112, SQ-QUERY-CONTRIBUTION-116-20261029, AP-XFER-ASF-20261028, AP-MEANTRAJ-TSP2026-20261029]
research_question_and_hypothesis: '新268/全300的透明W效果及强anchor取舍，隔离support gate与nearest整体'
testable_predictions: '正文主要/机制比较与全人口不变量、事前解释出口'
design_and_controls: '两R×四W，全16pairs；原32六臂预测只复用，新268八臂重算'
sample_materials_or_experimental_units: '固定V300及原32/新268/38共享identity组件；原056/114/116只读cache'
primary_outcome_and_estimand: '新增268同WS共同点MAE/RMSE及NEAR−MEAN/ANCHOR，NEAR−ALL机制比较；全300并列'
secondary_outcomes: '尾部/宏场景/组件/去最大组件、完整coverage/重复、论文位置属性及成本'
analysis_plan_and_uncertainty: '完整确定性开发人口，所有比较/不利/null保留，无IID显著性/独立测试/同行胜负'
resources_time_cost_and_risk: '一次CPU4/RSS4GiB/AS8GiB/1200s advisory，固定300，无训练/GPU'
ethics_safety_privacy_or_compliance: 'cached_reference仅评价；私有身份/坐标/weights留服务器；T_test封存'
stopping_and_interpretation_rules: '正文出口，失败保全不重试，不自动扩量/训练/改规则'
source_transfer_id: '复用TRANSFER-QUERY-CONTRIBUTION-20261029边界；本项不是完整跨域B复现'
adaptation_vs_method_attribution: '固定R只换W；相同W跨R另计；支持门控以原all-nearest同信息控制隔离'
evidence_status: partial
missing_evidence: [new268_effect, support_gate_increment, strong_peer_same_task_comparison, quality_identifiability, reference_generation]
next_skill: record-experiment_after_actual_execution
```
