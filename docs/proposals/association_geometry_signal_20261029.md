# 128：同候选静态几何与LOO信号的真实验证合同

2026-10-29，执行前Idea/实现规格1.0。消费[OA局部矩阵§7](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/association_output_action_matrix_20261029.md#7-2026-10-29-review后修订留出一致性先与完整pair几何对齐对象化保留拒配)的OA-C6r（P7/F15/17/18/22，M7，E15/17/18/22）及[已有设计](association_object_signal_pilot_20261029.md)§2，落实为一次原V32实际实验。执行状态只认HandOver，本文不单独提供READY。

## 问题、来源和可否证预测

两成员LOO逐query就是pair距离；指定局部欧氏多成员LOO-MSE可由全部pair距离平方和重构。原22维/score压缩不同，不等条件运动算子创造了新信息。这些是条件代数和代码事实；真实匿名多fragment里几何可辨性、有效覆盖和跨组件稳定性仍UN。Draves有限假设/谱系、MeanTraj对象集合及TPMBM未配分支的原文边界继承上游完整分析，不将本次软件检查计新精读或作者复现。

本项主模式Pattern5系统/表示审查，检验同候选上的LOO与完整pair/support静态控制，**不训练、不改分区或物理输出，不以AUROC差宣称完整系统提高**。若只有静态统计重组有价值，也可发展轻量A′；若可评覆盖很低或跨组件不稳，保留无定义人口，不能把无Q正例判负。

## 固定输入、候选与特征

- 固定056 `system/predictions/states-{sid}.npz` 的Q64 `native_states/support/sources/node_ids`（它们是预测，非原生事件），056 `association/final-scores.npz` 的全部字典序i<j pairs/scores、原阈值1.7320103112387182，114 WS原membership、manifest及其事前V32/9组件顺序。
- 原WS所有组（含singleton）保留；每个至少二成员组，按组外节点与组内全pair平均signed margin选最高两个单节点扩展，平分用原节点序；仅去除完全相同成员集合。候选不是同时输出对象，节点/事件不在这里重新解释。
- 完整静态score：sum/mean/min/max/median、negative比例、组大小/来源数/同源及异源pair数。全部内部pair逐Q球面距离和joint support保存private，可建立同信息静态组头；shared及full-Q median/P90/RMS分别报告，不能混权限。
- LOO逐成员整片段留出，仅在自己有support且其他成员有support时以其他支持成员的纬均值/经圆均值预测；单贡献成员exact复制，零圆resultant沿相应首成员角回退。无信号为null，不赋坏分；不拟合被留片段、真身份或参考。
- **冻结数值负控：** 二成员所有可评残差与同pair球面距离核`atol=1e−6m, rtol=1e−10`；多成员每query实际有支持集合m≥3，在同候选合法局部北东chart上核LOO-MSE与`Σpair distance²/(m−1)²`，`atol=1e−4m², rtol=1e−10`。超限即失败保全，不执行后放宽。
- 每场先落盘完整`private/features-{sid}.npz`，然后才打开缓存identities及完整pair labels，校验两者等价并评价pure/mixed；reference数组不读取。特征API不接身份/参考。来源metadata包含既有评价汇总用于人口绑定，不能称整个进程无评价材料。

## 评价、实际产物与停止

主结果：候选/节点覆盖及null分母、二成员和欧氏负控次数/最大差、固定负LOO median的pure/mixed AUROC。十个事前scorer为score sum/mean/min/median、shared/full pair median/RMS、LOO median/RMS，分数方向冻结；AUROC ties=.5，单类null。每个scorer各自定义人口与**所有scorer共同可定义人口**均报，不能拿不同人口直接定优劣。

全候选、原组、size1/2/3/4+、单/多来源、shared为0/(0,.5]/>.5完整分层；逐场/9组件/去最大组件及场景等权AUC，所有候选/原分组/节点均保留。pure/mixed仅后评价，同源共享偏差、平行异船、错误关联仍竞争解释。候选间和组件内相关，不作候选IID显著性。没有新raw/decoded F1、位置、coverage/duplicate方法增量，旧值只作背景。

一次attempt-1，原32固定终点，CPU≤4/RSS4GiB/AS8GiB/600s advisory，0模型/前向/优化/GPU/ZIP/CSV/reference数组/T_test。错误/非有限/绑定失配/恒等超限即失败保全，无自动重试/扩量/新头训练。private几何/身份/大矩阵留服务器；回传run/summary/全部逐场、真实launcher命令/退出/成本。完成后Record→Squeeze→Mode C，再决定同信息静态/条件头有无必要。

## 规格到验收

| 行为 | 已实现的可观察检查 |
|---|---|
| 先合法候选/完整信息后标签 | 无身份参数预测API；features落盘后才读identity/labels，失配保留已冻结特征和FAILED run |
| 同信息负控 | 日期线/不同support的二成员恒等、多成员平方重构；完整pair几何/support保存在同一候选artifact |
| 缺失人口不偷换 | singleton/无Q/null/单类AUROC保留；common_all_scorers与各自分母并报 |
| 一次终点/失败保全 | 独立output exist_ok=false，完整成功/第二场标签损坏的部分失败核验 |

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-ASSOCIATION-GEOMETRY-SIGNAL-20261029
plan_version: '1.0'
source_matrix_id: MC-ASSOCIATION-OUTPUT-ACTION-20261029-REVIEW-R3
selected_C_id: OA-C6r
source_P_ids: [OA-P7]
source_F_ids: [OA-F15, OA-F17, OA-F18, OA-F22]
source_M_ids: [OA-M7]
source_E_ids: [OA-E15, OA-E17, OA-E18, OA-E22]
research_question: LOO可辨性与覆盖能否超出当前score压缩，而其信息是否已由完整pair/support保留
hypothesis_and_mechanism: 可能有表征增量；不预设条件动作提供新状态信息
testable_predictions: 数值恒等负控、全分母及跨组件固定scorer AUROC，common定义人口
design_and_controls: 本文十scorer/同候选/完整pair几何/LOO和欧氏重构，无新分区
sample_materials_or_experimental_units: 原114事前V32/9组件，056全部原节点与pair
primary_outcome_and_estimand: 负控最大差和次数、覆盖/null、固定LOO median AUROC
secondary_outcomes: 所有scorer及共同人口/完整分层/场景组件/成本
analysis_plan_and_uncertainty: 确定性开发诊断，无候选IID推断，未知原因保留
resources_time_cost_and_risk: 单次CPU4/RSS4GiB/AS8GiB/600s advisory/32终点/0训练
ethics_safety_privacy_or_compliance: 特征先冻结再身份评价，reference数组/test不读，private留server
stopping_and_interpretation_rules: 错误/恒等超限保全，无自动重试/扩量/训练
status: diagnostic
evidence_status: partial
missing_evidence: [target_signal, matched_learned_heads, physical_mechanism, full_method_effect]
next_skill: record-experiment_after_actual_execution
```
