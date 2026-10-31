# 174：多数T混组受当前目标奖励，不支持只换更强搜索

2026-10-31。REC-ASSOCIATION-DECISION-174 → SQ-ASSOCIATION-DECISION-174；来源[Idea174 plan1.0](../proposals/association_decision_20261031.md)/Mode C1.80 C57/E178，实际结果verified，机制未确证。

## Record：实际执行

原173 T32全部860参考对象、跨源140079对/1509真对；四scorer与原阈值不改，32场全量复放WS与保存分区exact；零训练、新物理预测、V/test访问。对每个预测混组按T真实身份拆分，仅作诊断，直接重算J与逐假边求和一致；不是可部署方法、不是oracle救回成绩。前置两项单元测试通过。

冻结b82a38b，`association-code-run174`；原run `runs/LEVEL1-ASSOCIATION-DECISION-AUDIT-20261031-174-attempt-1`，launcher同前缀`-launch-attempt-1`。同发布完整命令见L2S174及request.json；CPU16–19/OMP等4线程/无GPU，RSS464510976字节。实际2026-10-31 06:23:48.299359Z–06:24:01.443587Z；launcher总13.263569s，内部分析10.029465s。`mtad-bb0d6b1f495b.service`真实Result=success/ExecMainStatus=0/inactive/dead/Restart=no，Linger=yes。无失败/重试/排除或计划偏差。34科学文件＋11 launcher原件完整回传[174证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_association_decision/174)，私人原输入不下传。

|T scorer|混合组|按真身份拆开提高J|拆开降低J|raw真但decoded漏掉（跨源）|raw负但decoded假连接（跨源）|
|---|---:|---:|---:|---:|---:|
|WS|147|2|145|313|17|
|CTX|145|2|143|267|18|
|BCE|135|2|133|237|16|
|STRUCT|150|1|149|276|19|

STRUCT相对BCE新出现16条raw跨源真边，其中12保留、4被构组抑制；相对原WS新56条，其中51保留/5抑制。这不能用来解释所有decoded退化，还包含原真边丢失及假边竞争。STRUCT最终跨源假连接372条，353本来raw为正、19为负后被组闭包纳入；不能概括为“错误全由负边传递造成”。

STRUCT对BCE救31/丢9对象，新890/丢269有效点；对WS救19/丢10，新358/丢248，与原173 T获失对齐。所有这些丢失对象都涉及拆真身份会降低J的混组，没有涉及正ΔJ拆分。全场真实身份分区J减预测J：WS−658.948、CTX−506.527、BCE−734.686、STRUCT−228.191；不表示J全局最优或参考身份无误。

## Squeeze：能回答什么、不能回答什么

DE：T错误分组多数在当前加和分数目标下有利，限制了“只提高J便能消除错并”的解释。少数正ΔJ混组仍存在搜索空间；单一按真身份拆分不穷尽所有搜索操作，因此不能证明更强搜索无价值。解码中没有按真值去修分组。

AI：现有完整场class-balanced BCE与固定k森林辅助对最终未知k组错误的学习重点可能仍不匹配；也可能是特征区分不足、标定或T标签/参考接口问题。T已参与训练和阈值选择，不能据此宣称V/新域分布规律或因果根因。低相关来源NMS、图搜索与我们的样本/目标不同，需要迁移条件和强对照。

最小后继应区分单边难度与真实构组错误反馈，保持输入/容量/曝光/辅助正负数量和部署规则不变。若两种难例训练同样有效，新增价值来自一般难例强化；若构组反馈有独立收益，才深入其具体错并/误拆救回机制；若都无增量，当前配方停止，转信息/目标代价条件。不因文献有先例取消有意义适配。

该分析回Mode C后才建立新Idea；当前报告不把训练候选写成结果。

```yaml
package_type: SQUEEZE_REPORT
squeeze_id: SQ-ASSOCIATION-DECISION-174
input_experiment_record_ids: [REC-ASSOCIATION-DECISION-174]
claim_scope: saved_T32_only_860_objects_no_prediction_or_fit
primary_result_with_uncertainty: 149_of_150_STRUCT_mixed_groups_lose_J_when_identity_split; descriptive_no_independent_generalization_CI
process_dynamics: exact_replay_32_scenes_4_models_13.263569_seconds_total
mechanism_and_intermediate_findings: score_objective_conflict_supported_for_this_intervention_not_unique_cause
errors_heterogeneity_and_boundary_conditions: all_cross_same_source_joint_no_joint_and_lost_rescued_retained_objects_complete
robustness_sensitivity_and_alternative_explanations: direct_J_and_edge_decomposition_agree; stronger_search_not_excluded; T_seen_labels
new_P: none
new_F: F276
new_M: none
open_questions: [decision_error_mining_increment, calibration, missing_input_information, physical_tail]
next_skill: deep-read_Mode_C
```
