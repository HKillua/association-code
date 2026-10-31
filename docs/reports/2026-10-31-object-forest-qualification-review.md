# 171：T对象/森林资格检查的技术失败与修复证据

## EXPERIMENT_RECORD — REC-OBJECT-FOREST-171

2026-10-31，plan1.0/[事前Idea](../proposals/structured_association_20261031.md)，C57/Mode C1.77。原T32/0训练/无GPU/不新开V/test，未改变168/170/159/152。冻结e70f426、新run/launcher attempt1；真实命令见[L2S171](../../HandOver/L2S-20261031-171.md)与[原request](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_structured_association/171/launcher/request.json)。

原unit `mtad-a0ea4b19a7df.service`：Linger=yes/Restart=no，2026-10-31T05:36:57Z启动，05:37:07.946543Z终止；launcher真实exit1/FAILED，系统服务failed/ExecMainStatus1，无自动重试。worker总10.591636s，science开始05:37:06.190599Z、内部.330745s，峰值RSS393932800B；完成0场/0更新/0预测。环境加载与托管计总墙时，不将science内部时间当全成本。

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: REC-OBJECT-FOREST-171
source_idea_id: IDEA-OBJECT-FOREST-QUALIFICATION-20261031-171
plan_version: 1.0
execution_status: FAILED
research_question: do_T_lost_objects_have_wrong_forest_edge_gradient_entry
hypothesis: C57_structured_supervision_may_touch_lost_object_bridges_not_causal_proof
planned_vs_actual_design_and_controls: planned_original_T32_all_objects_actual_zero_completed_scenes
actual_configuration: see_L2S171_and_original_request_no_recipe_change
samples_materials_or_data_provenance: original170_saved_T_partitions_outputs_and127_T_evaluation
results_with_uncertainty: no_scientific_estimate_KeyError_before_first_scene
all_runs_or_repetitions: one_attempt_no_retry
observations_and_deviations: no_summary_endpoint_unmet_unmatched_reference_rows
evidence_locations: linked_171_science_and_launcher_originals
known_limits_and_missing_evidence: T_qualification_unknown
next_routes: preserve_then_explicit_repair_design
next_skill: squeeze-experiment
```

原因：`point_coverage`用中央target_ids查所有aligned_target_ids，遇到没有参考的组输出（空target ID）抛`KeyError: ''`。原170评价本已保留这类组；不是人口数据坏/模型效果差。全部12个公共原件（science run＋11 launcher含stderr/真实退出）已回传，[证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_structured_association/171/)；身份/坐标继续留原170服务器。未产summary、没有对象/梯度科学结论；计划终点未达成，不填0为无效果。技术失败算一次attempt，不替换原run。

## SQUEEZE_REPORT — SQ-OBJECT-FOREST-171

只挤出可支持事实：保存输出并不都映射到中央参考目标；无参考输出仍须保留组/重复/覆盖其他统计，**不应被错误加进reference有效query**。失败后只读原首T场核验：WS/CTXCAL分别11/8个未匹配aligned row，其中有效参考位置query均0。此后验读取不是171已完成科学，未读取新V或选择阈值。

修复条件：仅对中央target不包含且所有eligible query为false的行不建立target-query；若不在中央人口却有任何有效参考query则明确报错。新增fixture涵盖两种情形，并保留所有目标/原组统计；4项unittest全部通过。修复只改统计映射，无边分数/森林/标签/噪声/目标/曝光变化。没有证明结构监督有效或目标不匹配是根因。

回流Mode C1.78（E176/F273）：技术边界，不新增机制/Idea连接；171固定一次关闭。**新172若发布，为明确修复资格任务，不是171自动续跑**，原冻结目录/日志不动。来源分析/方案与失败均保留，不因修格式宣布科研完成。

```yaml
package_type: SQUEEZE_REPORT
squeeze_id: SQ-OBJECT-FOREST-171
input_experiment_record_ids: [REC-OBJECT-FOREST-171]
claim_scope: technical_boundary_only
data_quality_and_provenance_gaps: unmatched_output_not_missing_or_corrupt_reference_evidence
primary_result_with_uncertainty: zero_completed_scientific_scenes_no_effect_estimate
process_dynamics: one_failed_attempt_10.591636_seconds_393932800_bytes_RSS
mechanism_and_intermediate_findings: unknown_target_lookup_failure_not_model_mechanism
errors_heterogeneity_and_boundary_conditions: unmatched_rows_have_no_eligible_reference_queries_in_first_T_scene
robustness_sensitivity_and_alternative_explanations: added_unmatched_zero_query_and_invalid_positive_query_regression
new_P: none
new_F: F273
new_M: none
evidence_index: Record171_original_stderr_and_post_failure_first_T_counts
open_questions: [completed_T_qualification]
next_routes: explicit_new_repair_Idea_not_original_run_retry
record_id: REC-OBJECT-FOREST-171
idea_id: IDEA-OBJECT-FOREST-QUALIFICATION-20261031-171
source_ids: [C57, E174, E175]
design: original_T32_zero_training_forest_qualification
actual_execution: failed_before_first_scene_zero_updates
results_with_uncertainty: no_scientific_estimate_technical_traceback_only
plan_deviations: [planned_T32_endpoint_unmet, post_failure_first_T_unmatched_row_read]
reproducibility: original_command_request_logs_service_exit_saved
evidence_status: verified_technical_failure
limitations: scientific_qualification_unknown
missing_evidence: [completed_T_qualification]
next_skill: deep-read_Mode_C
```

## 172：明确修复后的实际资格检查 — REC/SQ-OBJECT-FOREST-172

事前[plan1.1](../proposals/structured_association_20261031.md)、[L2S172](../../HandOver/L2S-20261031-172.md)，新50520ca/attempt1/`mtad-edf901f4936e.service`，没有覆盖或重投171。原T32/860中央目标、原170score/分区/保存输出、原127motion/eval；0训练/teacher/GPU/新预测/新V/test，ε.1/B4/seed20不改。05:39:34.978388Z启动至05:39:47.017795Z，launcher总12.170251s，science6.935841s、RSS440160256B；32/32成功、真实exit0，服务inactive/dead/ExecMainStatus0/Restart=no/Linger=yes。34科学＋11launcher原件完整回传[172证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_structured_association/172/)，身份坐标仍留原127/170服务器。

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: REC-OBJECT-FOREST-172
source_idea_id: IDEA-OBJECT-FOREST-QUALIFICATION-REPAIR-20261031-172
plan_version: 1.1
execution_status: COMPLETED_exit0
research_question: original171_T_qualification_after_explicit_population_mapping_fix
hypothesis: same_C57_gradient_entry_question_no_effect_claim
planned_vs_actual_design_and_controls: complete_original_T32_all860_objects_full_lost_rescued_retained_neither
actual_configuration: original171_noise_and_full_labels_unchanged_only_unmatched_mapping_fix
samples_materials_or_data_provenance: original170_and127_T_only_bound_nodes_manifest_order
results_with_uncertainty: table_below_exploratory_T_counts_no_causal_interval
all_runs_or_repetitions: one_new_172_attempt_original171_failure_preserved
observations_and_deviations: none_for172_first_T_checks_for171_labeled_posthoc
evidence_locations: linked_172_all45_public_originals
known_limits_and_missing_evidence: no_matched_training_or_deployment_effect_no_independent_domain
next_routes: squeeze_then_Mode_C
next_skill: squeeze-experiment
```

| T-only对象层 | 对象数 | 单片段 | 正类pair质量 | 被错误森林边触及 | 新/丢有效query |
|---|---:|---:|---:|---:|---:|
| 全部 | 860 | 73 | 2268 | 314 | 394/417 |
| 丢失 | 13 | 0 | 37 | 13 | 0/417 |
| 救回 | 17 | 2 | 51 | 15 | 394/0 |
| 前后保留 | 812 | 69 | 2142 | 268 | 0/0 |

剩18个为前后均未代表对象，不从分母删去。丢失目标片段数`3,2,4,3,3,3,3,4,2,2,3,2,3`，均有异类混合新组；不是singleton漏学的证据。类内正边质量只37/2268约1.63%，但不证明尺寸是原因。32场normalized gap范围0…1.174224，规则资格为true；**该梯度触及全部丢目标，也触及15救回/268保留目标，不是高精度坏边分类器**。森林选负类边的梯度方向向下，真实类内连接梯度方向向上；WS部署和物理效果仍未知。

独立一致性：本地从全部匿名object行重加，T新394/丢417点、救17/丢13目标与原170冻结后验一致；检查无漏场/原T32顺序、全860目标与所有对象层非互斥问题（lost/rescued/retained/neither相互排斥）。这不是第二次科学重算森林或独立最优性审计，最优性只由小图枚举/源码数学特例与rank软件验收支持，完整标签数据准确性未认证。

Squeeze：支持到“已有T丢对象确有可测的结构训练干预入口”；**不支持**BCE/WS是唯一根因、已减少错误、未知K推理获益、覆盖或尾部保证。对照必须保冻结CTX、同起点/全场曝光BCE与BCE＋结构项，不能把额外训练或T标定当新机制。真实T类数只用于训练，部署仍WS/不读取k。MC4少于原文100，训练/推理图目标不同，均为需要实测的适配条件。回Mode C1.79/E177/F274，C57保条件候选，新训练须另立具体合同；原171技术失败不删除。

```yaml
package_type: SQUEEZE_REPORT
squeeze_id: SQ-OBJECT-FOREST-172
input_experiment_record_ids: [REC-OBJECT-FOREST-172]
claim_scope: T_qualification_not_effect_or_causal_evidence
data_quality_and_provenance_gaps: bound_saved_nodes_central_population_no_reference_truth_certification
primary_result_with_uncertainty: all13_lost_objects_touched_also15_rescued_and268_retained
process_dynamics: one_attempt_32_scenes_zero_updates_12.170251_seconds_440160256_bytes_RSS
mechanism_and_intermediate_findings: candidate_structured_gradient_entry_not_unique_root_cause
errors_heterogeneity_and_boundary_conditions: all_lost_2to4_fragments_mixed_groups_not_singletons
robustness_sensitivity_and_alternative_explanations: matches_original170_T_point_target_transitions_no_second_forest_audit
new_P: none
new_F: F274
new_M: none
evidence_index: all_T_anonymous_rows_summary_and_original_launcher
open_questions: [matched_training, unknown_k_deployment, physical_tail]
next_routes: Mode_C_then_specific_matched_training_Idea
record_id: REC-OBJECT-FOREST-172
idea_id: IDEA-OBJECT-FOREST-QUALIFICATION-REPAIR-20261031-172
source_ids: [C57, SQ-OBJECT-FOREST-171]
design: original_T32_saved_output_full_label_forest_gradient_qualification
actual_execution: completed_32_scenes_860_objects_zero_training
results_with_uncertainty: exploratory_T_all_13_lost_touched_also_15_rescued_no_causal_or_V_claim
plan_deviations: none_for_172_original_171_failure_retained
reproducibility: original_request_logs_exit_service_and_all_anonymous_rows_saved
evidence_status: verified_T_qualification_only
limitations: no_deployment_effect_or_independent_domain_estimate
missing_evidence: [matched_structured_training, coverage_association_physical_joint_gain]
next_skill: deep-read_Mode_C
```
