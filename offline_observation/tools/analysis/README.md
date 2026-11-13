# 分析与核验工具索引

按用途和源码说明定位。目录含已结案pilot及仍被其他工具导入的共享函数；此索引不授予执行权限，不表示对应方法已通过科学验证。实际任务看[交接](../../HandOver/Handover.md)，结果与设计看[报告](../../docs/reports/README.md)/[提案](../../docs/proposals/README.md)。退役专用工具的来源及恢复方法见[历史入口](../../docs/history/README.md)。

## 工程计时

|边界|入口与原件|
|---|---|
|应用收到已分轨measurement CSV字节→清洗/运动/关联/分组/融合→全部组JSON bytes|[离线方法前缀计时入口](../offline/README.md#历史整链前缀计时)、[历史整链计时说明](../../docs/reports/2026-11-05-pipeline-online-timing.md)|
|缓存motion→单组后端输出；不包含完整前端|[单组后端计时说明](../../docs/reports/2026-11-05-deployment-timing.md)；原一次性脚本保存在[资料库原件](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261105_deployment_timing/originals/timing_script.py)，不在本仓工具目录|

两种边界不能互换；计时入口不含雷达回波检测、单传感器建轨或网络/显示，也不提供新的精度结论。

## 比较与指标

<details>
<summary>展开保存预测的比较与指标工具</summary>

|文件|源码说明|
|---|---|
|[compare_association_duration.py](compare_association_duration.py)|060: paired 1000/3000-update system comparison using frozen V caches only.|
|[compare_completion_policies.py](compare_completion_policies.py)|Paired 093/094 evidence from original completed artifacts, without inference.|
|[compare_deepaf_peer.py](compare_deepaf_peer.py)|099 cached peer comparison and prespecified two-learning-rate selection.|
|[compare_frozen_position_adapter.py](compare_frozen_position_adapter.py)|090 complete cached CPU comparison; no archive, inference or training.|
|[compare_physical_predictions.py](compare_physical_predictions.py)|Compare completed development predictions on their saved common support only.|
|[compare_position_harm.py](compare_position_harm.py)|074 paired final4000 harm-penalty vs linear Transformer, cached CPU audit.|
|[compare_sequence_budget_seeds.py](compare_sequence_budget_seeds.py)|097 CPU table: two within-seed budget changes, no inference or training.|
|[coordinate_metrics.py](coordinate_metrics.py)|Read existing090C arrays; no model, training, archive or rematching.|

</details>

## 科学验证、条件诊断与共享编排

<details>
<summary>展开实验入口与共享编排索引</summary>

|文件|源码说明|
|---|---|
|[action_conditioned_risk_pilot.py](action_conditioned_risk_pilot.py)|195：T侧候选—动作风险验证。|
|[action_conditioned_risk_validation.py](action_conditioned_risk_validation.py)|196：冻结O/MC动作评分的原V32验证，不训练或重评T。|
|[association_context_pilot.py](association_context_pilot.py)|168/169: controlled front-end context or anchor/support-aware regrouping.|
|[association_decision_audit.py](association_decision_audit.py)|174: bounded T-only score/partition/object audit; oracle is diagnostic only.|
|[association_geometry_continuation.py](association_geometry_continuation.py)|130: new bounded-memory successor to failed128, importing six complete scenes.|
|[association_geometry_signal.py](association_geometry_signal.py)|128: same-information group geometry and leave-one-fragment-out diagnostics.|
|[association_operating_point_pilot.py](association_operating_point_pilot.py)|170: same T-only raw false-acceptance budget for three frozen scorers.|
|[conditional_source_value.py](conditional_source_value.py)|125: T-component-heldout prediction of the benefit of a fixed soft action.|
|[cost_routing_pilot.py](cost_routing_pilot.py)|180: matched objective continuation, single-support positions only.|
|[decision_error_mining_pilot.py](decision_error_mining_pilot.py)|175: matched generic-hard vs actual WS-decision error auxiliary training.|
|[decision_followup_pilot.py](decision_followup_pilot.py)|Independent143/144 pilots: relative decisions or actual-coordinate mixture gain.|
|[decoded_coverage_calibration.py](decoded_coverage_calibration.py)|176 T-only grid selection; all policies share candidates and error budgets.|
|[decoded_coverage_pilot.py](decoded_coverage_pilot.py)|176: shared-grid objective and coverage ablation on frozen 175 scores.|
|[diagnose_pair_representation_dependency.py](diagnose_pair_representation_dependency.py)|108: fixed T32 input-dependency diagnostic, with no model/reference access.|
|[diagnose_three_scenes.py](diagnose_three_scenes.py)|103: fixed-population diagnosis of three previously selected V scenes.|
|[drift_memory_pilot.py](drift_memory_pilot.py)|189 T qualification / 191 fixed V validation of the same mean-velocity control.|
|[endpoint_separation_pilot.py](endpoint_separation_pilot.py)|160: native endpoint x velocity interventions, anonymous nearest readout.|
|[fitted_endpoint_pilot.py](fitted_endpoint_pilot.py)|190: T-only endpoint-state x velocity factorial with geometry control.|
|[fragment_identity_capacity.py](fragment_identity_capacity.py)|C53-Pa: conditional hard-edge capacity on existing056 score-cache labels.|
|[frozen_fusion_transfer.py](frozen_fusion_transfer.py)|Frozen transfer;066 output-domain audit, or063 first-scene diagnostic.|
|[frozen_score_decision_diagnostic.py](frozen_score_decision_diagnostic.py)|C53: fixed056 edge evidence, original partition plus eight fixed permutations.|
|[fusion_transfer_sources.py](fusion_transfer_sources.py)|062 bindings to the actual056/057 endpoints and complete saved OLS replay.|
|[generated_support_score.py](generated_support_score.py)|Summarize an existing107 TCM feature report; no model, ZIP or score forward.|
|[group_reference_counterfactual.py](group_reference_counterfactual.py)|163: fixed references vs fixed inputs; oracle analysis, never a deployment score.|
|[history_candidate_pilot.py](history_candidate_pilot.py)|178: one shared history-risk model, matched supported/all-history actions.|
|[history_tail_qualification.py](history_tail_qualification.py)|179: T-only opportunity, score and native-age attribution; no deployment.|
|[inspect_tmtaff_population.py](inspect_tmtaff_population.py)|One non-training TMTAFF data preparation with public count-only outputs.|
|[learned_group_oracle.py](learned_group_oracle.py)|061: fixed056 learned groups, three cached controls and a restricted selector.|
|[local_expression_pilot.py](local_expression_pilot.py)|159: fixed T128, query-local encoding x quadratic/residual neural scoring.|
|[low_support_attribution.py](low_support_attribution.py)|177 T-only saved-output attribution; every oracle is diagnostic, never deployed.|
|[mean_velocity_validation.py](mean_velocity_validation.py)|191: exactly the189 mean-velocity expression, now on frozen V32.|
|[mixture_opportunity_pilot.py](mixture_opportunity_pilot.py)|134: T-only reference mixture opportunity and frozen OOF mistake diagnosis.|
|[native_compatibility_qualification.py](native_compatibility_qualification.py)|182: T-only native kinematic conditions; fixed existing outputs, no training.|
|[native_cv_state_control.py](native_cv_state_control.py)|127: native-event CV filtering/smoothing, T-only scale choice and fixed V32.|
|[native_history_bridge_pilot.py](native_history_bridge_pilot.py)|181: native bracket mask, nearest endpoint and conditional-time bridge controls.|
|[native_motion_interface_qualification.py](native_motion_interface_qualification.py)|186: T-only production list-group interface and label-free arithmetic audit.|
|[native_motion_pilot.py](native_motion_pilot.py)|183/184/185 independent fixed-scope native-motion interventions.|
|[object_forest_qualification.py](object_forest_qualification.py)|171: T-only saved object loss and exact full-label forest qualification.|
|[object_hypothesis_pilot.py](object_hypothesis_pilot.py)|129: native temporal link candidates, T-frozen rejection and real WS unions.|
|[objective_difficulty_pilot.py](objective_difficulty_pilot.py)|162: same nonlinear source scorer, supervision objective x train difficulty.|
|[observation_common.py](observation_common.py)|Shared immutable cache/evaluation plumbing for observation-only pilots.|
|[observation_quality_pilot.py](observation_quality_pilot.py)|136: fixed offline observation-quality contribution pilot, T32/V32 once.|
|[observation_scale_pilot.py](observation_scale_pilot.py)|137: T-only noisy-observation scale selection and fixed T32/V32 recovery.|
|[ordered_shape_continuation.py](ordered_shape_continuation.py)|132: reuse immutable126 caches/norms; bound sampling-input file handles.|
|[ordered_shape_signal.py](ordered_shape_signal.py)|126: four matched 1000-update probes of ordered Q64 pair-distance shape.|
|[output_domain_control.py](output_domain_control.py)|Fixed original V32: reuse saved NEAR states, vary only declared position domain.|
|[query_contribution_control.py](query_contribution_control.py)|116: fixed original V32, two saved identity partitions x four query rules.|
|[query_contribution_full_control.py](query_contribution_full_control.py)|117: reuse the original six V32 predictions; evaluate fixed controls on V300.|
|[report_availability_factor_check.py](report_availability_factor_check.py)|Deterministic report-factor calculation; no dataset or author reproduction.|
|[robust_endpoint_pilot.py](robust_endpoint_pilot.py)|158: same-source current multiscale velocity, fixed source/readout/population.|
|[sampling_measure_qualification.py](sampling_measure_qualification.py)|192: fixed T native sampling/mean-measure qualification, no new predictions.|
|[set_conditioned_risk_pilot.py](set_conditioned_risk_pilot.py)|166: fixed159 base, own-only capacity control versus other-source quality.|
|[shared_component_control.py](shared_component_control.py)|Cross-group heldout relative correction x local Huber-distance reweighting.|
|[shared_component_full_control.py](shared_component_full_control.py)|120: import all original118 arms and extend the frozen recipe to remaining268.|
|[source_contribution_readout_pilot.py](source_contribution_readout_pilot.py)|140 bounded T32/V32 chord-risk/readout pilot; no raw events or neural calls.|
|[source_coverage_pilot.py](source_coverage_pilot.py)|155: fixed T128 coverage, same10 features/66-coefficient source head.|
|[source_mix_shrink_control.py](source_mix_shrink_control.py)|124: separately test fixed source mixing and empirical center shrinkage.|
|[source_preference_control.py](source_preference_control.py)|121: source-priority attribution on immutable original118 V32 predictions.|
|[source_preference_full_control.py](source_preference_full_control.py)|123: import121 V32 and120 eight controls; compute only268x2 priority arms.|
|[source_risk_full_benchmark.py](source_risk_full_benchmark.py)|147: frozen133 heads, original32 imported predictions and new268 inference.|
|[source_risk_pilot.py](source_risk_pilot.py)|133: fixed T32/V32 shared source error scores; general-K contribution pilot.|
|[source_selection_evidence_pilot.py](source_selection_evidence_pilot.py)|139: fixed T32/V32 single-factor radial-meter source-selection target.|
|[state_recovery_tail_probe.py](state_recovery_tail_probe.py)|142: independently bind zero-support outputs and replay their CV endpoints.|
|[static_partition_pilot.py](static_partition_pilot.py)|114: fixed056 static partition pilot, with prespecified conditional expansion.|
|[static_partition_position_control.py](static_partition_position_control.py)|115: truly recompute ordinary Mean for four frozen114 pilot partitions.|
|[structured_association_pilot.py](structured_association_pilot.py)|173: matched full-scene BCE vs BCE+partial FY, unknown-k WS deployment.|
|[t128_frozen_full_scope.py](t128_frozen_full_scope.py)|157: frozen155 head, full147 inputs and151 frozen-reference V300, no fitting.|
|[temporal_gate_fixed_pilot.py](temporal_gate_fixed_pilot.py)|194: explicit new task after the preserved 193 entrypoint import failure.|
|[temporal_gate_pilot.py](temporal_gate_pilot.py)|193: matched T-only task gate with mean, orderless blocks, ordered blocks.|
|[temporal_source_pilot.py](temporal_source_pilot.py)|149 zero-support candidates /150 temporal source evidence, fixed T32/V32.|
|[tmtaff_saved_training_trace.py](tmtaff_saved_training_trace.py)|Read-only107 TFM trace audit; no forward, reference, V arrays or new fit.|
|[truth_free_partition_pilot.py](truth_free_partition_pilot.py)|165: same signed evidence, split/move refinement, real regrouped physics.|
|[unsupported_recovery_pilot.py](unsupported_recovery_pilot.py)|145 fixed T32/V32: endpoint/velocity factors and source-balanced fallback.|
|[within_source_forecast_pilot.py](within_source_forecast_pilot.py)|156: forecast selection only within a source; fixed anonymous nearest source.|

</details>

## 核验与记账

<details>
<summary>展开保存输出的核验与记账工具</summary>

|文件|源码说明|
|---|---|
|[action_conditioned_risk_closeout.py](action_conditioned_risk_closeout.py)|195：独立评分/输出复核与上限分析。|
|[action_conditioned_risk_validation_closeout.py](action_conditioned_risk_validation_closeout.py)|196：V32保存输出、full模型动作与固定害/救回群体核验。|
|[association_context_closeout.py](association_context_closeout.py)|One read-only output audit followed by complete saved-attribute accounting.|
|[audit_association_context_outputs.py](audit_association_context_outputs.py)|Independent NumPy raw/decoded association, anchors/support and saved physics.|
|[audit_conditional_T_arrays.py](audit_conditional_T_arrays.py)|Independent125 T-only labels, component folds, saved heads and OOF audit.|
|[audit_conditional_value_arrays.py](audit_conditional_value_arrays.py)|Independent saved-array main-position audit for completed125 only.|
|[audit_followup_outputs.py](audit_followup_outputs.py)|Independent NumPy output audit for143/144/146; does not certify fitted heads.|
|[audit_full_fusion_benchmark.py](audit_full_fusion_benchmark.py)|Independent NumPy output audit for147, without production prediction/evaluation.|
|[audit_geometry_arrays.py](audit_geometry_arrays.py)|Independent saved-candidate labels/counts/rank AUC audit; no predictor calls.|
|[audit_group_reference_population.py](audit_group_reference_population.py)|Independent 163 population/reference audit; no purification state refit or neural inference.|
|[audit_mixture_opportunity.py](audit_mixture_opportunity.py)|Independent NumPy audit of134 T-only saved mixture-opportunity diagnostics.|
|[audit_object_output_arrays.py](audit_object_output_arrays.py)|129 independent saved-array audit with variable partitions and coverage.|
|[audit_observation_common.py](audit_observation_common.py)|Independent NumPy metric audit helpers for 136/137 saved artifacts.|
|[audit_observation_quality.py](audit_observation_quality.py)|One bounded independent NumPy audit of completed 136 saved predictions.|
|[audit_observation_scale.py](audit_observation_scale.py)|Independent saved-observation selection/position audit for experiment 137.|
|[audit_ordered_shape_arrays.py](audit_ordered_shape_arrays.py)|One-shot independent126 saved-score/statistics and training-record audit.|
|[audit_predicted_registration.py](audit_predicted_registration.py)|Read-only condition audit of predicted correspondences in returned 103.|
|[audit_raw_case_chain.py](audit_raw_case_chain.py)|105: compare three selected raw CSV chains with retained 103/104 evidence.|
|[audit_shared_control_arrays.py](audit_shared_control_arrays.py)|One-shot independent saved-array position audit for completed120/121/123/124.|
|[audit_source_contribution_readout.py](audit_source_contribution_readout.py)|Independent 140 audit: chord responses, thirteen ridge fits and three readouts.|
|[audit_source_risk_arrays.py](audit_source_risk_arrays.py)|Independent133 NumPy audit of saved source policies, T fits and V arrays.|
|[audit_source_selection_evidence.py](audit_source_selection_evidence.py)|139 independent saved-array audit: fixed features, meter-target Q and hard output.|
|[audit_temporal_source_outputs.py](audit_temporal_source_outputs.py)|Independent NumPy output audit of149/150; no production policy or fitting.|
|[audit_three_case_chain.py](audit_three_case_chain.py)|104: bounded audit of three fixed cases in retained 103 evidence.|
|[audit_variable_group_outputs.py](audit_variable_group_outputs.py)|Independent saved-output/population/J audit for165, allowing coverage changes.|
|[decision_error_mining_closeout.py](decision_error_mining_closeout.py)|175 saved-output closeout, adapted from the preserved 173 squeeze source; no tuning.|
|[decoded_coverage_closeout.py](decoded_coverage_closeout.py)|176 saved-output audit and candidate counts; inherited saved-physics decomposition, no tuning.|
|[drift_memory_closeout.py](drift_memory_closeout.py)|189: independent native-mean intervention plus complete T-only output audit.|
|[fitted_endpoint_closeout.py](fitted_endpoint_closeout.py)|190 independent least-squares/geodesic actions and original T harm cohorts.|
|[followup_tail_accounting.py](followup_tail_accounting.py)|146 post-science accounting of planned before/after zero-support strata.|
|[forecast_tail_accounting.py](forecast_tail_accounting.py)|Read-only156 tail/validation-horizon accounting. No new prediction or fitting.|
|[history_candidate_closeout.py](history_candidate_closeout.py)|178: saved-output audit, independent action/label accounting and tail strata.|
|[history_followup_closeout.py](history_followup_closeout.py)|180/181: independent saved metrics plus objective/native bracket accounting.|
|[low_support_attribution_closeout.py](low_support_attribution_closeout.py)|177 prescribed supported/all-fragment split, independently recounted on saved harm cases.|
|[native_motion_closeout.py](native_motion_closeout.py)|Independent new-head forward/native arithmetic and complete output accounting.|
|[saved_attribute_accounting.py](saved_attribute_accounting.py)|Read-only complete physical/paper attribute reports from frozen mps artifacts.|
|[temporal_gate_closeout.py](temporal_gate_closeout.py)|193: saved T metrics plus matched training, native phase and gate accounting.|
|[temporal_source_accounting.py](temporal_source_accounting.py)|Read-only Squeeze of149/150: anchor-side zero support and source-slot errors.|

</details>
