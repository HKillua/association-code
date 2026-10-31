# 190：潜在端点状态×运动估计的T侧干预

2026-11-01，IDEA-FITTED-ENDPOINT-FACTORIAL-190，plan1.0/research-pipeline-contract v1.0/workflow v2.0。消费MC2.00/C66（10/12，M120 AI）、SQ189/F295/E198和[独立代码/来源条件AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/latent_endpoint_conditions_20261101.md)/F296/E199。Pattern1：33个HARD公里害query仅一片段，移除OU记忆仍伤，不能靠换源；工作模型下的端点状态与速度相容性值得实际干预，不预设真实端点有错。

研究问题：固定178选中节点/权重和189所有控制，用全段CV位置拟合端点替代测量端点，能否保平均运动条件收益并减少纯组尾部？原模型已使用原生端点，不冒称换成了“首个原生端点”。原GAussian/IID/匀速假设若不成立，拟合端点可能抹去真实机动；边界与全部不利结果保留。不是GPTR/CTCRW作者复现或新OLS估计器，成熟状态估计可作有意义适配，不设绝对原创门槛。

## 受控操作与终点

原T32/860目标，0V/test。原189两头七策略base/O/PH/PC/PD/PG/MC各Mean+NN全部导入保留；新三策略：GL＝原生端点＋geodesic OLS NE速度（单独几何控），FL＝全段OLS拟合端点＋同OLS NE速度，FM＝全段OLS拟合端点＋原生报告均值速度。共两头各十策略×Mean/NN＝40臂。固定source等价类/任意K、所有选中节点/源等质量tie，无固定源名字、参考/oracle或V选门槛。

每片段全native真实t与lat/短弧解包lon，b=Σ(t−t̄)(x−x̄)/Σ(t−t̄)²、x̂_end=x̄+(t_end−t̄)b；速度转换在native均值纬度。只在原189全组单侧count0域，按冻结权重合各节点球面位置。GL/FL/FM同用NE geodesic推进，故FL−GL隔离端点状态、FM−MC隔离报告均值下端点、MC−GL及FM−FL隔离运动、GL−O检查几何；完整FM−O不当单因素因果。全部速航、其他zero/单多域保持189/178；分区/分数/coverage/raw-decoded不改。

同一场的输入native拟合只做一次并共享两前端；预计T2310次双坐标OLS、2310输入均值（实际数计），0新神经/teacher/reference/似然优化，继承原监督头不称全链无学习。每片段核原native_count/所有支持外旧状态/OLS NE速度与冻结cache绑定（坐标1e−10度、速度1e−8m/s浮点容差），不靠SHA/hash准入；若不符即失败保全，不暗改窗口/输入。端点报告不足2/selected末端缺报告或拟合端点非法纬度时，三新臂共同保持MC并计数，不删点/clip新端点。原生输入不覆盖。

主要：FL−GL、FM−MC与完整FM−O共同有效点MAE差，T12组件2000次seed20描述区间/全部组件/去最大；次要全部几何/运动直接控、MAE/RMSE/P95/P99、四属性/纸面指标、覆盖/重复/raw-decoded、纯混/0-1-多、新>1km害/救回/最大害及原base门槛变化。计划另核189原34/1 PG害和3/33救回在GL/FL/FM下的保留/解除/反转，GT/纯度仅分析。

如果拟合端点有局部收益但伤整体/尾部，不晋级；若能保均值/降低损害则保条件候选、先解释发生域，不自动V/全300或新seed；若几何已解释变化，不把增益全归端点。新输出在支持外单独换状态，边界可能不连续，不宣完整连续后验、协方差校准或在线端到端。没有物理噪声真值，不从改善反推原始观测坏或reference错。

## 资源、实现与检查

新run190/launcher attempt1，CPU8–11×4/无GPU/RSS8GiB/1800s advisory；原T32/40臂到固定全部评价。成功后规定一次新audit190/CPU16–19×4无GPU/1800s advisory。失败保全不重试/换配方，原152/178/188/189及失败冻结不改，不碰他人/值守/建守护。首场来源绑定.143299605s/157次双坐标OLS单列来源成本，非190科学或189重跑。

在native_motion_interventions增加fit_endpoint_models与fitted_endpoint_controls，旧函数不改；新fitted_endpoint_pilot/closeout复用真实保存评价/T-onlyaudit。验收核匀速无噪声时native/拟合端点相同、已知端点残差的解析线性拟合、速度×端点分离/日期线/前后时间/源匿名K3/list、共同缺报告/非法拟合域回MC；独立lstsq/球面状态重建与冻结数组，全T代码/cache绑定，完整T-only生命周期沿189已验证链复用。科学后主位置/人口/关系/尾部/primary独核，四属性同评价器全记账非独立双实现，native重算/保持域与原害病例核算；参数/坐标/身份/预测留server，公共原件回传，Record→Squeeze→Mode C。

三层检查：全段OLS拟合端点与160短窗Theil–Sen/127旧尺度RTS条件不同，不重复旧配方；近邻与来源范围沿AP，不宣首次。价值是验证支持外单片段恢复与状态相容性，正负均有明确出口；真实K2/T共享身份/复用开发域、工作模型误设/源间偏差和拟合endpoint边界不连续均限制归因/泛化。没有新超参数/按V挑比例，没有先选好方法再补论据。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-FITTED-ENDPOINT-FACTORIAL-190
created_at: '2026-11-01'
source_ids: [MC-MTAD-20261101-v2.00, SQ-DRIFT-MEMORY-ABLATION-189, AP-LATENT-ENDPOINT-CONDITIONS-20261101]
idea_id: IDEA-FITTED-ENDPOINT-FACTORIAL-190
plan_version: '1.0'
source_matrix_id: MC-MTAD-20261101-v2.00
selected_C_id: C66
source_P_ids: [P13, P14]
source_F_ids: [F295, F296]
source_M_ids: [M120_AI]
source_E_ids: [E198, E199]
research_question: same_candidate_latent_endpoint_state_and_motion_consistency_under_unsupported_singleton_forecast
hypothesis_and_mechanism: fitted_state_can_reduce_endpoint_residual_carryover_but_manoeuvre_or_bias_can_reverse_effect
primary_outcome_and_estimand: FL_minus_GL_FM_minus_MC_and_full_FM_minus_O_common_position_point_MAE
analysis_plan_and_uncertainty: original_T12_components_2000_seed20_descriptive_bootstrap_complete_population_tails_attributes_and_old_case_sets
resources_time_cost_and_risk: one_T32_40arm_CPU_science_and_one_mandated_CPU_audit_1800s_each_advisory_no_V_or_new_neural_training
ethics_safety_privacy_or_compliance: private_coordinates_identities_models_weights_predictions_stay_server
status: pilot
evidence_status: partial
missing_evidence: [full_T_binding, endpoint_velocity_geometry_effect_and_tail, physical_linear_noise_validity]
next_skill: record-experiment
```
