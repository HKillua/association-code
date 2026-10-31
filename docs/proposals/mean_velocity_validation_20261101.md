# 191：更简单的平均运动表达能否保住188的V条件收益

2026-11-01，IDEA-MEAN-VELOCITY-VALIDATION-191/plan1.0，research-pipeline-contract v1.0/workflow v2.0。来源MC2.01/C65、F294/F295/F297、E197/E198/E200/M119 AI与既有原生运动AP。Pattern5匹配验证，不产生新机制/新论文数。189原T32完整MC−PG MAE<.005m/区间跨0、原34/1害全保留，位置单点仍可差339.942m，故不能直接填V成绩；190拟合端点不晋级，不用于本次MC，不拼组合。

假设：189完全相同、V前固定的MC=原生端点＋全片段有效SOG/COG报告NE均值×真实时间，可以在V32保留188 PG的条件平均效果，并省去输入侧OU似然搜索。操作/输入/缺失规则逐项继承[Idea189](drift_memory_ablation_20261101.md)，不改mean_velocity_control数学或既有网络；只新固定V入口、显式绑定已完成189 T-only资格。不是严格等效检验/全300/新域或作者复现，也不预设MC每点和PG相同。

原178/188的固定V32/1427目标，两前端base/O/PH/PC/PD/PG/MC各Mean+NN共28臂，完整导入188原24臂，仅新MC；0本轮T评价/新神经/teacher/reference/似然优化，native均值计算计数。固定源等价类/任意K、178选中片段/tie/分组/阈值，canonical_all离线、0封存test。保留缺失/重复/全人口、速航/其他支持域与coverage/raw-decoded，不退源优先/固定双源比例，不沿V改方案。

主要MC−PG、MC−O共同位置MAE，原V9组件/2000次seed20描述bootstrap、全部组件/去最大；次要MAE/RMSE/P95/P99、四属性/纸面指标、coverage/重复/关联、纯混/0-1-多、对O及最初base的新>1km害/救回/最大害/旧害去留、MC-PG候选距离全分布。V784/655主参考点人口须与原件绑定，PG原害11/0及纯11/0作为原控制绑定，不限制MC新增害以免选择性删点。V是复用开发域、与T共享身份，不作确认性显著性宣称。

若均值近似PG但尾部有差，完整保留，不以均值小宣等效；若MC无额外条件成本且保持整体/尾部，保更简单候选、不自动全300或替换178；若MC更差，保原188，不追加β/比例/seed/按V门槛。新端点FM/FL已阴性，绝不事后加入。0新OU工作fit只是算法调用差异，原188已做输入拟合成本仍计，不以本次缓存对旧方法宣不公平速度比；全链时延未测，不宣在线/整体轻量已证。

实现：复用189共享run函数的显式(split,number)白名单，仅默认(T_core,189)和新固定(V_select,191)；独立CLI mean_velocity_validation.py，必需原189资格路径。通用closeout支持单一显式T或V，禁止伪造另一split；原T-only/双split默认回归仍核。验收真实保存→主位置关系/四属性/自定义native动作/条件链的2T-0V和0T-2V软件场景，加既有端点/几何/OLS/域检查；软件不是MTAD效果。原189/190冻结代码/run不变，new191各一次新checkout/run/launcher。

CPU8–11×4无GPU/RSS8GiB/1800s advisory，到V32全部28臂即止；失败保全，不重投/换配方/删点/加步。成功后一次新audit191/CPU16–19×4无GPU/1800s advisory，独立主位置/人口/关系/尾部/primary/native均值球面状态和全部原控制，四属性同评价器完整记账非独立双实现，784/655及PG旧害绑定。原152活动预算/设备、178/188/189/190和失败不动，不值守/建守护/触他人。公共原件回传，private留server，Record→Squeeze→Mode C；无自动T重跑/V300/新seed或192。

三层检查：均值CV成熟，不宣新算法或领域首创；新增价值是让已有条件效果更简单/可扩展且少依赖输入似然估计，同时明确其反例。固定信息/端点/权重/全部对照、公式V前冻结，避免把缓存/容量/标签差算机制。阅读108不增、真实K2/K3仅软件、离线和参考未知保留。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-MEAN-VELOCITY-VALIDATION-191
created_at: '2026-11-01'
source_ids: [MC-MTAD-20261101-v2.01, SQ-DRIFT-MEMORY-ABLATION-189, SQ-ONE-SIDED-PERSISTENCE-188]
idea_id: IDEA-MEAN-VELOCITY-VALIDATION-191
plan_version: '1.0'
source_matrix_id: MC-MTAD-20261101-v2.01
selected_C_id: C65
source_P_ids: [P13, P14]
source_F_ids: [F294, F295, F297]
source_M_ids: [M119_AI]
source_E_ids: [E197, E198, E200]
research_question: actual_V32_effect_and_tail_of_frozen189_MC_vs188_PG_and178
hypothesis_and_mechanism: simpler_mean_motion_expression_may_preserve_conditional_effect_without_claim_OU_cause
sample_materials_or_experimental_units: original_V32_1427targets_no_T_reevaluation_no_sealed_test
primary_outcome_and_estimand: MC_minus_PG_and_MC_minus_O_common_point_position_MAE
analysis_plan_and_uncertainty: V9_components2000_seed20_descriptive_no_independent_confirmation
resources_time_cost_and_risk: one_CPU_science_and_one_audit1800s_each_advisory_no_new_training_or_likelihood
status: pilot
evidence_status: partial
missing_evidence: [actual_MC_V_effect_tail_and_coverage]
next_skill: record-experiment
```
