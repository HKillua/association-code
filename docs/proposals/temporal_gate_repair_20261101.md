# 194：隔离CPU审计导入后检验原生阶段信息

2026-11-01，IDEA-TEMPORAL-GATE-194/plan1.0，research-pipeline-contract v1.0/workflow v2.0；消费MC2.06/C68与SQ193/F302/E205。Pattern5最小任务适配，技术修复后一次新任务，不复投193。[193完整科学设计](temporal_gate_20261101.md)全部按引用生效（来源、迁移条件、对照、信息隔离、标签、指标、停止与解释），与[失败Record/Squeeze](../reports/2026-11-01-temporal-gate-startup-failure.md)共同组成完整包。193无科学结果，不据其调参数或改假设。

## 问题→方案与验收

C68目标问题仍是原MC只保全段均值，阶段/顺序是否可改善单侧任务风险；T33单片段害不能都靠换源解决，未定噪声/未知未来或参考原因。193失败暴露训练入口借用ARMS常量导入CPU审计并隐藏GPU。**只隔离该常量依赖、新194入口明确绑定失败血缘、训练前记录设备环境并做一元素分配**；不改旧审计行为、神经网络/特征/标准化/抽样/目标/分组/选择/几何或原输入。

|行为契约|验收/证据|
|---|---|
|训练入口导入不修改外部设备环境|全新子进程设置可见性marker，导入194入口后原值不变；此前父进程已导入审计会掩盖此问题。|
|旧科学与输出兼容|原相位/顺序/K3/旋转/日期线/有限梯度、匹配学习及NumPy重建、T-only44臂保存到独审/四属性原测试继续通过。|
|新任务明确归属|194入口要求failure-run，校验193 FAILED/0更新/0V；新run及所有模型固定IDEA194，原失败不修改。|
|真实设备而非仅软件模拟|新冻结194按UUID与CPU8–11先保存最小CUDA分配结果；科学run也记录实际设备及预检；失败不改CPU或换配方续跑。|

科学额度保持：原T128/12组件leave+full13fold，39fit×128步＝4992GPU更新，batch1024/seed20/相同初始化抽样与共享normalizer/Adam.001-wd1e−4-clip1/固定末步；原T32/860对象44臂、0V/test。MEAN/BAG/ORDER各2145参数，同一O/MC时间估计读出，HALF控制独立、原28臂全部保留。K匿名源等质量tie不变；不是固定双源、FlowState模型复现或新方法的理论证明。

新science194独立checkout/run/launcher attempt1，GPU0 UUID GPU-c5a00c37-f181-90b3-f220-be6e7457f473，现场兼容共卡新增CUDA≤1GiB、CPU8–11×4/RSS8GiB/3600s advisory。成功后新audit194独立checkout/launcher attempt1，CPU16–19×4无GPU/1800s advisory，原temporal_gate_closeout全部核验必须通过。主位置/人口/关系/尾部独核、四属性同评价器记账；共享特征提取/无优化器重放边界不变。最小设备探针不是训练，也需记录成本。

主要完整T ORDER−BAG/MEAN/MC/O/HALF，两前端各报组件描述区间、去最大；全部44臂四属性/coverage/raw-decoded/纯混/支持域/原34–1害及3–33救回/新增公里害不省。只能T条件探索，不选V/最好seed/删点；没有同容量控制的增量不称顺序有用，不保完整风险不晋级。成功或失败均保原件，Record→Squeeze→Mode C后再决定；无自动V/全300/195/新seed。152活动预算、原193失败与178–192冻结不动。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-TEMPORAL-GATE-194
created_at: '2026-11-01'
idea_id: IDEA-TEMPORAL-GATE-194
plan_version: '1.0'
source_matrix_id: MC-MTAD-20261101-v2.06
selected_C_id: C68
source_P_ids: [P13, P14]
source_F_ids: [F295, F298, F300, F301, F302]
source_M_ids: [M122_AI]
source_E_ids: [E198, E201, E203, E204, E205]
research_question: phase_information_and_order_increment_after_preserved_import_failure
hypothesis_and_mechanism: original_193_scientific_question_unchanged_no_assumed_SSM_or_noise_cause
sample_materials_or_experimental_units: original_T128_fit_T32_860targets_evaluation_0V
primary_outcome_and_estimand: original_44arm_common_point_risk_and_tail_ORDER_BAG_MEAN_MC_O_HALF
analysis_plan_and_uncertainty: original_193_full_plan_by_reference_components_descriptive_CI_no_best_selection
resources_time_cost_and_risk: 39fits_4992updates_3600s_science_1800s_audit_advisory_one_attempt
stopping_and_interpretation_rules: full_T32_and_required_audit_or_preserve_failure_no_automatic_V_or_retry
status: pilot
evidence_status: partial
missing_evidence: [phase_increment, order_increment, full_task_risk_tail]
next_skill: record-experiment
```
