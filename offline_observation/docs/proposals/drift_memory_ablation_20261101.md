# 189：T侧平均运动与速度记忆的匹配分离

2026-11-01；IDEA-DRIFT-MEMORY-ABLATION-189，plan1.0，research-pipeline-contract v1.0，workflow_contract_version v2.0。消费MC-MTAD-20261031-v1.98/C65、P13/P14、F294/E197/M119(AI)与SQ188；来源与公式独立分析复用[AP-NATIVE-MOTION-TRANSFER](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/native_motion_transfer_20261031.md)。本次为Pattern5受控干预，不是新论文、OU作者复现或宣创新已成立。

## 连接、问题与预测

188 PG在V32有平均/RMSE/P99条件收益，但T HARD仍新增34个纯组>1km增害、STRUCT1个，原178不改。全部片段工作记忆很短只是线索，尚未在主点所选节点上测其作用。PG的位移为μh+(v0−μ)d；已有PC使用v0h，缺少同端点μh控制，不能把收益归于衰减。C65原评分9/12/条件候选保持，物理噪声、组/参考对应和真实曲率仍UN。

核心假设AI：若主要变化来自报告均值漂移，移除记忆项的MC=Exp(native_endpoint, μh)将接近PG；若记忆项有实质贡献，PG−MC的主点误差/尾部/候选距离会体现。结果接近不等统计等效或证明OU无用，结果不同也不证明真实动力学。另测T纯组害与合法输入条件的重叠，找后继解法的条件，不把相关特征变成部署门槛。

## 实际干预与完整对照

固定原188/178的HARD/STRUCT分组、raw scores/阈值、选中节点/源等质量tie、原生端点、缺报告规则和全部速航。新MC仅在count0且query在全组全部native之前/之后时，取该片段全部有效报告速度NE均值μ，按同一原生端点传播μh；raw kn显式一次转m/s。有效速度少于2或被选端点缺报告时同一节点退178状态，与188四臂合法域一致。任意源等价类/可变K、无固定名字或编号优先；离线canonical_all，不宣在线。

继承原T32/860目标、不选难例，**不读取V、不开封存test**。两头各base/O/PH/PC/PD/PG/MC×Mean+NN＝28臂。前六项完整数组从188按原臂导入并绑定178；只新生成MC，不重跑PG或原似然网格。0新增神经/前端/teacher/reference或似然优化，原生均值每片段每前端重算并计次数，不把统计均值计算叫零参数/零成本。继承178已监督头，不称全系统无学习；T前端见过标签，非全链LOCO。

主要：MC−PG完整共同有效位置点MAE差，按T12组件2000次seed20描述bootstrap；全部场景/组件、去最大并报，非独立确认。次要：MC−O/PH/PC/PD及PG−O、各自/共同人口MAE/RMSE/P95/P99、四属性/纸面口径、coverage/重复/raw-decoded、纯混/0-1-多、对O及最初base的>1km害/救回/最大害和旧害去留。主点PG-MC候选距离分布连续报告，>1m计数只作观测尺度、不作质量/部署阈值。

计划条件分析仅原T主参考对齐eligible点：全部/纯组/PG害与救回/MC害与救回、before/after；分母绑定188的451/344及纯组PG害34/1。输入字段为原生端点时距、组节点/源数、选中节点数、native样本/跨度/OLS fit_rmse、报告均值速度/离散度、均值-端点速度差/均值-OLS速度差/均值-全段位置secant差、PG-MC位置差。真值/纯度仅分析，不部署。完整量化分布与重叠，不沿T/V挑门槛，不宣某字段已是根因。私有逐点行留server，公共仅聚合。

## 资源、终点与解释

新run189/launcher attempt1，CPU8–11×4无GPU、RSS8GiB/1800s advisory，原T32/28臂一次到全部结果；非有限、输入绑定/软件错误/OOM即FAILED保全、不自动重投或改配方。成功后新audit189 checkout/launcher attempt1，CPU16–19×4无GPU/1800s advisory，完整规定收尾一次。152两活动预算/设备及原178/183/184/185/186/187/188冻结不动，现场兼容/FIFO，不建SSH值守/守护。

规定独审：保存主位置/人口/coverage/关系/尾部/primary独立复算，全部四属性同评价器记账（非独立四属性双实现）；独立从raw native报告重算μ和球面新MC、权重及保持域，全部冻结原188/base/O精确绑定。T条件记录核主点数、旧34/1害及profile人口/误差，与保存输出一致；原生描述量是共用提取的分析，非物理真值认证。science/audit含输入/评价/写盘成本各计，0新似然的原188成本引用保留。

若MC与PG无稳定净差，保平均运动适配价值，撤回记忆已证解释；若PG有局部额外价值，定位主点条件及不利例而非自动全300；若MC也伤HARD纯组，继续区分端点/平均运动与分组对应，不退源优先、不扫同信息loss。当前不授权V32/全300、重训、加seed或组合。

## 三层质量检查与实现约束

新颖性：均值CV/积分OU本身成熟，本次是C65归因缺口的可复核海上适配对照，不宣本领域首次；已比较本仓149/156/158/160/188，前者是不同状态/短窗/候选策略，不重复。源CTCRW正文Eq5/附录与代码范围沿AP，partial不增充分阅读数108。价值：正负均能判断是否需要OU工作拟合及何种输入条件适用，也保留更简单方法的可扩展潜力，不要求绝对原创。归因：同端点/节点/权重与旧全部控制、无新监督/参数搜索，V不访问；T复用/参考生成及实际主要K2仍限制泛化。

实现：association/evaluation/native_motion_interventions.py增加独立mean_velocity_control，旧函数数值不改；新tools/analysis/drift_memory_pilot.py/closeout只服务此合同。generic audit_variable_group_outputs及history_followup_closeout读取显式evaluation_splits，默认仍双split，仅支持非空无重复T/V集合；不伪造空V结果。保存属性工具已有此接口。验收核：报告均值与瞬时端点不同/β0及大β极限；kn/秒/前后时间/日期线/缺报告、K3匿名与list入口；完整T-only保存→主独审/四属性/自定义动作核，无V文件也成功，非法split拒绝。软件通过不作MTAD成绩。完成Record→Squeeze→Mode C后再定后继。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-DRIFT-MEMORY-ABLATION-189
created_at: '2026-11-01'
source_ids: [MC-MTAD-20261031-v1.98, SQ-ONE-SIDED-PERSISTENCE-188, AP-NATIVE-MOTION-TRANSFER-20261031]
idea_id: IDEA-DRIFT-MEMORY-ABLATION-189
plan_version: '1.0'
source_matrix_id: MC-MTAD-20261031-v1.98
selected_C_id: C65
source_P_ids: [P13, P14]
source_F_ids: [F294]
source_M_ids: [M119_AI]
source_E_ids: [E197]
research_question: whether_188_memory_term_adds_to_same_endpoint_native_mean_CV_and_T_pure_harm_conditions
hypothesis_and_mechanism: frozen_PG_minus_same_endpoint_mu_times_h_is_the_single_memory_term_intervention
sample_materials_or_experimental_units: original_T32_860_targets_no_V_no_sealed_test
primary_outcome_and_estimand: MC_minus_PG_common_position_point_weighted_MAE
secondary_outcomes: full_metrics_coverage_relations_tails_support_purity_and_input_conditions_above
analysis_plan_and_uncertainty: original_12_T_components_2000_seed20_descriptive_bootstrap_no_independent_test
resources_time_cost_and_risk: one_CPU_science_and_one_mandated_audit_1800s_each_advisory_RSS8GiB_no_new_fit
ethics_safety_privacy_or_compliance: private_coordinates_identities_native_parameters_weights_predictions_stay_server
evidence_status: partial
missing_evidence: [actual_MC_increment, selected_main_point_memory_contribution, pure_harm_visible_conditions]
status: pilot
next_skill: record-experiment
```
