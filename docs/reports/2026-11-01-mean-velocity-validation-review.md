# 191：平均运动表达的固定V32验证

2026-11-01，REC-MEAN-VELOCITY-VALIDATION-191 → SQ-MEAN-VELOCITY-VALIDATION-191，research-pipeline-contract v1.0/workflow v2.0。来源[Idea191/plan1.0](../proposals/mean_velocity_validation_20261101.md)、MC2.01/C65/F294–295/F297/E197–198/E200/M119 AI，189完整T资格后新立合同；不是沿旧189自动扩V。科学/规定独审各一次真实exit0、execution_status=completed/evidence_status=verified，探索性计算pilot。充分108＝72领域/相近＋36跨域不增。

## Record：问题、设计、血缘与实际量

检验完全相同的189 MC能否在V保住188条件效果、减少输入OU似然调用。MC=同原生端点＋同已选片段全部有效native SOG/COG报告NE均值×真实秒；不加190拟合端点，不改公式/权重/分组/阈值/候选或按V设cap。raw kn一次转m/s，球面大圆传播，少于两有效报告/缺端点报告时与188同节点退旧状态。只全组native支持前/后且count0改位置，其他zero、单/多支持与所有速航保持178 O。无源名称/编号优先或固定双源比例，可变K接口/真实仍主要K2、canonical_all离线。

原178/188固定V32/1427目标，两前端各base/O/PH/PC/PD/PG/MC×Mean+NN共28臂，原188全部24臂import exact。0本轮T评价/新神经、前端、teacher/reference监督或OU似然优化；继承178已监督风险头，不能称整系统无学习。输入均值计算实际次数、资源与起止见下。V是复用开发域、与T共享6053原A键/组件偏斜，不当身份独立/确认性新域，0封存test。所有缺失/困难/重复保留，first-anchor/first-claim对应不改；reference/纯度仅评价，不进入MC predictor。

实际完整命令/环境见[L2S191](../../HandOver/L2S-20261101-191.md)、[启动](../../HandOver/S2L-20261101-191-start.md)及原request。原178/188/189/190/152与失败原件冻结不动。

## Record：完整V位置结果

单位米；全部28臂/各自与共同点、所有场景/组件/去最大、四属性/论文指标完整保留原件。普通Mean未改变，下表全部NN动作不挑单列冠军。

|V策略|MAE|RMSE|P95|P99|
|---|---:|---:|---:|---:|
|HARD_NN|452.991515|1502.760453|2480.603107|5166.149782|
|HO_NN|355.760381|1329.099776|561.988102|5003.510220|
|HPH_NN|365.451518|1345.992150|576.196995|5208.192453|
|HPC_NN|803.331287|6869.342397|590.350999|9003.313622|
|HPD_NN|365.427900|1345.823139|576.196995|5204.960383|
|HPG_NN|353.072763|1310.308529|562.116035|4927.703218|
|HMC_NN|353.084018|1310.392606|562.116035|4927.415374|
|STRUCT_NN|427.240999|1447.518565|2105.656507|4842.083406|
|SO_NN|328.915570|1267.936207|503.993180|4359.930018|
|SPH_NN|337.491599|1195.460145|511.362108|4684.825022|
|SPC_NN|766.435708|7063.118459|519.217213|8054.190604|
|SPD_NN|337.473759|1195.362337|511.362108|4683.971200|
|SPG_NN|326.529710|1251.402467|505.428300|4309.512819|
|SMC_NN|326.529709|1251.456623|505.310791|4309.512819|

|V primary共同点MAE差|差m|9组件/2000次seed20描述95%区间|
|---|---:|---:|
|HPG_NN_vs_HO_NN|-2.687617772|[-3.224624589,1.132183451]|
|HMC_NN_vs_HO_NN|-2.676363118|[-3.209368678,1.142560736]|
|HMC_NN_vs_HPH_NN|-12.367499754|[-40.603081942,18.755652588]|
|HMC_NN_vs_HPC_NN|-450.247268995|[-802.144837327,-45.658672128]|
|HMC_NN_vs_HPD_NN|-12.343881917|[-40.570855063,18.665011941]|
|HMC_NN_vs_HPG_NN|+0.011254653|[-0.081202794,0.017712785]|
|SPG_NN_vs_SO_NN|-2.385860422|[-2.760577884,-0.058587571]|
|SMC_NN_vs_SO_NN|-2.385861231|[-2.761211333,-0.061094229]|
|SMC_NN_vs_SPH_NN|-10.961890312|[-19.847888954,51.917381863]|
|SMC_NN_vs_SPC_NN|-439.905998895|[-513.553601242,9.987063361]|
|SMC_NN_vs_SPD_NN|-10.944049844|[-19.826536813,51.919653622]|
|SMC_NN_vs_SPG_NN|-0.000000809|[-0.008318019,0.007362594]|

## Squeeze：完整尾部与原base门槛

|V比较|新增>1km害|改善>1km|纯组新增害|最大增害m|
|---|---:|---:|---:|---:|
|HMC_NN_vs_HO_NN|11|70|11|1999.163876|
|HMC_NN_vs_HPG_NN|0|0|0|70.145290|
|SMC_NN_vs_SO_NN|0|42|0|496.849295|
|SMC_NN_vs_SPG_NN|0|0|0|12.955100|

|相对最初base的门槛变化|178 O旧害|MC当前害|解除|新引入|持续|纯组旧→新|
|---|---:|---:|---:|---:|---:|---|
|HMC|128|138|1|11|127|28→39|
|SMC|107|92|15|0|92|19→19|

以上门槛与MC对O新害定义不同，均保留；全部支持/纯混/单多/eligible层和所有组件在原件，不将某一前端或子组冠军拼成新完整系统。原PG害/救回的去留另见原条件行聚合和核算，query非独立轨迹。

## Squeeze：解释、边界及决定

MC−PG完整MAE+0.011255/−0.000000809m，两描述区间均跨0；MC−178 O为−2.676363/−2.385861m，HARD区间[−3.209369,1.142561]跨0、STRUCT[−2.761211,−.061094]略低0。RMSE/P99两头均较178降低，保更简单表达的条件价值；不认MC比PG更准或记忆项无物理作用。去最大组件后MC−O +.081137/−.295681m，HARD收益依赖人口组成，不能只报全体平均。各组件差表见下，不拼两前端冠军，不自动替换178/扩大全300。

V主点eligible784/655，与原PG绑定。MC-PG位置距离均值5.944872/5.964121m、中位4.775475/5.004739m、最大81.894682/14.747510m；误差变化最大+70.145290m、最小−14.147100m，两头相对PG均0新增/解除公里害。因此不是每点等效，少量相反变化可在总体均值抵消。

原PG的HARD11个纯组公里害全部保留，集中于**一场景/一个两节点组的11个query**，不当11独立案例；与T189的33单节点query不是同一人口。MC在该批点比PG好2.09–2.32m，仍不足修复，最大对O增害1999.163876m。原HARD71个公里改善点保70，1点跌出1km改善门槛、没有反转成公里害；原STRUCT42个全保。HARD救回点MC较PG误差增加.476–10.588m，不把门槛数差包装为新灾难或因果。T原34/1害与V此处11/0同时保留。

|原V组件|H MC−O MAE m|S MC−O MAE m|H MC−PG m|S MC−PG m|
|---|---:|---:|---:|---:|
|component-0b9d7d23108cbf0a74ed5828|+0.000000|+0.000000|+0.000000|+0.000000|
|component-23608a7a22c5322f7254ffc1|-3.253910|-2.828295|+0.018119|-0.000448|
|component-75391b641c3cd7287e382451|-3.181239|-0.174379|-0.142419|-0.011284|
|component-9ea441151d6468bd21826aea|+0.000000|+0.000000|+0.000000|+0.000000|
|component-a077aa3eb774a8b670334e7c|+0.000205|+0.000205|-0.006772|-0.006772|
|component-c260c3bbf884e315f715f00b|-0.001021|-0.001021|-0.023645|-0.023645|
|component-c7082a697d934f0004db48b9|+1.500121|-0.479371|+0.013957|+0.010189|
|component-d091361fb95b75d6d2a1a9c5|-0.513863|-0.068328|-0.002154|-0.001674|
|component-fdbc675a9f9f6a7e1c67e3bb|+0.000000|+0.000000|+0.000000|+0.000000|

|前端|位置点coverage|覆盖目标/1427|重复组|raw跨源mPCA/mPFA %|decoded跨源mPCA/mPFA %|
|---|---:|---:|---:|---:|---:|
|HARD|98.891905%|1401|203|89.645511/25.956239|81.759255/14.924344|
|STRUCT|98.031747%|1380|104|94.448134/28.165712|81.765139/15.573316|

以上与对应178 O/188 PG exact不变。HARD71396/72196有效点、STRUCT70775/72196；coverage差别属于原前端，不是MC提升。速航exact不变，全部原控制/纸面指标/四属性保留，不能以ME冒充MAE。

DE支持到固定V32的动作对照与完整风险；不证明OU/报告噪声为根因，也不把不显著当严格等效。T189原34/1害、190原33单节点害与端点阴性仍保留，不能靠V某一头无公里新害宣长尾已解。原生位置与SOG/COG时间依赖、运动估计可靠性/可用历史仍待条件核验；下一不得先认某一噪声机理再选论文或沿V设门槛。

## Record：实际服务、成本、核验与偏差

科学/独审均冻结f82d8a8，新association-code-run191/audit191，原run `runs/LEVEL1-MEAN-VELOCITY-20261101-191-attempt-1`。科学CPU8–11×4无GPU/RSS8GiB/1800s advisory，原mtad-83ff3fb70d1f.service真实exit0/inactive/dead/Resultsuccess/Restartno/Lingeryes。程序2026-10-31 20:05:19.687289Z–20:13:12.174857Z（北京时间11月1日04:05–04:13），472.487609s/RSS2922741760B；实际7702次native均值计算（两前端重复计算，非7702独立样本/学习fit）。

规定一次独审CPU16–19×4无GPU/1800s advisory，mtad-a93aa61965c8.service真实exit0/inactive/dead/Resultsuccess/Restartno，payload20:14:15.737541Z–20:16:46.883357Z，RSS1414873088B。主位置/人口/coverage/关系/尾部/primary独核85.896751s，四属性同评价器完整记账24.647395s，独立native均值/球面动作/保持域/原控制及pure/base害核25.993752s，显式V-only条件与预算绑定4.544310s。

科学launcher481.399491835s＋规定独审151.260458693s＝632.659950528s；计划内旧害/旧救回条件行聚合.068981141s单列，共632.728931669s。聚合在规定动作/保存行绑定后执行，0新预测/拟合/门槛选择，源脚本和结果一并回传，不当独立物理认证。完整服务/实际命令、输入/环境、原启动与终态、每步成本均有原件。

主位置/人口/coverage/关系/尾部/primary独立核验；native报告均值/球面新状态/保持域/原强控及预算/split绑定独核；全部四属性同评价器保存数组完整记账，非四属性独立双实现。条件描述复用同提取器，不当物理诊断独立复现。18项READY前检查含2T-0V/0T-2V链、旧双split/list/K3/单位/缺报告/几何/OLS；软件验收不是MTAD效果，原T-only与默认旧模式回归保持。无科学/规定独审失败或配方/训练量/数据筛选变更，无T重跑/封存test/新seed/全300。

全部65公共原件49934283B、tar6867562B＋真实终态/传输清单回[191原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261101_drift_memory/191)，逐文件数/大小核齐。原坐标/身份/权重/报告参数/全部预测/逐点条件行留server；65文件包含计划旧病例聚合源脚本/结果，非复制实时队列。原152/178/188/189/190和全部失败未改，当前无自动192/全300/新seed或组合。Record→Squeeze→Mode C2.02/F298/E201、SR v4.3/OD v1.16。

0新增OU工作fit只表示本次MC不调用输入似然搜索，原188的拟合代价留原记录；全局pipeline/frontend/head训练成本不归零。当前大量旧控制从缓存导入，不能用本次墙时对旧188宣不公平速度比；无全链延迟实测、不宣在线端到端或跨域K≥3已证。更简单且能保住效果的适配也有新增价值，不要求绝对原创，但论文贡献仍需可靠整体/尾部及公平对比。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-MEAN-VELOCITY-VALIDATION-191
created_at: '2026-11-01'
source_ids: [REC-MEAN-VELOCITY-VALIDATION-191]
squeeze_id: SQ-MEAN-VELOCITY-VALIDATION-191
input_experiment_record_ids: [REC-MEAN-VELOCITY-VALIDATION-191]
source_idea_id: IDEA-MEAN-VELOCITY-VALIDATION-191
plan_version: '1.0'
evidence_status: verified
claim_scope: fixed_V32_1427targets_28arms_no_T_reevaluation_new_neural_reference_or_likelihood
primary_result_with_uncertainty: MC_minus_O_minus2.676363_minus2.385861m_H_CI_crosses0_S_slightly_below0_MC_minus_PG_both_CI_cross0
process_dynamics: science481.399492_audit151.260459_planned_cohorts0.068981_total632.728932_seconds
mechanism_and_intermediate_findings: simpler_expression_effect_not_OU_physical_cause_or_strict_equivalence
errors_heterogeneity_and_boundary_conditions: MC_preserves11_0_PG_harms_retains70of71_42of42_km_rescues_no_new_km_harm_vsPG
data_quality_and_provenance_gaps: reused_development_V_identity_overlap_and_unknown_reference_physical_generation
robustness_sensitivity_and_alternative_explanations: original_controls_and_native_actions_independent_four_attributes_same_evaluator
new_P: none
new_F: F298
new_E: E201
new_M: none
mode_C_input_bundle: P13_P14_F298_M119_still_AI_E201_C65
evidence_index: original_run_launchers_all_saved_summaries_audits_and_profiles_linked_above
open_questions: [reliable_singleton_mean_motion, native_position_velocity_dependence, full_V300_effect]
next_skill: deep-read_Mode_C
no_automatic_new_run: true
```
