# 187：双侧端点导数与形状约束的净增量

2026-10-31；REC-ENDPOINT-KINEMATICS-187 → SQ-ENDPOINT-KINEMATICS-187，research-pipeline-contract v1.0。来源[Idea187/plan1.0](../proposals/native_motion_postqualification_20261031.md)完整继承[184科学设计](../proposals/native_motion_interventions_20261031.md#endpoint-184)，Mode C1.95/C64/F288–291/E191–194与[原生运动来源AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/native_motion_transfer_20261031.md)。探索性计算实验，execution_status=completed，科学/规定独审各一次exit0、evidence_status=verified；充分108不增。

## Record：问题、实际设计与血缘

检验同源原生双侧zero域的端点速度信息及形状约束是否在完整178/181位置桥接强控上有净增量；不是把大gap/残差当可靠门槛，不预设GPTR的潜在状态条件能由带噪原生报告无代价替代。固定178 HARD/STRUCT组、风险分数/选源/tie权重，直接导入178 O及181 B圆周桥接/E源等权。新CL为经纬chart线性，CH为WNOA条件均值特例的Hermite，CM将各轴两端导数/secant比限制到[0,3]、零secant导数0。原生kn一次转NE m/s，真实秒与短弧经度。数学不越端点范围不保证真实船轨迹/端点正确。

仅count0且当前组同源严格双侧；三新臂共同合法域，选中端点缺报告/chart极点或CH非法纬度则全部回B并计数，不删点/不clamp原始观测。T/V观测均为canonical_all离线，方法在V前固定、不用参考选源/估参。单/多支持、未资格zero、全部速度航向exact保持178 O；分数/组/coverage/raw-decoded完全不改。可变K/源换序接口，真实主要K2，不退固定源；T_test封存。

原T32/860对象、V32/1427对象，全目标/重复/缺失保留；两头base/O/B/E/CL/CH/CM各Mean+NN共28臂，0神经/前端/reference拟合/teacher。原184软件失败保留，186真实T32资格成功后另立187，只修组入口list→array、不改公式/预算。实际[L2S187](../../HandOver/L2S-20261031-187.md)与原request保存全命令/环境，不覆盖旧失败。

## Record：完整位置结果

单位米，各自/共同人口、全部场景/组件及全部四属性在原件。

|策略|T MAE|V MAE|V RMSE|V P99|
|---|---:|---:|---:|---:|
|HARD base|454.615|452.992|1502.760|5166.150|
|HARD 178 O|392.233|355.760|1329.100|5003.510|
|HARD 181 B|390.259|356.142|1330.886|5011.074|
|HARD 181 E|394.947|360.842|1334.263|5011.074|
|HARD chart-linear|390.259|356.142|1330.886|5011.074|
|HARD Hermite|426.475|400.975|1440.037|5563.878|
|HARD 受限Hermite|393.584|359.764|1329.343|5009.328|
|STRUCT base|430.087|427.241|1447.519|4842.083|
|STRUCT 178 O|367.202|328.916|1267.936|4359.930|
|STRUCT 181 B|361.545|328.403|1268.273|4368.416|
|STRUCT 181 E|366.202|333.472|1272.058|4369.058|
|STRUCT chart-linear|361.545|328.403|1268.273|4368.416|
|STRUCT Hermite|400.534|372.810|1399.630|5412.318|
|STRUCT 受限Hermite|367.314|332.387|1267.511|4369.058|

普通Mean全部副本exact沿原HARD/STRUCT；V覆盖仍98.891905%/98.031747%、1401/1380对象、重复203/104，decoded跨源mPCA81.759255%/81.765139%、mPFA14.924344%/15.573316%；不宣覆盖/关联或速度航向改善，不拼两头冠军。

## Squeeze：全部主要比较与反证

下表为共同有效位置点、2000次seed20组件描述区间；负差较好。T12/V9组件、复用开发域且共享身份，非独立确认性检验，多臂探索不作单一显著性筛选。

|V primary比较|MAE差m|95%描述区间|
|---|---:|---:|
|HB_NN_vs_HO_NN|+0.381362|[-1.985454,2.427903]|
|HE_NN_vs_HO_NN|+5.081680|[2.639912,5.477787]|
|HCL_NN_vs_HO_NN|+0.381362|[-1.985454,2.427903]|
|HCH_NN_vs_HO_NN|+45.215024|[28.558991,81.186918]|
|HCM_NN_vs_HO_NN|+4.003545|[1.407576,4.175482]|
|HCL_NN_vs_HB_NN|+0.000000|[0.000000,0.000000]|
|HCH_NN_vs_HB_NN|+44.833662|[30.215045,79.504149]|
|HCM_NN_vs_HB_NN|+3.622183|[0.281377,4.878554]|
|HCH_NN_vs_HCL_NN|+44.833662|[30.215045,79.504149]|
|HCM_NN_vs_HCH_NN|-41.211479|[-79.104647,-25.594478]|
|HCM_NN_vs_HCL_NN|+3.622183|[0.281377,4.878554]|
|HCM_NN_vs_HE_NN|-1.078135|[-2.434557,0.252185]|
|SB_NN_vs_SO_NN|-0.512169|[-2.887233,0.207357]|
|SE_NN_vs_SO_NN|+4.556866|[1.205307,4.835137]|
|SCL_NN_vs_SO_NN|-0.512169|[-2.887233,0.207357]|
|SCH_NN_vs_SO_NN|+43.894121|[34.072644,108.582779]|
|SCM_NN_vs_SO_NN|+3.471241|[-0.665548,3.741035]|
|SCL_NN_vs_SB_NN|-0.000000|[-0.000000,0.000000]|
|SCH_NN_vs_SB_NN|+44.406290|[36.502488,108.924359]|
|SCM_NN_vs_SB_NN|+3.983410|[0.936207,5.619276]|
|SCH_NN_vs_SCL_NN|+44.406290|[36.502489,108.924359]|
|SCM_NN_vs_SCH_NN|-40.422880|[-107.795763,-30.914671]|
|SCM_NN_vs_SCL_NN|+3.983410|[0.936207,5.619277]|
|SCM_NN_vs_SE_NN|-1.085625|[-2.334751,0.594804]|

CH相对CL在V增加44.834/44.406m，T也增加36.215/38.988m，不是只在V突然失效。CM显著压下原CH的大退化，但对CL仍+3.622/+3.983m、对完整O+4.004/+3.471m；改善一个坏局部不等系统净增量。CL与B的表中三位数相同，细小数值差见原始primary，不把chart换法当新收益。CM在STRUCT的RMSE略低O，但MAE/P99和新增害更差，不拼单列冠军。去最大组件V CH−O仍+48.007/+60.800m、CM−O+3.129/+2.399m，负结果并非仅最大组件所致。

## Squeeze：资格、尾部与旧害去留

|V相对完整178 O|新增>1km增害|改善>1km|纯组新增害|最大增害m|
|---|---:|---:|---:|---:|
|HCL|41|16|20|20941.248|
|HCH|815|4|577|22634.641|
|HCM|92|10|69|17726.850|
|SCL|23|33|12|20941.248|
|SCH|709|10|496|28472.237|
|SCM|82|17|60|17726.850|

相对最初base门槛，原178 V害128/107，CH后911/767、纯574/487；CM后203/164、纯79/62；CL后159/111、纯41/24。CH救旧仅1/4，引新784/664；CM救旧5/9，引新80/66。与上表相对O的增害定义分开，不用少量救回遮盖新害。T CH害489/436、CM155/116，原104/75，T同样不支持改善。所有变化在有同源双侧资格的zero域，单/多域exact0；混组外纯组也显著受损，不能单归混组或reference错误。

完整原诊断聚合（[脚本原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_native_motion/analysis/native_diagnostic_summary.py)）核T原始group-query资格1012/1007、V1657/1486，所有joint fallback均0，非缺报告全部回退造成无效。选中source跨度T中位971.501/1011.427s、V771.085/831.164s，V最大3532.484/5269s；这包含重复输出组，**不等参考对齐主点分母**。完整min/median/p95/max与alpha保留native-diagnostic-summary.json；跨度大可能放大速度条件误差只是AI竞争解释，未做因果阈值验证，不能沿V设cap。

DE：当前直接端点报告速度条件有明显额外损害，单调限制减少其退化但未胜完整位置控制。AI/UN：原生报告可靠性、端点位置噪声、组/时序对应和真实曲率均可能影响；GPTR作者用估计潜在状态而本次用原始报告是已知适配差异，不等已证失败根因，不否定全部连续时间/导数条件方法。

## Record：真实服务、核验、成本与偏差

科学/独审均冻结3c3e0e3，独立association-code-run187/audit187、run `runs/LEVEL1-ENDPOINT-KINEMATICS-20261031-187-attempt-1`。科学CPU8–11×4/无GPU/3600s advisory，unit mtad-d90c0b551ff7.service；程序11:02:23.840652Z–11:13:34.675007Z、670.835088s、RSS2944856064B；launcher实际起止/环境/request原件完整。科学真实exit0/inactive/dead/Resultsuccess/Restartno/Lingeryes。

独审在科学完成后FIFO先于188，新CPU16–19×4无GPU/1800s advisory，mtad-b091163a9804.service真实exit0/inactive/dead/Resultsuccess/Restartno。主位置/人口/关系/raw-decoded/尾部/组件/primary独核140.484776s；同评价器保存数组四属性完整记账36.668090s，非四属性独立双实现；native动作/原生公式/保持域/纯混和旧害独核32.769107s、RSS1450225664B。独立Hermite/chart/限幅重建成立，native同时间流提取共用nc，不认证物理真值/作者复现。边界/跨度计数为已计划诊断的科学后只读聚合，0新预测/fit/评价选择，追加成本单列、脚本保存，不倒填成独审项目。

科学launcher677.458700180s＋独审218.516906821s＝895.975607001s；诊断聚合再加0.271110147s，合896.246717148s。不是部署延迟；此前184失败8.262476s与185失败9.215925s、共享186资格90.650868s另列保留，不能隐藏或双计为本科学训练。无新科学/独审失败、样本排除、配方/预算/选模变化或test访问；187是明确新合同非原184重试。15 READY前软件项含复用不作新MTAD证据。

全部96公共原件113233086B、tar16263387B＋真实服务/传输清单回[187原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_native_motion/187)，逐文件数/大小核齐。坐标/身份/权重/大预测留server，全部原失败与活动冻结不改。**保178；187当前CL/CH/CM配方均不晋级，不自动组合、全300、cap或新seed。** 回Mode C1.97/F293/E196、SR v3.8，C64保端点状态可靠性的未决价值，原始导数/限幅配方关闭自动扩大。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-ENDPOINT-KINEMATICS-187
input_experiment_record_ids: [REC-ENDPOINT-KINEMATICS-187]
source_idea_id: IDEA-ENDPOINT-KINEMATICS-187
plan_version: '1.0'
evidence_status: verified
claim_scope: T32_V32_28arms_zero_neural_updates_native_bilateral_zero_only
primary_result_with_uncertainty: V_CH_vs178_plus45.215_plus43.894m_intervals_above_zero_CM_plus4.004_plus3.471m_not_system_increment
mechanism_and_intermediate_findings: shape_constraint_reduces_raw_derivative_damage_does_not_prove_raw_velocity_noise_root_cause
errors_heterogeneity_and_boundary_conditions: CH_new_harms815_709_pure577_496_CM92_82_pure69_60
robustness_sensitivity_and_alternative_explanations: independent_native_math_controls_full_metrics_and_zero_fallback_conditions_verified_no_physical_truth_claim
new_F: F293
new_E: E196
new_M: none
next_skill: deep-read_Mode_C
no_automatic_new_run: true
```
