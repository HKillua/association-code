# 183：原生运动条件信息与历史候选风险读出

2026-10-31；REC-NATIVE-MOTION-FEATURES-183 → SQ-NATIVE-MOTION-FEATURES-183，research-pipeline-contract v1.0。上游[Idea183/plan1.0](../proposals/native_motion_interventions_20261031.md)、Mode C1.93/C63/F288–289/E191–192、[独立来源AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/native_motion_transfer_20261031.md)。探索性计算实验，execution_status=completed、科学/规定独审各一次真实成功、evidence_status=verified。不是全T2100/全V300或独立泛化确认，充分108＝72+36不增。

## Record：问题、实际操作与数据血缘

问题：T纯单支持历史误选是否缺少原生运动条件信息？不预设182的相关特征就是根因，不把报告自一致当真值。固定178原HARD/STRUCT组、分数/阈值、91系数ridge和1505参数神经风险头；在原12维后加入182事先定义的13维原生时刻/报告外推/局部速度/双侧条件，显式raw kn一次转m/s。13维不含绝对位置或源名；只对单支持query重新选历史候选。零/多支持和全部速航、分区/coverage/关联保持O，任意K接口不退固定源。

T128沿原178特征/参考监督/first-anchor标签；T32复用127 native，其余96只从151同archive提取指定T原生输入，绑定node/query/native count，不新增reference或V。当前组≥2节点且count1有标签用于训练；144061候选行、源等质量权重合27717.5，不把两前端视图或重复抽样当独立样本。T32十二leave＋full共13fold，从全部T128排除同组件；前端已经见过T标签，不称全链LOCO。

每fold冻结相应178头，学习y−old_score残差。25→32→32→1的新1921参数头：F0/OLDINFO把新13标准化维置0，F1/NATIVE保留；共享T-only normalization/初始参数、每步相同4096节点抽样、256步seed20/Adam .001/wd1e−4/clip1/log误差MSE。26新fit/6656真实GPU更新，0新ridge/前端更新/teacher。全T模型冻结后才V，固定末步不挑V；部署风险残差1505+1921＝3426，新增1921，不是全流水线总参数。

原T32/860对象、V32/1427对象，base/O/F0/F1各Mean+NN、两前端共16臂；全部困难、缺失和重复保留。普通Mean及原base/O均完整强控，不将不同前端冠军拼成方法。L2S与原request为完整实际命令/配置，不另补注册。

## Record：完整主要结果与不确定性

单位米，所有各自/共同人口、全部场景/组件/四属性原件保留。

|策略|T MAE|V MAE|V RMSE|V P99|
|---|---:|---:|---:|---:|
|HARD base|454.615|452.992|1502.760|5166.150|
|HARD 178 O|392.233|355.760|1329.100|5003.510|
|HARD OLDINFO|391.629|355.830|1329.985|5021.982|
|HARD NATIVE|390.090|359.395|1340.552|5000.568|
|STRUCT base|430.087|427.241|1447.519|4842.083|
|STRUCT 178 O|367.202|328.916|1267.936|4359.930|
|STRUCT OLDINFO|366.289|329.256|1268.536|4369.058|
|STRUCT NATIVE|365.789|330.819|1275.560|4444.897|

普通Mean各副本exact：HARD T1202.191/V1201.217m，STRUCT T1227.212/V1290.534m。HARD家族V71396/72196点＝98.891905%、1401/1427对象、重复203、decoded跨源mPCA81.759255%/mPFA14.924344%；STRUCT70775点＝98.031747%、1380对象、重复104、81.765139%/15.573316%。分数/组/全部raw-decoded与速航exact不变，不宣覆盖/关联/四属性学习改善。

|V共同有效点MAE差|点估计m|2000次seed20组件描述95%区间|
|---|---:|---:|
|F0−O HARD|+0.070|[-1.060,1.701]|
|F1−O HARD|+3.635|[-1.271,4.335]|
|F1−F0 HARD|+3.565|[-2.786,4.412]|
|F0−O STRUCT|+0.340|[-0.298,1.707]|
|F1−O STRUCT|+1.903|[-0.448,17.294]|
|F1−F0 STRUCT|+1.563|[-2.236,17.558]|

T12/V9组件，V复用开发域且T/V共享身份，区间非独立新域确认。F1−F0的T差−1.540/−.500m亦均跨0；不能用T小幅下降证根因。去最大组件V F1−O仍+.283/+6.232m，F0−O+.808/+.999m；F1−F0 HARD变−.525、STRUCT仍+5.233m，异质性和不确定性保留，不宣普遍恶化或某组件必因噪声。

## Squeeze：动力学、尾部与两种“新增害”定义

26条曲线/6656更新均有限。F0/F1各3328步，clip前梯度最大.073929/.093063，>1均0。full随机batch第1→256步loss F0 .070045→.063548、F1 .070045→.051064；批次变化、没有固定探针或优化器重放，不称收敛或过拟合根因。

|V新策略相对178 O|新增>1km增害|改善>1km|纯组新增害|最大增害m|
|---|---:|---:|---:|---:|
|H F0|40|27|12|11072.519|
|H F1|169|50|32|5654.717|
|S F0|34|18|15|11072.519|
|S F1|95|45|15|6766.217|

全部变化位于单支持域；F1该域平均差+8.685/+5.320m，纯组全部有效人口平均+.171/−.476m，不能把全部损害归为混组。F1−F0仍新增168/92害、救64/58；零/多支持和速航strict0。

**相对178的增害**与**相对最初base跨过1km门槛的点集**不同，必须同时看。以原175/173 base为共同门槛，V原178害128/107，F1降100/80，纯组28/19降14/8；F1救旧36/28，但引新8/1。T原104/75降83/65，纯组22/20降18/17。这个条件尾部信号存在，但并未防住相对O的169/95新害，均值/RMSE退化；两套统计分母/参考策略不同，不矛盾、不择优宣高害已解。F0 V害135/104，纯28/16；F1不是仅额外容量的同样行为。

DE：原生信息在当前容量/曝光/目标/动作域下，能改变部分旧误选，但未形成对匹配旧信息头和完整178的稳定系统净收益。HARD P99小幅下降，STRUCT P99上升，RMSE两头升，不能拼单列冠军。AI/UN：报告噪声、信息与目标/动作条件、组/参考对应和有限训练的交互仍未分开；不判原生运动无价值，不否定整类物理条件风险方法，也不从V再设门限/加步。

## Record：真实服务、完整核验、成本与计划偏差

科学冻结312b123，新association-code-run183及原run `runs/LEVEL1-NATIVE-MOTION-FEATURES-20261031-183-attempt-1`；[L2S183](../../HandOver/L2S-20261031-183.md)原命令。CPU0–3×4/GPU0 UUID兼容共卡，程序10:48:05.287346Z–10:57:19.093916Z、553.807467s、RSS1928024064B、Torch峰值显存62759424B；launcher payload10:47:57.625098Z–10:57:20.466437Z、562.979401s。科学unit mtad-1b97959ddccf.service真实exit0/inactive/dead/Resultsuccess/Restartno。

规定独审新冻结3c3e0e3/association-code-audit183，CPU16–19×4、无GPU/1800s advisory；payload11:02:18.963489Z–11:04:19.724270Z、launcher120.922287s，unit mtad-b06447c04d3a.service真实exit0/inactive/dead/Resultsuccess/Restartno。主位置/人口/关联/尾部/组件/primary CI独核51.484362s；四属性同评价器保存数组完整记账19.279842s，**非独立四属性双实现**；训练/动作/域独核36.104323s、RSS870596608B，144061行/13fold/norm/同抽样/26末步/曲线重建，独立NumPy与GPU评分最大差4.991889e−7。原生特征提取共用nc实现，不冒称其物理真值已认证、优化器重放或作者复现。

合科学＋规定收尾683.901688s，不是部署延迟。全部98公共原件38548944B、tar5984855B＋真实服务/传输清单回[183原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_native_motion/183)，逐文件数/大小核齐；坐标/身份/权重/原生参数/完整预测与scores留server。15软件检查含复用项不当15项新科学证据。无科学/独审失败或配方/样本/训练量变化；审计代码比科学晚，仅包含桥接/持续性list修复和187/188显式资格接口，不改变183科学输出/特征/评分。184/185旁路失败和186资格另外完整记账，不能从本次683.902s内消失或改为183训练失败。封存test未访问，离线canonical_all/真实主要K2，K3只接口测试。

**保178平均收益，183当前F0/F1配方不晋级、不自动全300/新seed/加步。** 保留原生信息减少部分base门槛旧害这一线索，但不据V拼策略；与独立双侧/单侧结果分别判断。交Mode C1.96/F292/E195，SR v3.7；未新增来源/机制已证或科学连接。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-NATIVE-MOTION-FEATURES-183
input_experiment_record_ids: [REC-NATIVE-MOTION-FEATURES-183]
source_idea_id: IDEA-NATIVE-MOTION-FEATURES-183
plan_version: '1.0'
evidence_status: verified
claim_scope: T128_13fold_26fit_6656_GPU_updates_T32_V32_16arms
primary_result_with_uncertainty: V_native_vs178_plus3.635_plus1.903m_and_vs_oldinfo_plus3.565_plus1.563m_all_intervals_cross_zero
process_dynamics: finite_gradients_no_clip_no_optimizer_replay_or_convergence_claim
mechanism_and_intermediate_findings: fewer_base_threshold_old_harms_but_new_O_relative_harms_and_worse_mean_RMSE
errors_heterogeneity_and_boundary_conditions: V_native_new_harms169_95_pure32_15_zero_multi_coverage_relations_speed_heading_exact
robustness_sensitivity_and_alternative_explanations: frozen_controls_component_analysis_numpy_scores_sampling_folds_actions_verified_no_physical_truth_claim
new_F: F292
new_E: E195
new_M: none
next_skill: deep-read_Mode_C
no_automatic_new_run: true
```
