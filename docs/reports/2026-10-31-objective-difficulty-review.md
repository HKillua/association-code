# 162：监督目标×训练难例质量 — Record→Squeeze

2026-10-31。52个真实GPU fit/6656更新、输出独审与计划内记账各一次exit0；四种新配方均未稳定超过冻结159。保留原小网络的平均收益，不自动加训练量或扩大V300。

## Record

EXPERIMENT_RECORD `LEVEL1-OBJECTIVE-DIFFICULTY-20261031-162-attempt-1`，source_idea_id `IDEA-OBJECTIVE-DIFFICULTY-SEPARATION-20261031`，plan1.0 / research-pipeline-contract v1.0，计算pilot、本聊天执行。来源Mode C1.70/SR-C1/F256/M108/E159；[事前Idea/spec](../proposals/rare_reversal_separation_20261031.md)、[L2S](../../HandOver/L2S-20261031-162.md)。假设AI：绝对风险/相对排序目标与难例训练质量分配的交互可能减少罕见错误；主要检验平均、误选、尾部及完整覆盖，不能只看少数救回。

独立冻结0da7496、checkout162、science/launcher attempt1。输入完整复用127 native、125 value、114 WS、133 source-risk、155 T128与159表达结果及level1-scene3000 manifest；[run.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_rare_reversal_separation/162/run.json)与原launcher `request.json`保存全部绝对路径、实际argv/env，输出前保存预测后开评价reference。0新teacher/rawCSV/test、0重训159/二次头。T128/28组件、166058源标签；原T32的12组件留出/full共13fold。full实际83029源对，旧二次头错957/易82072；各outer训练内独立定难度，13fold全部有两类、0fallback。不是OOF难度/物理健康标签，不取V定权。

共享旧10维、冻结159 T-only标准化clip±8/66系数二次基底、1441参数10→32→32→1残差网。fresh四臂同seed20/初始化/每步2048源对索引、128更新/Adam .001/weight_decay1e−4/clip1；每对两源都forward，query内无序对权1/C(K,2)。ABS为两源log误差MSE平均，PAIR为相对分数BCE，误差平局target .5。BALANCED仅改loss质量，使训练内难/易各占一半原query质量，不改变采样和源角色。全52fit共13631488源对/27262976源呈现。159旧网为已训256步/batch4096源标签的冻结强控，**与fresh128步不是同预算**；四臂内才是等预算因子比较。没有早停、按V选步数、恢复、追加或新seed。

原T32：37635/38224位置点、860目标/911预测组/86重复/35漏轨迹；V32：71071/72196、1427目标/1605组/217重复/39漏轨迹。D2是组×query多源可评分人口，T22627/V39847，不等同去重复后的主位置点。全部八臂固定056 raw分数/114 decoded WS组，未改变关联；0/1源域及speed/course仍旧输出，不认领这些模块增量。匿名slot反转层仅事后分层，未用作特征、采样权或部署优先级，真实仍K2。

北京时间02:02:44.528–02:05:49.870，程序185.341320s/launcher189.697753s，RSS1389047808B/CUDA峰78602752B。GPU1 UUID6811d9c0-b8a4-b5fd-0175-790964d42afc兼容共卡、CPU8–11×4/RSS8GiB/3600s advisory。原unit `mtad-fcfd2b696c6d.service`真实inactive/dead/success/ExecMainStatus0/Restart=no/Linger=yes。所有科学/独审/补项服务成功终态，未触碰152活动目录/他人进程。

| 同点位置/m | MAE | RMSE | P95 | P99 |
|---|---|---|---|---|
| T SHARED_Q_HARD | 619.476 | 2666.191 | 2436.283 | 10500.827 |
| T T128_Q_HARD | 616.574 | 2665.649 | 2468.556 | 10500.827 |
| T T128_RESIDUAL_HARD | 597.179 | 2654.206 | 2344.415 | 10500.827 |
| T ABS_UNIFORM | 597.171 | 2654.199 | 2344.415 | 10500.827 |
| T ABS_BALANCED | 600.417 | 2656.138 | 2361.101 | 10500.827 |
| T PAIR_UNIFORM | 604.677 | 2659.396 | 2408.234 | 10500.827 |
| T PAIR_BALANCED | 647.443 | 2687.389 | 2694.243 | 10541.991 |
| T SOURCE_EQUAL | 1311.223 | 2878.589 | 2544.797 | 10480.821 |
| V SHARED_Q_HARD | 547.721 | 1829.356 | 3125.980 | 7260.394 |
| V T128_Q_HARD | 540.295 | 1824.317 | 3094.506 | 7250.750 |
| V T128_RESIDUAL_HARD | 527.427 | 1813.299 | 2942.040 | 7260.394 |
| V ABS_UNIFORM | 527.447 | 1813.304 | 2942.040 | 7260.394 |
| V ABS_BALANCED | 527.987 | 1813.737 | 2956.659 | 7260.394 |
| V PAIR_UNIFORM | 535.208 | 1821.391 | 3077.702 | 7260.394 |
| V PAIR_BALANCED | 576.356 | 1860.193 | 3379.351 | 7298.620 |
| V SOURCE_EQUAL | 1276.441 | 2168.137 | 3097.435 | 7194.825 |

## Squeeze：目标、质量权重与平均/尾部效应

SQUEEZE_REPORT `SQ-OBJECTIVE-DIFFICULTY-20261031-162`，来源上列EXPERIMENT_RECORD；partial机制、完整冻结输出已限定核验。52fit各128步，末端loss均低于初始；full ABS_U .207907→.029373、ABS_B .519252→.369340、PAIR_U .540043→.072281、PAIR_B .659280→.428825。不同loss/权重的数值不可横比，固定步数也不是收敛认证。模型和样本/正则交互仍可成立，不把负结果推广到所有排序学习。

| candidate−baseline / m | T差 [组件95%描述区间] | V差 [组件95%描述区间] |
|---|---|---|
| ABS_UNIFORM_vs_WS_T128_RESIDUAL_HARD | -0.008294 [-0.012357, 0.000000] | 0.019916 [0.000000, 0.023516] |
| ABS_BALANCED_vs_WS_T128_RESIDUAL_HARD | 3.238422 [0.000000, 4.815584] | 0.559919 [0.000000, 0.661130] |
| PAIR_UNIFORM_vs_WS_T128_RESIDUAL_HARD | 7.498353 [1.678978, 10.538601] | 7.781367 [1.369495, 8.666215] |
| PAIR_BALANCED_vs_WS_T128_RESIDUAL_HARD | 50.264106 [25.819724, 129.908682] | 48.928494 [40.362288, 113.058079] |
| ABS_BALANCED_vs_WS_ABS_UNIFORM | 3.246717 [0.000000, 4.827940] | 0.540003 [0.000000, 0.637614] |
| PAIR_UNIFORM_vs_WS_ABS_UNIFORM | 7.506647 [1.678978, 10.538601] | 7.761451 [1.369495, 8.642634] |
| PAIR_BALANCED_vs_WS_ABS_BALANCED | 47.025684 [25.745432, 129.907646] | 48.368575 [39.794450, 113.058079] |
| PAIR_BALANCED_vs_WS_PAIR_UNIFORM | 42.765754 [14.853061, 124.878023] | 41.147127 [31.813175, 111.762111] |

2000次/seed20按组件重抽样，T12/V9组件；反复使用的开发域，非独立确认、非多seed。四臂内目标效应、权重效应及其条件依赖必须一起看：V ABS下平衡+0.540m，PAIR下+41.147m；训练质量平衡不是普遍有益。full难类仅约1.153%却分到50%质量，是本次操作强度而非已证明最优比例。

| 全D2，比较冻结159 | 误选 | 救回/新增 | 选择excess>1km | 实际新增损害>1km/最大增害m | 好/差/平场 |
|---|---|---|---|---|---|
| T ABS_UNIFORM | 154 | 2/0 | 90 | 0/0.000000 | 2/0/30 |
| T ABS_BALANCED | 217 | 4/65 | 134 | 45/4336.904317 | 0/6/26 |
| T PAIR_UNIFORM | 269 | 2/115 | 184 | 94/8160.914562 | 0/14/18 |
| T PAIR_BALANCED | 937 | 10/791 | 830 | 748/5381.933750 | 0/14/18 |
| V ABS_UNIFORM | 292 | 6/11 | 90 | 0/405.450572 | 0/1/31 |
| V ABS_BALANCED | 319 | 10/42 | 103 | 13/3523.011990 | 0/4/28 |
| V PAIR_UNIFORM | 474 | 10/197 | 264 | 174/6761.309385 | 0/13/19 |
| V PAIR_BALANCED | 1536 | 13/1262 | 1307 | 1217/8075.622185 | 0/16/16 |

| 匿名反转层 | 冻结159 | ABS_U | ABS_B | PAIR_U | PAIR_B |
|---|---|---|---|---|---|
| T 错误/154 | 154 | 154 | 150 | 154 | 146 |
| V 错误/290 | 278 | 289 | 283 | 289 | 286 |

冻结159反转错误T154/V278，选择excess合计T230003.218/V261380.724m；ABS_B对应150/283及228104.777/261543.453m，PAIR_B为146/286及213072.279/262876.227m。T的少数救回没有转为V及完整风险的稳定改善。相对fresh ABS_U，ABS平衡使V反转289→283，但全错误292→319、MAE+0.540m；不能只报“救回6点”。PAIR平衡相对PAIR_U反转289→286，但全错误474→1536、实际新增大损害明显。冻结网仍强于当前新配方。

| 去最大组件MAE/m | 冻结159 | ABS_U | ABS_B | PAIR_U | PAIR_B |
|---|---|---|---|---|---|
| T | 939.047065 | 939.047065 | 939.070048 | 945.077346 | 1003.874554 |
| V | 684.110515 | 684.110515 | 684.110515 | 687.584606 | 774.107920 |

完整每场/全部组件、纯混组/角色、支持层及去最大表保存在原JSON/gzip，未删不利行。零支持T1487/V2770点所有方法完全不变，V before1655/after1115，不解决零域回退。完整四属性含经纬度、speed/course与方向分层；speed MAE T23.363639/V23.310266m/s保持，航向也保持。ME不冒充MAE，印刷ESD仍未认证，不形成论文胜负。

独立保存输出审计268850exact/67968numeric/336null通过，程序13.917251s/RSS134881280B、launcher14.395633s；范围为完整主位置/人口/尾部/组件/主要bootstrap和不变域，**不独立重训52头/复算训练特征与选源诊断/认证truth**。方向/角色记账9.643537s/RSS144932864B，四属性10.308585s/RSS418889728B，补项launcher23.943113s；各实际调用起止/argv/returncode另存accounting-calls.json，不把程序时间混成全launcher成本。所有脱敏原件见[162证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_rare_reversal_separation/162/)，权重、坐标、身份、大预测留server。

计划偏差：初版READY早于本地fixture链通过；缺133字段修复后完整链通过，**实际提交前已通过**，没有科学失败重试。fresh128步弱于冻结159的预算差是事前约定，不倒填等预算；独审/补项按计划各一次，未以软件fixture充MTAD成绩。V反复开发、T128共源/真实K2、无新域/新传感器确认、零支持旧回退均保留。

DE支持当前目标/难例权重配方没有稳定升级；AI提示风险目标之外的构组/对应及上下文需独立处理，不能将“loss下降/少数T救回”写成机制认证。保留159平均候选、关闭本轮四配方扩大，不以单次失败否定相对学习。新操作必须有新的可辨信息或对应证据，不无依据加权/加步枚举。

```text
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-OBJECTIVE-DIFFICULTY-20261031-162
input_experiment_record_ids: [LEVEL1-OBJECTIVE-DIFFICULTY-20261031-162-attempt-1]
claim_scope: fixed_T128_features_and_WS_original_T32_V32_2x2_pilot
primary_result_with_uncertainty: tables_above_none_stably_beats_frozen159
process_dynamics: 52fits_6656updates_all_final_loss_lower_not_convergence
mechanism_and_intermediate_findings: AI_objective_difficulty_interaction_not_unique_cause
errors_heterogeneity_and_boundary_conditions: rare_common_pure_mixed_and_full_tails_retained
robustness_sensitivity_and_alternative_explanations: component_intervals_drop_largest_budget_difference_fixed_features
new_P: none
new_F: F259
new_M: none_M108_conditions_only
evidence_index: run_summary_Tsummary_all_scene_curves_population_audit_accounting_launcher_originals
open_questions: object_reference_coupling_and_residual_pure_errors
data_quality_and_provenance_gaps: K2_development_scope_no_independent_new_domain_no_truth_certification
next_routes: deep_read_Mode_C1.71
next_skill: deep-read Mode C
```
