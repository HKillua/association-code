# 192：采样密度假设的T侧资格与反证

2026-11-01，REC-SAMPLING-MEASURE-QUALIFICATION-192 → SQ-SAMPLING-MEASURE-QUALIFICATION-192，research-pipeline-contract v1.0/workflow v2.0。来源[Idea192/plan1.0](../proposals/sampling_measure_qualification_20261101.md)、MC2.03/C67/F299/E202/M121 AI、[代码/条件AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/sampling_measure_conditions_20261101.md)。computational diagnostic，execution_status=completed/evidence_status=verified，科学一次真实exit0；**没有新部署预测/目标误差，因此没有新方法成绩或目标指标独审**。

## Record：目的、输入、实际操作

当前MC按有效报告数等权估速度。检验T原生不规则密度是否真实存在、是否值得据此替换均值，并核输入工作可靠性及旧害/救回动作规模。不是重做137内点九rho/156历史专家选模，亦非190端点位置拟合。原189完整T资格绑定到178/127native cache；原T32/2310片段/580384事件、12组件，0V/封存test/新神经或reference/似然fit/teacher。读取旧189保存错误行/selected-node diag，仅评价条件关系，不进入新预测（本项没有新预测）。

每片段显式kn→NE m/s，分别计算报告数均值与原生时间梯形积分均值、权重TV/有效样本量、间隔CV/极值、缺报告/时段、速度离散度和经验均值标准误工作量。原178候选/tie/权重不变，所有T451/344主eligible与原34/1害、3/33救回精确绑定，原μ速度/离散/OLS差/n/span同原189数值一致。

节点权公式与独立区间积分表达在全部2310片段逐项核同，均2310合格/无报告缺失，0片段被删。该核验只认证代数/输入与保存病例，不认证真实噪声/参考或物理准确性；没有第二科学/audit作业。

## Record：全T输入结果

|T2310片段描述|均值|中位|最大|
|---|---:|---:|---:|
|事件数|251.248|200|719|
|可见时段秒|3199.564|2761.985|7190|
|间隔CV|0.000027613|0.000024217|0.000049924|
|最大间隔/中位间隔|1.000042551|1.000049526|1.000099877|
|均值速度表达差m/s|0.164693|0.073751|10.620496|
|报告速度离散度m/s|31.735534|33.506154|2124.424153|
|iid标准误工作量m/s|2.541341|1.912747|130.175085|

**DE：片段内部几乎均匀采样。** 当前数据不支持“原生不均匀采样密度导致34/1害”的候选解释；极小间隔变化也可能只是时间量化/浮点表示，来源未核，不把它认证为真实自适应采样。所有输入都保原件，没有从数据中挑均匀子集。

两种均值仍不相同，不构成不均匀密度解释的证据。在严格等距条件，梯形均值−算术均值＝[算术均值−(首速度＋末速度)/2]/(n−1)，所以端点半权本身就会改变结果。这里尚未独立分解实际差的端点与间隔残差，不能精确宣差异全部由端点造成。全体时间权有效n中位199.501，较原200略小，不能预设时间权更降噪。

## Squeeze：旧病例/所有主点与可靠性边界

|T人群|query数|不同组数|两均值速度差中位m/s|名义差×时距中位m|iid标准误工作量中位m/s|
|---|---:|---:|---:|---:|---:|
|HARD/all|451|71|0.080333|103.927850|2.385150|
|HARD/PG_harm|34|2|0.134128|418.942012|2.385150|
|HARD/PG_rescued|3|1|0.145200|189.907080|3.206175|
|STRUCT/all|344|63|0.080333|94.402050|2.190103|
|STRUCT/PG_harm|1|1|0.058977|167.288195|1.630346|
|STRUCT/PG_rescued|33|2|0.098330|268.315167|2.280792|

HARD34害只来自2个组，3个救回来自1组；STRUCT1害1组、33救回2组。条件高度相关，不能把点数当独立案例或据此训练门槛。名义差×真实时距是Σ质量×速度表达差模×|dt|，不是新的球面预测距离或误差改善，更不能认领“已降低419m”。输入差会有作用空间，但没有方向/收益认证。

旧HARD害的经验标准误工作量中位2.385m/s，救回3.206m/s；STRUCT害1.630、救回2.281m/s，不能凭“标准误大就回退”分开害和救回。报告离散度大及部分极端值是可见输入事实，不认证真实传感器噪声、运动突变或生成故障；方差/n依赖独立同分布，未核自相关/偏差，更不是可直接融合的校准Cov。

## Squeeze：回答了什么、如何取舍

已回答：全部原T内部间隔近均匀，无缺报告，当前不均匀密度假设缺目标证据。算术均值和时间均值虽有差，但不能据此把时间加权当作解决本问题的成熟方案。**关闭“修不规则采样密度”的当前解释/自动时间权验证，不为凑实验数继续上V。** 这不是否定时间连续模型、积分思想或适配价值，也不因公式有先例拒绝研究。

仍未解：恒定平均运动为何在纯组/单片段发生反转；报告时序、位置增量相容性、经验噪声/偏差与未知参考的关系。下一从时间顺序信息和运动条件本身找依据，不只沿同一均值权重/窗口/β补丁推进。178及MC/PG已有条件平均价值保留，不以本次资格替换输出、改coverage/关联或认定根因。

## Record：真实执行、资源与偏差

冻结3f0ba9c，新association-code-run192/原run `runs/LEVEL1-SAMPLING-MEASURE-20261101-192-attempt-1`；实际命令、环境、一次额度见[L2S192](../../HandOver/L2S-20261101-192.md)和原request。CPU8–11×4无GPU/RSS4GiB/1800s advisory/Lingeryes/Restartno，unit mtad-0983808a95d7.service真实exit0/inactive/dead/Resultsuccess/Restartno。程序UTC2026-10-31 21:11:20.542915Z–21:11:28.515699Z（北京时间11月1日05:11），7.972850373s/RSS434970624B；launcher21:11:11.260525Z–21:11:29.260875Z、18.158249401s，准备/输入/统计/写盘在内，不当部署延迟。

3项READY前软件检查覆盖不规则线性速度解析积分/同函数插值加密/时间单位缩放、常速/2点/缺失/严格时间以及保存旧候选权重的实际join；通过后发布，py_compile/diff通过。科学成功，无科学重试、删片段、改公式/预算、V调参或新预测。没有额外计划外目标评分；不存在把诊断数当训练数。

45公共原件158650B/tar93385B及真实终态/传输清单回[192原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261101_sampling_measure/192)，逐文件数/大小核齐；坐标/身份/两均值向量/完整逐点条件留server private。原152/178/188/189/190/191和失败均不改。Record→Squeeze→Mode C2.04/F300/E203、SR v4.5/OD v1.18；充分108不增，无自动193/全300/seed/新READY。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-SAMPLING-MEASURE-QUALIFICATION-192
created_at: '2026-11-01'
source_ids: [REC-SAMPLING-MEASURE-QUALIFICATION-192]
squeeze_id: SQ-SAMPLING-MEASURE-QUALIFICATION-192
input_experiment_record_ids: [REC-SAMPLING-MEASURE-QUALIFICATION-192]
source_idea_id: IDEA-SAMPLING-MEASURE-QUALIFICATION-192
plan_version: '1.0'
evidence_status: verified
claim_scope: fixed_T32_2310fragments_580384events_0V_new_predictions_training_or_target_metrics
primary_result_with_uncertainty: within_fragment_interval_CV_max0.000049924_and_gap_ratio_max1.000099877_no_informal_causal_CI
mechanism_and_intermediate_findings: irregular_sampling_density_premise_not_supported_time_vs_count_mean_can_differ_even_at_uniform_rate
errors_heterogeneity_and_boundary_conditions: harm_and_rescue_proxy_distributions_overlap_only2_1_and1_2_groups_no_calibrated_covariance
process_dynamics: one_launcher18.158249401_seconds_program7.972850373_no_training_curve
data_quality_and_provenance_gaps: original_all_T_bound_no_missing_reports_unknown_reference_physical_noise_or_time_quantization
robustness_sensitivity_and_alternative_explanations: all2310_node_weight_vs_interval_integrals_checked_saved_189_current_mu_and_population_bound_no_physical_guarantee
new_P: none
new_F: F300
new_E: E203
new_M: none
mode_C_input_bundle: P13_P14_F300_M121_still_AI_E203_C67_density_route_not_established
evidence_index: original_run_summary_all_T_scenes_launcher_and_terminal_manifest_linked_above
next_skill: deep-read_Mode_C
no_automatic_new_run: true
```
