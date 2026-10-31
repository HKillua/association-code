# 176：最终指标与覆盖约束的工作点对照未产生新增工作点

2026-10-31，REC-DECODED-COVERAGE-176 → SQ-DECODED-COVERAGE-176；[Idea176/plan1.0](../proposals/decision_operating_point_20261031.md)，Mode C1.83/C59/F280/E183。科学32T/32V十八臂、规定独审/全属性及候选核算均成功终态；原件verified，效应仅限既定开发协议。

## Record：固定输入与单独改动

冻结175已训练模型/全部原分数、WS未知K解码与159物理头，0optimizer/teacher训练，原T32/860参考对象、V32/1427对象，Q64。原WS/HARD/STRUCT×Mean/159六强控分数/分组/物理输出逐场exact绑定175。HARD/STRUCT各33个T跨源raw分位点＋原自身θ＋WSθ，实际各35/共70候选×32场＝2240次T构组，不临时加密。

三规则共享候选、raw mPFA≤.180990764192559与decoded mPFA≤.09679907414944734（均为原WS在T原点）的共同预算；mPFA=FP/(TP+FP)，场等权，有未定义场则整候选不可行，不删除分母。R最大raw mPCA；D最大decoded mPCA；C在D基础上约束有效位置点数、覆盖参考轨迹数不低于该头原点。全部平局同低decoded mPFA→高θ；R→D只换目标，D→C只加覆盖。

原点均在网格且符合共同预算。R相对175还引入共同预算/网格控制，不把这项比较纯归目标。校准raw判定显式使用部署时相同的分数平移，再与WS阈值比较，避免浮点边界两种定义混用。覆盖在T以first-anchor对应及参考valid计数，先与175实际覆盖exact绑定；输入构组不含身份，T监督指标可用身份。阈值/候选分组先落盘、V之前冻结；不沿V扫θ或覆盖底线。

## Record：标定选择与全评价

|头|候选数|共同预算内候选|再加双覆盖后候选|T覆盖底线：位置点/轨迹|原θ|Rθ|Dθ|Cθ|
|---|---:|---:|---:|---|---:|---:|---:|---:|
|HARD|35|2|2|37993/841|1.034393536288261|1.034393536288261|1.034393536288261|1.034393536288261|
|STRUCT|35|3|3|37745/834|1.5152517858726042|1.5152517858726042|1.5152517858726042|1.5152517858726042|

覆盖约束没有排除任何共同预算内候选，因而未激活；不能声称它带来保覆盖收益。D与R选择也相同；在既定网格内，没有找到仅改变选择目标带来的增量。不是一般性证明更细网格、不同预算或新解码都无空间；本轮不自动追扫。

九前端WS/HARD/STRUCT/HR/HD/HC/SR/SD/SC×Mean/159十八臂全部完成。T12/V9组件、2000次seed20重采样；参考/量测/权重/完整输出保持原服务器，封存测试未读。所有R/D/C变体对应原头的阈值、分组、物理评价结果相同。

|V完整链路|有效点覆盖%|覆盖目标/1427|decoded跨源mPCA%|decoded跨源mPFA%|MAE m|RMSE m|P99 m|重复输出组|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|WS＋159|98.4417|1388|78.9893|15.5502|527.427|1813.299|7260.394|217|
|HARD及HR/HD/HC＋159|98.8919|1401|81.7593|14.9243|452.992|1502.760|5166.150|203|
|STRUCT及SR/SD/SC＋159|98.0317|1380|81.7651|15.5733|427.241|1447.519|4842.083|104|

各自覆盖人口不同，共同点评价/获失、0/1/多源、所有属性及新增害仍完整报告；由于本次选择未改变，原175强控的平均/覆盖/尾部取舍没有改善。HARD既有条件价值仍保留，不能将旧收益重算作176新增收益。

## Squeeze：本轮排除了什么

DE：预定共同网格、共同预算下，raw召回、decoded召回与双覆盖约束三种选择没有分离；覆盖约束在可行候选中不激活。因此本轮没有支持“仅校准目标接口即可改善当前联合目标”的实证。当前固定配方结案，不加θ网格或沿V换预算。

AI/UN：原分数/未知K构组的证据、按query的历史支持与实际状态可用性仍是未解条件。不能把无工作点增量写成训练目标、表示或decoder均已最优。177的训练侧事实独立推进，不能拿其oracle选176参数；反例回流后先读与动作空间相符的成熟方法再形成下一Idea。

本轮不新增论文计数，充分107不变；不宣超过TMTAFF/PTFM/DeepAF，不改全V300基准，不把raw与decoded或ME与MAE混用。后继从本轮回流进入新来源/Idea，不延长本配方。


## Record：服务终态、完整原件与成本

科学冻结c003770，新`association-code-run176`，run `runs/LEVEL1-DECODED-COVERAGE-20261031-176-attempt-1`、launcher同前缀`-launch-attempt-1`。payload07:27:38.135529Z–07:36:21.588227Z、launcher523.604597s/程序516.779023s；CPU0–3×4/GPU0按UUID兼容共享，RSS1719685120B、Torch峰值显存22817792B。unit `mtad-53bd1504aa33.service`真实exit0/Resultsuccess/inactive/dead/Restartno/Linger=yes，3600s advisory未超；非仅提交成功。

规定收尾冻结74642a0，新`association-code-audit176`、同前缀`-audit-launch-attempt-1`，CPU16–19×4/无GPU、RSS926547968B；payload07:37:35.742009Z–07:39:26.125774Z，launcher110.507601s。unit `mtad-140820cb132f.service`真实同成功终态、Restartno。科学＋收尾634.112198s；独审62.438544s、全四属性21.609307s、校准/后验19.092438s已包含在收尾内，不重复相加。另一次只读原数组exact等价核3.358988s，明确为事后成本。

70个候选/2240个场景候选的raw/decoded计数、有效参考点/轨迹覆盖、共同预算、网格及选择规则全部独立复算；最大指标差2.22e−16；基于缓存候选分区，不冒称独立实现另一套WS求解器。原输出独审关联差≤1.11e−16/MAE差≤1.92e−9m，完整人口/共同点/新增害/所有组件与区间核齐。四属性完整原件保留；新R/D/C与原对应头在64场的分区、平移分数、Mean/NN全部四状态原输出也逐数组exact。原论文ESD/速度单位和数据独立性边界沿原合同，不因审计通过获得作者复现资格。

96份公共科学/逐场/科学与audit原launcher文件＋1传输/服务/等价检查清单完整回传[176证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_decoded_coverage/176)，逐文件大小一致；私人坐标/身份、模型、大预测及2240候选分区保持服务器。无实际科学故障、重试、候选临时扩展、V调参或数据排除；READY前4项软件验收通过。后验等价核时序明确，不倒填事前范围。

```yaml
package_type: SQUEEZE_REPORT
squeeze_id: SQ-DECODED-COVERAGE-176
input_experiment_record_ids: [REC-DECODED-COVERAGE-176]
claim_scope: frozen_175_T32_V32_18_arms_70_predeclared_grid_candidates
primary_result_with_uncertainty: R_D_C_select_original_thresholds; all_new_minus_own_controls_zero_with_zero_descriptive_interval; no_new_gain
process_dynamics: zero_optimizer_teacher_training;2240_T_clusterings;523.604597_science_plus110.507601_closeout_seconds
mechanism_and_intermediate_findings: coverage_constraint_inactive_in_common_feasible_grid_not_global_optimality
errors_heterogeneity_and_boundary_conditions: all_original_HARD_STRUCT_counterexamples_and_all_attributes_unchanged; V_reused_and_T_seen_labels
robustness_sensitivity_and_alternative_explanations: independent_candidate_count_selection_and_population_recount; exact_saved_output_equivalence; no_auto_finer_grid
new_F: F282
new_E: E185
new_M: none
next_skill: deep-read_Mode_C
```
