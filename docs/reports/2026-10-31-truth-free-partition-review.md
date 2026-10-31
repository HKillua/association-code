# 165：固定证据下无真值分组修正 — Record→Squeeze

2026-10-31。真实原T32/V32六臂一次exit0、限定独立输出/J核验与四属性记账通过。REF＋冻结小网络在V有平均及尾部信号，但T完整MAE略差、V丢2目标/新增140个>1km增害点，**暂不晋级或自动扩300**。保留有效分组条件的研究入口及原159平均候选，不拼局部冠军。

## Record

EXPERIMENT_RECORD `LEVEL1-TRUTH-FREE-PARTITION-20261031-165-attempt-1`，source_idea_id `IDEA-TRUTH-FREE-PARTITION-REFINEMENT-20261031`、plan1.0 / research-pipeline-contract v1.0，本聊天执行。来源Mode C1.72/C53/P05/P14/F260/M69/E163与114原操作反例；[事前Idea/spec](../proposals/object_conditions_context_20261031.md)、[L2S165](../../HandOver/L2S-20261031-165.md)。假设AI：原WS merge-only没有split/move出口，允许利用固定signed反证修正成员可能有益；竞争解释是J与真实物理/coverage目标不一致、过拆/支持域/anchor变化。

冻结d379bba/new checkout165，科学run与launcher attempt1。T用125保存scores及127原pairs，V用056 association/final-scores的node_ids/pairs/scores；完整原WS由同分数重解绑定membership。0teacher/新score/阈值/训练/rawCSV/test，模型输入不读identity/reference/label。分组API只接受完整scores/pairs/原membership，允许任意源数，源名不参与搜索；真实仍K2。

REF从WS开始，逐次比较全部节点→现存组或新singleton的ΔJ＝到新组signed和−到旧组其他成员signed和；取>1e−10最大正增益，同分按原节点/组最小索引，最多4N move。全部节点一次归属，保留每次move/J/tie；不保证全局最优、源置换下精确ties不声称唯一输出。CL为原114保守强控；三分区各真重建source聚合/10特征/普通Mean，以同冻结159 T组件留出/full网络打分恢复，WS_NN真实物理输出与159exact。每场全部六预测/分区存盘后开评价真值，官方anchor_identity_first只在评价使用，不读truth选组/anchor。

原T32/12组件860目标、V32/9组件1427目标全部保留，单源/零支持也输出；分区改变后各自及共同有效人口分开，不能把丢D2或新first-anchor当选源救回。分组变化影响位置、speed/heading的Mean与零支持回退，不称它们的新学习。普通WS_MEAN是node支持均值，**不同于source等权SOURCE_EQUAL**，不混称同一强控。训练网络仍1441参数＋冻结66系数底座，当前不是完整系统参数/延迟。

北京时间11:03:52.103–11:05:53.268，程序121.164377s/launcher138.586257s，RSS799735808B/CUDA24143872B；GPU0兼容共卡/CPU0–3×4/RSS8GiB/1800s advisory。原unit `mtad-d5cdf809afc3.service`真实inactive/dead/success/ExecMainStatus0/Restart=no/Linger=yes。完整实际argv/env与输入路径、退出/服务/64逐场与组件在[165脱敏原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_object_conditions_context/165/)，坐标、成员、逐点预测/分数/权重留server。

| 同组状态，米 | MAE | RMSE | P95 | P99 | 有效/可用位置点 | 预测组/重复/漏轨迹 |
|---|---|---|---|---|---|---|
| T WS_MEAN | 1311.271444 | 2878.594743 | 2544.797341 | 10480.821469 | 37635/38224 | 911/86/35 |
| T WS_NN | 597.178960 | 2654.205915 | 2344.415284 | 10500.826733 | 37635/38224 | 911/86/35 |
| T CL_MEAN | 1305.701734 | 2913.929615 | 2801.781718 | 11575.817852 | 38147/38224 | 1011/155/4 |
| T CL_NN | 654.135428 | 2709.305035 | 2624.311488 | 11594.618274 | 38147/38224 | 1011/155/4 |
| T REF_MEAN | 1311.009405 | 2878.608337 | 2539.653624 | 10637.304285 | 37722/38224 | 912/85/33 |
| T REF_NN | 597.588040 | 2654.584221 | 2346.411270 | 10637.304285 | 37722/38224 | 912/85/33 |
| V WS_MEAN | 1276.328389 | 2168.265557 | 3103.794978 | 7194.824823 | 71071/72196 | 1605/217/39 |
| V WS_NN | 527.427043 | 1813.298607 | 2942.040400 | 7260.393881 | 71071/72196 | 1605/217/39 |
| V CL_MEAN | 1278.511960 | 2370.757600 | 3399.791415 | 9231.506340 | 71946/72196 | 1762/344/9 |
| V CL_NN | 617.791411 | 2091.537585 | 3396.576034 | 9213.351455 | 71946/72196 | 1762/344/9 |
| V REF_MEAN | 1279.575249 | 2163.402943 | 3042.668127 | 7087.593225 | 71135/72196 | 1600/209/36 |
| V REF_NN | 517.947017 | 1801.182250 | 2840.382499 | 7166.747110 | 71135/72196 | 1600/209/36 |

| 分区变化，候选−基线 | 共同域MAE差/m [组件95%描述区间] | 新增/丢位置点 | 救回/丢目标 | 实际增害>1km/最大增害m |
|---|---|---|---|---|
| T REF_NN_vs_WS_NN | -4.149141 [-6.181232, 0.000000] | 87/0 | 2/0 | 19/2327.312754 |
| T REF_MEAN_vs_WS_MEAN | -3.502367 [-5.562994, 1.926383] | 87/0 | 2/0 | 123/2327.312754 |
| T CL_NN_vs_WS_NN | 53.714713 [-17.733512, 143.878680] | 563/51 | 33/2 | 875/28765.186111 |
| V REF_NN_vs_WS_NN | -9.246400 [-10.692408, 0.431011] | 163/99 | 5/2 | 140/13080.814365 |
| V REF_MEAN_vs_WS_MEAN | 3.049812 [-0.777427, 3.962083] | 163/99 | 5/2 | 759/13080.814365 |
| V CL_NN_vs_WS_NN | 93.035954 [37.554930, 258.744567] | 989/114 | 34/4 | 2284/39496.394612 |

## Squeeze：平均、覆盖与尾部不能合成一个胜负

SQUEEZE_REPORT `SQ-TRUTH-FREE-PARTITION-20261031-165`，来源上列Record，开发域/机制partial；限定保存输出和J操作已独立复核，不能认证物理truth、评分头推理或作者算法复现。

REF T改变3场/20move/J+40.401826，V改变8场/65move/J+107.902191，0预算耗尽，均达本次单节点局部最优。原raw分数/决策不变；decoded关系如下，不把raw/decoded混报：

| 关联，百分比 | all宏F1 | MSTSA宏F1 | 跨源mPCA | 跨源mPFA |
|---|---|---|---|---|
| T raw | 84.868918 | 82.641036 | 90.340194 | 18.099076 |
| T WS | 87.800621 | 86.639787 | 84.124794 | 9.679907 |
| T CL | 84.789272 | 83.068617 | 77.213493 | 7.990443 |
| T REF | 87.858077 | 86.707270 | 84.176777 | 9.600604 |
| V raw | 79.851073 | 76.939253 | 89.709923 | 27.055822 |
| V WS | 84.314318 | 81.350586 | 78.989324 | 15.550185 |
| V CL | 80.916555 | 77.628476 | 72.322797 | 14.480325 |
| V REF | 85.233430 | 82.218967 | 79.916089 | 14.756457 |

REF_NN V各自人口MAE527.427→517.947m，RMSE1813.299→1801.182m、P99 7260.394→7166.747m；但共同域527.905→518.659m，差−9.246400m，组件区间[−10.692408,+.431011]跨0。新增163点/5目标同时丢99点/2目标，不能只报净增64点/3目标。T共同域−4.149141m，却因新增87点/2目标的误差较大，完整MAE597.179→597.588m/P99+136.478m。CL虽然覆盖更多，NN的完整误差和尾部明显更差，不把CL当有效替换。

**同分区下原网络平均收益仍在。** REF V普通Mean1279.575m→NN517.947m；WS1276.328→527.427m，T也保持相对Mean优势。REF_MEAN V共同域反而+3.049812m，不能说更优J对任何融合器必然有益，也不能将分组与评分头两个收益混成单因果。V同anchor70776共同点平均−5.530821m，changed-anchor196点平均−1350.949739m；T同anchor37630点−4.043208m、changed5点−801.404141m。既有成员变化作用，也有对应代表变化，不全归某一路径。

V组件仅2改善/1变差/6相同；坏组件0b9d7d23共同MAE+5.978146m，最大23608a7a−10.968944m、c7082a69−2.297727m；去最大后仍−1.081174m，但不能据此宣稳定。T唯一非零改善在最大组件，去最大差0。全部场景符号、所有组件/去最大及七个主要比较的2000次seed20组件区间留原JSON；T/V为反复开发域，非独立多seed/新域确认。

### 新增大害的支持条件（科学后只读深挖）

`squeeze-group-accounting.json`及完整`squeeze-source.py`从原冻结输出/原native支持和身份只读重算，无新预测/fit。程序3.733731s/RSS137113600B，实际命令 `/usr/bin/taskset --cpu-list 0-3 /home/wangguangjie/djs/mtad-field-20261016/bin/python /tmp/mtad-squeeze-group165.py`；这是后验异质性分析，不倒填为事前机制验证。全共同域差/大害计数与科学结果复核。

- T新增19个>1km增害：全部same-anchor/成员改变、新组只有1源支持；V140个全部same-anchor/成员改变，其中新零支持22/单源118/多源0。**这些时刻hard评分器不作两源以上的选择，不能继续把这140点归为小网络多源选错。** 组变化→可用支持/回退改变是已定位操作条件，真实物理误差成因仍未知。
- V原纯→新纯也有60个大害，原混→新纯24、原混→新混55、原纯→新混1；T为7/1/2/9。纯度提高或保持不保证原生支持/物理输出安全，不能只用mixed-group减少评价好坏。
- V最坏例scene3772/q3，pure→pure/same-anchor，源支持1→0，近邻约2.002381min；原误差212.724m→13293.539m、增13080.814m。T最坏scene35/q58，mixed→pure/0→1，近邻.099730min，315.380→2642.693m。只是脱敏例子，不据此设时长阈值/删点或判定数据坏。
- 新覆盖V163点MAE207.904905m、丢失99点原MAE184.437678m；T新87点MAE2569.411708m/RMSE4907.991913m/P9913703.092187m。补覆盖也可能纳入难点，应保留它们并分开解释完整与共同误差。

| 分组后的其他属性 | speed MAE/mps | heading MAE/degree |
|---|---|---|
| T WS_NN | 23.363639 | 72.404521 |
| T REF_NN | 23.371708 | 72.418728 |
| V WS_NN | 23.310266 | 76.619756 |
| V REF_NN | 23.259543 | 76.504577 |

全部经纬/速度/航向、observed/missing及论文量完整保存；ME不冒充MAE，ESD未认证。网络本身未学习speed/heading，新分组仍会改变这些输出和人口。

### 核验、偏差和决策

独立输出/J核验171310exact/26287numeric/252null通过，程序11.799493s/RSS141639680B、launcher12.266175s；核完整原registry/first-anchor/合法分区、保存主位置/覆盖/尾部/组件/主要CI、每步实际score-J增益，**不重新前向159/认证truth/独立重放全部关联指标或四属性**。四属性同评价器记账程序7.356316s/RSS403501056B、launcher10.170940s，单独计成本。全部服务真实成功终态。

事前无科学重试/改阈值/删样本/V选分区；扩展软件fixture初版变量位置错误在READY前修正，合法覆盖变化fixture只验链路。后验支持/纯度深挖单列，不冒称预注册或独立新科学重复。固定输入/分数但搜索及终态组改变，J收益不等唯一物理机制；组件留出仅后端风险头，056前端没有重新按本轮组件留出；真实K2、离线全窗/旧anchor回退、T/V共享域仍是限制。

保留REF＋NN的条件信号和原小网络平均收益，不自动扩300或接主方法。下一连接需把**对象一致性与按query保留观测支持/回退**一起检查：同score目标下更纯/更高J仍可破坏支持；应由T反例决定合法支持条件/候选的独立Idea，再在全人口/同候选预算下比较，不能V追阈值或部署固定源优先。已有MeanTraj存在/匹配的成熟先例提供发展基础，不以先例否定价值，也不宣本领域首次。

```text
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-TRUTH-FREE-PARTITION-20261031-165
input_experiment_record_ids: [LEVEL1-TRUTH-FREE-PARTITION-20261031-165-attempt-1]
claim_scope: fixed056_truth_free_search_original_T32_V32_six_arms_not_global_or_peer_win
primary_result_with_uncertainty: V_common_minus9.246400_CI_cross0_with_coverage_losses_T_full_worse
process_dynamics: zero_updates_T20_V65_moves_all_local_optimum_no_budget_exhaustion
mechanism_and_intermediate_findings: all140V_new_harms_in_new_zero_or_one_source_support_not_multi_source_score_decisions
errors_heterogeneity_and_boundary_conditions: same_anchor_all_harms_pure_to_pure60V_lost2targets_full_component_table
robustness_sensitivity_and_alternative_explanations: CL_mean_controls_components_drop_largest_same_changed_anchor_and_independent_J_output_audit
new_P: none_P05_P13_P14_refined
new_F: F262
new_M: none_M69_M107_M108_conditions
evidence_index: original_165_all64_rows_summary_audit_attributes_posthoc_source_and_launcher
data_quality_and_provenance_gaps: K2_development_domain_no_truth_certification_or_full_forward_independent_replay
open_questions: truth_free_identity_and_query_support_conditions_residual_pure_errors
next_routes: deep_read_Mode_C1.73
next_skill: deep-read Mode C
```
