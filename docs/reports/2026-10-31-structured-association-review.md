# 173：结构监督有同曝光平均/有效点收益，未达覆盖与关联联合目标

2026-10-31。按独立来源分析→Mode C1.77→171技术失败/172明确修复资格→Record/Squeeze→Mode C1.79→[事前Idea173/plan2.0](../proposals/structured_association_20261031.md)→实际执行→本Record/Squeeze。前置13丢T对象均有相关森林梯度，同时影响15救回/268保留对象，不能把资格当安全坏边分类器。原171失败保留。

**结论**：同信息/容量/初始化/曝光下，结构项相对继续BCE的共同位置MAE−13.827m（组件描述区间[−89.959,−4.464]），有效点覆盖96.888%→98.032%；但跨源decoded mPCA82.939%→81.765%、mPFA14.486%→15.573%，并有933个对BCE新增>1km损害点。相对原WS覆盖仍98.032%<98.442%，救12/丢20目标，故**不整体晋级、不自动全V300**。保留C57训练关系监督的条件平均价值，关闭本次固定配方自动扩展；不据V继续调λ/阈值/步数。成熟迁移不要求绝对原创，但新增价值的范围须据实限定。

## EXPERIMENT_RECORD — REC-STRUCTURED-ASSOCIATION-173

### 实际设计、输入与训练

[L2S173](../../HandOver/L2S-20261031-173.md)明确READY后一次领取，冻结7be3dae/新checkout `association-code-run173`，独立`LEVEL1-STRUCTURED-ASSOCIATION-20261031-173-attempt-1`和同名launch-attempt1，未改变旧活动或失败原件。输入原168 T128缓存/CTX权重和192维normalizer，127 native-motion/evaluation、125/114/133原控制、151额外T evaluation index、159冻结后端及主manifest；完整路径/argv见[原request](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_structured_association/173/launcher/request.json)。T节点对齐、原WS/CTX工作点/分区/物理输出真实绑定通过。

- T128/775705唯一all-pairs；两头各6209参数，共用冻结056 5697参数及原CTX关系输入，后端159不重拟合。两头都从168原CTX末权重起，fresh Adam .001/clip1，seed20同每轮场序，四完整轮128场，**每头512更新/3102820pair呈现，共1024真实GPU更新**。128场每场恰4遍，无删场/earlystop/V选步。
- A原冻结CTX；A′同完整场曝光的class-balanced BCE；A′+B同BCE＋λ1的partial FY。上三角Gaussian ε.1/B4，无/有约束解同噪声，按max(n−k,1)归一。真实T identities/k仅训练/监督，rank0也保BCE；原文B100及负距离嵌入与本适配不同，非作者复现。未知对象数部署仍原WS，不读GT k或标签，不固定源/一对一容量。
- 四scorer同170原T32 raw跨源mPFA预算.1809907642，257分位＋原θ、固定tie，1032候选。θ WS1.7320103112/CTX2.0085254542/BCE1.7320103112/STRUCT1.5152517859，**模型与阈值全部V前冻结**。T32参与前端训练，不是独立标定/前端LOCO；T后端159仍leave-component，V为full。
- 原T32/V32，四前端×Mean/159NN八臂；预测/分数/分区先保存才读取V标签评价。中央T860/V1427目标，V可用参考位置72196/91328，19132无参考点依原协议不在位置分母，所有臂相同、未删难例。图构组不含reference，评价第一anchor对应仍是现有协议，不认证物理真值或未知K跨域泛化。

### 原run、设备与退出

科学unit `mtad-d17a77e64770.service`，systemd Linger=yes/Restart=no/RuntimeMax=infinity；GPU0 UUID `GPU-c5a00c37-f181-90b3-f220-be6e7457f473`兼容共卡，启动前17125MiB free，CPU0–3/OMP-MKL4，不碰他人进程。worker05:44:05.668990Z至05:54:35.754424Z，真实exit0/COMPLETED，服务inactive/dead/ExecMainStatus0。**总630.246547s**（含启动/准备/训练/评价），science内部622.537408s，RSS2297462784B，CUDA峰值allocated249623040B；没有超3600s建议或加步。缓存/搜索成本也包含，不能用GPU显存或6209参数替全链路效率。

完整V后一次新audit checkout，同冻结7be3dae、CPU16–19/无GPU，unit `mtad-23e6b8dd48d6.service`，05:55:09.334986Z至05:55:50.083823Z，worker总40.913979s/exit0、同服务成功终态。限定保存输出独审17.020400s：301503 exact/52202 numeric/336 null checks，位置/覆盖/尾部/组件/主要CI与NumPy raw/decoded关联通过；不复算网络训练/独立认证物理truth。四属性核算9.845032s，采用原评价器不是独立属性实现；后验支持/纯度/anchor与全部T网格独立sorted-prefix核7.177171s、最大指标差3.33e−16。audit峰RSS601481216B。全流程科学＋规定收尾总671.160526s；另171/172失败/资格、局部阅读开发及后验本地分析不混作同一GPU训练时间。

### 同V32完整有效位置与跨源指标

下表关联为**decoded跨源场宏**，覆盖为有效参考位置point coverage，MAE/RMSE/P99为米制大圆位置误差。不是论文ME/作者ESD或同行排名；各自产出人口不同，共同域另报。

| 前端＋同159后端 | mPCA↑ | mPFA↓ | point覆盖↑ | 代表目标/1427 | MAE m↓ | RMSE m↓ | P99 m↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原WS | 78.9893% | 15.5502% | 98.4417% | 1388 | 527.427 | 1813.299 | 7260.394 |
| 冻结CTX/同T标定 | 80.6839% | 15.3688% | 97.6938% | 1381 | 432.801 | 1477.093 | 4933.121 |
| 同曝光继续BCE | 82.9392% | 14.4857% | 96.8876% | 1364 | 438.835 | 1452.974 | 4673.059 |
| BCE＋结构项 | 81.7651% | 15.5733% | 98.0317% | 1380 | 427.241 | 1447.519 | 4842.083 |

raw与decoded分开：原WS raw mPCA89.7099/mPFA27.0558%，CTX90.3258/26.5309，BCE93.0240/27.2970，STRUCT94.4481/28.1657。STRUCT−BCE **raw召回+1.4241pp**、描述CI[+.8730,+3.2317]，但decoded召回−1.1740pp、CI[−4.6304,+1.7146]；decoded误报+1.0876pp、CI[−.1098,+3.9190]。它们不是同一指标，不能拿raw召回上升冒充构组后联合改善。对原WS decoded召回+2.7758pp CI[−.4512,+7.5205]，误报+.0231pp CI[−1.2174,+2.9987]。所有区间均9组件/2000重采样/seed20、开发性描述，V重复使用且T共享身份，不称独立确认/因果效应。

### 共同域、获失与尾部反例

| STRUCT相对 | 共同位置MAE差/95%组件描述区间 m | 新/丢query | 救/丢目标 | 新增>1km增害点 | 最大增害 |
|---|---|---:|---:|---:|---:|
| 原WS | −103.561 [−242.942,−85.015] | 418/714 | 12/20 | 1131 | 64.640km |
| 冻结CTX | −6.421 [−105.088,+7.834] | 862/618 | 20/21 | 1120 | 16.918km |
| 同曝光BCE | −13.827 [−89.959,−4.464] | 1213/387 | 29/13 | 933 | 16.918km |

STRUCT有效位置70775，比BCE69949多826，比CTX70531多244，**代表目标却比CTX少1**；point coverage不等对象coverage。对原WS的群组1605→1484、重复217→104、代表1388→1380；不是新增所有目标覆盖。原强控CTXCAL与170 exact，历史有利/不利结果均保留。

相对WS，STRUCT1131大害按新支持0/1/多源分121/770/240，same-anchor958，pure→pure231。最坏scene3835/query1仍1→2源、same-anchor，7.645km→72.285km、增加64.640km；另一纯→纯可从2源变0源，增害15.416km。多源、纯组或same-anchor都不能认证状态相容，不能把全部尾部归零支持/小网选错；也不因P99总体下降说单点灾难消失。完整计数、5个匿名最坏例及原私有全数组可复核。

跨源decoded相对WS：真边救253/丢142，假边去477/增460；raw假边去237/增3805。共享支持跨源decoded假边去313/增340，无共同支持去164/增120。聚类闭包对raw变化的非线性取舍可见，但没有分离证明WS是唯一瓶颈。

T32结果：原/CTX/BCE/STRUCT MAE597.179/521.951/509.158/430.087m，point覆盖98.4591/98.3989/97.1222/98.7469%，代表825/829/812/834。STRUCT相对BCE救31/丢9、共同差−86.278m；T与V增量大小不同，不能把T已见标签上的改善外推。

全组件及去最大保留：[完整summary](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_structured_association/173/science/summary.json)。V去最大19场组件后WS/CTX/BCE/STRUCT MAE684.111/564.573/543.983/486.858m；最大组件本身CTX404.917优于STRUCT414.622。相对BCE的各组件完整MAE6改善/3相同/0变差，相对冻结CTX为5改善/2相同/2变差；完整人口可能不同，不能以去最大或全均值替全部异质性。

Mean强控V MAE1276.328/1256.212/1290.623/1290.534m，STRUCT不超原Mean或冻结CTX Mean；位置收益与冻结159读出交互，非新的连续混合兑现。四属性原件含经纬度论文口径/米制位置、speed mps及圆周heading、observed/missing与全场：V speed MAE WS23.310/CTX23.215/BCE22.989/STRUCT23.041mps，heading76.620/75.826/75.711/75.957°。速度航向仍随分组的Mean，无新速度/航向学习，不能宣四属性全面提高。

### 原件、偏差与标准包

[完整173公共原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_structured_association/173/)含73初始科学文件（64逐场＋T/V完整summary/scene-summary/训练population/curve/sampling/T网格/run）、3服务器限定收尾、科学/审计各11launcher、事后本地matched关联区间原码/结果和closeout/squeeze原码。原身份、坐标、权重及大预测不回传，继续原私有目录；现有tar只是同73原件传输容器，不替代逐场正文。没有少报失败、T/V场或臂。全部7项本地测试在READY前通过。

计划内科学/核验按固定配方完成，无训练预算/终点偏差。后验支持/纯度/anchor/边转移与本地matched关联CI为科学后分析，标明事后且无新预测/拟合/阈值选择；本地CI内部.129500s/峰RSS112771072B，不作服务器训练成本。当前飞书检索器重开不可达，规范沿用主仓此前全文采纳版本，不称本轮重新在线通读；无流程内容改写。171技术失败/172修复为独立记录，不藏进173或称173重试。

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: REC-STRUCTURED-ASSOCIATION-173
source_idea_id: IDEA-STRUCTURED-ASSOCIATION-20261031-173
plan_version: 2.0
execution_status: COMPLETED_exit0
research_question: does_cluster_relation_auxiliary_help_beyond_matched_BCE_exposure
hypothesis: C57_average_and_coverage_gain_without_association_tail_cost
planned_vs_actual_design_and_controls: specified_T128_two_heads_512_each_T32_V32_eight_arms_all_complete
actual_configuration: frozen7be3dae_L2S173_original_request_and_T_calibration
samples_materials_or_data_provenance: bound168_127_125_114_133_151_159_and_main_manifest
results_with_uncertainty: full_tables_common_CI_and_components_above
all_runs_or_repetitions: one_science_one_prespecified_closeout_no_retry
observations_and_deviations: prespecified_complete_posthoc_decomposition_and_local_matched_CI_labeled
evidence_locations: linked_all_public_originals_private_states_and_weights_server_only
known_limits_and_missing_evidence: repeated_V_shared_T_ids_K2_offline_no_fullV300_new_head_or_peer_ranking
next_routes: squeeze_then_Mode_C_no_automatic_new_training
next_skill: squeeze-experiment
```

## SQUEEZE_REPORT — SQ-STRUCTURED-ASSOCIATION-173

1. **可保留的增量**：匹配曝光/容量/信息与同T规则后，STRUCT对BCE有效点/共同位置平均有条件增量，raw跨源召回也增。不是仅多训练；同曝光BCE自身decoded关联较好却丢更多覆盖，故不能用单一边指标选模型。训练完整场成本及原法物理优势均要保留。
2. **不能晋级的代价**：STRUCT对BCE decoded召回/误报点值更差、933新增大害；对冻结CTX平均区间跨0，对原WS覆盖仍低/新大害1131。四属性/Mean/组件亦不全面赢。当前固定λ1/B4/4轮配方不扩大，保权重与反例供后续机制分离，不把一次未联合成功否定所有结构监督。
3. **新可区分问题**（AI）：raw真边改善并未同向变成decoded指标，候选为训练forest与推理WS目标/工作点失配、错误桥边的图闭包，以及状态/代表对应耦合；原额外假边/源状态不相容、T过拟合也是竞争解释。尚无分离干预，不能写“已证明WS根因”或“源数多反而坏”。
4. **下一个有效动作范围**：先在原T保存输出上分离哪些真边被恢复却在WS结果丢失、哪些错并造成对象/时刻支持损害，对照成熟可变对象数/多片段构组与可学习决策机制，再形成同预算可替换构组的具体Idea；保现有平均强控。不直接按V网格换λ/阈值/源优先，也不永久冻结原anchor。当前无新READY或自动全300。

回Mode C1.80，E178/F275，P14/F275/M111/E178/C57，107篇不增。全source分析、正负结果、原失败和未采纳理由保留。

```yaml
package_type: SQUEEZE_REPORT
squeeze_id: SQ-STRUCTURED-ASSOCIATION-173
input_experiment_record_ids: [REC-STRUCTURED-ASSOCIATION-173]
claim_scope: matched_development_T128_and_original_T32_V32_conditional_signal
data_quality_and_provenance_gaps: all_saved_and_bound_but_reference_generation_and_independent_domain_unknown
primary_result_with_uncertainty: STRUCT_minus_BCE_common_MAE_minus13.826820_CI_minus89.959177_to_minus4.464296_with_decoded_and_harm_tradeoff
process_dynamics: 1024_GPU_updates_4_complete_passes_each_630.246547_seconds_science_plus40.913979_closeout
mechanism_and_intermediate_findings: raw_vs_decoded_divergence_AI_not_unique_WS_root_cause
errors_heterogeneity_and_boundary_conditions: coverage_below_WS_1131_new_harms_0_1_multi_121_770_240_and_largest_component_limits
robustness_sensitivity_and_alternative_explanations: matched_BCE_same_T_calibration_Mean_full_and_common_components_all_attributes_no_V_selection
new_P: none
new_F: F275
new_M: none
evidence_index: original_science_audit_attribute_calibration_and_posthoc_sources
open_questions: [train_decode_objective_gap, physical_compatibility, joint_coverage_association_tail_gain, independent_domain_K3plus]
next_routes: Mode_C_then_T_counterexamples_and_mature_decision_sources_before_new_Idea
next_skill: deep-read_Mode_C
```
