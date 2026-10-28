# 112生成支撑记账：真实执行记录、独立复核与深挖

2026-10-29。**固定V16的生成区占48.1078%；在原四属性reference均有效的生成query上，TCM的冻结特征MSE高于state-hold，差为+0.008865220。** 14场更差、2场有收益。该结果只描述107固定TCM端点的支撑和特征误差；没有TAM分数或物理解码，不能认定生成区已导致关联错误或TCM普遍无价值。

本文件先依`record-experiment`记录实际，再依`squeeze-experiment`分析。只复核下载的小型112产物，不运行模型、原始数据或新服务器任务；不改state/Handover/主矩阵。SC/DE/AI/UN含义沿主流程：运行与数组报告DE不自动成为因果机制DE。

## 1. EXPERIMENT_RECORD：实际做了什么

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-LEVEL1-GENERATED-SUPPORT-20261029-112
package_id: EXP-LEVEL1-GENERATED-SUPPORT-20261029-112
source_idea_id: IDEA-GENERATED-SUPPORT-SCORE-20261029
plan_version: '1.1'
execution_status: completed
created_at: '2026-10-29'
research_type: computational_cached_report_diagnostic
source_matrix_id: MC-C52-STEPWISE-20261029
selected_C_id: C52L-Cb
evidence_status: partial
research_question: '§1.1：固定V16的O/G分母与同mask model-hold冻结特征差异'
hypothesis: '输入生成支撑可占较大比例；不预设其有害或能解释关联，见§1.1/2.3'
planned_vs_actual_design_and_controls: '§1.1全部计划/实际；同端点state_hold和缺参考分母完整保留'
actual_configuration: '§1.2实际命令/工作目录/env与§1.4的Q64/cap128/128维目标'
samples_materials_or_data_provenance: '§1.3既存107报告到112固定V16；无新原始数据/权重读取'
results_with_uncertainty: '§1.4–1.5全部分母/误差与16场异质性；无独立重复/CI'
all_runs_or_repetitions: '112单次attempt1、exit0；不覆盖107既有失败，见§1.1–1.2'
observations_and_deviations: '§1.1无plan1.1范围偏差；§1.5有利/不利结果与§2.3未知'
evidence_locations: 'E112-RUN/SUM/STAT/EXIT/REQ/CODE/PLAN，见§1.6'
known_limits_and_missing_evidence: '未本地重读大journal；参考缺失、无TAM/pair masks/物理因果/精确环境血缘，见§1.2–1.4/2.3'
next_routes: '本文件§2先squeeze，再Mode C；下一小核验另定具体范围'
next_skill: squeeze-experiment
```

### 1.1 问题、计划与实际

研究问题：在已有107 TCM评估中，provided O与generated G占多少query；在同一原参考有效分母上，model与state_hold的冻结特征误差有何差异？这是Pb评分问题的输入/中间量核验，不是对“GG影响身份分数”的完整验证。

上游为[Idea plan1.1](../proposals/generated_support_score_diagnostic_20261029.md)、[112独立合同](../../HandOver/L2S-20261029-112.md)、[C52小矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/c52_stepwise_problem_matrix_20261029.md)。实际入口为[generated_support_score.py](../../tools/analysis/generated_support_score.py)。

| 项目 | 事前计划1.1 | 实际事实 |
|---|---|---|
| 选择 | 原manifest V300数值排序，`random.Random(20261029).sample(...,16)`再数值排序，不看成绩 | 名单完全匹配；16场全部保留，2379片段 |
| 输入 | 已完成107 TCM的run/summary/既存completion-evaluation；不读权重/ZIP | 原报告216,833,440B，只读一次；run记录checkpoint/archive/model均false |
| 模型/训练 | 0新前向、0训练、0GPU，原TCM2424固定端点 | run记录gpu_forwards=0、optimizer_updates=0、model_constructed=false；未换checkpoint/阈值 |
| 支撑 | O=输入首末时刻内的既有插值嵌入；G=before+after | 完整152256=2379×64 query；O79009、G73247，和与逐场重聚合一致 |
| 误差/对照 | model与state_hold同原四属性reference有效mask，按SSE/feature元素数池化 | 逐场/逐区分母一致；inside差为0，外侧有正负异质性 |
| 参考访问 | 本次不读reference CSV，但使用已有reference派生指标 | 明确`source_report_reference_derived=true`；不能写为“没有使用参考信息” |
| 资源/执行 | CPU≤2、RAM预算4GiB、AS≤8GiB、建议600s，一次失败保全 | 单次COMPLETED/exit0；程序8.201467s、launcher8.746398s，峰RSS540,606,464B=515.5625MiB，AS8,589,934,592B |
| 全V | 原顶层汇总只作重叠的既有source，不当独立复验 | 公有结果保留原全V300/24599片段的顶层指标，另节列出并明确重叠 |

实际未发现相对plan1.1的科学范围/选择偏差；plan1.0的T32/可选新前向已在执行前由plan1.1改成已有报告V16复用，不能混称事后改样本。没有重跑、补抽、删难例或多端点择优。107固定056子任务另有schema失败，见[原失败及证据](2026-10-29-107-failure-discovery.md)；112消费已完成TCM产物，不能覆盖该失败为成功。

### 1.2 时间线、环境与命令

launcher记录payload于**2026-10-28 20:36:38.665 UTC**开始，20:36:47.198 UTC结束；北京时间为**2026-10-29 04:36:38–04:36:47**。systemd后端，advisory600s，`returncode=0`、`timed_out=false`、`deadline_exceeded=false`，退出文件同为0。

真实工作目录`/home/wangguangjie/djs/association-code-run112`；解释器`/home/wangguangjie/djs/mtad-field-20261016/bin/python`。命令读取原107的`.../LEVEL1-TMTAFF-UNIT-BASELINE-A-20261027-107-attempt-1/tcm`，原manifest为`configs/data/mtad-level1-scene3000.json`，写新`LEVEL1-GENERATED-SUPPORT-20261029-112-attempt-1`。`CUDA_VISIBLE_DEVICES=''`，OMP/MKL/OPENBLAS线程环境各2。精确argv/cwd/env见[E112-REQ](#16-证据索引)。公有run/launcher原件未附精确代码SHA、Python完整版本/物理CPU型号，不用当前HEAD倒填这些字段；本次输出包含plan_version1.1和真实源路径。

### 1.3 数据血缘与独立复核层级

原107 TCM固定训练normalizer/嵌入与完整V评估 → 原`completion-evaluation.json`的逐fragment区域计数/特征误差 → 112固定V16池化及原全V顶层附录 → 本次逐场独立重聚合。

本地实际核[E112-RUN/STAT/EXIT/REQ/SUM]：

- 重新从原manifest按既定seed计算16场，与run、summary、per_scene顺序完全一致。
- 对2方法×5区×4计数字段共**40个池化计数**，直接对16场求和均一致；10个SSE/feature分母重算MSE均吻合，SSE最大浮点差`1.4552×10^-11`。
- 16场均核`fragment×64=all=O+G`，各方法`all=inside+before+after`、`G=before+after`，并核model/hold参考与特征元素分母一致；逐场model−hold差值重算一致。
- **复核限制**：原216.8MB逐fragment大journal仍留服务器，本地没有再次逐fragment重建112输出，也没有原逐query masks/latent数组；因此“独立核验”指公有逐场产物与总表/执行证据的重聚合及代码审查，不冒称重新验证原始物理样本/每个latent值。

固定16场为：700、1293、1582、2038、2153、2416、2479、2576、3156、3772、3835、3916、3956、3981、4272、4473。没有按结果再挑有利或不利场。

### 1.4 主要结果：完整分母与实际model/hold值

MSE单位为**冻结T内normalizer归一后的四状态，加实际query时间编码，经冻结ReLU投影形成的128维嵌入，每feature元素平方误差**。它没有米、m/s、度的直接物理解释。state_hold是在请求query时刻重新嵌入可见端点状态，不是把旧时刻latent常量搬到新时刻。O是观测派生插值支撑，不能当成全部原生测量点。

| 区域 | 全部query | 可评query | 缺完整参考query | feature元素分母 | model MSE | hold MSE | 差model−hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全窗all | 152256 | 125844 | 26412 | 16108032 | 0.015743702471 | 0.012437103096 | +0.003306599375 |
| O/inside | 79009 | 78906 | 103 | 10099968 | 0.012276511064 | 0.012276511064 | 0 |
| before | 39230 | 25370 | 13860 | 3247360 | 0.020801225742 | 0.012166036544 | +0.008635189198 |
| after | 34017 | 21568 | 12449 | 2760704 | 0.022479275528 | 0.013343475188 | +0.009135800340 |
| **G=before+after** | **73247** | **46938** | **26309** | **6008064** | **0.021572289226** | **0.012707069325** | **+0.008865219901** |

G主要估计的原误差分子为model **129607.69429494208**、hold **76344.88575794778**，统一除以6,008,064；没有平均两个方向MSE或逐场MSE来代替池化。all的原SSE为253600.06319437877与200337.25465738447；O共同为123992.3688994367。

G占全部query **48.107792%**；其中参考覆盖只有**64.081805%**，缺失**35.918195%**。O参考覆盖为99.869635%，all为82.652900%。保留缺参考分母，不能把46938个可评生成query说成全部73247都已验证；不能把O与G的非同分布均值直接当“外推造成误差”的因果比较。该V16没有“完全无O query”的片段（计数0），不能据此否定其他场的无O情况。

**inside的0是model−hold之差为0，不是inside误差为0。** 两条路径在O区使用相同观测派生插值，因此这一对照符合操作结构；它不认证插值/参考物理正确或网络已学会恢复。

### 1.5 全部16场：支持与反例一起保留

下表全部为G区；“可评/缺参考”采用相同query定义。正差不利于model，负差有利；不把场/fragment/query当独立统计重复。

| 场景 | 片段 | O/G全部query | G可评/缺参考 | model MSE | hold MSE | 差model−hold |
|---|---:|---:|---:|---:|---:|---:|
| 700 | 179 | 5892 / 5564 | 3962 / 1602 | 0.024206125 | 0.014724170 | +0.009481954 |
| 1293 | 10 | 303 / 337 | 205 / 132 | 0.018159241 | 0.007438906 | +0.010720334 |
| 1582 | 51 | 1731 / 1533 | 1130 / 403 | 0.035813440 | 0.014064376 | +0.021749064 |
| 2038 | 12 | 350 / 418 | 314 / 104 | 0.029369715 | 0.013656216 | +0.015713499 |
| 2153 | 16 | 478 / 546 | 313 / 233 | 0.047456989 | 0.007863094 | +0.039593895 |
| 2416 | 14 | 455 / 441 | 415 / 26 | 0.014232459 | 0.015555377 | **−0.001322918** |
| 2479 | 41 | 1509 / 1115 | 634 / 481 | 0.021665839 | 0.023336918 | **−0.001671080** |
| 2576 | 28 | 964 / 828 | 650 / 178 | 0.014890414 | 0.007114371 | +0.007776042 |
| 3156 | 102 | 3888 / 2640 | 1750 / 890 | 0.017459875 | 0.014339335 | +0.003120540 |
| 3772 | 999 | 31928 / 32008 | 18624 / 13384 | 0.019392626 | 0.012525674 | +0.006866952 |
| 3835 | 686 | 24429 / 19475 | 14476 / 4999 | 0.018129140 | 0.011716143 | +0.006412997 |
| 3916 | 16 | 333 / 691 | 251 / 440 | 0.015358484 | 0.013353777 | +0.002004707 |
| 3956 | 54 | 1793 / 1663 | 1155 / 508 | 0.049158655 | 0.013277940 | +0.035880715 |
| 3981 | 49 | 1320 / 1816 | 882 / 934 | 0.073949976 | 0.013896146 | +0.060053831 |
| 4272 | 11 | 212 / 492 | 271 / 221 | 0.025391951 | 0.014285686 | +0.011106265 |
| 4473 | 111 | 3424 / 3680 | 1906 / 1774 | 0.016876864 | 0.013260435 | +0.003616429 |

G与all均**14场不利/2场有利**；before为15不利/1有利（2479），after为15不利/1有利（2416），inside16场差值均0。2416是after收益抵过before不利，2479是before收益抵过after不利，不能把正收益反例折叠掉。

### 1.6 证据索引

| E | 实际原件及定位 | 本次核验内容 |
|---|---|---|
| E112-RUN | [112 run.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/112/run.json) | COMPLETED、plan1.1、固定16场、源路径/报告字节/一次读取、0前向/训练、资源 |
| E112-SUM | [112 summary.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/112/summary.json) | 16场全部per_scene、完整区域分母/分子、全V旧顶层、解释边界 |
| E112-STAT/EXIT | [launch-status](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/112/launch-status.json)、[exit_code](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/112/exit_code.txt) | systemd真实退出0、时间、未超时 |
| E112-REQ | [launch-request](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/112/launch-request.json) | 实际argv/cwd/env、advisory600s |
| E112-CODE | [汇总入口](../../tools/analysis/generated_support_score.py)、[原writer](../../association/evaluation/tmtaff.py:441) | O/G统计对象、feature target、state_hold重嵌入、逐fragment与顶层schema；汇总不是TAM |
| E112-PLAN | [Idea1.1](../proposals/generated_support_score_diagnostic_20261029.md)、[112合同](../../HandOver/L2S-20261029-112.md) | 事前选择与目标、上限、失败和结论范围 |

原始大journal路径在E112-RUN的`source_tcm_run`下；未复制原node/identity/坐标到本报告。此记录的实际配置、数据来源、所有运行/重复、结果及偏差均见§1.1–1.6；本次只有一次来源复用诊断，不是一次新的TCM训练/评测。

## 2. SQUEEZE_REPORT：能多知道什么、还不能知道什么

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-GENERATED-SUPPORT-20261029-112
package_id: SQ-GENERATED-SUPPORT-20261029-112
input_experiment_record_ids: [EXP-LEVEL1-GENERATED-SUPPORT-20261029-112]
source_idea_id: IDEA-GENERATED-SUPPORT-SCORE-20261029
plan_version: '1.1'
evidence_status: partial
next_skill: deep-read Mode C
```

### 2.1 主要解释与场景异质性

**DE：生成区域在这个实际输入中很大，且当前TCM端点相对简单hold的特征优势不成立于V16池化。** 这是比“生成可能有风险”的泛泛猜测更具体的测量，但仍不是Pb后半段的“生成区TAM分数不利主导”。没有TAM输出或身份错误，不能跨过该缺口。

两大场3772/3835合占片段**70.8281%**、G可评query **70.5186%**，故主要池化结果带有明确的人口权重；场景符号计数补充异质性，但不能替代主要池化，也不能把16个场视为无共享身份的独立重复。最大的逐场G差在3981（+0.060053831），不是片段最多的场；因此不能只凭“大场多”给出根因。没有事后删两大场来制造另一个主成绩。

参考缺失高度偏向G：26309/26412个缺完整参考query发生在G。当前G比较对model/hold使用相同mask，因此两方法差异不是二者评分人口不同；但它只覆盖64.08%的生成query，缺失机制可能与时间/目标寿命/包内reference有关，不能扩为全部生成区。公有汇总没有“完全无参考片段”的逐fragment计数，不能由缺query总数反推该人数。

### 2.2 原全V顶层：反向收益必须保留，但不是独立复验

以下来自**同一107 TCM原报告的旧全V顶层**，包含上述16场；只用于理解已有source的条件性，不叫“额外300场独立确认”。本地从其before/after分子分母重新合成G，没有再次读大journal或跑模型。

| 原全V范围 | query/可评/缺参考 | model MSE | hold MSE | 差model−hold |
|---|---|---:|---:|---:|
| all：300场/24599片段 | 1574336 / 1229715 / 344621 | 0.020276073312 | 0.018691418284 | +0.001584655028 |
| O | 771450 / 769237 / 2213 | 0.017885779927 | 0.017885779927 | 0 |
| before | 396339 / 246455 / 149884 | **0.023670645080** | **0.025593554622** | **−0.001922909543** |
| after | 406547 / 214023 / 192524 | 0.024958247320 | 0.013638974488 | +0.011319272832 |
| G（本次由旧顶层两端重聚合） | 802886 / 460478 / 342408 | 0.024269102540 | 0.020037252032 | +0.004231850508 |

原全V G占50.998389%、参考覆盖57.352850%。**原全V before的model实际优于hold，与V16 before池化方向不同**；after的差更不利。这说明条件/场景权重与延展方向影响结果，反证“补全在所有条件都无用”的说法。V16是事前随机开发样本，其方向差不意味着选择规则违规，也不支持只改抽样直到得到想要的方向。

### 2.3 机制链与替代解释

| 层级 | 现在有什么 | 仍缺什么/不能推出什么 |
|---|---|---|
| 输入支撑→生成占比 | O/G完整分母、G约半窗；DE | 没有每对两个支撑的交集，不能由单轨计数推出pair GG占比或“全窗评分被GG主导” |
| 生成→特征误差 | 同端点、同target与mask、model/hold真实特征MSE正负；DE | 没有分布无偏保证；latent MSE受normalizer/冻结投影/时间编码影响，不等于米制恢复 |
| 训练目标→多步延展 | TCM训练是原生相邻嵌入预测；实际query含长时域rollout，此差别可由代码核 | 是否是teacher forcing、长视界误差积累、模型容量/训练曝光不足或reference/目标本身变化导致，当前只是AI/UN |
| 特征误差→关联/融合损失 | 112没有该层输出 | 107 TAM未训练；056异构score不能接来补链。TFM可能过滤/放大该误差，其真实作用需独立共同人口评价 |
| 源内正收益 | TMTAFF原Fig9有TCM正面关联收益；本次2416/2479及旧全V before也有正特征效果 | 不能因为一次有限端点的不利池化就否定TCM全部用途；原论文收益也不能担保当前端点/当前协议 |

state_hold与model的比较是固定端点的替代输出对照，可以直接说“在这个feature目标上的误差差异”；它不是多个训练budget/架构随机化后识别出的普遍因果效应。没有训练轨迹/多seed或新的物理对照数据，本轮跳过收敛/独立重复/误报漏报/校准分析，明确不是已经排除这些解释。

### 2.4 下一项有区分度的核验入口

优先继续复用同一原报告、同一V16的逐fragment**native one-step forward/reverse model–hold**与query before/after结果，保持全部fragment及缺参考分母。原writer已保存one_step，不需新GPU；可按事先固定的G占比区间和片段原生点数区间查看“原生一步已弱”还是“原生一步可用、长query/参考目标下转弱”。这能缩小训练不足/目标不一致/延展累积等解释，**不能单凭两个不同target的误差差就认定rollout是因果主因**。需另有具体小计划才执行，本文件没有新增运行。

同时保留2416/2479的正收益条件，不能只分析最坏场。原报告有独立的距离段边缘汇总，但没有`G×distance`逐query联合表；不能把两个边缘表强拼成“生成视界分层”。若后续要分离真正视界或验证身份影响，要建立合适联合记录/合法训练TAM；若要检验物理恢复，则应在固定107/共同输入与分组下核真实TFM结果，不能拿latent误差换算成米制收益。任何新干预仍须单独Idea/预算，不自动补训练。

### 2.5 可回流的最小事实包（不改主矩阵）

```yaml
package_type: SQUEEZE_REPORT
squeeze_id: SQ-GENERATED-SUPPORT-20261029-112
input_experiment_record_ids: [EXP-LEVEL1-GENERATED-SUPPORT-20261029-112]
claim_scope: fixed_V16_existing107_TCM_support_and_reference_derived_feature_errors_only
data_quality_and_provenance_gaps: public_per_scene_reaggregation_verified; original_fragment_journal_not_reparsed_locally; missing_G_reference_35.9182percent; precise_code_SHA_and_environment_not_in_public_run
primary_result_with_uncertainty: G73247_of152256_queries; evaluable46938; modelMSE0.021572289226_vs_hold0.012707069325; delta+0.008865219901; 14scenes_worse_2better; no_independent_repeat_or_CI
process_dynamics: no_new_training_or_forward; one_existing_report_read; program8.201467s; peakRSS540606464B
mechanism_and_intermediate_findings: G_support_condition_present_DE; harmful_GG_TAM_score_path_UN; inside_difference0_is_shared_interpolation_not_zero_error
errors_heterogeneity_and_boundary_conditions: selected2416_2479_gain; full_V_overlapping_before_gain_after_loss; G_reference_coverage64.0818percent; pooled_weights_concentrated
robustness_sensitivity_and_alternative_explanations: 40count_and10MSE_checks_match; per_scene_signs_retained; finite_budget_training_and_target_rollout_reference_differences_not_identified
new_P: none; C52L-Pb remains conditional and its TAM part unverified
new_F:
  - F112a_DE: fixedV16_G_fraction48.107792percent; all2379fragments_have_some_O_query
  - F112b_DE: same_mask_G_feature_model_worse_pooled_with_two_positive_scene_counterexamples
  - F112c_DE_existing_source: same_overlapping_fullV_before_model_better_after_worse
new_M: AI_candidate_native_one_step_target_vs_query_rollout_target_mismatch; no_identified_causal_mechanism
evidence_index: E112-RUN,E112-SUM,E112-STAT, E112-EXIT,E112-REQ,E112-CODE,E112-PLAN
open_questions: per_fragment_gain_distribution; native_one_step_vs_query_conditions; genuine_pair_GG_energy_and_legal_trained_TAM; physical_TFM_effects_and_reference_semantics
next_routes: deep-read_Mode_C_then_bounded_discriminating_diagnostic_if_authorized
next_skill: deep-read Mode C
```

**决定边界：** 本次完成了计划内真实支撑/特征记账并独立重聚合；Pb的“生成区影响身份评分”仍未验证。保留不利结果、明确的正收益反例与可发展的条件化路径，不直接删掉TCM或指定新网络。
