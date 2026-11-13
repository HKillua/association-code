# 五个研究 skill 的规范调用流程

> 版本：research-pipeline-contract v1.0
>
> 适用范围：自然科学、医学、工程、计算机、社会科学、心理学、商科、教育、艺术研究，以及计算实验、实验室实验、临床研究、田野研究、调查、模拟、工程测试和混合方法研究。

## 一、规范调用顺序

这五个 skill 不按名称列出的顺序调用，而按研究信息从“来源理解”逐步变成“可检验研究”，再回到“证据积累”的顺序调用：

```text
analyze-paper
    │  ANALYSIS_PACKAGE：研究/论文事实、证据、限制、候选 P/F
    ▼
deep-read（仅 Mode C）
    │  MODE_C_PACKAGE：P/F/M/E + P×F 矩阵 + C 连接点
    ▼
discover-innovation
    │  IDEA_PACKAGE：假设、预测、对照、指标、最小验证设计
    ▼
实际执行实验/研究
    ▼
record-experiment
    │  EXPERIMENT_RECORD：实际配置、所有运行、结果、不确定性、偏差
    ▼
squeeze-experiment
    │  SQUEEZE_REPORT：机制、动态、异质性、稳健性、新 P/F/M
    ▼
deep-read（Mode C 增量合并，形成下一轮 MODE_C_PACKAGE）
    └──────────────────────────────────────────────┘
```

其中：

1. `analyze-paper` 负责“理解外部研究来源”，不直接把作者贡献改写成创新点。
2. `deep-read` 负责把来源中的问题、现象和机制放回同一证据矩阵；它只支持 Mode C，不负责最终方案设计。
3. `discover-innovation` 负责把一个有证据的 `C` 连接点转成可检验研究想法。
4. 实际执行由用户或其他执行工具完成；执行本身不是这五个 skill 的隐含步骤。
5. `record-experiment` 先记录事实，再由 `squeeze-experiment` 深挖事实中还能支持什么。
6. `squeeze-experiment` 产生的新现象和新限制必须先回到 `deep-read` Mode C；不能把深挖报告直接冒充为新的 Idea 卡片。

## 二、统一交接协议

每个交接包必须包含以下元数据：

```text
package_type：包类型
package_version：版本
package_id：唯一 ID
created_at：生成日期
source_ids：来源或上游包 ID
evidence_status：verified / partial / unknown
evidence_locations：可定位的页码、图表、代码、日志、样本或记录位置
missing_evidence：缺失证据；无则写 none
next_skill：建议的下游 skill
```

通用证据状态含义：

| 状态 | 含义 |
|:--|:--|
| `verified` | 当前材料中有可定位的直接证据，且关键边界已记录 |
| `partial` | 有证据但存在未解决的条件、替代解释或数据缺口 |
| `unknown` | 当前材料不足以判断；不能当成阴性结果或已验证结果 |

ID 关系必须保持可追溯：

```text
analysis_id → source_ids / P-candidates / F-candidates
matrix_id   → S/P/F/M/E/C IDs
idea_id     → matrix_id + selected C/P/F/M/E IDs
experiment_id → idea_id + plan_version
squeeze_id  → experiment_id(s) + evidence locations
```

下游不得从文字语气猜测上游结论；如果必填字段缺失，应标记缺口并继续做范围内分析，不能静默补全。

## 三、五类标准产出包

### 1. `ANALYSIS_PACKAGE`：analyze-paper → deep-read

最小字段：

```text
package_type: ANALYSIS_PACKAGE
analysis_id
source_identity_and_version
research_question
study_type_and_context
design_units_samples_or_materials
intervention_input_or_procedure
measurements_and_analysis
results_with_uncertainty
claim_evidence_implementation_map
limitations_biases_and_ethics
reproducibility_and_transferability
candidate_P
candidate_F
evidence_index
unresolved_questions
next_skill: deep-read Mode C
```

`candidate_P` 和 `candidate_F` 只是候选素材，必须在 Mode C 中重新核验、编号并补充机制和边界。

### 2. `MODE_C_PACKAGE`：deep-read → discover-innovation

最小字段：

```text
package_type: MODE_C_PACKAGE
matrix_id
scope_and_inclusion_rules
source_map
P：具体问题
F：有证据支持的现象
M：机制链、条件、替代解释
E：证据索引和定位
P_by_F_matrix
C：候选连接点、状态、评分、未知项
evidence_status
recommended_C
first_low_cost_check
next_skill: discover-innovation
```

`C` 只能是问题—现象连接点，不能提前写成完整方法或实验方案。

### 3. `IDEA_PACKAGE`：discover-innovation → record-experiment

最小字段：

```text
package_type: IDEA_PACKAGE
idea_id
source_matrix_id
selected_C_id
source_P_ids / source_F_ids / source_M_ids / source_E_ids
research_question
hypothesis_and_mechanism
testable_predictions
design_and_controls
sample_materials_or_experimental_units
primary_outcome_and_estimand
secondary_outcomes
analysis_plan_and_uncertainty
resources_time_cost_and_risk
ethics_safety_privacy_or_compliance
stopping_and_interpretation_rules
status
next_skill: record-experiment
```

`record-experiment` 用此包中的“计划值”与实际执行逐项对照；因此 Idea 卡片不能只写“想法”，必须写清楚对照、主要结果和解释规则。

### 4. `EXPERIMENT_RECORD`：record-experiment → squeeze-experiment

最小字段：

```text
package_type: EXPERIMENT_RECORD
experiment_id
source_idea_id
plan_version
execution_status
research_question_and_hypothesis
planned_vs_actual_design_and_controls
actual_configuration
samples_materials_or_data_provenance
all_runs_or_repetitions
results_with_uncertainty
observations_and_deviations
evidence_locations
known_limits_and_missing_evidence
next_routes
next_skill: squeeze-experiment
```

记录必须覆盖所有运行、异常、排除和计划外变化；“完成但未观察到清晰差异”也是有效状态，不应被改写成“没有价值”。

### 5. `SQUEEZE_REPORT`：squeeze-experiment → deep-read Mode C

最小字段：

```text
package_type: SQUEEZE_REPORT
squeeze_id
input_experiment_record_ids
claim_scope
data_quality_and_provenance_gaps
primary_result_with_uncertainty
process_dynamics
mechanism_and_intermediate_findings
errors_heterogeneity_and_boundary_conditions
robustness_sensitivity_and_alternative_explanations
new_P
new_F
new_M
evidence_index
open_questions
next_routes
mode_C_input_bundle
next_skill: deep-read Mode C
```

`mode_C_input_bundle` 至少要把新发现分成 `P/F/M/E` 四类，并附上来源为 `experiment_id` 或 `squeeze_id` 的证据位置。若只有解释没有直接证据，标为 `AI` 或 `UN`，不得写成已确认机制。

## 四、调用时的最小检查

每次调用前后按以下清单检查：

| 阶段 | 调用前必须有 | 调用后必须交付 |
|:--|:--|:--|
| `analyze-paper` | 来源及版本 | `ANALYSIS_PACKAGE` |
| `deep-read` | 一个或多个来源包/实验包 | `MODE_C_PACKAGE`，且含 P/F/M/E/C |
| `discover-innovation` | 一个 `C` 连接点 | `IDEA_PACKAGE`，且含预测、对照和解释规则 |
| 实际执行 | 已批准的 Idea/协议和资源 | 原始数据、日志、图表、样本或操作记录 |
| `record-experiment` | 实际执行证据及计划版本 | `EXPERIMENT_RECORD`，且含所有运行和偏差 |
| `squeeze-experiment` | 一个或多个实验记录及原始证据 | `SQUEEZE_REPORT`，且含新 P/F/M/E 或明确无新增 |

如果某阶段材料不足，输出仍可生成，但必须将状态设为 `partial` 或 `unknown`，并在 `missing_evidence` 中列出补证据路径。

## 五、边界与禁止跳步

- 不允许 `analyze-paper → discover-innovation` 直接跳过证据矩阵；除非已有等价的 `MODE_C_PACKAGE`。
- 不允许 `discover-innovation → squeeze-experiment`；没有实际执行和 `EXPERIMENT_RECORD`，不能深挖实验。
- 不允许把 `EXPERIMENT_RECORD` 中的单次结果直接当成新机制；先由 `squeeze-experiment` 分析，再交 `deep-read` 做 P/F/M/E 语境化。
- 不允许把 `SQUEEZE_REPORT` 直接当成 `IDEA_PACKAGE`；要重新经过 Mode C 和创新筛选。
- 不允许因一次不理想结果就删除问题、隐藏偏差或把当前设置下未观察到的增益写成普遍结论。
- 原始证据、分析推断和待复验假设必须分栏或分层保存。
