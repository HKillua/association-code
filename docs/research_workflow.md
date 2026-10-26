# MTAD科研流程：项目强制规范

采用日期：2026-10-27。用户明确指定以下两份材料为整个项目必须遵循的科研方法，并要求覆盖飞书全文，不能只选五技能与避免死循环两节。

- [飞书《视频文案》](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd)：2026-10-27通过浏览器逐节、滚动读取全文七个主题；页面显示最近修改为10月13日22:27，不把访问日写成原文更新日。
- 用户附件 `5-skill-research-workflow.md`，版本 `research-pipeline-contract v1.0`：全文已读，完整字段与禁止跳步规则保存在本文附录。源文件在本地 `/Users/a1/Downloads/skills/5-skill-research-workflow.md`。

这是**用户采纳的项目约定**，不是网页自行授予权限。覆盖本地阅读、研究设计、服务器实验交接、结果分析和写作；旧技能路由、历史方案与本规范冲突时，以用户最新要求和本规范为准。后续网页变化须实际核读并记录差异，不自动改项目规则。下文是来源要点的归纳及本项目落地解释，不冒称逐字原文。

## 1. 飞书全文阅读覆盖与落地

| 已读主题／定位 | 原文覆盖的内容 | 本项目落实 |
|---|---|---|
| [第1章](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#Wq7gdDT11otUQsx36z6co1qhnjb)：项目组织 | 数据／权重、只读来源、演进代码、实验产物、研究管理与写作分工；论文解析→笔记→实验→报告→写作 | 原PDF／公开代码快照保持来源可辨；自己的适配另记差异。重用现有目录与冻结运行树，不照抄一套新目录、分支或MinerU依赖 |
| 第1章：必需记录与可选材料 | 现象、写作素材、论文笔记、索引、最近分析、决策沿革、宏观计划、全部实验、写作读取清单；可选推理链、完整解题路径、创新类型、章节写法、工作量、骨架改造位置、骨架全景及外部工具分析 | 每类信息都有现有载体，映射见下表；可选分析按当前问题调用，不为数量另开八份文档 |
| 第1章：baseline与论文阅读 | 优先近两年高引用、可复用实现及后续改进较多的基础；论文原文分析、P×F矩阵、组合探索、实验深挖 | 引用量是检索线索，源码可用性与可信效果另核；同源论文不只登记协议，还要提取问题、作用机制、条件、反例及与本项目的关系 |
| 第1章：1a–4b、骨架与创新类型 | 观察→归类→追问→因果链；替换→分离改动→消融→核查因果；已被多次改造的位置仍可能有空间；区分新问题、新解法、迁移与微调 | 因果链先标假设，靠代码／观测／干预验证。维护“已知／未知／尝试过”的骨架定位，不以论文数量判定某位置一定有问题或没有机会；创新分级用来反思贡献，不当作期刊录用标准 |
| [第2章](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#HiPYdeilTofYRfxHl1PcHnYYnId) | 按潜在收益与可解决性选方向；深挖到模块／运算；寻找领域B机制、领域C迁移经验，改造A使条件成立 | 先定要解释的具体失败，再检索同类机制；不以模块名称／能接张量替代机制适配；不得直接断言适配后已解决问题 |
| [第3章](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#QxYsdMheSoUZfGxPbYecO8Din7b) | 目录可灵活；baseline没有大量后继改造也能研究；强调实验信息利用、探索组合；检查问题是否仅特定数据集存在；理论源方法和应用迁移场景区分 | 工具与目录服务研究；没有源码或领域C不能无期限卡住。保留探索空间，同时记录边界，不将MTAD特有现象说成通用规律 |
| [五技能全流程](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#S2RFdTEteoRR0mxr52dcsM0SnKb) | Analyze、Mode C、Idea、实际执行、Record、Squeeze、回流；各自责任与标准产出 | 严格按下节及附件契约交接，不能把读了技能名、列了模块或写了包名当作完成 |
| [怎样做实验少走弯路](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#RUVSdvvaPoBqyXxOF8QcToLmnMe) | 修好局部缺陷未必改善整个模型；副作用、交互和评估噪声可能抵消收益；还须衡量代价 | 每个方案同时写局部机制预测、系统主要指标、可能退化项和计算成本；局部变好不自动晋级，整体未涨也不自动证明该机制必然有害 |
| [科研死循环](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#PoBfd1q3aoGzkRxbYFYcswhqnWd) | 先查已有解法与条件；尽量单变量、必要时研究交互；看整体；提前写成功／回退线；无效时回退而非叠补丁；每次实验增加知识 | 不拿实验次数、文档数当进展。关键旧结果未解释就不叠下一版；无新增证据时结束当前配方，保留负结果和有价值的新现象 |
| [跨领域方法迁移全文](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#QifSdePVUomGQwxR0npcYUe8nCf) | A的具体问题→B的机制及生效条件→C如何迁移→改造A；区分必要条件与原实现偶然选择；比较A、A′、A′+B；没有C不等于不可迁移 | 明确输入／分布／结构／目标条件：已满足、可建立、未知、冲突。先把A改成满足条件的A′，再验B是否有额外净收益；C案例缺失如实记，不虚构或无限搜索 |

### 既有文件承接全部记录职责

| 记录职责 | 当前载体，不另建平行系统 |
|---|---|
| 文档索引／当前结果／宏观推进 | [README](../README.md)、[研究状态](research_status.md)、[交接](../HandOver/Handover.md) |
| 已发现现象、最近实验、全部运行与偏差 | `docs/reports/`当前分析、`HandOver/results/`必要汇总、服务器原日志／预测；结案见[历史摘要](history/README.md)及Git原件位置 |
| 论文原件、阅读笔记、来源与证据索引 | 研究资料库 `biye-ronghe/papers/`、`docs/surveys/2026-10-26_mtad_actual_use/`，主仓仅保必要决策与来源链接 |
| 关键决策、方向调整 | [目标订正](reports/2026-10-26-goal-reconciliation.md)、研究状态、提案版本与Git历史 |
| baseline骨架、改造位置、方法／外部机制 | `docs/specs/`、对应 `docs/proposals/` 的实现关系、来源机制与已知／未知项 |
| 可用于写作的素材、写作读取清单 | 现有结果分析中的可支持结论／不可支持结论及证据位置；写作时从索引提取，不只摘最佳数字、不复制全部日志 |

### 对原文不同表述的处理

- 第3章保留随机组合以发现意外的探索建议；后文五技能契约要求证据交接与可检验设计。**探索组合可以是发现性研究，但须如实标探索、写明目的／范围／成本／记录与出口，仍走Idea和真实记录，不能把任意拼接冒称机制已成立。** 这条解释既不删除原文建议，也不将其变成当前自动开跑许可。
- 早期“反复追问找到根因”“很少改造处大概率无问题”等是研究启发，不是已证实规律。沿用后文对AI因果解释须验证、未知可标记的要求；不把“上一阶段几乎完全了解”解读为无限诊断或必须证明所有未知。
- a–d是作者用于说明贡献类型的分法，不强制本课题只能选a/b，也不推翻用户接受成熟机制与小幅海上适配的要求。引用高、改造多、模块新，均不能替代科学证据。

## 2. 必须遵守的研究顺序

`analyze-paper → ANALYSIS_PACKAGE → deep-read（仅Mode C）→ MODE_C_PACKAGE → discover-innovation → IDEA_PACKAGE → 实际执行 → record-experiment → EXPERIMENT_RECORD → squeeze-experiment → SQUEEZE_REPORT → deep-read（Mode C增量合并）`

- **先独立读懂来源，再问能否用于我们。** 每篇说明真实阅读覆盖、问题、样本／设计、机制与公式、结果／消融／不确定性、限制、实现与适用条件；SC作者主张、DE直接证据、AI分析推断、UN未知分开。不能围绕预选方案只搜支持句。
- **Mode C只做连接。** 有独立P/F/M/E编号、包含支持与不支持关系的P×F矩阵、C状态／取舍／未知；C不是完整网络或训练合同。问题与现象不能都变成一个实验分数或模块名。
- **Idea才做方法与实验设计。** 从有证据的C出发，给可被否证的预测、竞争解释、对照、主要指标／统计对象、不确定性、资源、终点和解释／回退规则。探索尚未通过筛选就标候选，不能因已写方案倒补一条看似事先存在的证据链。
- **跨域分析嵌入主流程。** `research-workflow`是协调器；`transfer-ml-methods`是从Mode C到Idea期间的辅助分析，不另起一条跳过矩阵的主链，不覆盖五技能的责任。补读B/C的重要来源先形成来源分析，再增量合并；正式计划最终归到一个Idea。
- **先记录事实，再分析，再回矩阵。** 全部运行、异常、排除、配置与计划偏差如实保留；单次好分数不直接写成机制。已完成历史实验可从记录阶段接入，缺原Idea就明确标缺，禁止倒写事先计划。已有等价分析可映射复用，不要求重训。
- **完整性是科学内容，不是文件数量。** 包可作为已有文档中的结构化小节；字段可引用已存在证据。统一元数据、ID链与必要边界必须齐或显式partial/unknown、missing_evidence。补能影响决策的缺口，不建立自动准入框架、额外模拟资格链或冗余检查。

附录给出全部最小字段和禁止跳步条款，不以这几条摘要代替正文。

## 3. 本轮具体纠正（2026-10-27）

独立只读审查确认：本轮同源论文笔记有原文定位和有价值的反例，但完整标准交接尚未完成，不能说五技能流程已经走完。

| 当前材料的问题 | 处理与下一步 |
|---|---|
| 精读素材的统一元数据、候选P/F、声明—证据—实现映射与缺证据项不齐 | 保留原事实；按论文补可复用的ANALYSIS映射，未补齐标partial，不因包名存在就宣布完成 |
| 部分C素材提前写成模型／C-D对照；缺完整P×F和统一来源ID | 设计想法暂存为未经筛选的草稿；先建问题／现象／机制／证据矩阵，再决定哪一个连接值得设计 |
| [观测约束融合](https://github.com/HKillua/association-code/blob/903f679/docs/proposals/observation_constrained_fusion.md)已被“主推”，而新精读正在围绕它找依据 | 收回已完成筛选的优先地位，标为待规范筛选候选；保留内容和历史，不默认必须实现它 |
| 098/100事实到新候选的记录→深挖→矩阵映射没有完整呈现 | 复用既有真实结果补链，区分实测失败、替代解释和待验因果；不重跑历史、不把本轮资料整理冒称新实验 |

本节是10月27日流程纠正记录，不是实时待办；当前阶段只看[研究状态](research_status.md)。科研仍先来源分析和Mode C，再由discover-innovation选方向。**不是暂停找创新，也不是为了格式重读所有论文；是停止先定答案、再拿阅读为答案背书。** 102是既有独立指标任务，原状态不因本次流程纠正被取消；本次没有新READY、训练或生产代码修改。

---

## 附录：用户提供的五技能契约

以下为用户附件正文，保留其版本、字段与规则；项目落地解释均在上文，不混入原契约。

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
