---
name: record-experiment
description: >-
  Record a completed experiment or empirical study in a reproducible,
  discipline-agnostic format. Captures the question, design, samples or
  materials, conditions, configuration, data provenance, results, uncertainty,
  deviations, observations, and next actions. Works for computational,
  laboratory, clinical, engineering, field, survey, simulation, and mixed-
  methods research. Use when a run, pilot, benchmark, study, or experiment is
  complete and needs a durable record. Triggers on: "记录实验", "记录这次实验",
  "实验结果", "log experiment", "record this run", "experiment is complete",
  "实验做完了".
---

# Record Experiment — 通用实验记录

## 0. 在五 skill 流水线中的位置

本 skill 是执行事实层，消费 `discover-innovation` 生成的 `IDEA_PACKAGE`（若有），并结合实际日志、原始数据、样本或操作记录生成 `EXPERIMENT_RECORD`，交给 `squeeze-experiment`。它记录实际发生了什么，不替执行者补写计划外事实。

```text
IDEA_PACKAGE + 实际执行证据
              ↓
      record-experiment
              ↓ EXPERIMENT_RECORD
      squeeze-experiment
```

## 1. 适用范围与目标

本 skill 适用于计算实验、模型训练/评估、数值模拟、实验室实验、临床或行为研究、工程测试、田野研究、调查、A/B 测试、设计研究和混合方法研究。

目标是留下一个“别人能理解、自己以后能复核”的实验事实记录，而不是只记录一个最好看的指标。记录应同时保留：研究问题、实际操作、原始证据、结果及不确定性、偏差、意外观察和下一步。

如果只是读论文或整理文献，使用 `analyze-paper` / `deep-read`；如果要深挖已完成记录中的机制、稳健性和隐藏发现，再交给 `squeeze-experiment`。

## 2. 输入配置

执行前从用户、上下文、日志、实验本或项目文件中提取，并先声明：

```text
记录 ID：{唯一 ID}
实验/研究名称：{名称}
领域与研究类型：{领域 / computational / lab / clinical / field / survey / ...}
研究问题与主要假设：{问题}
记录日期与实验日期：{日期}
记录保存位置：{路径；如未指定则只在对话中输出}
相关计划/协议/预注册：{路径或无}
上游 Idea 包：{IDEA_PACKAGE ID/路径；无则写无}
```

路径不存在时标记为“未提供”，不要用猜测补齐。涉及受试者、患者、动物、敏感数据、危险材料或受限设备时，记录脱敏后的 ID、合规状态和受限证据位置，不把原始隐私数据复制到普通笔记。

## 2A. IDEA_PACKAGE 输入契约

若实验来自 `discover-innovation`，先保存原始 `IDEA_PACKAGE` 的 `idea_id` 和 `package_version`，至少读取：

```text
idea_id
source_matrix_id / selected_C_id
research_question
hypothesis_and_mechanism
testable_predictions
design_and_controls
sample_materials_or_experimental_units
primary_outcome_and_estimand
analysis_plan_and_uncertainty
stopping_and_interpretation_rules
```

这些是“计划值”。记录时必须另列“实际值”，并将每项差异写入偏差表；没有 Idea 包时，允许从协议、预注册或用户描述建立记录，但 `source_idea_id` 写 `none`。

## 3. 最小必填字段

在写入前检查以下字段。缺失字段可以保留空值并标记缺口，但不能伪造：

| 类别 | 必填信息 |
|:--|:--|
| 目的 | 研究问题、主要假设或估计目标 |
| 对象 | 实验单位、样本/材料/数据、来源和纳入排除规则 |
| 设计 | 组别/对照、时间线、重复、随机化/盲法（适用时） |
| 操作 | 干预、处理、输入、设备、环境或软件配置 |
| 结果 | 主要结果、单位、聚合方式、基线/对照及不确定性 |
| 证据 | 原始数据、日志、图表、代码、协议、样本或仪器记录位置 |
| 偏差 | 实际执行与计划的差异、异常和可能影响 |
| 状态 | 完成、部分完成、中断、探索性、pilot 或数据不完整 |

## 4. 工作流

```text
Step 1  收集实验事实
   ↓
Step 2  核对设计、数据血缘和执行偏差
   ↓
Step 3  记录结果、对照和不确定性
   ↓
Step 4  分离观察、解释和待复验假设
   ↓
Step 5  更新计划状态与后续路由
   ↓
Step 6  追加持久化记录并输出摘要
```

### Step 1：收集实验事实

从用户描述和原始记录中提取：

- 实验 ID、名称、日期、版本、操作者、设备/环境
- 研究问题、假设、主要和次要结果
- 样本、实验单位、材料、数据来源、组别和重复数
- 输入/干预/处理、关键参数、软件/硬件/试剂/仪器配置
- 训练、反应、测量、随访或运行时长
- 结果文件、日志、代码、协议、仪器导出和图表位置
- 发生的异常、重跑、排除、手工操作和未完成部分

如果是计算研究，记录代码版本、环境、随机种子、数据切分、训练/分析命令、checkpoint 或结果选择规则；如果是实验室/临床/田野研究，记录批次、操作者、时间、地点、设备校准、协议偏差和伦理/注册信息；如果是调查/质性研究，记录问卷/访谈版本、招募、编码本、编码者和缺失处理。

### Step 2：核对设计、数据血缘和偏差

建立简短数据血缘：

```text
原始材料/输入 → 清洗或预处理 → 中间数据 → 分析/测量 → 结果图表 → 结论
```

检查：

- 实际样本、组别、重复和时间线是否符合计划？
- 数据是否发生泄漏、标签变化、不可逆筛选、缺失或异常处理？
- 关键配置、设备、试剂、环境、代码和依赖是否可追溯？
- 哪些计划外变化可能影响主要结果？
- 是否存在安全、隐私、受试者负担、动物福利、环境或知识产权风险？

不要把偏差直接写成结果原因；先记录事实，再标注可能影响和待核验事项。

### Step 3：记录结果、对照和不确定性

优先记录主要结果，再记录次要结果和资源/可用性结果。通用表格：

| 结果 | 实验条件 | 对照/基线 | 差异或效应 | 不确定性 | 实际意义 | 证据位置 |
|:--|:--|:--|:--|:--|:--|:--|
| {主要结果} | {条件} | {对照} | {数值/方向/主题} | {CI/SD/范围/可信区间/重复差异} | {解释} | {位置} |

注意：

- 无对照时明确写“无直接对照”，不要制造方向性比较。
- 没有数值指标时，记录定性主题、代表性案例、协议偏离或工程验收结果。
- 不只记录最佳一次运行；同时保存所有重复、失败运行、排除样本和选择规则。
- 将统计显著、实际意义、机制证据和可复现性分别描述。
- 如果结果未达到预期，写“在当前设置下观察到的结果”和可能解释，不把一次结果扩张为普遍结论。

### Step 4：分离观察、解释和待复验假设

将每个重要发现拆成三层：

| 层级  | 写法               | 示例占位                  |
| :-- | :--------------- | :-------------------- |
| 观察  | 原始数据/日志/图表直接显示什么 | “条件 B 的响应在第 3 个时间点上升” |
| 解释  | 当前最合理的机制或原因      | “可能与处理后的滞后效应有关”       |
| 待复验 | 如何区分替代解释         | “加入时间错位对照并增加独立批次”     |

对意外现象、零结果、异常值、错误案例、子组差异和资源变化也采用同样格式。记录“新发现/复现/方法验证/技术性观察”的来源属性，避免把复现他人方法写成自己的新现象。

### Step 5：更新计划状态与后续路由

把计划状态分为：

- `completed`：按主要设计完成，主要证据可用
- `partial`：只完成部分条件、样本、时间点或分析
- `interrupted`：运行中断，记录已获得证据
- `pilot/exploratory`：用于估计可行性或探索，不当作确认性结论
- `data-incomplete`：关键数据或记录缺失，暂不能做完整解释
- `completed-no-clear-difference`：完成且未观察到清晰差异，保留不确定性范围

然后给出条件化下一步：

| 当前发现 | 下一步路由 |
|:--|:--|
| 需要从多次实验中找规律 | `squeeze-experiment` |
| 出现新的问题/现象，需放回文献语境 | `deep-read` Mode C，整理为 P/F/M/E |
| 有可靠问题/现象，准备提出新研究 | `discover-innovation` |
| 只缺一次重跑或数据补齐 | 明确补齐项，不重写原记录 |
| 发现安全、伦理、隐私或合规风险 | 先暂停相关操作并按项目流程上报 |

### Step 6：持久化记录

只有在用户要求或项目工作流要求时写入文件。写入前读取已有记录，找到正确的实验位置，采用追加或版本小节，保留历史，不直接覆盖旧结果。若是计划内实验，可同步计划状态；计划外实验应标为计划外，不伪装成预先注册结果。

## 5. 标准记录模板

```markdown
## [EXP-{ID}] {实验/研究名称} — {实验日期}

### 记录元数据
- 记录日期：
- 包类型：EXPERIMENT_RECORD
- 包版本：research-pipeline-contract v1.0
- 实验 ID：
- 包 ID：{同实验 ID}
- 来源 IDs：{idea_id、协议、预注册或其他上游来源 ID}
- 生成日期：
- 来源 Idea ID：
- 计划版本：
- 领域/研究类型：
- 负责人/操作者：
- 代码/设备/协议/材料版本：
- 记录状态：completed / partial / interrupted / pilot / exploratory / data-incomplete
- 计划/预注册/协议：

### 研究目标
- 研究问题：
- 主要假设或估计目标：
- 主要结果：
- 次要结果：

### 设计与实际配置
- 实验单位、样本/材料/数据来源：
- 组别、对照、重复、随机化/盲法：
- 输入/干预/处理：
- 关键参数、环境、设备或软件配置：
- 实际时间线：
- 资源与成本：

### 数据与证据血缘
- 原始数据/材料：
- 预处理/清洗：
- 中间数据与分析脚本：
- 日志、代码、图表、仪器记录或样本位置：
- 随机性、版本和环境：

### 结果
| 结果 | 条件 | 对照/基线 | 差异/效应 | 不确定性 | 意义 | 证据位置 |
|:--|:--|:--|:--|:--|:--|:--|

### 观察与解释
| 类型 | 观察 | 当前解释 | 待复验/替代解释 | 证据位置 |
|:--|:--|:--|:--|:--|
| 主要观察 | | | | |
| 意外现象 | | | | |
| 技术/质量 | | | | |

### 偏差与限制
- 计划与实际差异：
- 排除、缺失、异常和失败运行：
- 可能影响结果的因素：
- 伦理、安全、隐私或合规事项：

### 结论边界与下一步
- 当前证据支持：
- 当前证据不能支持：
- 下一项最低成本检查：
- 后续路由：squeeze-experiment / deep-read Mode C / discover-innovation / 补数据
- 下一 skill：squeeze-experiment
```

## 6. 标准交接产出：EXPERIMENT_RECORD

为了让 `squeeze-experiment` 可以直接开始，记录至少提供以下包字段：

```text
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id
source_idea_id
plan_version
execution_status
research_question
hypothesis
planned_vs_actual_design_and_controls
actual_configuration
samples_materials_or_data_provenance
results_with_uncertainty
all_runs_or_repetitions
observations_and_deviations
evidence_locations
known_limits_and_missing_evidence
next_routes
next_skill: squeeze-experiment
```

缺失字段由 `squeeze-experiment` 标记为待补证据，不得从结果反推配置。新现象必须保留“观察—解释—待复验”三层；后续由 `squeeze-experiment` 形成 `SQUEEZE_REPORT`，再交给 `deep-read` Mode C。

## 7. 禁止行为

- 不只记录一个最佳指标、最佳样本或最好一次运行。
- 不把计划外调参、后验排除或选择规则隐藏在结果中。
- 不把观察、解释、因果结论和文献中的已知规律混为一谈。
- 不凭缺失日志、代码、数据、协议或仪器记录补写细节。
- 不把一次实验的结果写成普遍结论；明确不确定性、边界和待复验项。
- 不覆盖已有记录；追加新版本并保留原始记录和证据位置。
- 不复制受限原始数据、个人信息或敏感材料到不受控的实验笔记中。
