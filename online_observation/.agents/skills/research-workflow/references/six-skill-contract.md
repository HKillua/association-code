# 六技能交接契约

版本：`research-pipeline-contract v2.0`。基于压缩包中的 v1.0 流程，增加可选 ML 迁移分支、入口路由与迁移包，不改变五个原有技能的专业产出职责。

## 原始规则与兼容方式

[原始说明](original-five-skill-workflow.md)保留五类包的字段和证据回流规则。源技能继续生成自己的 v1.0 包；总控在工作流索引中记录 `workflow_contract_version: research-pipeline-contract v2.0`。旧包保持原始版本，不因加入总控就改称按 v2.0 生成。

兼容已有材料时使用映射或补充层，不覆盖源包：

| 可能存在的字段差异 | 总控处理 |
| --- | --- |
| `research_question`、`hypothesis` 分开；下游要求 `research_question_and_hypothesis` | 合并引用两个原字段，保留原文与来源 |
| 元数据或 `evidence_locations` 缺失；已有 `evidence_index` | 从真实信息补充元数据和证据索引引用；无法获得则列缺口 |
| Mode C 表格／笔记包含等价 P/F/M/E/C，但没有包名 | 建立 ID 与字段映射；不因形式重读所有文献 |
| 旧 `next_skill` 指向 `discover-innovation`，本轮需要先迁移 | 保留源包字段，索引中另记 `actual_next_stage: transfer-ml-methods` |

正式研究方案需可追溯的 Mode C 材料。独立来源分析、记录历史实验、单独迁移判断不强制补齐整条上游链。历史实验没有 Idea 时将 `source_idea_id: none`，不能事后伪造预先假设。

## 共享元数据与证据状态

沿用：`package_type`、`package_version`、`package_id`、`created_at`、`source_ids`、`evidence_status`、`evidence_locations`、`missing_evidence`、`next_skill`。原包缺失的字段可在总控补充层保存。`created_at` 是真实生成日期，不能倒填为实验日期。

包级 `verified` 表示所述已知事实有直接证据且边界已记录；`partial` 表示重要缺口未解；`unknown` 表示材料不足。即使论文事实已核实，迁移有效性和创新性仍可未知。每个关键论断另保留 `SC/DE/AI/UN`，两种层级不互相代替。

```text
analysis_id → source_ids / 候选 P/F
matrix_id → P/F/M/E / connection_C_id
transfer_id → matrix_id（如有）/ connection_C_id（如有）/ A、B、迁移案例来源
idea_id → matrix_id / selected_C_id / transfer_id（可选）
experiment_id → idea_id（可无）/ plan_version / 原始执行证据
squeeze_id → experiment_id(s) / 新 P/F/M/E
后续 matrix_id → 先前 matrix_id / squeeze_id / 新增来源
```

`selected_C_id` 保持原 Idea 接口；总控映射到 `connection_C_id`。迁移案例使用 `transfer_case_ids`，不要用连接点 ID 代替案例文献。

## 新增 TRANSFER_PACKAGE

总控对迁移报告的结构化包装，由 `transfer-ml-methods` 的实质输出整理而来。完整研究方案交接时使用；短问答只保留相关判断及证据，无需空填全部字段。

```yaml
package_type: TRANSFER_PACKAGE
package_version: research-pipeline-contract v2.0
package_id: 与 transfer_id 相同
transfer_id: 唯一迁移包 ID
created_at: 实际生成日期
source_ids: 原矩阵、目标材料、源方法与案例来源 ID
source_matrix_id: 已有 matrix_id；独立迁移判断无矩阵时为 none
connection_C_id: 原连接点 ID；独立迁移判断未建立时为 none
target_A:
  task_and_baseline_version: 目标任务、Baseline 与版本
  data_and_evaluation_protocol: 数据与评价协议
  failure_observations: 失败现象、条件与来源；未实测的假设明确标注
  candidate_mechanisms_and_alternatives: 候选机制、竞争解释和证据状态
source_B:
  source_ids_and_version: 核验过的具体方法版本及来源
  original_problem: 源方法实际解决的缺陷
  core_mechanism_and_locations: 核心作用链及公式、图表或代码位置
  enabling_conditions: 条件、证据与必要性状态
  failure_boundaries: 原方法的失效边界或相关未知项
transfer_case_ids: 已有迁移案例来源；未找到时为 none 并记录检索范围
case_adaptations_and_limits: 案例的实际改造与不能直接推广的部分
condition_gap_table: 条件在 A 中已满足、可建立、不满足或未知；如何测量
adapted_A_prime: 修改位置、建立条件的方法、代价与新混杂
preserved_or_modified_B: 保留或修改的源机制及理由
transfer_hypothesis: 条件 S 下，B 经机制 M 改善 Z；可观察预测与否证结果
attribution_design: A、A′、A′+B 的设计与可比性；必要额外对照
condition_and_mechanism_measurements: 条件、目标现象、机制、整体效果与成本
decision: worth-minimal-validation / needs-evidence / unsuitable-now
evidence_status: verified / partial / unknown
evidence_locations: 定位或索引引用
missing_evidence: 明确缺口与补证据路径；无则为 none
next_skill: discover-innovation；用户只问可迁移性时为 none
```

没有案例不表示没有人迁移过。没有 A′ 的必要性证据时，不自动断言“必须修改 A”；无条件适配的迁移可令 A′ 与 A 相同并说明，A 与 A′ 实验可合并，不伪造三组差异。A+B、破坏条件、移除核心运算、匹配容量或预算等附加对照由假设决定。

## IDEA_PACKAGE 的增量

沿用原五技能契约的全部 Idea 字段，迁移方案附加：

| 字段 | 内容 |
| --- | --- |
| `source_transfer_id` | 迁移包 ID；未走分支时为 none |
| `transfer_condition_commitments` | 哪些条件已满足、哪些先验证；不得把拟建立条件写成事实 |
| `adaptation_vs_method_attribution` | A→A′ 与 A′→A′+B 的比较、机制测量和限制 |
| `plan_version` | 实验计划版本，记录阶段引用它核对偏差 |

`discover-innovation` 核对迁移方案与研究问题的对应、新颖性检索、资源限制和对照可实施性；保留已有条件分析，不重新生成同义假设。已有 Idea 时修订原 Idea 的版本，保留原版本及修改依据。

## 实验记录、深挖与回流

记录逐项保留方案的组别、条件测量、机制预测和实际偏差；对应运行不存在时写未执行，不补结果。深挖分别回答适配是否有收益、B 在适配系统中是否有增量、条件／机制观测是否支持解释、整体任务价值及成本是否可接受。

| 真实观察 | 解释边界与后续 |
| --- | --- |
| A′ 改善而 A′+B 无可靠增量 | 收益可能来自适配；检查 B 实现、测量灵敏度与条件 |
| 总指标改善而机制预测未出现 | 证据不足以支持所提机制；考察容量、预算或其他解释 |
| 条件建立但目标现象未改善 | 当前设置挑战迁移假设；核对机制匹配与实现边界 |
| 缺少运行／日志或多项变化同时发生 | 保持部分或未知状态，列明归因缺口 |

先形成 `SQUEEZE_REPORT`，再由 Mode C 增量纳入新 P/F/M/E，与文献证据对照。只在新证据或明确竞争解释值得检验时再进入迁移或创新设计；不以“还没涨点”为由无边界重复。

## 可选工作流索引

长期推进且用户要求保存时，可用项目现有文档或建立 `research/workflow-index.md`，记录实际路径，无须每轮复制完整包：

```yaml
workflow_contract_version: research-pipeline-contract v2.0
task: 用户的研究任务
current_stage: 实际所在阶段
requested_endpoint: 本轮终点
selected_connection_id: 当前连接点或 none
packages: 包 ID、类型、版本、路径与关键证据状态
stage_history: 阶段、输入 ID、输出 ID、生成日期、推进或暂缓理由
active_unknowns: 影响决策的未知项
next_discriminating_step: 下一项能改变判断的检查
```

没有写入要求时在对话中保留必要交接信息。不得用索引状态替代原始数据或实际操作授权。

## 可选文献资产映射

使用文献库或本地知识库时，按 [文献资产与知识库规范](literature-knowledge-base.md) 在上述索引或现有来源清单中补充映射，不复制所有正文，也不升级源包版本。本节是 v2.0 总控的可选补充层，旧索引无需补齐才能继续研究。

| 补充信息 | 用途 |
| --- | --- |
| 来源 ID、标准标识、具体版本及实际原件位置 | 保证结论可回溯到本轮使用的材料 |
| 可选 Zotero 连接上下文、库／条目／附件定位 | 连接文献库条目，未连接时不编造 |
| 全文可用性与实际阅读覆盖 | 区分保存、读取和证据核验 |
| 分析笔记与包 ID／版本／路径 | 复用专业产出，不另造竞争性分析包 |
| 下游依赖与待复核原因 | 来源变化后仅定位受影响材料，不静默改写历史 |

来源清单与专业包分别维护信息；缺少前者不代表后者不存在。知识库问答可引用这些路径定位实际证据，不生成新的研究包类型。
