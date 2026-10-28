# 四个矩阵问题的小范围验证：实际记录与深挖

2026-10-29。遵循`Idea → 实际执行 → record-experiment → squeeze-experiment → Mode C`。用户明确授权四项并行小验证，初步支持后可继续有限服务器验证；每项保留独立问题/证据/停止条件，不把软件通过或某一项支持合并成新方法成功。执行队列只看[交接](../../HandOver/Handover.md)，当前决定只在[研究状态](../research_status.md)。

## A. 报告资格因子：条件数值核验已完成

### EXPERIMENT_RECORD

| 字段 | 实际事实 |
|---|---|
| experiment_id / Idea / plan | LOCAL-REPORT-AVAILABILITY-20261029-attempt-1；[IDEA-REPORT-AVAILABILITY-FACTOR-20261029](../proposals/report_availability_factor_check_20261029.md)，1.0，C52/P13 |
| 目的/对象 | 两个已指定报告、两个合拆解释、空间比.2；固定p1=p2=.5，p3=.1/.25/.5/.75/.9。无抽样数据/MTAD/作者原run。 |
| 实际操作 | `python3 tools/analysis/report_availability_factor_check.py --output /Users/a1/Projects/association-code/runs/LOCAL-REPORT-AVAILABILITY-20261029-attempt-1`；逐簇log Bernoulli因子再作合拆比，独立核解析值1e−12。 |
| 环境/时间/终态 | 本地macOS标准库Python3；2026-10-28 20:18:44.689741–20:18:44.690348 UTC（北京时间10月29日04:18:44）；0.000693s，真实退出0，COMPLETED，5/5条件。 |
| 输入/对照/血缘 | Wolf已读公式/核心代码→条件假设→固定数值计算→完整summary。A两个合法来源、bad已知不可报告却计miss、A′条件修正；可报告但缺轨反对照保留miss。pD与空间项不重估。 |
| 原件 | [run](/Users/a1/Projects/association-code/runs/LOCAL-REPORT-AVAILABILITY-20261029-attempt-1/run.json)、[全条件summary](/Users/a1/Projects/association-code/runs/LOCAL-REPORT-AVAILABILITY-20261029-attempt-1/summary.json)、[程序](../../tools/analysis/report_availability_factor_check.py)。原目录不覆盖；无排除/重试/计划偏差。 |
| 资源/权限 | 0模型/GPU/优化/ZIP/reference/test；无个人/身份数据。数值误差容差不是效果阈值，没有统计CI或独立泛化。 |

主要结果为H1合/H2拆的条件相对评分，无真值正确率含义：

| p3 | 原A | 已知不可报告却计miss | 条件适配A′ | 来源确可报告但缺轨：正确保留miss |
|---|---:|---:|---:|---:|
| .10 | .8 | .888889 | .8 | .888889 |
| .25 | .8 | 1.066667 | .8 | 1.066667 |
| .50 | .8 | 1.6 | .8 | 1.6 |
| .75 | .8 | 3.2 | .8 | 3.2 |
| .90 | .8 | 8.0 | .8 | 8.0 |

### SQUEEZE_REPORT / 回流内容

- **DE数值**：全部5条件与解析`.8/(1−p3)`一致，4条件合拆次序翻转，适配A′恢复原.8。可报告反对照保留miss是正确事件模型，不能一概删缺报告因子。
- **AI机制**：同一额外miss因子按簇重复，K簇解释额外乘`(1−p3)^K`；偏置来自事件资格错配而非这里已固定的几何/pD重估。确定性核验不提供独立因果实验或原作者性能复现。
- **UN目标**：MTAD没有可靠availability生成说明；当前056/107也没有这种逐簇miss乘积。故该核验**不能解释其FP/尾部，不能认证该问题在MTAD发生**。unknown资格也不能直接设unity。
- **价值与分支**：为概率成组路线提供明确的适配条件和反对照；只有合法建立真实报告资格、且系统确使用miss模型时才转目标效应验证。本项不自动服务器扫数据/训练，不以数值通过冒称可改善幅度。

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: LOCAL-REPORT-AVAILABILITY-20261029-attempt-1
source_idea_id: IDEA-REPORT-AVAILABILITY-FACTOR-20261029
plan_version: '1.0'
execution_status: COMPLETED
research_question_and_hypothesis: '已知不可报告的错误miss是否改变合拆条件评分'
planned_vs_actual_design_and_controls: '5固定p3和全部对照完成，无偏差/重试'
actual_configuration: '本地标准库CPU，固定p1=p2=.5、空间比.2、log乘积，0数据/训练'
samples_materials_or_data_provenance: '已读Wolf公式→事前条件设定→确定性数值，非MTAD/模拟抽样'
all_runs_or_repetitions: '一次，5条件全部保留；不是5个独立实验样本'
results_with_uncertainty: '表中全部比值；解析容差1e-12，无抽样CI'
observations_and_deviations: '4/5翻转，适配全恢复；无计划偏差'
evidence_locations: '本节run/summary/程序链接'
known_limits_and_missing_evidence: '无真实availability与MTAD效果；当前系统无该因子'
next_skill: squeeze-experiment
```

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-REPORT-AVAILABILITY-FACTOR-20261029
source_experiment_ids: [LOCAL-REPORT-AVAILABILITY-20261029-attempt-1]
verified_findings: '固定条件下K次miss造成明确评分倍率，资格适配可消除；正常可报告miss必须保留'
mechanism_status: '条件数学机制支持，非目标根因/方法效果'
negative_and_boundary_findings: 'p3=.1未翻转；当前056不含操作；unknown资格未建模'
uncertainty: '确定性有限条件，无总体估计'
candidate_mode_c_update: '细化C52L-Ma的适配条件与当前目标不适用边界'
next_skill: deep-read_Mode_C
```

## B. Fragment合法关系与容量

结果应来自独立[事前Idea](../proposals/fragment_identity_capacity_20261029.md)及真实一次运行，本节不预填结果。

## C. 固定分数之后的决定

结果应来自[事前Idea](../proposals/frozen_score_output_decision_diagnostic_20261029.md)及真实一次运行，本节不预填结果。

## D. 生成支撑与关联评分

输入资格与可计算范围见[独立Idea/审计](../proposals/generated_support_score_diagnostic_20261029.md)。不能把107未训练TAM的输出当有效关联评分，也不能拼上另一模型056分数认证因果。
