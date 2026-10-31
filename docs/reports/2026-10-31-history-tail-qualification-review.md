# 179：历史候选新增害的T侧条件分离

2026-10-31，REC-HISTORY-TAIL-QUALIFICATION-179 → SQ-HISTORY-TAIL-QUALIFICATION-179，research-pipeline-contract v1.0。来源[Idea179/plan1.0](../proposals/history_tail_qualification_20261031.md)、Mode C1.87/C60、F283/E186。execution_status=completed，描述性计算核查；T32/全部860目标、两种原固定组、0V/test/训练/新部署。不是新方法效果或因果认证。

## Record：实际执行、对象与完整性

代码fcb4aca，独立`association-code-run179`，run `runs/LEVEL1-HISTORY-TAIL-QUALIFICATION-20261031-179-attempt-1`，原launcher同前缀`-launch-attempt-1`；实际argv/env见[L2S179](../../HandOver/L2S-20261031-179.md)。CPU16–19×4/无GPU/RSS4GiB/900s advisory；payload09:05:12.184609Z–09:05:44.316045Z，launcher32.243969s、程序27.694008s/RSS498098176B。unit`mtad-acb3599b0192.service`真实exit0/inactive/dead/Resultsuccess/Restartno，Linger沿用已核yes。

只读178 T原raw/partitions/policy/leave-component参数及127原生事件/运动/评价，base与175数组exact绑定。HARD/STRUCT全部共同有效点37993/37745与178独审一致，新增>1km害104/75及多支持零变化亦一致。按原首anchor首输出对应，重复/未知组另计，不真值净化、改标签或筛难例；所有有效点均描述，不只选害点。原生times/offsets独立于reference产生时距特征，reference仅作后验误差与纯度分析。

全部45公共原件761222B＋终态服务/文件大小清单回[179证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_history_tail/179)，逐文件大小核齐；tar446826B。私有逐query/目标对应及上游原坐标/权重留server。一次科学、无失败/重投/配方偏差。READY前软件用例核有好候选/全坏候选、原生时距、重复/无效/纯组，不能当MTAD分数；本轮是保存结果资格核查，不是另一次预测或物理真值独审。

## Record：主要结果与条件

“可避免害”定义组内最佳既有片段误差≤base误差＋1km，oracle不可部署。所有新害中base本身已有近乎相同节点候选；本次问题并非这些点完全缺少较好组内状态，也不代表全部绝对大错如此。

|T固定组/域|新增害|oracle可避免害|base已有片段候选|所选传播时距中位数秒|top1/2分数差中位数|所选特征clip±8比例|
|---|---:|---:|---:|---:|---:|---:|
|HARD 全部|104|104|104|1261.373|.325125|0|
|STRUCT 全部|75|75|75|914.762|.329630|0|
|HARD 纯单支持|19|19|19|2132.538|.120702|0|
|STRUCT 纯单支持|20|20|20|1969.264|.137206|0|
|HARD 纯零支持|3|3|3|698.757|.112770|0|
|STRUCT 纯零支持|0|0|0|未定义|未定义|未定义|
|HARD 混零支持|11|11|11|914.762|.104125|0|
|STRUCT 混零支持|38|38|38|735.048|.416843|0|

纯单支持改善>1km分别822/817点，所选age中位162.567/160.504秒，top间隔.252830/.249448。单支持全部新增害间隔中位H.327922/S.235678，与改善点.259597/.249567相近，分布有重叠；大间隔也存在害，不能把分数间隔当已校准置信度或直接设小间隔门限。害的age更大是关联，但最短新增害仅13.579/14.968秒，不能将全部错误归为超长传播，亦不能据此在V设age cap。

HARD零支持1108点，全部组原生时间包络前/后189/262，内部657；STRUCT967点，前/后121/223，内部623。零支持新增害分别14/38，其中组包络内部9/33、整体结束后各5，整体开始前0。内部包络仅保证组中两侧有事件，不保证同一个源有两侧、不保证事件属于同一真实目标；源内桥接资格仍须由可见输入建立。HARD纯零的3个新增害都位于组时间包络内部，所选片段却在自身native区间外传播。

## Squeeze：已回答与竞争解释

DE：这些178新增害均存在较好当前候选，不能用“组内所有状态都坏”解释；新纯单支持错误不能全部归混组。clip没有发生，因此±8截断不是这些点的直接触发条件。DE同时限制简单小margin或长age门限：误差与评分间隔/年龄关系并非无重叠的分界。

AI：按log误差回归训练的评分与实际物理选择代价可能存在接口差异，合法T误差成本可作为独立监督对照；但没有loss替换前，不能宣log变换是根因。AI：内部缺口若有合法同源两侧事件，可以研究条件插值以减少单片段外推；错误组/源偏差、噪声端点或单侧事件会破坏条件。不是“零支持都应该插值”，也不证明当前事件能正确连接或GP covariance已校准。

本核查无区间/新泛化分数，T已参与上游训练；引用的V反例仍仅来自178原记录，不访问V新数据。全部原强控和当前平均收益保留。回Mode C1.88/F284/E187后，把候选选择成本和零域原生历史条件与独立来源合并，再各立可归因方法实验；不将此诊断当完成本轮改进目标。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-HISTORY-TAIL-QUALIFICATION-179
input_experiment_record_ids: [REC-HISTORY-TAIL-QUALIFICATION-179]
source_idea_id: IDEA-HISTORY-TAIL-QUALIFICATION-179
plan_version: '1.0'
evidence_status: verified
claim_scope: T32_all_saved_main_points_two_frozen_fronts_no_V_training_or_new_deployment
primary_result_with_uncertainty: all104_75_new_harms_have_better_existing_candidates_and_base_node_no_clip_no_independent_generalization
process_dynamics: one_CPU_run_32.243969_seconds
mechanism_and_intermediate_findings: score_cost_alignment_and_internal_native_gap_are_candidates_not_proved_causes
errors_heterogeneity_and_boundary_conditions: pure_one19_20_zero_internal9_33_outside_after5_5_margin_age_overlap
robustness_sensitivity_and_alternative_explanations: full_denominators_and_harms_bound_to178_independent_audit_GT_oracles_not_deployed
new_F: F284
new_E: E187
new_M: none
next_skill: deep-read_Mode_C
```
