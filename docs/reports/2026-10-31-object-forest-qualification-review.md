# 171：T对象/森林资格检查的技术失败与修复证据

## EXPERIMENT_RECORD — REC-OBJECT-FOREST-171

2026-10-31，plan1.0/[事前Idea](../proposals/structured_association_20261031.md)，C57/Mode C1.77。原T32/0训练/无GPU/不新开V/test，未改变168/170/159/152。冻结e70f426、新run/launcher attempt1；真实命令见[L2S171](../../HandOver/L2S-20261031-171.md)与[原request](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_structured_association/171/launcher/request.json)。

原unit `mtad-a0ea4b19a7df.service`：Linger=yes/Restart=no，2026-10-31T05:36:57Z启动，05:37:07.946543Z终止；launcher真实exit1/FAILED，系统服务failed/ExecMainStatus1，无自动重试。worker总10.591636s，science开始05:37:06.190599Z、内部.330745s，峰值RSS393932800B；完成0场/0更新/0预测。环境加载与托管计总墙时，不将science内部时间当全成本。

原因：`point_coverage`用中央target_ids查所有aligned_target_ids，遇到没有参考的组输出（空target ID）抛`KeyError: ''`。原170评价本已保留这类组；不是人口数据坏/模型效果差。全部12个公共原件（science run＋11 launcher含stderr/真实退出）已回传，[证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_structured_association/171/)；身份/坐标继续留原170服务器。未产summary、没有对象/梯度科学结论；计划终点未达成，不填0为无效果。技术失败算一次attempt，不替换原run。

## SQUEEZE_REPORT — SQ-OBJECT-FOREST-171

只挤出可支持事实：保存输出并不都映射到中央参考目标；无参考输出仍须保留组/重复/覆盖其他统计，**不应被错误加进reference有效query**。失败后只读原首T场核验：WS/CTXCAL分别11/8个未匹配aligned row，其中有效参考位置query均0。此后验读取不是171已完成科学，未读取新V或选择阈值。

修复条件：仅对中央target不包含且所有eligible query为false的行不建立target-query；若不在中央人口却有任何有效参考query则明确报错。新增fixture涵盖两种情形，并保留所有目标/原组统计；4项unittest全部通过。修复只改统计映射，无边分数/森林/标签/噪声/目标/曝光变化。没有证明结构监督有效或目标不匹配是根因。

回流Mode C1.78（E176/F273）：技术边界，不新增机制/Idea连接；171固定一次关闭。**新172若发布，为明确修复资格任务，不是171自动续跑**，原冻结目录/日志不动。来源分析/方案与失败均保留，不因修格式宣布科研完成。

```yaml
package_type: SQUEEZE_REPORT
record_id: REC-OBJECT-FOREST-171
idea_id: IDEA-OBJECT-FOREST-QUALIFICATION-20261031-171
source_ids: [C57, E174, E175]
design: original_T32_zero_training_forest_qualification
actual_execution: failed_before_first_scene_zero_updates
results_with_uncertainty: no_scientific_estimate_technical_traceback_only
plan_deviations: [planned_T32_endpoint_unmet, post_failure_first_T_unmatched_row_read]
reproducibility: original_command_request_logs_service_exit_saved
evidence_status: verified_technical_failure
limitations: scientific_qualification_unknown
missing_evidence: [completed_T_qualification]
next_skill: deep-read_Mode_C
```
