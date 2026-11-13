---
handover_id: L2S-YYYYMMDD-ONLINE-TASK 或 S2L-YYYYMMDD-ONLINE-TASK
direction: LOCAL_TO_SERVER 或 SERVER_TO_LOCAL
status: DRAFT 或 READY 或 RUNNING 或 COMPLETED 或 FAILED 或 STOP_REQUESTED 或 CANCELLED
updated_by: LOCAL_AI 或 SERVER_AI
updated_at: 实际时间及偏移
task_id: 本轮任务
next_actor: USER 或 SERVER_AI 或 LOCAL_AI
---

# 本轮任务或回执

- 执行前依据：链接本次适用的来源分析、Mode C 连接和 Idea。给出要回答的问题、基线与对照、报文与发布时间权限、场景、指标、结束与回退条件。
- 资源与命令：固定尝试次数、设备及预算，记录实际完整命令、独立输出目录和退出码读取方式。
- 状态与结果：领取或结束时间、实际设备、耗时、真实退出码、完成范围、覆盖与指标、配置偏差和原件位置。未完成项按事实标记。
- 实验结案：关联实际 Record、Squeeze 和 Mode C 回流。运行失败或偏离设计时保留原证据、影响和后续判定。

各方更新自己的独立 L2S/S2L。当前任务登记在[交接](Handover.md)，按[完整科研流程](../docs/research_workflow.md)和[现场约定](../docs/server_autonomy.md)执行。已存在的方案和报告可直接引用。
