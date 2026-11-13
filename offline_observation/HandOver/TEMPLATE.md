---
handover_id: L2S-YYYYMMDD-NNN 或 S2L-YYYYMMDD-NNN
direction: LOCAL_TO_SERVER 或 SERVER_TO_LOCAL
status: DRAFT 或 READY 或 RUNNING 或 COMPLETED 或 FAILED 或 STOP_REQUESTED 或 CANCELLED
updated_by: LOCAL_AI 或 SERVER_AI
updated_at: 实际时间及偏移
task_id: 本轮任务
next_actor: USER 或 SERVER_AI 或 LOCAL_AI
---

# 本轮任务或回执

- 执行前依据：链接本次适用的来源分析、Mode C 连接和 Idea。给出要回答的问题、基线与对照、输入权限和人口、指标、结束与回退条件。
- 资源：固定更新/尝试次数及建议时长、单卡/CPU线程、并发和尝试次数。没有授权不重试。
- 命令与输出：完整实际命令、唯一新输出目录、托管服务和退出码读取方式；不记录文件/代码摘要门槛。
- 状态与结果：领取/结束时间、实际设备、退出码、完成更新、人口/指标、小型证据及大产物服务器路径。未完成如实记录。
- 回传范围：聚合指标和必要逐场指标；完整预测/路由/逐组树留服务器。精简文件使用明确名称并说明省略字段，不上传整个run，不伪装成原始完整summary。
- 实验结案：关联实际 Record、Squeeze 和 Mode C 回流。运行失败或偏离设计时保留原证据、影响和后续判定。

各方更新自己的独立 L2S/S2L。当前任务登记在[交接](Handover.md)，按[完整科研流程](../docs/research_workflow.md)和[现场约定](../docs/server_autonomy.md)执行。结案后的过时材料按[归档规则](results/README.md)总结，完整版本留在 Git 历史。
