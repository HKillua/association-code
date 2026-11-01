# 当前交接

更新：2026-11-02。这里只维护执行、待核验和保全事项；研究目标/结论看[研究状态](../docs/research_status.md)。历次完整合同和回执从[交接记录索引](README.md)查找，整理前正文已[完整保留](../docs/history/2026-11-02-research-process.md#原交接正文)。

## 当前执行

**152A/B原全量训练仍在运行，没有新增READY。** [原合同](L2S-20261030-152.md)、[不可变启动回执](S2L-20261030-152-start.md)。全T2100/46126组，fresh seed20各一次，冻结bda61e5，比较对象仍原147。GPU2/3按合同UUID；不得加157/178、改预算或活动checkout。

|任务|固定额度|北京时间2026-11-02 00:06:08最后有界核验|原服务|
|---|---|---|---|
|152A PTFM|150epoch/427800更新|RUNNING/training，完成70epoch/201525更新，0全V场|`mtad-e358bc6e1a62.service`，active/running、Restart=no|
|152B DeepAF|20epoch/80800更新|RUNNING/training，完成10epoch/43975更新，0全V场|`mtad-ea2cca7761dd.service`，active/running、Restart=no|

运行中的Result=success/ExecMainStatus=0不当终态成功；训练loss不能替代固定末轮物理评价。终态后按原合同取必要公共原件，核完整输入/命令/退出、人口、指标、尾部和成本，Record→Squeeze→Mode C。保持同一run与launcher，不重投或续旧148预算。

## 本轮收尾与保全

- 194科学与规定独审均已成功，66公共原件及完整记录已核齐，无待审/待领；193技术失败独立保全。[194终态回执](S2L-20261101-194.md)、[193失败回执](S2L-20261101-193.md)。原science/audit和其他已结案目录冻结，无自动195/V/全300/新seed。
- 132：用户暂拿不到系统OOMD日志、要求保留现场；1000更新权重、计划、normalizer、缓存及失败产物不改，不重训/重启。[回执](S2L-20261029-132.md)。原107/113及143/144/146也不重跑。
- 其他100–194历史任务与失败/修复记录从[记录索引](README.md)、[完整报告](../docs/reports/README.md)按需读取，历史READY不提供当前执行许可。135仍为保留草案。

## 执行与回传边界

按[服务器规则](../docs/server_autonomy.md)一次任务、独立冻结目录、最多四单卡/FIFO；失败保全，不自动重试或替换配方，不触碰他人进程。墙时advisory、不加训练额度，不建SSH值守/守护或跨聊天消息。

数据、权重、原身份/坐标、大预测及服务器runs原位保留；公共小型原件和完整回执按原合同回主仓/资料库。本次仅整理文档入口及导航，无新科学任务或自动跟进。Git主仓可同步文档，活动checkout不得pull。
