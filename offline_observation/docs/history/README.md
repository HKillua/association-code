# 研究历史摘要

当前目标/结论只看[研究状态](../research_status.md)，执行只看[交接](../../HandOver/Handover.md)。本页只导航；历史READY、预算与“下一步”不恢复成任务。

## 研究推导与原过程

|所需材料|入口|
|---|---|
|结案实验、失败、Record/Squeeze与反例|[完整报告索引](../reports/README.md)|
|来源→矩阵→Idea、强控及未采纳理由|[全部提案](../proposals/README.md)；矩阵从研究状态材料表进入|
|原合同、启动和终态|[交接记录索引](../../HandOver/README.md)|
|脱敏数值与日志原件|[结果索引](../../HandOver/results/README.md)|
|2026-10-30状态/交接完整文本|[过程全文](2026-10-30-research-process.md)|
|2026-11-02状态/交接完整文本|[过程全文](2026-11-02-research-process.md)|
|两轮整理的范围和验证|[整理记录](2026-11-02-repository-organization.md)|
|2026-11-10存储瘦身、工具退役与入口纠正|[整理与恢复记录](2026-11-10-repository-maintenance.md)|

## 保留的研究事实

原历史页逐阶段的数值、阴性结果、解释边界和去留依据已完整无损保存在[历史页全文gzip](2026-11-02-history-full.md.gz)。日常直接读对应实验报告，不再维护另一份长篇成绩表。解压内容与精简前全文逐字节相同，不以当前判断改写历史。

```sh
gzip -dc docs/history/2026-11-02-history-full.md.gz
```

## 本轮精简（2026-10-27）

本节为旧整理定位：完整范围、验收与恢复命令见上面的历史页全文；删除前版本[903f679](https://github.com/HKillua/association-code/tree/903f679)。

## 2026-10-30退役入口与重复文件清理

完整去留依据及通用测试迁移记录见历史页全文；删除前版本[cba6054](https://github.com/HKillua/association-code/tree/cba60543a329e1c67007217101b9ae5bdf39f082)。原模型/训练/评价与复现工具继续保留，不因旧实验结案清理共享依赖。

## 原件与恢复

更早的单文件定位：[9eefd2c](https://github.com/HKillua/association-code/tree/9eefd2c)、[1a25183](https://github.com/HKillua/association-code/tree/1a25183)、[e7e4a11](https://github.com/HKillua/association-code/tree/e7e4a11)、[5776cd3](https://github.com/HKillua/association-code/tree/5776cd3)、[612afea](https://github.com/HKillua/association-code/tree/612afea)。提交号仅定位历史；按需读取单文件，不恢复退役队列。数据、真实权重、预测和服务器runs保持原位。
