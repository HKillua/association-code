# 当前交接

更新：2026-10-30。只维护发布、领取与待核验事项；研究目标和下一步只读[研究状态](../docs/research_status.md)。旧长入口的[完整交接沿革](../docs/history/2026-10-30-research-process.md#原交接正文)已保留，原L2S/S2L和结果文件均在原位。

## 当前执行

**154/155/156已Record/Squeeze→Mode C1.66；157科学完成/独审RUNNING；158 READY由本聊天领取直接提交**：[157固定头完整评价](L2S-20261030-157.md)、[158当前多尺度速度](L2S-20261030-158.md)。156配方不扩大；157不新拟合/拼状态冠军，158仅独立反例机制pilot。新run各一次，152活动冻结目录不动。

**152A/B真实同一launcher/systemd均RUNNING，已实际训练；149/150/151/153已结案回流。** [152合同](L2S-20261030-152.md)、[启动原件与非终态回执](S2L-20261030-152-start.md)：全T2100/46126组，PTFM150epoch/427800更新、DeepAF20epoch/80800更新，fresh seed20各一次；冻结bda61e5，unit分别`mtad-e358bc6e1a62.service`、`mtad-ea2cca7761dd.service`，GPU2/3按合同UUID。21:44:24实际training日志为epoch1第55/27更新；run.json每25更新落盘为50/25，两路T初始探针/PTFM归一化已完。不能把RUNNING当训练/全V评价完成。151原launcher/服务终态成功，整数曝光已核，旧148不续跑。新run各一次，无自动后继/抢卡/SSH值守。

149/150原独审入口失败保留、新v2限定复算成功；153原数值审计失败与新精度分离复核分别记录。完整[149/150 Record/Squeeze](../docs/reports/2026-10-30-temporal-unsupported-review.md)、[153 Record/Squeeze](../docs/reports/2026-10-30-tmtaff-training-physical-review.md)已回Mode C1.64，当前两配方未稳定升级、不扩大。151数据准备不等训练完成。

143/144/146科学及限定独立输出审计均COMPLETED/exit0，Record→Squeeze→Mode C1.61收尾完成；145原技术失败保留。此前批次已结案，不重复领取。

## 保全与执行边界

- 132：用户明确暂拿不到系统级OOMD日志、先保留现场。SUMMARY1000更新权重、计划、normalizer、缓存和失败产物不动；完整评价及另三臂未完成，不重启/重训或自动发布继续任务。[S2L132](S2L-20261029-132.md)。
- 107/113及143/144/146不重跑或续旧额度；新比较和科学验证另立明确合同。135仍为保留草案，不是READY。
- 按[服务器规则](../docs/server_autonomy.md)执行一次、独立冻结目录、最多四单卡/FIFO；失败保全，不自动重试或替换配方。历史READY和命令不提供当前运行许可。
- 原数据、权重、完整预测和服务器runs保持原位；本轮仅授权新合同执行，未新增自动跟进。

## 完成记录入口

[历史研究摘要](../docs/history/README.md)按实验链接完整Record/Squeeze；[保留结果索引](results/README.md)提供证据位置；本目录原`L2S-*.md`和`S2L-*.md`保留。需要旧过程时读[原交接正文](../docs/history/2026-10-30-research-process.md#原交接正文)，不从历史叙述恢复队列。
