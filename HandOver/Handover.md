# 当前服务器任务

只按本页当前名单领取新任务；历史READY、旧配置或历史摘要都不构成开跑授权。已领取任务继续原隔离checkout，不重复启动。正常最多两个单卡任务，可共卡，不终止他人作业；时长仅提醒，失败不自动重试。

## 新任务队列

**用户2026-10-20明确回复“批准039—041三组”。** 各从头teacher1000步、完整V及正常完成后一次原四场CPU动态检查，累计预计7—8个单卡小时，各4小时仅提醒。待035/037释放名额，按039→040→041领取，可在两个名额内并行；无下游或T_test。具体设置与命令见独立交接和[三组规格](../docs/specs/cooperative-decoder-ablations.md)。

| 任务 | 状态/下一执行方 | 唯一干预 |
| --- | --- | --- |
| [039](L2S-20261020-039.md) | **READY / SERVER_AI，待领取** | 仅decoder pre-norm，encoder仍六层post-norm |
| [040](L2S-20261020-040.md) | **READY / SERVER_AI，待领取** | 仅decoder保留前两层，encoder仍六层 |
| [041](L2S-20261020-041.md) | **READY / SERVER_AI，待领取** | 仅teacher身份BCE贡献为0，保留物理self/cross与clip |

## 在途任务

按当前收到的回执，尚无035/037终态成绩。发布新交接不改变在途执行目录，也不代表039—041已启动。

| 任务 | 当前回执与范围 |
| --- | --- |
| [035 baseline关联](L2S-20261020-035.md) | **RUNNING / 已领取**，[S2L](S2L-20261020-035.md)。12:23在`association-code-run035`、GPU0提交；冻结033，新TAM1000，每100步完整V，8小时仅提醒，无TFM/JOINT/T_test。不得重复提交或自动延长。 |
| [037圆周TCM](L2S-20261020-037.md) | **RUNNING / 已领取**，[S2L](S2L-20261020-037.md)。14:02:43在`association-code-run037`、GPU1提交；一次TCM4000及完整V，4小时仅提醒，无自动TAM/TFM/JOINT/T_test。新旧latent MSE不可跨空间直接排名。 |

022—025继续**HOLD_COMPLETION_REVIEW、未运行**；旧草案已总结清理，不恢复。019—034、036、038已结案并分析，已结束的轮次不得重领；负结果与必要服务器路径见[历史摘要](../docs/history/README.md)，近期比较证据见[分析索引](../docs/proposals/repair_before_retraining/README.md)和[结果说明](results/README.md)。

本地负责研究/代码/分析，服务器负责执行/运行监督，push不会唤醒服务器。完整职责见[现场约定](../docs/server_autonomy.md)，新回执使用[模板](TEMPLATE.md)；共享A身份隔离、T_test封存不变。
