# 当前交接

更新：2026-10-26，服务器仍以`1a25183`为最近已审回执；本地新增100。历史命令与READY不恢复为授权。RUNNING是最近回执，不是实时监控。

| 任务 | 当前状态 | 本地决定／入口 |
|---|---|---|
| 100A/B/C＋CPU整表 | **WAITING_RESOURCE，尚无领取回执**。09:00Z 现场核查：四张获准GPU（c5a00c37/6811d9c0/efbe9d07/1043f26a）全部存在他人 compute 进程（user `hejingjing`，6.7GB×4 加若干评估进程，index0/1/3 util 100%、index2 util 0%但显存 7.1GB）；无卡满足"无compute进程、util=0、≤256MiB"，未创建 100 checkout／输出，未提交任何臂。按合同等待，不结束他人进程。冻结 `b0433af` 已核为 commit |
| 096A | **COMPLETED（执行验收通过，负结果）**。unit `mtad-11e76d7fa8c3.service`、index0，08:09:10Z→**2026-10-25 16:20:44Z**退出0，launcher **29,493.9 s（8h11m）**、`time_advisory_issued=true`（advisory继续未截断）、`deadline_exceeded=false`；0训练／融合前向39,788／阈值沿用094A。**实际预测组覆盖3.3557%（15,267/454,952），位置MAE 58,587.02m（RMSE 78,610.67、P95 160,927.78、P99 289,946.72），同组anchor插值583.53m（2.238%覆盖，不同点集不作归因）**。见[回执](S2L-20261025-096A.md) |
| 096B／097A及097表 | COMPLETED | 096B完整系统负结果；097为第二seed预算控制。原096-summary等待096A，仍由同一队列收尾 |
| 096-summary | **COMPLETED**。287 s、退出0、零前向：共同点 learned 59,898.42 对 hold 59,916.82 m（learned好18.40m，0.031%），逐场179/300、组件24/38；各自人口覆盖3.3557%/3.5678%，自身MAE 58,587.02/57,215.08——两系统均不可用。见[回执](S2L-20261025-096-summary.md) |
| 093A/B、094A/B及配对表 | COMPLETED，已审 | 物理/关联均明显失效，不能当可信同行或把失败唯一归因补全；原件保留供096收尾 |
| 098A/B/C/D、两seed表与总表 | **全部COMPLETED；当前配方不接受** | 双seed门槛均未过，停止扩展、不追加训练。[总回执](S2L-20261025-098-summary.md)、[历史合同](https://github.com/HKillua/association-code/blob/9eefd2c/HandOver/L2S-20261025-098.md) |
| 099 DeepAF | **HOLD：真实MTAD适配与比较尚未完成** | 不领取A/B/C，不用旧ZIP READY；自建模拟不作为放行门槛。[暂停交接](L2S-20261025-099.md)、[适配边界](../docs/reports/2026-10-25-deepaf-method-audit.md) |

098 seed23关系版572.775m，差于同预算564.834m；seed21关系版568.684m，优于583.976m但仅2.618%，未达预定3%。两seed终态bias-off几乎相同，功能门槛也未过；不抹掉第二seed正信号，也不晋级当前配方。科学解释见[研究状态](../docs/research_status.md)。

## 当前执行约束

- 100是本次唯一新READY；096A及其CPU收尾继续原合同。099保持HOLD，098结案，不删除服务器输入、权重、预测或runs。
- 096—097原有限控制服务持有该批接续权，不手工重交096A或CPU表。状态缺失先按原完整output/record-dir查服务，不能另建队列；真实终态和释放事实由服务器记录。
- 活动checkout/config冻结，本地主仓清理不热改运行。最多四单卡、现场FIFO、不动他人进程；失败保全不自动重试。
- T_test封存、默认056＋原linear不变，066仍REJECTED；不得把同行适配失败算我方获胜。
- 只核配置、实际输入、更新、命令/退出和数值，不设额外SHA门槛。push不唤醒服务器。

## 保留证据与历史

[结果索引](results/README.md)登记仍使用的来源与近期回执。工作树只留096B及直接来源小摘要、097预算总表、098两个seed与最终总表。已审逐臂日志、旧交接和重复材料均总结后移出工作树；[历史摘要](../docs/history/README.md)给出`9eefd2c`等恢复位置。服务器原件不受影响。
