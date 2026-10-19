# 当前服务器任务

只从本表领取；当前领取/终态由服务器更新对应S2L。push不自动唤醒服务器。001—018已结案并从工作树归档，结论与原件找回方式见[历史摘要](../docs/history/README.md)。

用户同意按参考≤24h＋候选合计≤4h拆分，[配置和授权](../docs/proposals/repair_before_retraining/README.md)。5种最终方法、6个GPU作业，各一次无重试，最多两个单卡并发；每项独立输出和累计额度。

| 任务 | 前置、当前状态与工作 |
| --- | --- |
| [019](L2S-20261019-019.md) | **FAILED**；[回执](S2L-20261019-019.md)及原件保留。实际argv比原交接多一个右括号，退出1、无成功marker；原交接可compile，不能采纳回执的来源归因。**服务器已接受该更正并追加「服务器更正」一节**：复算证实原文（`7c849f1`/`612afea`）payload **373 字节且 compile 通过**，服务器实际传入 **374 字节**，`difflib` 显示唯一差异为 **index 341 处 insert `)`** ——**多余括号由服务器转录时插入，先前"错在请求文本"的 published 归因作废**。019 保持 FAILED、不改状态、不重开额度。 |
| [026补核](L2S-20261019-026.md) | **COMPLETED**（[S2L-026](S2L-20261019-026.md)，提交 `2026-10-19T06:42:35Z`＝本地 14:42:35，5.19 秒完成）。**成功条件全部达成**：`status.json=COMPLETED`、`returncode=0`、`timed_out=false`、`exit_code.txt=0`、`stdout` 22 B、`stderr` 0 B、`payload-survived.txt` 42 B。**决定性证据（019 缺失项已补齐）**：`probe-start.json` 的 payload 自身 cgroup ＝ **`…/user@1018.service/app.slice/mtad-748bb2c505e6.service`**，与 AI 自身 **`…/session-9554.scope`** 明确不同 —— 实证运行在**用户管理器的独立 transient service** 内，非 AI 进程树。三 PID 分离（launcher 1916959 已退出／worker=unit 主进程 1916962／payload 1916964）；**marker 晚于提交 launcher 返回约 5 秒写出**；journal 归属 `systemd[1911324]`；`systemctl show` **`Result=success`、`ExecMainCode/Status=0/0`、`NRestarts=0`**（`MainPID=0`／`ControlGroup` 空属卸载后真实回显，**未补值**；`RuntimeMaxUSec=infinity`／`TimeoutStopUSec=1min30s` 是**卸载后缺省回显**，实际生效值为 `launch.json` 里的 `RuntimeMaxSec=26.989754s`／`TimeoutStopSec=3s`／`Type=exec`）。**防转录错误**：命令块由脚本从交接文件抽出（861 B、SHA256 `fd14fc2e…620afb`），执行前后复核哈希未变、payload `compile()`＋`bash -n` 通过后才运行，全程未手工重打。**边界**：未宣称实测跨过 AI 运行时重启（本轮未故意重启、未触碰他人会话）。**⇒ 020/021 的托管前置由本轮替代并满足；019 仍为 FAILED、原件不改。** 24h 参考＋候选合计 4h、最多两单卡、每项一次无重试继续生效；022–025 仍待 021 完整 COMPLETED。回传 14 文件 / 3,933 B。 |
| [020参考](L2S-20261019-020.md) | **RUNNING**（[S2L-020](S2L-20261019-020.md)）。026 已 COMPLETED 并实证托管 ⇒ 替代"019 通过"前置，故启动。TCM1000/TAM1000/TFM300/JOINT60＝**2,360 更新**、lr .002×3／.0002、**微批 pair8／track4**、batch_tracks64、每100步验证、`singleton=observed`、`validation_batches{tracks:4,pairs:1024}`、**`longitude_loss_mode=shortest_arc`**、`train_position_affine`、**`max_hours=23.98333…（86,340 秒）／外层 86,400`**。冻结 worktree `../association-code-run020`；经 `launch_detached.py`＋`systemd-run --user` 独立 service，**AI 可断开**；`--max-seconds` 传绝对剩余量、grace 30。**不重打命令**：从交接文件抽块＋`bash -n`＋执行前后复核 SHA256。不从 017 恢复、不挪用结余、一次无重试。 |
| [021共享](L2S-20261019-021.md) | **RUNNING**（[S2L-021](S2L-20261019-021.md)）。同上由 026 满足前置。`model_variant=support_residual_v2`＋`support_time_policy=train_absmax_v1`，teacher1000→一次完整V身份阈值→utility200＝**1,200 更新**、`--phase shared`、`max_seconds=9540`／外层 9,600、routes 为 plain／scalar-u／attribute-u／learned-attribute（**本项不做四臂适配与物理评价**）。冻结 worktree `../association-code-run021` **将由 022–025 复用**（同 `shared.pt`／normalizer／S／身份 records／阈值），本轮结束后服务器不删不改。与 020 并行＝2 个单卡作业，符合上限 2；一次无重试。 |
| [022主方法](L2S-20261019-022.md) | READY；021完整COMPLETED后，attribute-u适配300＋完整V，≤1200秒。 |
| [023普通融合](L2S-20261019-023.md) | READY；同一021端点，plain适配300＋完整V，≤1200秒。 |
| [024统一效用](L2S-20261019-024.md) | READY；同一021端点，scalar-u适配300＋完整V，≤1200秒。 |
| [025普通门控](L2S-20261019-025.md) | READY；同一021端点，learned-attribute适配300＋完整V，≤1200秒。 |

候选9600＋4×1200=14400秒，共4GPU小时。优先021→022→023，再024/025；020占一卡时候选一次跑一项，020结束后可两臂并行。021失败则四臂不读半成品；单臂失败不阻塞其他具备前置的臂，020失败不阻塞候选。各项不互转结余，不重开旧预算。

四臂复用021冻结代码和完整shared.pt、normalizer/S、固定身份records/阈值；同初始化与seed22采样、各自新Adam及输出，不重复公共训练、不继承上一臂。服务独立托管，AI可断开；重连查同一任务状态后领取下一项，SUBMITTED不等于COMPLETED。

**已取消：** 001—018重领/续训/补评价、006旧查询、重复取件/ZIP审计/环境smoke/成本探测、CDF及本轮额外TSA训练。019旧正文“尚无新GPU READY”及020/021正文“019通过”均按本页后续授权/026替代前置解释，历史原文不回改。服务器的runs原manifest、权重及运行产物不属于本次Git工作树清理范围。

研究规则见[数据契约](../docs/data_contract.md)和[研究说明](../docs/research_brief.md)，执行权限见[现场约定](../docs/server_autonomy.md)。T_test封存，不改人口、指标、精度或模型来掩盖失败。核实际命令、退出码、配置、完成状态、人口与数值，无SHA/hash准入。
