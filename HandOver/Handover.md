# 当前服务器任务

只从当前队列领取；旧L2S的READY不是新任务。操作见[028保全并停止](L2S-20261019-028.md)，配置与[本地评审](../docs/proposals/repair_before_retraining/README.md)以最新用户决定为准。push不会唤醒服务器。

**2026-10-19最新用户决定：先补全→再关联→再融合，并明确“保留已有权重后停止020和021”。当前唯一可领取操作为[028保全并停止](L2S-20261019-028.md)，不新开GPU实验。022—025保持HOLD，027旧启动块不再是执行授权。发布交接不等于服务器已停止，收到原终态才更新实际状态。**

时长政策仍为建议与提醒；**本次用户明确的停止指令立即优先执行**。T_test封存、不重试、不启动新实验。在途020/021直到真实退出仍有旧固化截止，不靠Git热取消。

| 项目 | 最新状态与下一步 |
| --- | --- |
| [028保全并停止](L2S-20261019-028.md) | **COMPLETED**（[S2L-028](S2L-20261019-028.md)）：按序执行完毕。**021 attempt-3 在指令到达前已于 `10:20:53Z` 自然 COMPLETED**（returncode 0、1,200 更新、六阶段全 COMPLETED、`shared.pt`／`utility.pt`／`teacher.pt`／identity-records 齐、`test_opened=false`）⇒ 未发任何信号。020 先保全再停：副本 144 MB 在 `runs/STAGEWISE-COMPLETION-20261019-028-preserved/`，两份权重都用 `torch.load(map_location='cpu', weights_only=True)` 一次确认可读；随后 `kill -TERM 1917300`（实时核实 MainPID＝worker_pid）**只发一次**，未用 systemctl stop、未补发、未动他人进程。终态 launcher `INTERRUPTED/rc=1/timed_out=false/deadline_exceeded=false/elapsed 12,945.908`、训练器 `FAILED`（原值保留）`cursor{stage1,update400,completed1400}`；`checkpoint.pt` 与 before-stop **逐字节相同** ⇒ **无新权重，TAM 300→400 的 100 步参数已丢失**；第 4 次 V 被打断（`FAILED, 190.514s, copied=false`）。unit 现 failed，**未 reset-failed**。`stop_reason=user_requested_completion_first`。022—025 一次都没领取（无 runs／无 S2L），代码目录 `../association-code-run022-025`（`9b06f80`）保留不删。回传 16 件／2.0 MB。 |
| [019](S2L-20261019-019.md) / [026](S2L-20261019-026.md) | 019转录错误FAILED；026退出0及独立服务托管已核，两项不重跑。 |
| [020参考](S2L-20261019-020.md) | **STOPPED／终态（经用户指令，非方法失败、非训练完成）**。保住的是 **TCM 1,000 步 ＋ TAM 前 300 步参数**（`checkpoint.pt` 83,309,324 B，`cursor{stage1,update300,completed1300}`）＋ 1,400 行 history ＋ **3 次完整关联 V** ＋ margin 校准全量。V 曲线 **macro-F1 .4245588143 → .4252587185 → .4356723172**（`@300` 超同线 017 的 .4329854，但 017 无 tp/fp/tn 落盘，不称全面优于）；`@300` micro precision 0.1797484 ＝ **先验 1.5815 倍**、recall 0.6848650、FPR 0.4008、`negative_scene_count=0` ⇒ 过并仍不可测；同 checkpoint 的 `mstsa .4641416`／`itsa .4020363` 属不同人口不排名。**无完整物理终评**。实测 TAM 单价在加速：23.218 → 22.027 → **11.130 秒/步**（`history` 末 100 步均值 10.972）⇒ 此前的跨截止条件情景前提已被实测改写，但作业按指令停止、不重启不续训。`test_opened=false`。 |
| [021共享](S2L-20261019-021.md) | **COMPLETED（自然完成，非被停）**：`ended_at 10:20:53Z`、`returncode 0`、`exit_code.txt 0`、`timed_out=false`、`deadline_exceeded=false`、`elapsed 10,511.966`（进程内 `run.elapsed 10,486.880` ＝ 六阶段相加逐项自洽：prep 125.66＋31.95／**teacher 1000 步 9,273.17**／identity 选择 12.21／**utility 200 步 1,040.51**／export 3.38）。**`completed_updates = 1,200`、六阶段全 COMPLETED**、`shared.pt` 14,813,820／`utility.pt` 9,110,758／`teacher.pt` 25,849,550／`identity-records`（index＋64 NPZ，872 KB）齐，`shared_format=cooperative-shared-v1`、`representation_version=support_residual_v2`、`support_time_scale_tau=360.0`、`resume_supported=false`、**`test_opened=false`**。**唯一身份阈值 `0.5957852602005005`，selection macro-F1 = 0.6303206221894191**（`candidate_count 29,003`、`scene_count 64`）——同人口（V_select 64 场景）的候选线历史值是 018 的 **.347575**（teacher 仅 100 步）与 015 的 .347370；同期 020 参考线 TAM@300 是 .4356723。逐对 P／R 未算（028 禁止新分析），但 64 个 identity NPZ＋固定阈值已回传可复算；⚠ 端点的 teacher 段是物理恢复＋身份 BCE **混合**训练，不称纯补全模型。四臂 **HOLD**，未领取；恢复下游需新的明确执行安排。 |
| [027修订](S2L-20261019-027.md) | COMPLETED，旧校准取件已核，不重复。四臂独立代码目录已备，尚未启动。 |
| [022主方法](L2S-20261019-022.md) | HOLD_COMPLETION_REVIEW；暂停领取，原计划attribute-u300＋全V，按027两层advisory，建议20分钟。 |
| [023普通融合](L2S-20261019-023.md) | HOLD_COMPLETION_REVIEW；暂停领取，原计划plain300＋全V，同上。 |
| [024统一效用](L2S-20261019-024.md) | HOLD_COMPLETION_REVIEW；暂停领取，原计划scalar-u300＋全V，同上。 |
| [025普通门控](L2S-20261019-025.md) | HOLD_COMPLETION_REVIEW；暂停领取，原计划learned-attribute300＋全V，输出仍COOPERATIVE-LEARNED。 |

**本地结论与下一步：**

- 020首两次关联验证仅小幅改善，误关联仍多。相较同场景“全接受”macro-F1 .347362有提升，不称全随机。p_fa=1−precision；无纯负场景只使该特定指标为null，FP仍可测。计数/人口、训练覆盖和成本详见[复核](../docs/proposals/repair_before_retraining/README.md)。
- 原人口与协议一致，不能扩展为已排除所有时间/标签/监督问题。当前无已证实需停训的实现故障；研究推进顺序已由用户改为补全优先，020/021现已获明确保全后停止授权，以028执行。不为本次回执重扫数据、重选阈值或另开诊断。
- 020 checkpoint元数据、run/history/coverage已收齐本次中期需要，**不重复取同一快照**。以后正常落盘更新再回，selected须标真实阶段/步数；CPU只读元数据导出已获准，不需再次审批。权重留服务器、不构造模型或额外forward。
- 新四臂启动配置已设为advisory，时间是建议；在途020/021旧限制仍存在。成本外推不是失败概率；不从头重跑、不缩步、不注入进程。接近旧截止且未完成时按下文报告。

**当前回传只按028收尾清单：** 保留检查点副本、实际阶段/步数、前后状态与日志、真实停止命令及退出码。若已自然完成，保留已有完整报告/identity-records/预测，不追加计算补齐缺失产物；权重留服务器。缺少终评不是本次收尾失败，不为产生shared.pt或漂亮结果继续训练。

四臂当前HOLD；原共享前置和027命令仅保留供后续规划，不构成启动许可。[补全优先路线](../docs/research_brief.md)明确下一步证据要求。收到028终态后，再确定独立补全评价的具体配置，不自动恢复/重训，不改原状态冒充可续训。

时长建议政策用于以后重新授权的作业；当前停止来自用户明确决定，不受旧“等超时再询问/继续固定工作”规则阻挡。保留旧硬截止风险直到实际退出，不能说发布028后服务已经停止。

001–018已结案，不重领/续训/补评；CDF退役，本轮不额外训练TSA。[历史摘要](../docs/history/README.md)。
