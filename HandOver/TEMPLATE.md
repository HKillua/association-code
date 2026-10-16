---
handover_id: L2S-YYYYMMDD-NNN 或 S2L-YYYYMMDD-NNN
in_reply_to: none 或上一条 handover_id
direction: LOCAL_TO_SERVER 或 SERVER_TO_LOCAL
status: READY 或 RUNNING 或 COMPLETED 或 FAILED 或 BLOCKED
updated_by: LOCAL_AI 或 SERVER_AI
updated_at: 实际 ISO8601 时间及偏移
task_id: 本轮任务 ID
run_id: 本次执行唯一 ID
code_commit: 已存在的完整 Git SHA
next_actor: SERVER_AI 或 LOCAL_AI
---

用法：复制本模板为 `HandOver/<handover_id>.md`，文件名与 frontmatter 的 `handover_id` 完全一致（本地 `L2S-YYYYMMDD-NNN`，服务器 `S2L-YYYYMMDD-NNN`）。每个 `handover_id` 一个新文件，创建后在 `HandOver/Handover.md` 索引登记；一方只更新自己那份文件（服务器在同一份 `S2L` 内从 RUNNING 推进到终态），不覆盖对方文件，也不再复制到 `history/`。

# 本地 AI → 服务器 AI／服务器 AI → 本地 AI（保留实际方向）

## 任务与依据

目标、计划/配置版本、关联假设；只验环境时明确无模型实验。

## 输入、环境、资源和前置条件

数据路径/哈希、环境、GPU、时间/内存预算、必需文件；未知写 unknown。code_commit 是代码版本，交接文档提交可不同；核对代码目录没有偏离该提交。

现场自主处置遵循 `docs/server_autonomy.md`。本轮明确写出：允许的同版本环境修复范围、CPU/GPU执行预算、GPU等待预算/检查间隔、最多尝试次数、attempt输出路径；所有失败尝试和终止宽限均计入相应预算。未给等待预算时不自行等待，未给重试次数时为0。已授权的传输/缓存/路径/资源问题无需逐次审批，研究语义变更仍由本地决定。

## 精确命令与输出

从仓库根目录运行的命令、预期新输出位置、命令退出码的记录方式。输出不得覆盖旧 run_id。

## 完成、失败与停止条件

必须回传的证据、允许重试范围及次数、何时停止/回退单源/报错。没有授权的重试写 0；不能用无限重试代替问题反馈。

## 服务器实际执行回执

领取/结束时间；实际 code_commit 和文档 commit；环境、设备、命令、退出码、相关进程/终止状态；结果/日志路径与小型产物，较大文件路径/大小/哈希；所有失败与偏差。未执行的项写未执行，不能替本地预填成功。

## 下一步与责任方

服务器只提供事实与建议；研究判断、方法/协议/代码修改及下一轮下发由本地负责。
