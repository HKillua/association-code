# 交接索引（不是任务内容）

本文件只做两件事：指出当前该读哪个交接文件、列出已有哪些轮次。**任务、命令和结果一律写在各自以 `handover_id` 命名的文件里。**

每轮交接 = 一个新文件，创建后不覆盖成别的轮次：

- 本地下发 → `HandOver/L2S-<YYYYMMDD>-<NNN>.md`（`LOCAL_TO_SERVER`）。写完即固定，服务器不修改它。
- 服务器领取 → 新建 `HandOver/S2L-<YYYYMMDD>-<NNN>.md`（`SERVER_TO_LOCAL`，`status: RUNNING`），推送后再启动。
- 服务器终态 → **只更新自己那个 `S2L-*.md`**；同一轮的状态演变由 Git 历史保留。
- 每次状态变化同步更新下方「当前待处理」。索引里没有的轮次不领取。
- 旧的 `HandOver/history/` 目录已废除：请求文件本身就永久留在 `HandOver/`，不需要再复制一份。

## 当前待处理

| 文件 | direction | status | task_id | run_id | next_actor |
| --- | --- | --- | --- | --- | --- |
| [S2L-20261016-003.md](S2L-20261016-003.md) | SERVER_TO_LOCAL | **RUNNING** | ENV-003 | ENV-20261016-003 | SERVER_AI（执行中） |

本地已审阅 ENV-002 终态：依赖元数据读超时，CPU/CUDA 验收未执行。新 ENV-003 保留旧现场，使用新环境和新输出；先保存主 wheel，再下载依赖并离线安装，A 最多 60 分钟，B 单卡 CUDA 最多 10 分钟。服务器只领取上方新请求，旧轮次不重跑。详见[失败核验](../docs/env_002_failure_record.md)和[最新方案](../docs/env_003_plan.md)。

服务器 AI 已于 16:42 领取 ENV-003：领取核对全部通过（冻结范围与 `5796636` 逐字节一致，三个新目录均不存在，`/home` 剩余 963 G），原请求未修改；ENV-002 回执中 `actual_commit` 被误写为 `db5f59d` 的更正已按本请求第 5 条记入该回执，本轮实际执行 SHA 一律以脚本写出的 `actual_commit.txt` 为准。

## 全部轮次

| 轮次 | 请求 | 回执 | 结论 |
| --- | --- | --- | --- |
| ENV-001 | [L2S-20261016-001.md](L2S-20261016-001.md) | [S2L-20261016-001.md](S2L-20261016-001.md) | COMPLETED / partial：4×3090 + torch 2.4.1+cu121 可见，但探测用的 `pytorch_env`（Python 3.9.18）不满足 `pyproject.toml` 的 `>=3.10`、`torch>=2.5,<2.11`（由 ENV-002 处理）。回执 15:33 补记：MTAD 四文件已在 `/home/wangguangjie/djs/data/mtad/`，Level2 MD5 与官方值一致。GPU 授权已由 ENV-002 明确（FIFO、最多 4 张空闲卡），本行早期“分配未确认”不再有效 |
| ENV-002 | [L2S-20261016-002.md](L2S-20261016-002.md) | [S2L-20261016-002.md](S2L-20261016-002.md) | **FAILED**：NumPy 1.26.4 装成、torch 主 wheel 显示 780.5 MB 下载完成，但 typing_extensions 元数据读超时；耗时 1959 s、内外退出码 1；CPU/CUDA 检查未执行。旧 wheel 是否仍存在未知，保留半成品 venv。原始 actual_commit 为 a6d6a04，与回执 db5f59d 不同，但执行代码相同，详见本地审计 |
| ENV-003 | [L2S-20261016-003.md](L2S-20261016-003.md) | [S2L-20261016-003.md](S2L-20261016-003.md) | **RUNNING**（16:42 领取）：新环境、显式保存主 wheel、有限网络重试/持久缓存、离线安装、CPU 与单卡 CUDA 验收；无真实数据训练授权 |

MTAD 四文件已在服务器，Level2 MD5 已匹配。本轮不要求用户重新上传数据，也无需重复确认空闲 GPU 的使用授权。

布局变更：2026-10-16 用户指示，从「覆盖 `Handover.md` + 归档 history」改为上述一方一个文件；ENV-001 两份文件由服务器 AI 从旧布局迁移而来，内容未改，迁移记录见 `S2L-20261016-001.md`。
