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
| [S2L-20261016-002.md](S2L-20261016-002.md) | SERVER_TO_LOCAL | **FAILED**（readiness: environment_setup_failed） | ENV-002 | ENV-20261016-002 | LOCAL_AI |

服务器 AI 已于 16:12 回传 ENV-002 终态：A 阶段安装在下载 torch 2.5.1+cu121 的 PyPI 依赖元数据时读超时退出（退出码 1，未触及 44.5 分钟预算），B 阶段单卡 CUDA 验收**未执行**。原请求 `L2S-20261016-002.md` 未修改，现场与半成品 venv 目录均保留。

## 全部轮次

| 轮次 | 请求 | 回执 | 结论 |
| --- | --- | --- | --- |
| ENV-001 | [L2S-20261016-001.md](L2S-20261016-001.md) | [S2L-20261016-001.md](S2L-20261016-001.md) | COMPLETED / partial：4×3090 + torch 2.4.1+cu121 可见，但探测用的 `pytorch_env`（Python 3.9.18）不满足 `pyproject.toml` 的 `>=3.10`、`torch>=2.5,<2.11`（由 ENV-002 处理）。回执 15:33 补记：MTAD 四文件已在 `/home/wangguangjie/djs/data/mtad/`，Level2 MD5 与官方值一致。GPU 授权已由 ENV-002 明确（FIFO、最多 4 张空闲卡），本行早期“分配未确认”不再有效 |
| ENV-002 | [L2S-20261016-002.md](L2S-20261016-002.md) | [S2L-20261016-002.md](S2L-20261016-002.md) | **FAILED**：A 阶段 NumPy 1.26.4 装成、torch 2.5.1+cu121 主 wheel 780.5 MB 已下完，但在 PyPI 依赖元数据（typing_extensions）读超时 → `--retries 0` 直接退出码 1（耗时 1959 s，未触及 2670 s 预算）；`pip check`、两套 CPU 测试与 B 阶段单卡 CUDA 验收全部未执行；半成品 venv 保留，重跑会先被脚本第 20 行 `test ! -e` 拦住，处置权在本地 |

ENV-002 已给出环境版本、安装脚本、执行预算和GPU选择规则；数据上传由用户处理，本轮不依赖真实数据。

布局变更：2026-10-16 用户指示，从「覆盖 `Handover.md` + 归档 history」改为上述一方一个文件；ENV-001 两份文件由服务器 AI 从旧布局迁移而来，内容未改，迁移记录见 `S2L-20261016-001.md`。
