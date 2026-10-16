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
| [L2S-20261016-004.md](L2S-20261016-004.md) | LOCAL_TO_SERVER | **READY** | ENV-004 | ENV-20261016-004 | SERVER_AI |

本地已审阅 ENV-003 与 FIELD 回执，并修复重复输入编码的浮点测试契约；本地 95 项 tests、15 项旧审计与 4 项状态记录器回归通过，服务器 pinned 版本与 CUDA 尚待验收。新 ENV-004 复用 FIELD 环境，先 CPU、后单卡合成 CUDA；原 ENV-003 失败及证据保留。详见[本轮请求](L2S-20261016-004.md)与[修复依据](../docs/env_004_plan.md)。

用户已授权服务器在同版本、冻结研究协议和总预算内自主处理网络/缓存/环境/资源/进程问题，规则见[现场授权边界](../docs/server_autonomy.md)。ENV-004 CPU 最多 15 分钟、GPU 等待最多 20 分钟、CUDA 执行最多 10 分钟；CPU/CUDA 各最多两次尝试，成功阶段不重跑，所有现场处置写入同轮 S2L。服务器只领取上方新请求，旧轮次不重跑。

## 全部轮次

| 轮次 | 请求 | 回执 | 结论 |
| --- | --- | --- | --- |
| ENV-001 | [L2S-20261016-001.md](L2S-20261016-001.md) | [S2L-20261016-001.md](S2L-20261016-001.md) | COMPLETED / partial：4×3090 + torch 2.4.1+cu121 可见，但探测用的 `pytorch_env`（Python 3.9.18）不满足 `pyproject.toml` 的 `>=3.10`、`torch>=2.5,<2.11`（由 ENV-002 处理）。回执 15:33 补记：MTAD 四文件已在 `/home/wangguangjie/djs/data/mtad/`，Level2 MD5 与官方值一致。GPU 授权已由 ENV-002 明确（FIFO、最多 4 张空闲卡），本行早期“分配未确认”不再有效 |
| ENV-002 | [L2S-20261016-002.md](L2S-20261016-002.md) | [S2L-20261016-002.md](S2L-20261016-002.md) | **FAILED**：NumPy 1.26.4 装成、torch 主 wheel 显示 780.5 MB 下载完成，但 typing_extensions 元数据读超时；耗时 1959 s、内外退出码 1；CPU/CUDA 检查未执行。旧 wheel 是否仍存在未知，保留半成品 venv。原始 actual_commit 为 a6d6a04，与回执 db5f59d 不同，但执行代码相同，详见本地审计 |
| ENV-003 | [L2S-20261016-003.md](L2S-20261016-003.md) | [S2L-20261016-003.md](S2L-20261016-003.md) | **FAILED**：A 在 `download_bootstrap` 用满 3570 秒，wrapper **124**，`wheels/` 落盘 0 个文件；全程卡在 `pypi.org` / `files.pythonhosted.org` 的 120 秒读超时，torch 主 wheel 阶段未开始。脚本 trap 的 `setup_exit_code.txt=0` 与 `download_bootstrap exit=0` 是 SIGTERM 残留值、不作成功依据。env003 venv 与 package-cache 原样保留，实际执行 SHA `2be9490`。**FIELD 补充（非本轮 A/B）**：根因是本机 IPv6 出口不通，强制 IPv4 + TUNA 主索引后同一组 pin 装成，`pip check`、导入断言、`tools/pilot` 15 项全过，但仓库 `tests` 93 项有 1 项失败（`test_models.py:128` 的 `rtol=0,atol=0` 逐位相等断言，实测差 `2.4e-7`）；CUDA 记 `deferred_no_free_gpu`（18:06 四卡全有他人进程），未抢占、未占用 GPU |
| ENV-004 | [L2S-20261016-004.md](L2S-20261016-004.md) | 待服务器新建 S2L-20261016-004.md | **READY**：复用已装好的 FIELD 环境，验收修订后的 CPU 测试与单卡 CUDA；开放有界现场处置，固定研究语义与依赖 pin |

MTAD 四文件已在服务器，Level2 MD5 已匹配。本轮不要求用户重新上传数据，也无需重复确认空闲 GPU 的使用授权。

布局变更：2026-10-16 用户指示，从「覆盖 `Handover.md` + 归档 history」改为上述一方一个文件；ENV-001 两份文件由服务器 AI 从旧布局迁移而来，内容未改，迁移记录见 `S2L-20261016-001.md`。
