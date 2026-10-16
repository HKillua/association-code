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
| [S2L-20261016-005.md](S2L-20261016-005.md) | SERVER_TO_LOCAL | **RUNNING** | ENV-005 | ENV-20261016-005 | SERVER_AI（执行中） |

ENV-004 的 CPU 验收已通过：95 项 tests、15 项旧审计、32 秒，内外退出码及源码哈希核对一致。CUDA 尚未执行，原始等待日志 18 次采样均无空闲卡。本地已修复后续补跑被整仓 HEAD 相等检查误拦的问题；新 ENV-005 先复用现有 FIELD 环境补齐新契约的 CPU 快照，再补单卡 CUDA。详见[本轮请求](L2S-20261016-005.md)与[修复及审计记录](../docs/env_005_plan.md)。

新校验保留 CPU/CUDA 各自实际 SHA，以受检文件集合与哈希、环境字段、成功凭据决定是否复用；仅交接提交不会误拦，代码/配置/依赖变化仍拒绝。已核验本仓库的 editable freeze 条目按真实路径、项目身份和安装版本规范化，其他依赖 VCS pin 不放宽。旧 ENV-004 没有新环境快照，不能伪造迁移；旧回执和结果保留。

服务器有[有限现场处置权](../docs/server_autonomy.md)：本轮 CPU 最多15分钟、GPU等待最多20分钟、CUDA最多10分钟，CPU/CUDA各最多两次尝试；同 pin 修复、FIFO 选卡和预算内前台等待可自主，研究语义和资源规则不变。只执行当前 READY，不重开旧轮次，不共享他人占用的 GPU。

## 全部轮次

| 轮次 | 请求 | 回执 | 结论 |
| --- | --- | --- | --- |
| ENV-001 | [L2S-20261016-001.md](L2S-20261016-001.md) | [S2L-20261016-001.md](S2L-20261016-001.md) | COMPLETED / partial：4×3090 + torch 2.4.1+cu121 可见，但探测用的 `pytorch_env`（Python 3.9.18）不满足 `pyproject.toml` 的 `>=3.10`、`torch>=2.5,<2.11`（由 ENV-002 处理）。回执 15:33 补记：MTAD 四文件已在 `/home/wangguangjie/djs/data/mtad/`，Level2 MD5 与官方值一致。GPU 授权已由 ENV-002 明确（FIFO、最多 4 张空闲卡），本行早期“分配未确认”不再有效 |
| ENV-002 | [L2S-20261016-002.md](L2S-20261016-002.md) | [S2L-20261016-002.md](S2L-20261016-002.md) | **FAILED**：NumPy 1.26.4 装成、torch 主 wheel 显示 780.5 MB 下载完成，但 typing_extensions 元数据读超时；耗时 1959 s、内外退出码 1；CPU/CUDA 检查未执行。旧 wheel 是否仍存在未知，保留半成品 venv。原始 actual_commit 为 a6d6a04，与回执 db5f59d 不同，但执行代码相同，详见本地审计 |
| ENV-003 | [L2S-20261016-003.md](L2S-20261016-003.md) | [S2L-20261016-003.md](S2L-20261016-003.md) | **FAILED**：A 在 `download_bootstrap` 用满 3570 秒，wrapper **124**，`wheels/` 落盘 0 个文件；全程卡在 `pypi.org` / `files.pythonhosted.org` 的 120 秒读超时，torch 主 wheel 阶段未开始。脚本 trap 的 `setup_exit_code.txt=0` 与 `download_bootstrap exit=0` 是 SIGTERM 残留值、不作成功依据。env003 venv 与 package-cache 原样保留，实际执行 SHA `2be9490`。**FIELD 补充（非本轮 A/B）**：根因是本机 IPv6 出口不通，强制 IPv4 + TUNA 主索引后同一组 pin 装成，`pip check`、导入断言、`tools/pilot` 15 项全过，但仓库 `tests` 93 项有 1 项失败（`test_models.py:128` 的 `rtol=0,atol=0` 逐位相等断言，实测差 `2.4e-7`）；CUDA 记 `deferred_no_free_gpu`（18:06 四卡全有他人进程），未抢占、未占用 GPU |
| ENV-004 | [L2S-20261016-004.md](L2S-20261016-004.md) | [S2L-20261016-004.md](S2L-20261016-004.md) | **COMPLETED / partial**（`cuda_smoke: deferred_no_free_gpu`）：A 一次通过——32 秒内内外 rc 均 0、`validation.json=COMPLETED`、`pip check` 无问题、95 项 tests OK、15 项旧审计 OK、CLI OK，pin 与 `association` 导入来源已核验，实际执行 SHA `6db0695`，未做任何环境修复。B 未执行——20 分钟前台等待 18 次检查，四卡 UUID 始终有他人 compute 进程，`free_cards` 全空，故未初始化 cuda 截止时间、未创建 `runs/`，未抢占、未后台轮询。**CUDA 通路仍无证据，`ready_for_native_profile` 未取得** |
| ENV-005 | [L2S-20261016-005.md](L2S-20261016-005.md) | [S2L-20261016-005.md](S2L-20261016-005.md) | **RUNNING**（23:10 领取，冻结检查全通过）：修复纯回执提交误拦；在已有 FIELD 环境刷新 CPU 契约，再补单卡 CUDA。源码增删改、环境差异和失败凭据仍拒绝；独立输出/预算、不共享他人卡 |

本地复核 ENV-004 的 22:30 补记：整仓 HEAD 相等检查确实会阻止后续补跑，但不是已经发生的 CUDA 失败；从 CPU 提交到回执提交仅 HandOver 变化，旧 CPU 通过结论仍有效。ENV-005 已替换该检查，不再要求纯文档提交前后 HEAD 相同。等待日志实际跨度1167秒、含19/20秒两个短间隔，不能称全部至少60秒；离散采样不能证明四小时连续无空档，也没有 smoke 显存实测。原始服务器叙述保留在其 S2L，本地核验口径见 env_005_plan。

MTAD 四文件已在服务器，Level2 MD5 已匹配。本轮不要求用户重新上传数据，也无需重复确认空闲 GPU 的使用授权。

布局变更：2026-10-16 用户指示，从「覆盖 `Handover.md` + 归档 history」改为上述一方一个文件；ENV-001 两份文件由服务器 AI 从旧布局迁移而来，内容未改，迁移记录见 `S2L-20261016-001.md`。
