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
| [S2L-20261017-010.md](S2L-20261017-010.md) | SERVER_TO_LOCAL | **RUNNING** | FIRST-FIT-010 | FIRST-FIT-20261017-010 | SERVER_AI（执行中） |

用户于2026-10-17直接授权不固定哪张卡、允许共卡。服务器可按实时剩余显存/负载自主选卡，已有compute进程不再构成拒绝条件；后续任务可在各自预算内同配置换卡。无需逐卡请示，不终止他人进程，模型/数据/评价标准保持冻结，详见[现场授权](../docs/server_autonomy.md)。该授权覆盖历史轮次的空闲卡限制。

**ENV-005 已通过 CPU 和真实 CUDA，保持 readiness: ready_for_native_profile。** CPU 95项测试、15项旧审计通过；CUDA外层67秒、合成pilot34.02秒、进程峰值分配18500096B。内外rc=0、COMPLETED及回传hash一致，实际使用当时空闲的GPU1，尚无共卡实测。本地核验复制的pilot_run.json哈希与acceptance相符，两份report_sha256相同；完整report留服务器。CPU实际提交2af59d6与CUDA的ebc9522不同但受检源码/环境相同，修复后的契约按设计放行。完整事实和等待历史保留在[S2L-005](S2L-20261016-005.md)。

**当前FIRST-FIT-010直接训练一个baseline。** 009已终态，008原始JSON/包装器与预先提交哈希核验通过；009整理留痕缺口不阻塞本轮。用户要求减少过度设计，取消额外14步探测，固定A0/CE/seed20、TCM100/TAM100/joint20、单作业最多6小时，复用007 prepared，完整V_select验证，不读测试集。GPU可共卡；本轮目标是实际学习曲线与验证指标，不要求好成绩才回传。代码冻结于`d13ac65dc59b9c9b16b59a93f7044eb49aefb568`，不启动其他fit/heads。

GPU-POLICY-006的授权与设备查询由007承接：若006尚未领取，不再单独执行；若服务器已有006领取或完成记录，先结束短查询并保留/补交真实回执后再执行007。**本地尚未收到006回执，不代填成功或失败。** 旧请求保持原样；执行以007中的衔接规则为准。

**007服务器终态（FAILED / prepare_ok_profile_timeout）**：真实准备全部通过，输入核验/prepare/验收合计73秒；四侧512/128/64/128场景、33/27/17/38组件，manifest内部hash `0b4df901…`。profile外层rc124、1784秒完成9/200步，均值184.85秒，前5步之后4步均值181.17秒；原始profile/run保持RUNNING，未伪造成功。显存峰值未保存，acceptance门控跳过，剩余预算不足重试。数据准备可以复用。**本地复核限定**：启动时GPU1无其他compute进程不证明全过程无竞争；单个最大节点场景不代表全训练均值或数学下界；22,831,200次计划更新的成本关口仍未通过。读取参考/测试为0是源码契约字段，并非独立系统访问追踪。完整原始回执保留于[S2L-007](S2L-20261017-007.md)，本地证据与算术复核见[成本审计](../docs/native_007_cost_audit.md)。

## 全部轮次

| 轮次 | 请求 | 回执 | 结论 |
| --- | --- | --- | --- |
| ENV-001 | [L2S-20261016-001.md](L2S-20261016-001.md) | [S2L-20261016-001.md](S2L-20261016-001.md) | COMPLETED / partial：4×3090 + torch 2.4.1+cu121 可见，但探测用的 `pytorch_env`（Python 3.9.18）不满足 `pyproject.toml` 的 `>=3.10`、`torch>=2.5,<2.11`（由 ENV-002 处理）。回执 15:33 补记：MTAD 四文件已在 `/home/wangguangjie/djs/data/mtad/`，Level2 MD5 与官方值一致。GPU 授权已由 ENV-002 明确（FIFO、最多 4 张空闲卡），本行早期“分配未确认”不再有效 |
| ENV-002 | [L2S-20261016-002.md](L2S-20261016-002.md) | [S2L-20261016-002.md](S2L-20261016-002.md) | **FAILED**：NumPy 1.26.4 装成、torch 主 wheel 显示 780.5 MB 下载完成，但 typing_extensions 元数据读超时；耗时 1959 s、内外退出码 1；CPU/CUDA 检查未执行。旧 wheel 是否仍存在未知，保留半成品 venv。原始 actual_commit 为 a6d6a04，与回执 db5f59d 不同，但执行代码相同，详见本地审计 |
| ENV-003 | [L2S-20261016-003.md](L2S-20261016-003.md) | [S2L-20261016-003.md](S2L-20261016-003.md) | **FAILED**：A 在 `download_bootstrap` 用满 3570 秒，wrapper **124**，`wheels/` 落盘 0 个文件；全程卡在 `pypi.org` / `files.pythonhosted.org` 的 120 秒读超时，torch 主 wheel 阶段未开始。脚本 trap 的 `setup_exit_code.txt=0` 与 `download_bootstrap exit=0` 是 SIGTERM 残留值、不作成功依据。env003 venv 与 package-cache 原样保留，实际执行 SHA `2be9490`。**FIELD 补充（非本轮 A/B）**：根因是本机 IPv6 出口不通，强制 IPv4 + TUNA 主索引后同一组 pin 装成，`pip check`、导入断言、`tools/pilot` 15 项全过，但仓库 `tests` 93 项有 1 项失败（`test_models.py:128` 的 `rtol=0,atol=0` 逐位相等断言，实测差 `2.4e-7`）；CUDA 记 `deferred_no_free_gpu`（18:06 四卡全有他人进程），未抢占、未占用 GPU |
| ENV-004 | [L2S-20261016-004.md](L2S-20261016-004.md) | [S2L-20261016-004.md](S2L-20261016-004.md) | **COMPLETED / partial**（`cuda_smoke: deferred_no_free_gpu`）：A 一次通过——32 秒内内外 rc 均 0、`validation.json=COMPLETED`、`pip check` 无问题、95 项 tests OK、15 项旧审计 OK、CLI OK，pin 与 `association` 导入来源已核验，实际执行 SHA `6db0695`，未做任何环境修复。B 未执行——20 分钟前台等待 18 次检查，四卡 UUID 始终有他人 compute 进程，`free_cards` 全空，故未初始化 cuda 截止时间、未创建 `runs/`，未抢占、未后台轮询。**CUDA 通路仍无证据，`ready_for_native_profile` 未取得** |
| ENV-005 | [L2S-20261016-005.md](L2S-20261016-005.md) | [S2L-20261016-005.md](S2L-20261016-005.md) | **COMPLETED / readiness `ready_for_native_profile`**（`cuda_smoke: completed_attempt_1_on_free_card`）：A 通过——39 秒／900 秒，内外 rc 0，`validation.json=COMPLETED`、五项 checks 全 0、95 项 tests 与 15 项旧审计 OK，新快照齐全，CPU 实际提交 `2af59d6`。B 于 10-17 13:01 用**本轮从未动用的 CUDA 预算**在空闲卡 GPU1 上 67 秒完成——内外 rc 0、`acceptance.json` 与 `pilot_run.json` 皆 COMPLETED、pilot 34.02 秒、峰值分配 18500096 B、UUID 绑定、report hash 一致；`allocation_status=free_at_prelaunch_check`，**故未使用 10-17 的共卡授权**。CUDA 实际提交 `ebc9522` ≠ CPU `2af59d6`（仅 HandOver 差异）而 `code_sha256` 相同，新契约按设计放行。**CUDA 尝试 1/2；首份真实 CUDA 通路证据已取得**（合成 smoke，不代表正式配置资源，不构成研究结论） |
| GPU-POLICY-006 | [L2S-20261017-006.md](L2S-20261017-006.md) | 本地未收到；服务器已有记录需保留回传 | 007承接未领取的查询与授权，不再重复；已领取006先收尾。旧请求保留，不预填执行结果 |
| NATIVE-PROFILE-007 | [L2S-20261017-007.md](L2S-20261017-007.md) | [S2L-20261017-007.md](S2L-20261017-007.md) | **FAILED / prepare_ok_profile_timeout**：真实准备73秒通过，可复用；profile外层rc124、1784秒完成9/200步，均值184.85秒（warmup后181.17），原报告RUNNING保留，峰值unknown。启动快照无其他进程不代表全程无竞争；单场景TAM投影不作完整训练实测。008补齐分段成本与显存证据，详见本地审计 |
| PROFILE-DIAG-008 | [L2S-20261017-008.md](L2S-20261017-008.md) | [S2L-20261017-008.md](S2L-20261017-008.md) | **服务器COMPLETED；本地原件复核未闭合**：profile外层/内层rc0、973.63s，冻结执行代码相符；accept摘要6步分段均值152.6095s，allocated11.497GiB，六个正例计数0。原profile/run与实际wrapper未入提交，009只读补交。首尾快照不证明全程独占，峰值不能推出仅容一个作业；零正例在已知稀疏分布下概率至少50.49%，不能单独判采样bug。单场景1个warmup后样本不作全训练成本结论，详见本地审计 |
| EVIDENCE-009 | [L2S-20261017-009.md](L2S-20261017-009.md) | [S2L-20261017-009.md](S2L-20261017-009.md) | **COMPLETED / readiness `original_evidence_ready_for_local_review`**：一次收集 exit 0、`COLLECTED`，`profile_run_hash_link_verified=true`、`errors=[]`；`profile.json` COPIED 19,090 B `c3c47005…`、`run.json` COPIED 2,998 B `68b04b09…` 均与本请求预期 SHA 一致，`acceptance.json` `f64aaf1f…` 相符，五个副本经绝对路径 `cmp` 与原件逐字节一致；三份真实 008 wrapper 齐全，`6894fd74…`、`67c74c50…` **与 008 声明相符**，phaseC 此前未声明、本轮记录 `efa76c13…`；`check-ignore` rc=1、暂存清单已验证含全部原件。**全轮 56 秒／300 秒，GPU 使用 0**，未改 007/008 任何产物。如实记录服务器两处自写缺陷：收集脚本 review 块误把 `collection.json` 的 `files` 当 dict 解析（`AttributeError` + `set -e` 中断尾部）、`cmp` 相对路径算错一度误报不一致，均在**同一 deadline 内**以正确结构与绝对路径补齐并复核，第一版 `review.txt` 被补齐版覆盖（同轮同标签，无其他删改）。本回执同时更正 008 的两处过度表述 |
| FIRST-FIT-010 | [L2S-20261017-010.md](L2S-20261017-010.md) | [S2L-20261017-010.md](S2L-20261017-010.md) | **RUNNING**（19:32 领取；冻结核对全通过，prepared 两件套 SHA 预核成功；19:31 GPU0／GPU1 各 24,247 MiB 空闲、util 0%）：单A0/CE/seed20，TCM100/TAM100/joint20，最多6小时；现有完整验证选模，测试集封存，真实训练结果优先 |

本地复核 ENV-004 的 22:30 补记：整仓 HEAD 相等检查确实会阻止后续补跑，但不是已经发生的 CUDA 失败；从 CPU 提交到回执提交仅 HandOver 变化，旧 CPU 通过结论仍有效。ENV-005 已替换该检查，不再要求纯文档提交前后 HEAD 相同。等待日志实际跨度1167秒、含19/20秒两个短间隔，不能称全部至少60秒；离散采样不能证明四小时连续无空档，也没有 smoke 显存实测。原始服务器叙述保留在其 S2L，本地核验口径见 env_005_plan。

服务器对 008 回执的两处自我修正（008 原文保留不改写）：所写「未共享／近独占」只有**首尾两次**离散 `nvidia-smi` 查询为据，不能证明 974 秒运行期间无竞争，准确说法是「启动与结束时该卡无他人 compute 进程，期间未监控」；「单卡同时只容得下一个这种配置」是由 peak reserved ≈11.8 GiB 对 24,576 MiB 做的**算术推断，非并发实测**。与本请求的复核意见一致。



MTAD 四文件已在服务器，Level2 MD5 已匹配。本轮不要求用户重新上传数据，也无需重复确认空闲 GPU 的使用授权。

布局变更：2026-10-16 用户指示，从「覆盖 `Handover.md` + 归档 history」改为上述一方一个文件；ENV-001 两份文件由服务器 AI 从旧布局迁移而来，内容未改，迁移记录见 `S2L-20261016-001.md`。
