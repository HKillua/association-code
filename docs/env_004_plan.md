# ENV-004：复用 FIELD，修正数值测试契约，重新验收 CPU/CUDA

日期：2026-10-16。来源：`S2L-20261016-003.md` 的 FIELD 结果与当前代码审查。本轮不重做 ENV-003 安装，不更改模型/划分/评价/版本 pin，不启动真实 MTAD 实验。

## 已知事实与判断

1. 原 ENV-003 的安装 wrapper 返回 124，内部 EXIT trap 却记录 0；原失败不改写。新验收必须正确处理 TERM/INT/HUP，仍以外层超时退出码及各阶段完成证据为准。
2. 服务器回执报告：FIELD 已装成 Python 3.11.7、NumPy 1.26.4、torch 2.5.1+cu121，pip check、导入及旧审计 15 项通过。原始 FIELD 产物仍在服务器，需本轮采集和重新核验，不能仅按叙述认定当前环境已通过。
3. 服务器报告同一批次重复轨迹的编码有约 2.38e-7 最大绝对差。源码实际只有一个 TAM encoder，整个批次一起编码，评分复用 embedding；并非左右两个独立参数网络。
4. 本地 macOS/Python3.12.14/torch2.10.0 下原测试通过。同轨迹批量与单独编码的 float32 差异为 2.38e-7（grad）/3.58e-7（no_grad），float64 差异降至约 4.44e-16/5.55e-16。支持有限精度计算路径解释，但未确认服务器具体内核；不能把它写成已在服务器复现。
5. [PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)明确讨论浮点计算顺序、平台/版本及批量计算的数值差异。共享参数不要求不同批次位置逐位相等。

## 修改与验收规格

| 条件 | 修改/输出 | 验证 |
|---|---|---|
| 共享 TAM、评分不重复编码 | 对实际 TAM 调用做 spy，检查整批输入；两分支都覆盖 | 模型单测 |
| 相同输入应数值一致 | 仅重复输入的 float32 编码使用 rtol=1e-5、atol=1e-6，保留近零绝对容差 | 模型单测；服务器 pinned 环境再验 |
| 轨迹编码不受批次其他轨迹/顺序影响 | 对比单条、整批及重排后的编码 | 新独立行为测试，能发现错误的跨轨迹信息混入 |
| 精确相同 embedding 的分数仍为零 | 单独构造完全相同 embedding，CE/contrastive 均保持精确零断言 | 新单测；不修改生产模型，不 round/clamp |
| 数据不被修改/遮罩/权重保存的严格契约 | 保持其他精确断言 | 全套 tests |
| 使用已有 FIELD 且导入当前代码 | 新 CPU 验收脚本记录 executable/prefix、包版本、源码来源、sitecustomize、原 FIELD 证据 | CPU 阶段；无自动安装 |
| 取消/超时不伪报成功 | 新阶段记录器显式处理信号；completed 只在任务确实完成后写入 | 本地信号/失败回归；服务器外 wrapper 记录 |
| CUDA 只占空闲的一张卡 | UUID 绑定、进程占用检查、查询失败拒绝启动 | 服务器现场执行；本地无 CUDA |
| 合成流程真实完成 | 内外 rc=0、完成状态、产物/设备/哈希验证 | 新 CUDA 验收脚本与服务器证据 |

修改范围：`tests/test_models.py`；新增 `tools/setup/*004*` 及其记录/测试辅助文件；授权规则与交接文档。`association/`、`configs/`、数据规则及已有 S2L/L2S 文件保持原样。

## 现场自主权与执行边界

用户本轮允许服务器承担有限现场决策，详见 [server_autonomy.md](server_autonomy.md)。网络/缓存/同版本环境修复/资源调度可由服务器在预算内处理；模型、测试语义、指标、划分、版本 pin 和实验预算由本地控制。现场处理纳入同轮回执，不再游离为单独 FIELD 任务。

ENV-004 总上限 45 分钟：CPU 验收与现场修复最多 15 分钟，GPU 等待最多 20 分钟，CUDA 执行最多 10 分钟。各阶段执行上限包含失败尝试和 TERM/KILL 宽限。CPU 与 CUDA 各最多两次尝试，即每阶段至多一次针对已修复临时故障的补跑，不是重跑完整流程。确定性代码或测试失败不得重试，成功阶段不重跑；CUDA 补跑可引用同代码下已成功的 CPU attempt。第二次使用新 attempt 输出。服务器只处置本轮进程。

阶段首次开始记录不可重置的截止时间，补跑仅使用剩余时间，不能重置 timeout 获得完整新预算；各阶段额度不可挪用。GPU 等待累计最多 20 分钟，检查间隔至少 60 秒；截止后只做必要终止与证据归档。

CPU 成功且 CUDA 成功：`status: COMPLETED / readiness: ready_for_native_profile`。CPU 成功但无空闲卡：`status: COMPLETED / readiness: partial / cuda_smoke: deferred_no_free_gpu`。测试、依赖或执行失败：`status: FAILED`，缺前提尚未执行：`status: BLOCKED`。没有本地新轮次，不进入正式数据准备、profile 或训练。

## 本地检查记录

原模型 11 项测试在本地通过；修订后模型 13 项、全套 tests 95 项、旧数据审计 15 项均通过。新增状态记录器 4 项回归通过（包含 TERM/INT/HUP、命令失败、未验收即退出），三个 shell 脚本通过 bash -n，7 段内嵌 Python 通过语法解析，git diff --check 通过。服务器 pinned 版本与 CUDA 验收留给 ENV-004，不能使用本地结果替代。
