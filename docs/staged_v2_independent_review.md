# Staged training v2 独立审查记录

日期：2026-10-17。审查范围为新增 `association/staged.py`、`association/sampling.py`、`association/pipeline.py` 的 probe 门控及 [v2 规范](specs/staged-training-v2.md)。本次是本地工作树审查，未执行真实 MTAD/CUDA 训练，未提交、推送或下发服务器任务。审查者此前实现了 config/CLI/reference/profile 的路由；因此本记录的独立实现审查结论针对上述由其他代理实现的范围，不将自己的路由代码称为独立复核。

**结论：首次审查发现的两项 P2 已修复并复核关闭。当前范围未发现剩余已确认 P1/P2。** 这表示本地软件验证通过，不表示成本、收敛、数据 Level 对齐或论文结果复现已通过。

## 已发现及修复的问题

|问题|首次证据|修后复核|
|---|---|---|
|P2：失败记录写入错误遮蔽原始错误，`run.json` 留在 RUNNING|原异常处理先 `persist()` 再 `run.finish()`；注入原始数据 `ValueError` 和 FAILED progress 写入 `OSError` 后，外抛变为 OSError，终态仍是 RUNNING；初次 persist 还处在 try 外|`staged.py` 初次 persist 已移入 try（约199–201行）；异常处理分别尽力写 progress 和 run，记录 `evidence_write_errors`，再重新抛出原异常（约385–399行）。新增回归测试已实际运行通过。另单独注入首次 progress 写入失败，确认 run 仍落 FAILED 且原错误保留|
|P2：逐步显存证据被顶层 progress 覆盖，最终 history 无法回查|原版本仅在 persist 更新顶层峰值，history row 和 stage report 都未记录各步显存|每个 CUDA 更新的 row 已保存 `process_peak_allocated_bytes`、`process_peak_reserved_bytes`，遥测异常另记 `memory_measurement_error`（约307–313行）。字段明确表示进程累计峰值的逐步快照；并非本步或本阶段独立峰值。代码路径已核，未在真实 CUDA 上测量|

如果存储本身持续不可写，无法保证落盘终态；修复后的行为是尽力分别保存、保留原异常，并为记录错误添加信息，不再用记录失败替换研究执行失败。进程被 SIGKILL 或机器断电仍可能只留下最近一次 RUNNING 快照，这不应被报告成完整运行。

## 关键合同复核

- 采样只返回原 scene 的候选索引；正负独立选 scene，合并同 scene，再构造 scene 内节点映射。未跨 scene 造边，未删同源正例，完整候选数组用于验证与最终缓存。类别内按 scene 均匀和类内不足时放回的规则与规范一致。
- TCM 阶段 optimizer 仅包含 completion 参数，input embedding 固定；冻结 TAM 阶段 `set_completion_trainable(False, include_embedding=True)` 保持 TCM eval，优化器只更新 TAM/分类头。每阶段保存前后模块 state hash，并对受保护模块强制相等；不是只打印“已冻结”。
- 本地额外注入两个错误更新：TCM 阶段偷偷改变 embedding、冻结 TAM 阶段偷偷改变 TCM 参数。两者均被 after-hash 检查拒绝，run 落 FAILED。正常三阶段真实小模型测试也确认各阶段改变的模块集合符合预期。
- completion 缓存使用 `no_grad()` 的普通 CPU tensor，只缓存 `complete()`，TAM 始终重算并反传；条目字节容量受限，参数版本变化或重新解冻会拒绝旧缓存。冻结阶段结束清空缓存，joint 不读取它。LRU 容量限制针对持久条目，不是模型、当前 batch 和全部数据的总内存上限。
- 正式 fit 对完整 V_select 选阈值和权重，tie 保留较早快照；joint 从已选冻结 TAM checkpoint 开始。最终恢复全阶段最佳 checkpoint 后才写 T_head/V_select 缓存。测试和静态路径未发现用 V_select 更新参数、归一化读取 V_select、训练提前读取 T_test，或 probe 产生 eligible checkpoint 的情况。
- probe 只加载 T_core，正式 pipeline 拒绝 probe 配置。失败和预算耗尽保留 FAILED，正式 loader 不接受失败产物；已有输出目录不覆盖。

## 实际执行的验证

本次独立审查使用仓库 `.venv/bin/python`，只使用合成数据和 CPU。

|检查|结果|
|---|---|
|`python -m unittest discover -s tests -p test_staged.py -v`|10项通过；含真实梯度、模块冻结、缓存输出/梯度、joint 起点、完整验证/缓存重载、SIGTERM、预算与记录失败回归|
|`python -m unittest discover -s tests -p test_sampling.py -v`|12项通过|
|`python -m unittest discover -s tests -p test_pipeline_contracts.py -v`|6项通过|
|临时故障注入：首次 progress 写入失败|保留原错误，run 为 FAILED|
|临时故障注入：TCM阶段 embedding 误更新|after-hash 拒绝，run 为 FAILED|
|临时故障注入：冻结 TAM 阶段 TCM 误更新|after-hash 拒绝，run 为 FAILED|
|`git diff --check`|通过|

独立审查实际运行28项测试，另有上述3个临时故障注入场景。此前路由开发时的29项通过记录属于开发验证，不重复计入本次独立审查数量。

## 审查版本与未覆盖范围

以下是本次复核时工作树内容 SHA-256；后续有实质改动时应重新核查对应部分，而不是沿用本记录宣称通过。

|文件|SHA-256|
|---|---|
|`association/staged.py`|`48f1f6134016dc7843ea3ecee55d2ea8ce8648248dad98705572747d96fb5493`|
|`association/sampling.py`|`fed0873b7983a237d00091c0d2846ae0c2f40557861308f2d0a79f54d29186e1`|
|`association/pipeline.py`|`8354391d9cba1588fba97ade0304a40c152e0aa263ad3206906df47898bab4a8`|
|`tests/test_staged.py`|`796486010a9021c414783a4e9cafd659b7f62551b12f7fe1a053197ad691e970`|

没有实测 GPU 显存、吞吐、共享 GPU 竞争、完整 MTAD fit、训练收敛或 full-validation 的真实耗时。CPU 输出/梯度近似一致不证明跨 CUDA kernel 逐位等同；进程累计峰值也不能直接当作某一阶段独立显存或整卡占用。

Level1/Level2 与论文 Dataset1/Dataset2 的对应关系仍是执行前的数据核验事项，本代码审查不替代它。009仍按原只读取证任务处理；本记录不创建010、不授权新服务器实验，也不把14步探针当作训练收敛验收。
