# 候选方法：本地实现与核验

2026-10-18。用户授权 review 后本地实现 CPU/GPU；本记录是软件验证，**没有真实 MTAD 候选训练或性能结论**。行为规格见[规格](../../specs/cooperative-completion-fusion.md)，命令、退出码、设备和逐模块参数见[verification.json](verification.json)。

## 已实现

- 原生时间编码器与共享物理查询解码器：单片段补全、多片段输出经纬度/速度/航向，保留两点短轨；reference 只进入监督和评价。
- 独立身份头与有方向、随查询时刻变化的效用头。冻结教师以同一输入比较单轨/辅助恢复误差，保留负效用。
- 首版完整训练入口：教师 → 效用 → 普通融合、仅身份门控、身份×效用门控、普通可学习门控四臂；固定相同身份分数、阈值、预测分组，匹配各臂初始化和采样/更新数。
- 全目标评价、单源/简单融合/真实分组诊断、缺失区与覆盖统计；最终状态保存及推理重载。复用现有原生 ZIP、manifest、参考对齐接口，T_test 封存。

最终使用效用的身份头及单步训练接口已实现，但首版 CLI **只做固定身份比较**；其后续完整身份阶段尚无正式训练计划。检查点不支持优化器续训，代码明确拒绝覆写输出；不把保存优化器状态称为支持精确恢复。

## 复核结果

独立复核发现并修复：效用路由漏乘身份分数、训练入口未拒绝错标签/缺候选/非法参考、数据准入耗时漏计预算。另加入仅身份门控对照，防止把身份过滤收益归因于效用。相关回归已通过；没有已知未关闭的执行缺陷。这不证明研究假设或新颖性成立。

| 检查 | 实际结果 |
|---|---|
| 全仓 CPU 测试 | 410/410，17.385 秒，含新增 39 项候选测试 |
| 原生审计测试 | 15/15，0.073 秒 |
| CPU 小模型四臂完整流程 | COMPLETED，真实优化、完整合成 V/物理融合、检查点 |
| Apple GPU/MPS 小模型四臂完整流程 | COMPLETED，同上，未回退 CPU |
| CPU 与 MPS 完整规模 | 各一次前向/反向，有限梯度，输出 64×4；真实输入长度 2/128 |
| NVIDIA CUDA | 设备不可用，CLI 明确退出 1；没有 CUDA 实测结果 |

完整尺寸为 encoder/decoder 各 6 层、128 维、8 头、FF256，Q64、cap128，FP32。实例含所有候选/对照头共 **2,156,814** 参数；普通路径实际涉及 2,006,277，效用路径 2,105,929，额外 query-pool＋效用头 99,652（约 4.97%）。普通可学习门控路径也是 2,105,929。没有通过缩小正式模型来通过尺寸检查；小模型仅用于快速验证整个执行流程。CPU/MPS 秒数含不同启动和并发环境，不构成硬件性能比较，不能外推服务器训练时间。

## 使用

在已安装项目依赖的环境，从仓库根目录执行。每次使用新的输出目录：

```bash
python tools/experiments/cooperative_fit.py check --device cpu --output runs/cooperative-check-cpu --cpu-threads 4
python tools/experiments/cooperative_fit.py check --device mps --output runs/cooperative-check-mps --cpu-threads 4
python tools/experiments/cooperative_fit.py shape-check --device cuda:0 --output runs/cooperative-shape-cuda --cpu-threads 4
```

`check` 不接受原生数据路径；`shape-check` 不开数据、不做优化。`fit` 必须显式提供 archive、manifest、alignment、config、device 与新输出路径；没有默认正式训练更新数，也没有发布新的 READY。

正式配置形状为 `scope=cooperative-physical-v1`、`train_splits=[T_core]`、`data.time_unit_seconds=20`、完整 `CooperativeConfig` 与规格列出的全部 training 字段。沿用已批准 cap128/显式时钟/gap600。候选的更新数、误差尺度解释、mask 训练量、累计成本和预算须在正式执行前列清；本次没有挪用参考013的额度。

剩余科学问题：真实缺失区是否有可预测效用、教师效用能否迁移到调整后的解码器/多成员组、是否优于普通同容量门控、海上近邻是否已覆盖本机制。下一步应围绕这些比较安排有界真实实验，而不是继续扩建框架。
