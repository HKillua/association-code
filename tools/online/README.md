# 在线路线代码入口

在线与[离线](../offline/README.md)持续并行。当前目标与待办只看[研究状态](../../docs/research_status.md)，执行和冻结范围看[交接](../../HandOver/Handover.md)。

## 入口与兼容

从仓库根目录查看命令：

```bash
python -m tools.online --help
python -m tools.online pipeline-timing --help
```

[pipeline_timing.py](pipeline_timing.py) 是原 `tools/analysis/deployment_pipeline_timing.py` 的真实迁移，参数、场景、模型和输出格式保持不变。原脚本命令、`python -m tools.analysis.deployment_pipeline_timing` 与旧模块导入继续可用；新旧导入指向同一模块对象。

实际计时需原协议指定的服务器目录和冻结权重，使用 `--root`、`--output`、`--time-policy advisory`；`--completed-run` 仅接收原失败前四条有效记录。本次迁移没有新开计时任务，[原报告](../../docs/reports/2026-11-05-pipeline-online-timing.md)保留既有证据。

## 实现范围

- 完整单次链路：已接收 measurement CSV 字节→清理/运动拟合→关联/分组→融合恢复→JSON 序列化。单次 `Pipeline.call` 不含模型初始化、传输及等待观测的时间。
- 固定工程协议比较全窗口控制和两个相邻一秒的前缀；每次重算全前缀，输出仍是历史 Q64 区间，不保证对应最新截止时刻，也未实现持续身份、增量缓存和长期队列。
- [association/online/replay.py](../../association/online/replay.py) 独立提供 `csv_rows`、`ReceivedMeasurements`、`preprocess`：验证输入格式，从收到的来源/片段边界生成匿名元数据，拒绝未来观测，复用共享清理/运动模块；不打开真实 label/reference，至少需要两个有效片段。
- PTFM/DeepAF 组预测复用 [fusion_peer_inputs.py](../../association/models/fusion_peer_inputs.py)，在线入口不导入离线训练编排。`association/online` 可随包安装，`tools/online` 仅从源码仓运行。

## 评价边界

用户指定每秒更新一次，最大延迟尚未指定；更新频率、计算延迟、确认等待和稳定吞吐须分开报告。两个前缀的工程计时不证明持续 1 Hz 或在线精度。离线可共享解析、单位、模型与适用指标；未来信息权限、查询时刻、窗口和延迟/规模协议需分别定义。

TSA-cTFER 是在线关联候选，现有离线 encoder 适配不包含原在线决策；固定关联后的融合另设控制，不能将局部对照称为全系统胜负。对照资格与在线缺口见[路线审查](../../docs/reports/2026-11-10-offline-online-route-review.md)，后续工作统一维护在研究状态。
