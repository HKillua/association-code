# 在线观测

本目录维护在线观测的输入、运行、状态、关联、融合和评价代码。当前工程控制仅依赖 Python 标准库。MTAD 原包、冻结名单和数据定义沿用项目约定。方法和下一步见[研究状态](../docs/research_status.md#下一步)，历史 Q64 计时见[原工具入口](../tools/online/README.md)。

## 工程回放

从仓库根目录运行：

```bash
python -m online_observation causal-replay --help
python -m online_observation causal-replay \
  --archive PART1 PART2 PART3 \
  --manifest configs/data/mtad-level1-scene3000.json --scene 692 \
  --start 0 --stop 7200 --stale-after-seconds 60 \
  --output outputs/online-692.jsonl
```

- ZIP 分卷按文件字节顺序提供。回放只打开冻结开发名单内指定场景的量测成员。也可通过 `--measurements` 提供同格式的开发量测 CSV。JSONL 含片段键和预测坐标，按私有产物保留。
- 回放从 `--start` 冷启动，按量测时刻安排理想到达，逐秒输出到含 `--stop` 的最后一轮。片段以 `(source, batch)` 标识。最新整行的航速和航向用于位置外推，超过配置的陈旧时长停止输出。这个按 CSV 整行交付的运动控制用于检查回放、生命周期和输出格式。航速与航向的实际可用时刻见[数据约定](../docs/data_contract.md#在线字段可用时间)。
- JSONL 记录查询时刻、量测时刻、数据年龄、生命周期事件和逐秒计时。服务时间含交付、更新、预测和状态序列化。预读和排序单计，文件写入与设备传输在计时范围之外。排队是固定每秒释放一轮的串行模型。按协议的 `ticks` 核对结果完整性。

```bash
python -m unittest discover -s online_observation/tests
```
