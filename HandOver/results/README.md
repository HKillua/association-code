# 结果保留规则

这里保留**当前研究决策需要的摘要**，不是服务器运行目录的镜像。2026-10-22清理后，旧Level2的大型预测包、训练中间快照、重复准备缓存、启动与显卡日志已移除；服务器原始数据、权重和运行目录未修改。

| 实验 | 保留目的 |
| --- | --- |
| 052 | Level1数据包与成员数量事实 |
| 053A／054A／055A | 普通完整系统、错误归因与一致性拆分的已结案证据 |
| 053B | 一份TFM冻结诊断报告，说明旧baseline尚未有效复现 |
| 056 | 固定final1000关联与六臂系统基线；062的服务器输入来源 |
| 057 | 三种监督的final1000配置、损失定义和系统结果；062的冻结模型来源 |
| 058／061 | 真值分组及学习分组下的受限Oracle诊断、人口和支撑分层 |
| 059A／059B | 配对采样、实际曝光与完整系统结果 |
| 060 | 3000更新终点、1000前缀复现证据和共同点比较 |
| 062／063 | 原迁移失败与首场未复现的原始回执、诊断摘要；完整系统尚无结果 |

关键实验目录通常只留三份文件：

- `compact-summary.json.gz`：原聚合指标及run／配置／终点信息。`provenance.source_files`登记被摘录的原文件，`records`保存数值；关联训练曲线仅留验证快照和最后一次训练记录。057三臂合并在同一个文件中。
- `scene-metrics.json.gz`：300场的三属性误差、人口／覆盖和共同点摘要，及已有的分组／归因统计。它省略逐目标匹配数组、逐场细分层和路由。
- `server-artifacts.txt`：原始服务器文件路径登记。其“已回传”清单和大小是当时的历史记录，不代表清理后的Git文件清单。

原聚合数值未重新计算或取整；新增的`system_population`仅将原逐场人口计数相加。各摘要明确记录省略范围，**不能替代完整summary、预测NPZ或checkpoint校验输入**。062必须读取服务器上的完整056／057运行目录，不能将这些摘要改名后作为源。

所有被清理的Git原件可在[清理前5776cd3](https://github.com/HKillua/association-code/tree/5776cd3/HandOver/results)恢复；历史研究结论与更正见[研究历史摘要](../../docs/history/README.md)。未重复搬运原件到新的归档目录。

用Python读取摘要：

```python
import gzip, json
with gzip.open("HandOver/results/LEVEL1-LEARNED-ORACLE-20261022-061/compact-summary.json.gz", "rt") as stream:
    result = json.load(stream)
summary = result["records"]["summary"]
```

后续回传继续采用：紧凑聚合摘要＋必要逐场指标＋服务器路径。大型预测、全部中间步及启动日志留在服务器；真有新的复算问题时按需取回，不默认加入Git。
