# 当前数据选择：Level1

用户2026-10-21决定改用数据集1。旧Level2结果只作开发历史；050已另请求停止保全，实际回执看[交接索引](../../HandOver/Handover.md)。

## 已完成的本地准备

| 文件 | 作用 |
| --- | --- |
| [mtad-level1-scene3000.json](mtad-level1-scene3000.json) | 从本包5000场标签生成，seed20，2100训练/300验证/600封存测试，2000未用；scene0固定训练 |
| [mtad-level1-alignment.json](mtad-level1-alignment.json) | 仅训练/验证2400场的offset与gap600；不含测试映射 |
| [mtad-level1-population.json](mtad-level1-population.json) | 真实CPU准备报告：2400/2400通过时间一致性检查，零训练更新、零测试量测/reference读取 |

数据准备逐场检查实际用了约357秒。训练155134条、验证24599条合法片段，均无量测节点隔离；训练27632584候选对、验证4439092对。全部原生观测分别43020903和6810574点。该准备阶段没有拟合normalizer或训练模型；后续053—055已得到普通系统开发结果，见[交接索引](../../HandOver/Handover.md)，当前仍无Level1学习模型成绩。

仓库保留可执行名单/映射与约15KB的全人口汇总；重复的2400场成功时钟细节留本地`outputs/level1-preparation/population-full.json`，不另堆日志。汇总保留全部计数、失败列表和可行偏移唯一性统计；下方命令可独立复算完整证据。

| 训练侧统计 | 9001 | 9002 |
| --- | ---: | ---: |
| 原生平均采样间隔 | 10.018秒 | 20.073秒 |
| 原生最大相邻间隔 | 13.334秒 | 40.000秒 |
| cap128后平均间隔 | 29.990秒 | 32.198秒 |
| cap128后保留原生点比例 | 33.59% | 62.56% |
| 全场Q64落在本片段首尾之间的比例 | 49.29% | 47.61% |

间隔均值按相邻间隔加权；支撑比例按片段query加权，含尚无有效真值的query。全部训练片段合计仅48.54%的Q64 query在自身支撑内，26.87%距最近原生观测超过1800秒；验证分别49.00%、26.05%。密集采样不等于整场所有查询都有观测，也不能据此声称更换数据就解决补全。

时钟规则为`min_reference_time_temporal_support_v1`：固定取label命名身份的最小reference时间作为offset，检查label/量测边界、偏移后reference范围及端点共时。179733条有效片段全部通过；372场offset非零，最大3008秒。587场的时间范围约束不能唯一确定offset，所以报告只说“支持既定规则”，不声称唯一识别真实时钟。offset/gap仍按场显式加载，不让模型误差选择它。

报告中训练20307、验证3937个reference issue是事件数，不是坏行数。已核前8训练场（0、3、7、9、10、11、14、17）的157事件为153个同身份同时间状态冲突、4个完全重复。已有target逻辑按属性屏蔽冲突点及跨冲突插值，完全重复可合并；没有删除整场或把缺真值写成0。该8场分布不外推到完整2400场。

允许跨场共享A，不能称身份独立；两包共享身份及旧Level2开发必须披露。名单是项目自定，并非TMTAFF作者公布名单。测试标签仅用于一次性划分/依赖审计，测试量测和reference未打开。

## 数据文件与复现

本地目录：`/Users/a1/Downloads/02_金海豚竞赛_航迹关联/数据集/`，文件`MTAD航迹关联数据集_level1_part1.zip`至`part3.zip`，共2681532343字节。三个文件是同一ZIP的连续字节分卷。reader和模型CLI支持`--archive PART1 PART2 PART3`，也支持单个完整ZIP；无需合并或全面解压。Level由ZIP内部数据成员目录识别，不以外部文件名或场景数猜测，显式Level1名单拒绝Level2包。

已提交的数据准备结果可直接使用，服务器无需重复扫描。需要独立复现时使用新输出路径：

```sh
python tools/experiments/prepare_scene_data.py \
  --archive PART1 PART2 PART3 \
  --manifest configs/data/mtad-level1-scene3000.json \
  --clock-policy min_reference_time_temporal_support_v1 \
  --alignment-output outputs/level1-check/alignment.json \
  --summary-output outputs/level1-check/population.json
```

`PART1 PART2 PART3`须替换为实际有序路径；命令不训练、不读测试。不要覆盖已固定名单，也不要把Level2编号、offset、统计或optimizer用于Level1。

## 下一轮配置

- [TCM一遍覆盖模板](../tmtaff-tcm-level1.json)：155134条/每批64=2424步，`ordered_fragment`，固定末尾不足64条不回卷；原TCM架构与损失保持033口径，8小时仅提醒。这是一遍覆盖的起点，不是充分收敛或原论文复现成功的证明。
- [关联配置](../evidence-association-level1.json)：完整canonical观测，1000更新固定终点；记录实际场景/配对覆盖，step0必须复现053纯OLS工作点。[056交接](../../HandOver/L2S-20261022-056.md)已获用户批准并READY，按可用名额执行一次。
- [融合helper质量](../evidence-fusion-level1.json)、[Direct-only](../evidence-fusion-level1-direct-only.json)、[固定混合收益](../evidence-fusion-level1-mixed-action.json)：只改变辅助监督，统一正bias初始化、1000更新和α=0.5；必须给当前`--ols-working-point`，不继承旧14.2km或启用051符号辅助损失。[057交接](../../HandOver/L2S-20261022-057.md)已获用户单独批准并READY，可与056按可用名额并行。

当前056关联与057融合均READY，TCM模板不在本次授权内；实际启动状态以服务器回执为准。参考编码器仍cap128/全场Q64，候选完整观测的收益须先与相同输入的OLS比较，再作同预算归因。后续训练预算和执行只看独立交接。
