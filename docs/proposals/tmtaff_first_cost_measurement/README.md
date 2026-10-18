# 首次全尺寸成本测量：具体待审配置

2026-10-18，**CONFIGURED_NOT_AUTHORIZED，方法/数据配置已确认，新增GPU额度未批准**。这份方案完整列出拟执行参数，用户可以在运行前审阅；它不是交接索引中的READY任务。

[完整CLI配置](../../../configs/tmtaff-time-cost-probe.json)给出模型、融合、四阶段测量、数据和划分参数；[576场景对齐参数](../../../configs/tmtaff-time-reference-alignment.json)给出已核offset及gap600。[资源提案](proposal.json)记录已批准项与仅剩的预算决定。参数文件已配齐，不猜offset、补默认loss或重建划分；文件存在不表示任务READY。

已批准：真实时间条件TCM、连续SmoothL1 β=.001、TCM后训练侧一次CL margin校准。构造器 `model.margin=1` 仅兼容占位；`margin_policy=train_negative_median_v1` 必须在任何TAM更新前替换成实际中位数，校准失败不退回1。

用户2026-10-18已确认的数据方案：

- 保留所有有效片段，每条最多128个真实点，保首尾，按原canonical索引均匀选点；T_core保78.62%点，9.26%长轨受截断，V保97.35%。256点备选保90.89%训练点，但history更长，尚无实测成本；不靠变更模型或精度降成本。
- 固定 `time_unit_seconds=20`，仅缩放时间编码，不把数据重采样成20秒，也不随轨迹或验证集调整。完整池已查时间顺序和支撑判定；1个内部FP32时刻alias如实保留披露。
- 用已核逐场景offset，参考只在自身实际时间范围内按逐属性有效端点插值，最大gap600秒；不外推真值、不跳过坏点、不按参考覆盖删目标。训练/V监督覆盖76.7648%/76.1107%；gap180备选覆盖57.8780%/51.8733%，并使57个训练身份全无监督。较长插值不等于准确运动真值，需披露参考生成方式和覆盖。

模型维度128、6层、8头、FF256、dropout.1；共同输出64点，TFM独立双encoder和64步自回归。FP32、seed20、CPU4；有效batch64、TFM/joint微批1，TCM预训仍整批64。四阶段各2更新、固定学习率，附带一次64批校准、两次完整V关联选模和最终全目标融合。**这是成本测量，8次更新不能代表充分训练或方法效果。**

资源待审：单作业、一次无重试、累计最多1 GPU小时，含准备/校准/验证/终评/写盘；外层现有timeout在3570秒发TERM，最多30秒清理，累计不超3600秒。可动态选兼容GPU、共卡，不终止他人进程。预算不足可能拿不到后两阶段步时，不能保证此上限内完成，也不将失败自动续为第二轮。

现有证据只有旧CE路径耗时和新模型参数/工作量，没有新时间适配＋TFM的GPU实测。因而目前**无法给出可信的完整训练小时数**；1小时是提议的测量上限，绝非总训练成本预测。测到各阶段时间后，再把90%/95%/99%抽样覆盖候选的更新量换算为正式资源方案。成本公式及边界见[完整计划](../../tmtaff_complete_plan.md)。

原manifest服务器路径已经明确：`runs/PREP-NATIVE-20261017-007-attempt-1/manifest.json`。本地012仍待复制原件；冻结名单足够完成本地数据诊断，但不替代完整组件图及全archive字段。本轮不构造假的manifest，不重复prepare。CLI仍要求原manifest、显式alignment/config、新输出目录及当时GPU，仅待GPU测量额度批准后才写独立L2S；本地完整原manifest装配尚未执行，服务器使用已有原件。

## 拟执行命令（只供批准前审阅）

从服务器仓库根目录运行；`task_device`由服务器按实时兼容GPU填写，例如选中设备的`cuda:<索引>`，不能把示例固定为某张卡。Python/ZIP/manifest路径来自既有007/010回执。新输出不存在才启动，原日志和退出码由服务器按既有交接记录。

```bash
timeout --signal=TERM --kill-after=30s 3570s \
 /home/wangguangjie/djs/mtad-field-20261016/bin/python \
 tools/experiments/tmtaff_complete_fit.py \
 --archive /home/wangguangjie/djs/data/mtad/数据集level2.zip \
 --manifest runs/PREP-NATIVE-20261017-007-attempt-1/manifest.json \
 --alignment configs/tmtaff-time-reference-alignment.json \
 --config configs/tmtaff-time-cost-probe.json \
 --output runs/TMTAFF-TIME-COST-20261018-013-attempt-1 \
 --device "$task_device" --cpu-threads 4
```

这条命令未执行。没有RUNNING/READY回执。测量预算若获批，读取、校准、8次更新、全部验证/终评和失败开销均计入同一上限，不能另行先跑一次“免费warmup”。验收看完成状态、各阶段实际更新/耗时、校准样本与m、验证与融合总耗时、选定设备allocator峰值及参考/输入覆盖；超时或缺阶段就如实FAILED，保留产物，不缩规模或跳评价改称成功。最终开发指标仅验证链路，不作为收敛性能或创新有效证据。
