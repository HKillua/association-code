# 运行入口

先看[交接索引](../../HandOver/Handover.md)。当前正式数据为Level1；旧Level2实验已结案或请求停止。模型配置不是训练授权，新任务须有具体交接；不要根据旧命令自行恢复预算。

| 入口 | 用途 |
| --- | --- |
| `tmtaff_complete_fit.py` | 参考模型四阶段，或显式`--phase tcm/tam/tfm`；TAM/TFM用`--initialize-tcm-from`冻结移交，resume只恢复同阶段 |
| `tmtaff_complete_evaluate.py` | 已有完整快照的冻结评价，不重训或重选阈值 |
| `cooperative_fit.py` | 候选补全及早期联合结构，`--phase teacher`输出单轨补全及可选预测 |
| `evidence_association_fit.py` | 普通OLS底座上的几何/全证据关联，`check`只用合成数据，`fit`才读真实数据 |
| `evidence_fusion_fit.py` | 固定OLS和分组的属性效用融合；`--mode check/fit`区分合成检查和训练 |
| `evidence_fusion_evaluate.py` | 固定关联分数和融合权重的连接评价，支持完整目标视图 |
| `tsactfer_fit.py` | 独立TSA-cTFER离线适配，非原在线方法 |
| `make_scene_manifest.py` | 旧共享A元数据到场景协议的历史转换器；不能拿Level2名单替代Level1准备 |
| `prepare_scene_data.py` | 已指定场景名单的原生时钟/人口准备；Level1需要自己的名单与映射 |
| `launch_detached.py` | 使用服务器服务管理器托管一次作业，独立保存进度和退出状态 |
| `../setup/gpu_allocation.py` | 按明确授权和兼容性选GPU，允许共卡，不终止他人进程 |

每个入口都支持`--help`。真实训练显式给archive、manifest、alignment、config、新output和设备；CUDA不可用时不静默转CPU。数据集1三个文件是连续分卷，需要先提供可读的完整ZIP或分卷读取支持；目前没有已经生成的Level1正式名单。

保留原生时间、有效短轨和全部候选；不按真值删难例。normalizer只拟合训练，测试保持封存。输出记录真实配置、阶段更新、覆盖、指标和终态；程序正常退出不能代替方法效果。

参考方法细节见[训练规格](../../docs/specs/complete-training-refactor.md)，候选关联/融合见[方法定义](../../docs/specs/evidence-association-fusion.md)。旧消融、执行命令与失败结果从[历史摘要](../../docs/history/README.md)查阅，已完成的一次性诊断入口已删除。
