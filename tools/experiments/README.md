# 运行入口

先看[交接索引](../../HandOver/Handover.md)。当前正式数据为Level1；旧Level2实验已结案或请求停止。模型配置不是训练授权，新任务须有具体交接；不要根据旧命令自行恢复预算。

| 入口 | 用途 |
| --- | --- |
| `tmtaff_complete_fit.py` | 参考模型四阶段，或显式`--phase tcm/tam/tfm`；TAM/TFM用`--initialize-tcm-from`冻结移交，resume只恢复同阶段 |
| `tmtaff_complete_evaluate.py` | 已有完整快照的冻结评价，不重训或重选阈值 |
| `cooperative_fit.py` | 候选补全及早期联合结构，`--phase teacher`输出单轨补全及可选预测 |
| `evidence_association_fit.py` | 普通OLS底座上的几何/全证据关联，`check`只用合成数据，`fit`才读真实数据 |
| `evidence_fusion_fit.py` | 固定OLS和分组的属性效用融合；Level1真实fit必须提供当前`--ols-working-point`；`--mode check`仅合成检查 |
| `evidence_fusion_evaluate.py` | 历史048固定关联分数/融合权重的连接复算；仅接受旧selected_cap128，不能直接用于新Level1候选 |
| `level1_ols_working_point.py` | 仅V_select、零训练计算当前Level1纯OLS工作点，不必先运行关联fit |
| `level1_system_evaluate.py` | 仅读V_select，普通/学习关联×普通/学习融合；完整目标、共同有效点、分组与路由数组保存；可不带任何学习权重 |
| `../analysis/tfm_information_probe.py` | 049原始验证输入导出TCM特征并检查0/100/300各层及单侧替换响应；CPU零更新，不读reference或测试 |
| `../analysis/system_attribution.py` | 054A仅复算053A保存数组；完整目标失代归因、独立片段/固定组anchor控制、共同点与覆盖重复代价；无archive/模型/训练入口 |
| `tsactfer_fit.py` | 独立TSA-cTFER离线适配，非原在线方法 |
| `make_scene_manifest.py` | `--archive`仅读标签生成Level1新名单；旧`--source-manifest`转换仅供历史兼容 |
| `prepare_scene_data.py` | 已指定train/V的时钟、人口和原生/选点密度统计；Level1用显式temporal_support策略，不需旧alignment |
| `launch_detached.py` | 使用服务器服务管理器托管一次作业，独立保存进度和退出状态 |
| `../setup/gpu_allocation.py` | 按明确授权和兼容性选GPU，允许共卡，不终止他人进程 |

每个入口都支持`--help`。涉及原生数据的上述入口均支持`--archive 一个完整ZIP`或`--archive part1 part2 part3`（顺序不可颠倒）；不需合并或全面解压。真实训练显式给manifest、config、新output和设备，补全/融合还需alignment；CUDA不可用时不静默转CPU。已生成Level1名单与准备说明见[数据入口](../../configs/data/README.md)。

候选使用`configs/evidence-association-level1.json`、`configs/evidence-fusion-level1.json`的`canonical_all`政策。关联step0输出`ols-working-point.json`，融合以`--ols-working-point 路径`加载；核同一Level、完整训练/V名单、输入政策与数据设置。此文件的阈值来自纯OLS，不能用学习分数阈值替代。配置预算是模板，不表示训练READY。

保留原生时间、有效短轨和全部候选；不按真值删难例。normalizer只拟合训练，测试保持封存。输出记录真实配置、阶段更新、覆盖、指标和终态；程序正常退出不能代替方法效果。

参考方法细节见[训练规格](../../docs/specs/complete-training-refactor.md)，候选关联/融合见[方法定义](../../docs/specs/evidence-association-fusion.md)。旧消融、执行命令与失败结果从[历史摘要](../../docs/history/README.md)查阅，已完成的一次性诊断入口已删除。
