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
| `../analysis/group_refinement.py` | 055A复现053原OLS分组/关联统计与054三臂；仅新增原组内first-fit一致性拆分，五臂完整系统及同anchor/helper共同点诊断；CPU零训练 |
| `../analysis/oracle_diagnostics.py` | 058只读053缓存，原OLS/真值身份分组×四种输出；区分覆盖恢复、代表切换与固定候选选择收益，CPU零训练 |
| `../analysis/compare_association_duration.py` | 060对056完整系统缓存比较，核同人口/输入与OLS控制，分别报告共同点、同/换代表、新/丢覆盖、G/U/D和场景宏平均；CPU零训练 |
| `../analysis/learned_group_oracle.py` | 061只读056 final1000缓存，逐数组复现三普通控制，固定学习分组/代表作Anchor/Mean受限选择；报告组/唯一目标分母、可见支撑与点级分位数，CPU零训练 |
| `tsactfer_fit.py` | 独立TSA-cTFER离线适配，非原在线方法 |
| `make_scene_manifest.py` | `--archive`仅读标签生成Level1新名单；旧`--source-manifest`转换仅供历史兼容 |
| `prepare_scene_data.py` | 已指定train/V的时钟、人口和原生/选点密度统计；Level1用显式temporal_support策略，不需旧alignment |
| `launch_detached.py` | 使用服务器服务管理器托管一次作业，独立保存进度和退出状态 |
| `../setup/gpu_allocation.py` | 按明确授权和兼容性选GPU，允许共卡，不终止他人进程 |

每个入口都支持`--help`。涉及原生数据的上述入口均支持`--archive 一个完整ZIP`或`--archive part1 part2 part3`（顺序不可颠倒）；不需合并或全面解压。真实训练显式给manifest、config、新output和设备，补全/融合还需alignment；CUDA不可用时不静默转CPU。已生成Level1名单与准备说明见[数据入口](../../configs/data/README.md)。

候选使用`configs/evidence-association-level1.json`、`configs/evidence-fusion-level1.json`的`canonical_all`政策。关联step0输出`ols-working-point.json`，融合以`--ols-working-point 路径`加载；核同一Level、完整训练/V名单、输入政策与数据设置。此文件的阈值来自纯OLS，不能用学习分数阈值替代。配置预算是模板，不表示训练READY。

系统评价学习关联时，必须提供`--association-run`、`--association-checkpoint`及原scores/summary/data-audit三项；`--association-endpoint final`使用固定终点，`selected`为兼容默认。入口重算对应权重分数，并核验相邻`baseline-scores.npz`/`baseline.json`精确复现OLS；不能混用选中步数的阈值。`baseline`端点不加载权重。`--ordinary-controls anchor mean_supported attribute_hybrid`生成实际三种输出，默认仍只mean；纯OLS调用无须关联权重。详细一次性命令见[056](../../HandOver/L2S-20261022-056.md)。关联`sampling_audit.json`只计成功optimizer更新对应的实际抽样。

059配对配置为`evidence-association-level1-paired-uniform.json`与`evidence-association-level1-paired-hard-negative.json`，仅`sampling.mode`不同；两臂seed20261022、1000更新，独立干预seed20261023。原采样器不改；B在原batch后将首64负槽改从同场景T内OLS前10%负池抽取，允许抽回原配对或与保留样本重合。`paired_sampling_audit.json`分别记录来源槽数、实际变化数、最终池命中及各来源曝光/重复/分数分布；只计成功更新。无`sampling`字段时保持旧路径，A与同seed原路径逐值兼容。执行权限见[059A](../../HandOver/L2S-20261022-059A.md)/[059B](../../HandOver/L2S-20261022-059B.md)，配置存在不代表已批预算。

060配置`evidence-association-level1-3000.json`与056仅更新数不同；`fit --prefix-reference 056的association目录`只读源final1000核对，绝不续训。1000保存prefix权重/Adam、分数、采样与比较报告；数值差异明确标记，不影响固定3000终点，协议错配失败。普通旧调用无此参数时行为不变。完整命令及新增GPU预算状态见[060](../../HandOver/L2S-20261022-060.md)。

融合Level1三配置仅`training.utility_objective`不同：原文件为`helper_quality`，`-direct-only`与`-mixed-action`为两项对照；共享`model.utility_output_bias=0.01`，固定`mixed_action_alpha=0.5`，符号BCE关闭。历史配置省略新字段仍用原初始化/监督。三臂命令和已批预算见[057](../../HandOver/L2S-20261022-057.md)，训练诊断保存在`run.json`。

保留原生时间、有效短轨和全部候选；不按真值删难例。normalizer只拟合训练，测试保持封存。输出记录真实配置、阶段更新、覆盖、指标和终态；程序正常退出不能代替方法效果。

参考方法细节见[训练规格](../../docs/specs/complete-training-refactor.md)，候选关联/融合见[方法定义](../../docs/specs/evidence-association-fusion.md)。旧消融、执行命令与失败结果从[历史摘要](../../docs/history/README.md)查阅，已完成的一次性诊断入口已删除。
