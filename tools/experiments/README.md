# 实验入口

在仓库根使用 `python tools/experiments/<脚本>.py --help`。真实实验按[当前交接](../../HandOver/Handover.md)中的独立READY运行，接口和历史配置本身不授权训练。

| 脚本 | 用途 |
| --- | --- |
| `tmtaff_complete_fit.py` | 默认完整参考四阶段；`--phase tcm`只训补齐；`--phase tam --initialize-tcm-from`从已完成TCM开始独立关联训练 |
| `tmtaff_complete_evaluate.py` | 加载选中检查点，仅评价，不重新训练/校准/选阈值 |
| `cooperative_fit.py` | 候选训练；v2支持teacher独立补全评价，以及plain/scalar-u/attribute-u/learned-attribute融合臂 |
| `tsactfer_fit.py` | TSA-cTFER离线关联适配，不含原在线指派，也不是完整融合系统 |
| `completion_check.py` | 固定4个V场景补齐诊断；双模型、候选残差、`--reference-feedback`反馈辨析或`--candidate-dynamics`逐层动态；无训练 |

`completion_check.py --candidate-residuals`复用同一入口，只推理既有teacher，每片段一次；参考checkpoint仅用于原数据契约核对。输出逐场景`predictions-*.npz`（预测、位置底座、原真值、逐属性mask及片段/时间/区域）、`residual-diagnostics.json`（最终学习权重、正则、分区域原属性损失）及`candidate-summary.json`。分区域loss为区域内均值，计算全局贡献时乘该属性的区域有效点数/全部有效点数；位置底座没有速度/航向预测。它不还原训练梯度、不生成额外裁剪或改写权重。

`completion_check.py --reference-feedback --reference-checkpoint /已完成033目录`只做原生短程反馈辨析，不需要`--candidate-checkpoint`，其余archive/manifest/alignment/output/device参数沿用原入口。同初始prefix memory、同目标时间，对比真实native历史与预测历史，双向至多3步；第1步检查相同，第2/3步分别报告。`reference-feedback.json`保留所有片段、不可评短轨、逐方向/步SSE和分母，按元素汇总。无reference CSV、自然query重评或参数更新，不能替代完整补齐成绩。范围见[036](../../HandOver/L2S-20261020-036.md)和[综合排查](../../docs/proposals/repair_before_retraining/tcm-diagnosis.md)。

`completion_check.py --candidate-dynamics --candidate-checkpoint /已完成034/teacher.pt --device cpu --cpu-threads 4`只读取已完成teacher的原生输入；不接受reference checkpoint，不读取reference或alignment内容，保留alignment参数仅兼容入口。加载源run.json核数据契约，每片段一次encode/decode，输出`candidate-dynamics.json`记录query、decoder各层、head和物理输出的时间变化及冻结残差均值后的输出差异；无目标、误差评价或训练。现有权重、hooks、冻结标记及模式不变。范围见[038](../../HandOver/L2S-20261020-038.md)及[规格](../../docs/specs/cooperative-dynamics-diagnostic.md)。

参考完整流程和候选可用 `--save-predictions` 保存实际预测、真值、mask及分组，供 `tools/analysis/compare_physical_predictions.py` 做全人口覆盖/共同有效点比较。进度和失败原件保留；不为新增报表重跑已结束实验。

旧pilot/profile/成本probe入口已删除。017、018均已结束；候选修正版和共享/独立臂入口已完成。用户最新把训练时长改为建议与提醒，固定训练计划和当前依赖以[交接索引](../../HandOver/Handover.md)为准。研究方法见[候选规格](../../docs/proposals/candidate_revision_after018/README.md)，旧阶段见[历史摘要](../../docs/history/README.md)。


正式服务器训练从`launch_detached.py`的一次性systemd用户服务提交，已启动任务不依赖AI前台会话。示例接口：

```bash
python tools/experiments/launch_detached.py --cwd /冻结执行目录 --record-dir /新启动记录目录 \
  --time-policy advisory --max-seconds 建议时长秒数 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES=GPU-实际UUID --env OMP_NUM_THREADS=4 \
  -- /既有环境/bin/python tools/experiments/tmtaff_complete_fit.py 完整已批参数 --time-policy advisory
```

显式传递MKL/OPENBLAS线程等所需环境；服务不继承AI shell全部环境。payload是argv，不通过shell展开。`launch.json`的SUBMITTED不是模型成功；重连读取`status.json`/exit_code及模型run/summary。超建议时长会生成`time_advisory.json`，AI读取后询问用户；AI离线时不能保证即时通知。必须在launcher和训练CLI两层启用advisory，默认hard仍兼容旧任务。在途作业的旧限制不会因仓库更新消失。用户管理器须Linger=yes，`session`只供本地核验。026已经完成服务器托管核实，新队列见[当前方案](../../docs/proposals/repair_before_retraining/README.md)。


候选v2可拆成共享阶段与独立融合臂：`fit --phase shared`先保存shared.pt、identity-records/与唯一阈值；随后`fit --phase arm --shared-from /已完成021输出 --route attribute-u --save-predictions --time-policy advisory`（其余臂替换route）。每臂独立输出和耗时，只加载同normalizer/S/模型/loss/身份记录，不重训teacher/u或选阈值。默认full保持旧行为，shared不接受--save-predictions。当前GPU队列以[交接索引](../../HandOver/Handover.md)为准。

`fit --phase teacher --save-predictions`保留原teacher self/cross/BCE训练，在固定端点做完整V独立片段补全，不进入关联选阈值、utility或融合；配置的utility/adapt更新数必须为0。输出`teacher.pt`、`completion-evaluation.json`及`completion-predictions/`逐场NPZ。该NPZ是独立补全格式，按逐属性mask/region统计，不能直接当分组融合格式交给`compare_physical_predictions.py`。`model.speed_output_policy`缺省为原`physical_softplus`；`scaled_log`只对v2生效并要求相应新权重。[031/032 A/B方案](../../docs/proposals/repair_before_retraining/030-repair-analysis.md#6-已实现的第一项与ab2026-10-20)已获用户继续指令，只按当前索引的独立READY执行。

本地收到两组完整结果后，先将gzip JSON和预测tar.gz还原为各自run目录的原层级，再执行`python tools/analysis/compare_completion_predictions.py --left /031结果目录 --right /032结果目录 --output /新的对比.json`。只读取已完成teacher预测，核完整人口与共有真值/mask一致，输出整体、来源、逐场的各区域/距离误差、raw log-speed MSE及速度分布；不重新推理或选阈值。

`tmtaff_complete_fit.py --phase tcm`要求非TCM更新全部为0，固定端点输出`completion-evaluation.json`和未选模`checkpoint.pt`；不支持`--save-predictions`。模型仍为原一步TCM，特征误差不能与物理米误差比较。默认`full`保持四阶段。[规格](../../docs/specs/tmtaff-tcm-phase.md)与[033任务](../../HandOver/L2S-20261020-033.md)说明具体范围；用户已批准一次4000更新，只按当前索引READY领取。

候选新可选`model.heading_output_policy=observation_residual`保留anchor可见航向底座并学习圆周修正；默认`learned_absolute`保持。新头需要新训练，不能将032旧权重静默改公式。用户已单独批准[034](../../HandOver/L2S-20261020-034.md)一次1000步及完整V评价；033已完成，不重启；CPU验收不代表实际效果。

候选v2新增三个独立可选消融：`model.decoder_norm_first=true`只改decoder归一化顺序；`model.decoder_num_layers=2`保留原完整初始化后裁decoder、encoder不变；`training.teacher_identity_weight=0`仅使teacher身份BCE贡献为0，仍计算原BCE并保留物理self/cross和全局clip。默认分别false/null/1，旧权重/配置兼容；非默认decoder语义会保存并校验，不能直接换旧checkpoint的forward。teacher日志`identity_bce_raw`为原BCE，`loss_components.identity_bce`为实际加权贡献。[039—041规格](../../docs/specs/cooperative-decoder-ablations.md)及DRAFT交接有完整命令，当前尚非开跑授权。

`tmtaff_complete_fit.py --phase tam --initialize-tcm-from /已完成033输出`只复制并冻结completion与输入embedding，独立初始化TAM/优化器/预算；配置仅`TAM_FROZEN`非零，不接受物理`--save-predictions`。按原规则在T_core校准margin，每次完整V验证保存全部关联原分数及overall/ITSA/MSTSA/paper_metrics，终评读取selected缓存，不重复推理或运行未训练融合。保留源目录及分数缓存；这不是恢复033/020。用户已批准[035](../../HandOver/L2S-20261020-035.md)一次1000步及完整V评价，可与034并行，按当前索引READY领取，详见[规格](../../docs/specs/tmtaff-tam-phase.md)。

原生时间模型可显式配置`model.heading_representation=circular_sincos`；缺省`linear_zscore`兼容旧权重。新模式用保存的T_core均值/尺度还原航向并转sin/cos，内部投影为5维，仍用PE4加零列；训练、评价与快照保持同一变换并核对normalizer。跨模式不允许恢复/转移旧权重，需从头训练。新旧特征MSE不可直接比，下游关联另行验证；[规格](../../docs/specs/tcm-circular-heading.md)与[037](../../HandOver/L2S-20261020-037.md)列明已反馈批准的一次TCM4000及完整V命令，无自动TAM。
