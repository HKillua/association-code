# 实验入口

在仓库根使用 `python tools/experiments/<脚本>.py --help`。真实实验按[当前交接](../../HandOver/Handover.md)中的独立READY运行，接口和历史配置本身不授权训练。

| 脚本 | 用途 |
| --- | --- |
| `tmtaff_complete_fit.py` | 完整参考的四阶段训练、恢复及关联/融合评价 |
| `tmtaff_complete_evaluate.py` | 加载选中检查点，仅评价，不重新训练/校准/选阈值 |
| `cooperative_fit.py` | 候选训练及plain/identity/utility/learned四臂比较；当前修正版用`support_residual_v2` |
| `tsactfer_fit.py` | TSA-cTFER离线关联适配，不含原在线指派，也不是完整融合系统 |
| `completion_check.py` | 复用020/021保全权重，固定4个V场景做独立补全诊断；无训练、关联或融合 |

`completion_check.py --candidate-residuals`复用同一入口，只推理既有teacher，每片段一次；参考checkpoint仅用于原数据契约核对。输出逐场景`predictions-*.npz`（预测、位置底座、原真值、逐属性mask及片段/时间/区域）、`residual-diagnostics.json`（最终学习权重、正则、分区域原属性损失）及`candidate-summary.json`。分区域loss为区域内均值，计算全局贡献时乘该属性的区域有效点数/全部有效点数；位置底座没有速度/航向预测。它不还原训练梯度、不生成额外裁剪或改写权重。

参考和候选可用 `--save-predictions` 保存实际预测、真值、mask及分组，供 `tools/analysis/compare_physical_predictions.py` 做全人口覆盖/共同有效点比较。进度和失败原件保留；不为新增报表重跑已结束实验。

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
