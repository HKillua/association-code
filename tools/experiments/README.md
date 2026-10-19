# 实验入口

在仓库根使用 `python tools/experiments/<脚本>.py --help`。真实实验按[当前交接](../../HandOver/Handover.md)中的独立READY运行，接口和历史配置本身不授权训练。

| 脚本 | 用途 |
| --- | --- |
| `tmtaff_complete_fit.py` | 完整参考的四阶段训练、恢复及关联/融合评价 |
| `tmtaff_complete_evaluate.py` | 加载选中检查点，仅评价，不重新训练/校准/选阈值 |
| `cooperative_fit.py` | 候选训练及plain/identity/utility/learned四臂比较；当前修正版用`support_residual_v2` |
| `tsactfer_fit.py` | TSA-cTFER离线关联适配，不含原在线指派，也不是完整融合系统 |

参考和候选可用 `--save-predictions` 保存实际预测、真值、mask及分组，供 `tools/analysis/compare_physical_predictions.py` 做全人口覆盖/共同有效点比较。进度和失败原件保留；不为新增报表重跑已结束实验。

旧pilot/profile/成本probe入口已删除。017、018均已结束；候选修正版和共享/独立臂入口已完成。用户已同意按参考24h＋候选合计4h拆分，当前队列依赖与正式配置以[交接索引](../../HandOver/Handover.md)为准。研究方法见[候选规格](../../docs/proposals/candidate_revision_after018/README.md)，旧阶段见[历史摘要](../../docs/history/README.md)。


正式服务器训练从`launch_detached.py`的一次性systemd用户服务提交，已启动任务不依赖AI前台会话。示例接口：

```bash
python tools/experiments/launch_detached.py --cwd /冻结执行目录 --record-dir /新启动记录目录 \
  --max-seconds 已批准的累计剩余秒数 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES=GPU-实际UUID --env OMP_NUM_THREADS=4 \
  -- /既有环境/bin/python tools/experiments/tmtaff_complete_fit.py 完整已批参数
```

显式传递MKL/OPENBLAS线程等所需环境；服务不继承AI shell全部环境。payload命令是argv，不通过shell展开。`launch.json`的SUBMITTED不是模型成功；重连读取`status.json`/exit_code及模型run/summary。systemd失败unit保留供Result/ExecMainStatus取证。用户管理器须可跨断开存在（Linger=yes），`session`仅本地核验，不可静默作为抵抗runtime回收的替代。当前019只做CPU验证，新训练配置与预算见[修复后方案](../../docs/proposals/repair_before_retraining/README.md)。


候选v2可拆成共享阶段与独立融合臂：`fit --phase shared`先保存shared.pt、identity-records/与唯一阈值；随后`fit --phase arm --shared-from /已完成021输出 --route attribute-u --save-predictions`（其余臂替换route）。每臂有自己的输出与限时，只加载同normalizer/S/模型/loss/身份记录，不重训teacher/u或选阈值。默认full入口保持旧行为，shared不接受--save-predictions。正式配置和当前GPU队列以[交接索引](../../HandOver/Handover.md)为准，旧草案已从工作树删除。
