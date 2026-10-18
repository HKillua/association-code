# 历史关联pilot说明

CDF1关联pilot已经退役。当前CLI不再提供`run-pilot`或CDF `smoke`，专用`cdf1_validation.py`入口也已删除；不把另一种实验替换到旧命令名下。旧pilot曾组合关联参考、第三片段修正和测试集评价，它不是当前“补全—关联—融合”路线的执行入口。

原协议、旧命令和实验产物保留在Git历史与各轮次原始交接中。本页旧全文可从历史提交`7766b86:docs/pilot_execution.md`查看；此版本号仅用于查阅历史，不是执行准入条件，也不授权重跑旧实验。退役依据及保留范围见[CDF1决策](decisions/retire_cdf1_and_focus.md)，原缓存的独立复算见[012报告](results/evidence012_local_review/README.md)。

当前入口如下：

- 完整TMTAFF时间适配参考：`tools/experiments/tmtaff_complete_fit.py`。
- 已有选中检查点的独立评价：`tools/experiments/tmtaff_complete_evaluate.py`。
- 共享补全/融合候选：`tools/experiments/cooperative_fit.py`。
- 独立关联诊断TSA：`tools/experiments/tsactfer_fit.py`，不承担完整融合比较。
- 共享CLI仍保留`prepare/train-reference/profile/probe-training`供已有数据及关联路径使用，保留接口不意味着要重新prepare或训练旧模型。

详细接口见[实验工具](../tools/experiments/README.md)，实际执行只以[当前交接](../HandOver/Handover.md)及其独立READY为准。014只评价013短训检查点；新完整参考和候选训练尚待具体配置与预算。T_test封存，不继承旧pilot的训练规模、重试次数或测试集访问安排。设备与累计资源规则见[GPU说明](gpu_execution.md)。
