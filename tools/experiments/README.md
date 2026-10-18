# 当前实验入口

这些脚本提供接口，**不是服务器执行授权**。服务器仅领取[交接索引](../../HandOver/Handover.md)指向的独立READY，按其中数据、配置、累计预算与重试次数执行；push不会启动任务。

| 入口 | 用途与边界 |
| --- | --- |
| `tmtaff_complete_fit.py` | TMTAFF时间适配近似参考：原生输入、四阶段训练、保存/恢复、完整关联及物理融合评价。正式训练需另有明确配置和预算。 |
| `tmtaff_complete_evaluate.py` | 仅加载已有选中检查点完成评价；不训练、不重新拟合normalizer、不重校准margin、不重选阈值。014已授权单次最多1GPU小时，实际状态见当前索引。 |
| `cooperative_fit.py` | 共享补全/融合候选；首轮固定身份比较普通、仅身份、身份×效用和普通可学习门控四臂。已有合成检查，尚无真实MTAD训练授权或效果。 |
| `tsactfer_fit.py` | TSA-cTFER独立离线关联适配，不含原在线两阶段指派，也不是完整补全/融合系统。代码保留，无新增训练任务。 |

在仓库根使用`python tools/experiments/<入口>.py --help`查看必需参数；正式数据运行采用交接中的完整命令、显式设备和新输出目录。CPU/MPS合成核验不等于真实数据训练，也不能替代CUDA成本测量。

参考和候选的`fit`可显式加`--save-predictions`，逐场景保存实际参与物理指标的预测/参考/有效掩码与分组。默认关闭，014仅评价入口不受影响。候选`progress.json`和`run.json`另记录准备、teacher、utility、身份评分及各臂适配/评价阶段，失败保留最后进度和原错误。阶段墙时与原history中的纯更新耗时分别解释；进程启动和CLI末尾准备报告仍由交接的外层硬时限计入总预算。

完整结果可用`tools/analysis/compare_physical_predictions.py`离线比较：同时报告各自全人口覆盖和共同有效点误差，不重新匹配目标或调整阈值；中断/不完整人口拒绝生成完整比较。配置、调用例子和成本限制见[首轮完整任务提案](../../docs/proposals/first_full_comparison/README.md)。该目录为待确认提案，不是READY。

CDF1活动实现及`cdf1_validation.py`已退役；共享CLI的`run-pilot`与CDF `smoke`也已移除。原结果、分数缓存和只读复算保留，详见[退役决策](../../docs/decisions/retire_cdf1_and_focus.md)和[012复核](../../docs/results/evidence012_local_review/README.md)。不重新领取旧011任务。

数据划分及T_test封存规则保持。当前重点是[完整参考与候选的两张比较表](../../docs/research_brief.md)，既有TSA/关联旧分数不与新原生时间结果直接混排。配置、数据范围、实际完成状态和结果内容用于核验，不设SHA/hash/fingerprint准入门槛。
