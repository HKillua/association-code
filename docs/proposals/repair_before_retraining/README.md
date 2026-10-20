# 当前结果与复算入口

这里保存仍支撑当前决策的分析，不维护任务队列；执行只看[交接索引](../../../HandOver/Handover.md)，目标和论文比较只看[研究说明](../../research_brief.md)。结案的旧训练安排、授权快照和重复汇报已合入[历史摘要](../../history/README.md)。

## 近期实验

| 分析 | 关键结论 | 当前用途 |
| --- | --- | --- |
| [030原量测与偏移](030-repair-analysis.md) | 观测信息未保住；真值配对下跨目标共享校准有潜力，端点局部运动不稳定 | 关联/融合设计依据，oracle不作算法成绩 |
| [031/032速度对照](031-032-analysis.md) | 速度MAE降低33.1%，航向和支撑内位置退化，学习动态近常量 | 保留速度参数化，不能验收补全 |
| [033增训TCM](033-analysis.md) | 4000步有收益，完整V仍670/1635片段差于保持 | 035固定补全端点、037同量表示对照 |
| [034航向底座](034-analysis.md) | 优于绝对预测，但仍差于观测控制；消去学习动态几乎不改误差 | 039—041共同对照，完整V预测保留 |
| [036短程反馈](036-analysis.md) | h2 −0.515%、h3 +4.310%，不足以确诊主要故障 | 避免盲目新增free-running训练 |
| [035独立TAM](035-analysis.md) | 最佳在600步、F1约.5129，仍低于位置控制；继续训练到1000步反退化 | 参考关联尚弱，分清训练量与表示作用 |
| [037圆周TCM](037-analysis.md) | 本空间query优于hold23.78%，新旧latent不直接排名 | 物理/关联收益待验证，同口径成本已纠正 |
| [039—041消融](039-041-analysis.md) | 动态幅度恢复未变成整体精度优势；两层较小，去BCE仅速度改善 | 不盲目叠加模块或宣布补全修好 |
| [038逐层动态](038-analysis.md) | 四场86轨每层时间变化幅度均下降，最终/输入缩小倍数中位1059 | 039—041干预依据，不能把幅度等同准确率 |
| [TCM综合问题](tcm-diagnosis.md) | 区分目标/时距、表示几何、短轨权重及候选动态 | 机制与解释边界，避免重复排查 |

## 复算与控制

同名`.py`和JSON保存主要分析的算法与数值；不修改模型或自动启动实验。030三个脚本和`association-distance-local-check.py`、`completion-base-local-check.py`需要本地Level2 ZIP，其余多数仅读已返回文件。034复算还读取原V量测；所有脚本仍使用原路径，当前输入副本保留范围见[回传说明](../../../HandOver/results/README.md)。007 manifest通过`git show 612afea:...`读取，勿将其误当现有待办。

- `031-032-analysis.py`：完整V同人口比较；`032-temporal-analysis.py`：消去已学动态的事后控制。
- `033-analysis.py`：020/029/033共同四场及完整V；`tcm-history-synthesis.py`、`tcm-mechanism-metadata.py`：训练和时距统计。
- `034-analysis.py`、`036-analysis.py`、`038-analysis.py`：各报告的复算。
- [位置距离关联控制](association-distance-local-check.json)、[原分数](association-distance-local-records.npz)：F1 .615627；021 F1 .630321仅高1.47个百分点，跨源无优势。020不是完整baseline，不据此宣布创新成功。
- [观测位置底座控制](completion-base-local-check.json)：保留来源/支撑分层，9001/reference复用限制降噪解释。

所有结果为单seed、已暴露开发人口，场景/片段相关，不声称独立测试或统计显著性。不重跑已完成诊断来补过程留痕；旧命令和审批原件从Git历史查阅。
