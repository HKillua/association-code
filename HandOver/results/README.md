# 保留结果索引

只读证据，不能代替服务器模型输入。当前任务只见[交接](../Handover.md)，事实解释见[既有研究分析](../../docs/reports/2026-10-26-experiment-review.md)及[102/103回执核验与深挖](../../docs/reports/2026-10-27-returned-diagnostics-review.md)。

| 保留目录 | 目的 |
|---|---|
| `LEVEL1-COORDINATE-METRICS-20261026-101` | 当前102关联的既有坐标结果、mask与尾部证据 |
| `LEVEL1-PAPER-METRICS-20261026-102` | 300场完整论文维度及独立属性人口；原启动/退出记录齐全 |
| `LEVEL1-THREE-SCENE-DIAG-20261027-103` | 三场全部声明query、来源关系与确定性输入重放证据；私有坐标/端点留服务器 |
| `LEVEL1-THREE-CASE-CHAIN-20261027-104` | 三个固定案例各5个query的原链核查；服务器数值/日志与本地由脱敏回执补绘的SVG分别保留 |
| `LEVEL1-TAIL-DIAG-20261027` | 服务器事后派生件；`d6f79c8`改写其中方向/轴序/排名解释，原版本可从Git `a68fde4`恢复；以本地审计为结论入口 |
| `LEVEL1-NATIVE-QUERY-20261026-100-summary` | 三臂合并的全人口、逐场/组件对照；不再保三套重复端点表 |
| `LEVEL1-QUERY-RELATION-20261025-098-*` | 两seed比较与总表，保留原晋级失败的可核依据 |
| `LEVEL1-SEQUENCE-BUDGET-20261025-097-summary` | 两seed强预算控制的来源 |

已审093/094/096机器副本、100逐臂冗余返回、重复自动report移出工作区，原件在Git [`903f679`](https://github.com/HKillua/association-code/tree/903f679)，更早来源见[历史摘要](../../docs/history/README.md)。104原始回执未被本地改写；tail文件被服务器后续提交修改，其原版本和订正需分别查看，不将精简摘要冒称原始日志。

完整权重、成功batch日志与逐点NPZ仍在服务器原runs；原路径由各 `original-registration.json` 和旧Git回执保存。按需恢复单文件，不恢复整套旧READY，不新建archive副本。
