# 双报端点覆盖、预测复用与错接预警

## Record

`EXPERIMENT_RECORD` v1.0，`REC-ONLINE-DISJOINT-REPORT-20261113`，completed。`IDEA-ONLINE-DISJOINT-REPORT-20261113`，C-ON43。核两批实报是否分别覆盖两条候选边、早报是否直接进入后报的对侧预测，以及相对残差能否减少正确接回的误拦。 [冻结计划](../proposals/online_disjoint_report_closure_20261113.md) · [纯传感器来源](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_disjoint_report_sources_20261113.md)

冻结提交 `8151123`，原28场，24场稀疏，482个固定窗、1,477个单报读出。只读 C42 快照，先写全部无标签分差，再连接既有匿名评价。零新量测、标签、训练、选型、新场、V或封存测试。Apple M2、Python 3.12.14，CPU线程上限2。北京时间16:46:14–16:46:16，一次真实退出0，进程墙时1.30秒、程序0.081秒、最大 RSS 214,515,712 B。126项在线测试通过。逐场复算1,945个双报关系、15,560项分支残差和遮蔽读出。 [日志](/Users/a1/Projects/biye-ronghe/docs/evidence/20261113_online_mtad/disjoint_report_closure_attempt_1.log) · [核验](/Users/a1/Projects/biye-ronghe/docs/evidence/20261113_online_mtad/disjoint_report_closure_attempt_1/verification.json)

```sh
.venv/bin/python -m online_observation.disjoint_report_closure_gap \
  --manifest configs/data/mtad-level1-scene3000.json \
  --anchor-run PRIVATE_C42_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
.venv/bin/python -m online_observation.disjoint_report_closure_squeeze \
  --summary NEW_PRIVATE_DIRECTORY/summary.json --output NEW_PRIVATE_DIRECTORY/aggregate.json
```

## Squeeze

`SQUEEZE_REPORT` v1.0，`SQ-ONLINE-DISJOINT-REPORT-20261113`。主人口28个稀疏错接窗、56焦点边、2个共享A组件，与142个同形正确接回比较，其中48个同组件。20／76个接回时换源另报。候选、发布时间和原生量测时钟保持。 [逐场读出](/Users/a1/Projects/biye-ronghe/docs/evidence/20261113_online_mtad/disjoint_report_closure_attempt_1/summary.json) · [聚合](/Users/a1/Projects/biye-ronghe/docs/evidence/20261113_online_mtad/disjoint_report_closure_attempt_1/aggregate.json)

**双报何时可用。** 原28个目标中，15个在接回前取得两分支均端点不交的双报，其中6个在接回当批仍有同类双报。6个首次到接回批才有，7个从未有。正确对照对应42个在严格接回前有、62个当批才有、38个没有；按最新时刻同刻比较，双报支持是15／37个。目标在取得两报时两次来报的间隔中位2.975秒（首个合格），正确对照4.571秒。原236个稀疏窗有82个严格提前支持、79个当批才有。28个目标有8个提前出现任一异源互指、24个在接回当批有，后者没有提前量。

**异源复用。** 全482窗内，按预定规则选出的618次严格提前互指对照及429次接回批互指对照，后报的互指预测均以该前报为最近一次更新。四端两种完整配对下，异源两报恰在一分支互指，同源不同片段可在两分支都端点不交，因此两类也受报告源影响，不能将跨类差异归因于复用本身。遮蔽只替换显式报文坐标，保留原批前预测。即使端点不交，两分支仍可能共享预测历史或潜在源误差。

### 接回前最后一条实报的同刻比较

| 读出 | 错接命中（28） | 正确接回误拦（142） | 同组件误拦（48） |
|---|---:|---:|---:|
| 原单报 | 9 | 87 | 29 |
| 原单报同刻屏蔽 | 10 | 65 | 31 |
| 上轮单锚一致性 | 12 | 31 | 10 |
| 端点不交，双实报相对残差 | 13 | 28 | 9 |
| 同一双报，遮蔽晚报 | 12 | 23 | 8 |
| 同一双报，遮蔽早报 | 9 | 15 | 5 |
| 同一双报，遮蔽两报 | 9 | 10 | 4 |
| 两次单报都支持阻止接回 | 0 | 4 | 0 |
| 端点不交且晚报增加正向支持 | 7 | 21 | 6 |
| 端点不交且两报各有正向增量 | 6 | 18 | 5 |
| 继续保留 I22 | 0 | 0 | 0 |

在同支持15／37组，双实报相对遮晚报新增1个命中、7个误拦，解除2个误拦。要求晚报增加支持后，同支持平衡正确率44.95%，低于遮晚报58.92%、两报均遮66.49%和单锚一致性59.82%。两次单报同向是另一个对照，条件过严且没有命中。主规则最新命中中位提前1.725秒，曾出现警报的口径为9／28与27／142、命中中位提前1.783秒。两个目标组件最新分别命中5／21、2／7，误拦6／41、0／7，后一个组件的7个正确对照均无同刻端点不交双报。仅换源子群为6／20命中、18／76误拦。接回当批主规则1／28、11／142，提前量为零。

当前两报差分有提前信号，晚报的同刻增量仍不足以保护正确接回。符合事前使用条件的全人口净效应为 −14，未启用隔离新场。同延迟发布、长错链实际缩短、原正确配对及首报留存尚未测。 [Mode C v49](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v49-双报供给与预测复用回流)
