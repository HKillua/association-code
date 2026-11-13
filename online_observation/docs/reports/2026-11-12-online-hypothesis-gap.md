# 短错段间隙里的新报文能否区分真假持续

## Record

`EXPERIMENT_RECORD` v1.0，`REC-ONLINE-HYPOTHESIS-GAP-20261112`，completed。来源 `IDEA-ONLINE-HYPOTHESIS-GAP-20261112` plan v1.0、C-ON37。问题是原 KF 撤边而区间臂 I22 持续的当刻及随后，已到新报文能否可靠区别错配延续和正确持续。同 MTAD 来源的 MH-T2TA 以整条输入作互斥候选和分数，Draves 等按新量测更新源轨组合并以期末最佳评价，IET RSN 2026 的 δ-GLMB 单独比较 N-scan 深度并每扫描发布。现实数据需要分别核新证据、假设保留和发布时钟。 [原文证据](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_hypothesis_gap_sources_20261112.md) · [事前设计](../proposals/online_hypothesis_gap_20261112.md)

原 MTAD Level1 T_core 的 28 开发场／28 共享 A 组件，其中 24 场稀疏。C-ON28 的原候选逐事件读取，C-ON36 的 KF 和 I22 发布保持固定。100% 对账两份快照的原生时间和键、KF 发布、配准及候选边事件计数。102,404 原生事件、102,485 报告、940,693 候选边事件。原 48 场只用于上一轮，本轮不读取其报文和 A-B。冻结方案只比较两组均有两条边的等量动作。原生风险窗在 KF 撤下共同边、I22 经四端换配继续发布时触发，至任一组改变、候选失效、四端失鲜或最多 60 秒。之后是否重接或属于长错段由评分确定。两条焦点边在同一次换配中只计一组。

批前对照在当前时刻用四端各自的 KF 预测。新报对照只把当批报文实际位置代入同一组，比较两组平方距离总差。正号支持原 KF 撤下焦点边，负号支持 I22 保留。分别统计原缓存差、自然四端无新报及组内两种方案的真边数。整组真边数相同的对照保留为平局，不计方向准确率。两种读出都不发布新配对。后验补查只读已经冻结的风险窗，取严格晚于起点、早于退场的首个合格新报及首个来自起点未报来源的报文。两者采用相同的时间和候选权限。

```sh
.venv/bin/python -m online_observation.hypothesis_gap_diagnostic \
  --archive PART1 PART2 PART3 --manifest configs/data/mtad-level1-scene3000.json \
  --graph-run PRIVATE_C28_OUTPUT --interval-run PRIVATE_C36_DEVELOPMENT_OUTPUT \
  --output-dir NEW_PRIVATE_DIRECTORY
```

Apple M2、Python 3.12.14，CPU 线程上限 2。原冻结提交 `f910865` 于北京时间 18:41:45–18:41:55 完成 28 份无标签捕获，汇总写入因 `asdict` 生成 `Counter` 元组键退出 1，进程墙时 9.75 秒。保留原快照、部分导出与日志。修复 `e65e4a5` 用同入口 `--resume-frozen` 替代 `--graph-run`，18:44:34–18:44:39 从原冻结快照重新导出，真实退出 0，进程墙时 4.34 秒，程序 3.493 秒，最大 RSS 245,661,696 B。恢复汇总可复核已保存的合格窗、原生计数和状态，首次运行的场级起点过滤计数未单独保存。后验 `46efbbb` 于 18:49:07–18:49:14 从同 28 场冻结窗中提取晚到证据，仅重读其中有严格晚报的 11 场量测，真实退出 0，墙时 6.95 秒，程序 5.878 秒，最大 RSS 241,729,536 B。三项均零训练、零选型、零新场、V、封存测试或位置参考。100 项在线测试通过。 [原执行日志](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/hypothesis_gap_attempt_1.log) · [恢复日志](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/hypothesis_gap_scoring_repair_1.log) · [后验日志](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/hypothesis_gap_later_attempt_1.log)

后验入口为 `python -m online_observation.hypothesis_gap_later`，沿用 `--archive`、`--manifest` 和原 `--graph-run`，以 `--gap-run` 指向原捕获目录，以新目录接收结果。每项在全部场的无标签对照冻结后才读 A-B。 [恢复逐场结果](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/hypothesis_gap_scoring_repair_1/summary.json) · [严格晚报逐场结果](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/hypothesis_gap_later_attempt_1/summary.json)

## Squeeze

`SQUEEZE_REPORT` v1.0，`SQ-ONLINE-HYPOTHESIS-GAP-20261112`。得到 482 个四端组／932 个焦点边窗，分布 13 个共享 A 组件。稀疏场 236 组／462 焦点窗，涉及 10 组件。全体 92 组撤边方案真边更多、384 组保留方案真边更多、6 组真数相同。焦点全错、全对、混合分别 99／352／31 组。432 组在原 KF 接回时结束，36 组 I22 撤下、10 组原图失资格、2 组其它指派、1 组失鲜、1 组到窗口上限。

| 口径 | 有报告组／方向可评分组 | 屏蔽／实报选对 | 变对／变错 | 稀疏场有报告组／变对／变错 |
|---|---:|---:|---:|---:|
| 首个合格涉事报文 | 479／473 | 346／225 | 43／164 | 235／17／68 |
| 严格晚于撤边的首报 | 259／254 | 163／137 | 30／56 | 121／13／26 |
| 严格晚于撤边且来自起点未报来源 | 193／189 | 120／95 | 14／39 | 92／7／20 |

**错误延续与正确持续。** 479 个首个合格新报中 477 个就在 KF 撤边当刻，全为单源报文。新报增量支持撤边的比例，在应撤边组为 73／92，在应保留组反而为 329／381。首报新旧差值的方向 AUC 为 0.378，屏蔽与实际分差的 AUC 为 0.684 和 0.557。对两类等权，准确率从 0.578 降到 0.559，稀疏场从 0.604 降到 0.548。首报净变对 −121 组，逐一剔除共享 A 组件仍为 −122 至 −81，稀疏为 −52 至 −33。即使限制在上一步确已完整发布保留组、四边前后都合法的 450 组，仍是 40 变对／159 变错。条件人口是原 KF 正在撤边的竞争组，局部结果不外推到全部报文。

**自然无新报对照。** 404 组观察到四端无涉事新报的事件，80 组在同一连通分量有外部端点报告。实际和屏蔽差值逐项恒等，稀疏对应 176／31。完整风险窗都无涉事新报的只有 3 组，均为正确焦点持续且撤边时同竞争分量有外部更新。无新报对照界定本批输入，样本量不足以估计长期无报下的错误率。

**等待带来的证据。** 严格后到首报的中位到达时间 2.526 秒，稀疏 2.173 秒。首个来自起点未报来源的报文中位 2.620 秒，稀疏 2.194 秒。严格后到首报的净变对为 −26 组，组件剔一仍为 −28 至 −17；稀疏为 −13 组，剔一仍为 −15 至 −8。来自另一来源的条件也未改善，稀疏净 −13，剔一 −14 至 −7。后到首报加入实测后，两类等权准确率从 0.590 降至 0.568，稀疏从 0.626 降至 0.582。不同来源报告可能携带共同误差，源换侧不等独立见证。 [首个新报合计](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/hypothesis_gap_scoring_repair_1/aggregate.json) · [严格后到合计](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/hypothesis_gap_later_attempt_1/aggregate.json)

**稀疏长错链能触及多少。** 仅与原 KF 短错段重叠的 I22 长段，共包含 117 个可观测间隙，939.926 错对秒，涉及 4 组件。稀疏子集 60 个间隙、407.874 对秒，来自 2 组件，全部在起点有涉事报文，随后原 KF 在间隙结束重新接回该边。严格后到首报覆盖稀疏 42／60 间隙，支持撤下的分差增量有 28 个，实际由保留翻到撤下有 12 个，信号时点至原间隙结束剩余 43.204 对秒。另一来源覆盖 20 个，6 个翻转，剩余 53.400 对秒。两个口径可选中不同间隙，剩余时间不能相加或折算成真实缩短的长链。相反方向的正确持续对照均有较多误撤。当前实验无新的发布策略、首报或确认延迟估计。

**去留。** 几何新报有覆盖，但未满足事前约定的真假区分和净正确性条件。保留原 KF 与 I22 为对照，本轮收束在机制诊断，尚未冻结新留场或实施多假设策略。下一条证据应能区别真正换配与同航噪声引起的短时翻转，再检验分支记忆是否带来等时增益。若继续从原 28 组件隔离，未用池有 1,633 场／25 组件，每组件每规模层取一场最多 40 场。把已有 76 场全作为开发会耗尽组件隔离新场。 [Mode C 回流](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v37-新报的区分度和长错段覆盖)
