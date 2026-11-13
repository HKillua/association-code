# MTAD 接回触发报同刻预测对照

2026-11-14，READY。原 28 场、原候选与接回批，一次本地 CPU，线程上限 2。C28 为资料库 `20261112_online_mtad/competing_reports_attempt_1`，C54 为 `20261114_online_mtad/rejoin_next_report_attempt_1`。原分卷、C37／C42／C45／C51 沿 [C54 输入](L2S-20261114-online-rejoin-next-report.md)。逐场冻结候选复用、形状复用、四路径和动作，再连接 C54 匿名标签。 [冻结方案](../docs/proposals/online_rejoin_boundary_report_20261114.md)

```sh
.venv/bin/python -m online_observation.rejoin_boundary_diagnostic \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --graph-run PRIVATE_C28_OUTPUT --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT \
  --shape-run PRIVATE_C45_OUTPUT --async-run PRIVATE_C51_OUTPUT --next-run PRIVATE_C54_OUTPUT \
  --output-dir NEW_PRIVATE_DIRECTORY
```
