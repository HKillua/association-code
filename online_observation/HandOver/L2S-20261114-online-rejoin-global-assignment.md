# MTAD 原接回批全局指派复算

2026-11-14，READY。原 28 场、原候选、原接回批，线程上限 2，一次本地 CPU。C55 为资料库 `20261114_online_mtad/rejoin_boundary_report_attempt_1`，分卷及 C28／C37／C42 路径沿 [C55 输入](L2S-20261114-online-rejoin-boundary-report.md)。全部场证据和动作冻结后，连接匿名 C55 及原 A-B 评价重解边。 [冻结方案](../docs/proposals/online_rejoin_global_assignment_20261114.md)

```sh
.venv/bin/python -m online_observation.rejoin_global_diagnostic \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --graph-run PRIVATE_C28_OUTPUT --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT \
  --boundary-run PRIVATE_C55_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```
