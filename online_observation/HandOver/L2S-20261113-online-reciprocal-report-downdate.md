# 在线异源互指单报去复用诊断

2026-11-13，READY。本地一次CPU，原28场固定候选和发布，按原生时钟重建局部状态，先冻结对照，再连接旧匿名评价。零训练、选型和新场。 [事前计划](../docs/proposals/online_reciprocal_report_downdate_20261113.md)

```sh
.venv/bin/python -m online_observation.reciprocal_report_downdate_gap \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT \
  --closure-run PRIVATE_C43_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
.venv/bin/python -m online_observation.reciprocal_report_downdate_squeeze \
  --summary NEW_PRIVATE_DIRECTORY/summary.json --output NEW_PRIVATE_DIRECTORY/squeeze.json
```

保留全人口和去复用同支持两种分母，严格提前与接回当批分列。新留场由事前判据决定。
