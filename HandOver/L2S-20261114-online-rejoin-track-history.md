# MTAD 接回前单轨历史诊断

2026-11-14，READY。原 28 场，原接回批，一次本地 CPU，线程上限 2。运行 [冻结方案](../docs/proposals/online_rejoin_track_history_20261114.md)。原分卷及 C37/C42 路径沿 [C51 输入](L2S-20261113-online-rejoin-async-innovation.md)。C51、C52 分别为资料库 `docs/evidence/20261113_online_mtad/rejoin_async_innovation_attempt_1`、`rejoin_common_innovation_attempt_1`。输出在新私有目录，逐场动作冻结后连接 C52 匿名真值。无新留场授权，直到满足本次冻结条件。

```sh
.venv/bin/python -m online_observation.rejoin_track_history_diagnostic \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT \
  --async-run PRIVATE_C51_OUTPUT --common-run PRIVATE_C52_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```
