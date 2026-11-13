# MTAD 接回前后报与单次更新消融

2026-11-14，READY。原 28 场、原候选和原接回批。一次本地 CPU，线程上限 2。输入原分卷及 C37/C42/C45 路径沿 [C51 输入](L2S-20261113-online-rejoin-async-innovation.md)，C51 为 `20261113_online_mtad/rejoin_async_innovation_attempt_1`，C53 为 `20261114_online_mtad/rejoin_track_history_attempt_1`，均在资料库 `docs/evidence`。输出到新私有目录，所有场完成证据和动作冻结后连接匿名标签。 [冻结方案](../docs/proposals/online_rejoin_next_report_20261114.md)

```sh
.venv/bin/python -m online_observation.rejoin_next_report_diagnostic \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT --shape-run PRIVATE_C45_OUTPUT \
  --async-run PRIVATE_C51_OUTPUT --history-run PRIVATE_C53_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```
