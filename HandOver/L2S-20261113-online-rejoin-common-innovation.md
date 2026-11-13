# MTAD 双报共同创新与同源留出

2026-11-13，READY。只用原 MTAD 28 场，一轮本地 CPU，线程上限 2。按 [C51 输入](L2S-20261113-online-rejoin-async-innovation.md) 定位原始分卷、C37、C42、C45，读取资料库 `docs/evidence/20261113_online_mtad/rejoin_async_innovation_attempt_1`。逐场证据与动作冻结后连接匿名结果及供体身份审计。 [冻结计划](../docs/proposals/online_rejoin_common_innovation_20261113.md)

```sh
.venv/bin/python -m online_observation.rejoin_common_innovation_diagnostic \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT --shape-run PRIVATE_C45_OUTPUT \
  --async-run PRIVATE_C51_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```
