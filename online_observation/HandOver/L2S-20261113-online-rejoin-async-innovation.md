# MTAD 原生时钟的异步接回增量

2026-11-13，READY。原 MTAD 28 场一轮本地 CPU，线程上限 2，先冻结断口血缘与各原生时钟动作，再连接匿名 C50。原始分卷和 C37/C42/C45 路径沿 [C49 输入](L2S-20261113-online-rejoin-witness.md)，C50 为资料库 `docs/evidence/20261113_online_mtad/rejoin_fresh_displacement_r1`。 [冻结设计](../docs/proposals/online_rejoin_async_innovation_20261113.md)

```sh
.venv/bin/python -m online_observation.rejoin_async_innovation_diagnostic \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT --shape-run PRIVATE_C45_OUTPUT \
  --fresh-run PRIVATE_C50_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```
