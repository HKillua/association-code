# MTAD 去缓存复用的对侧新报位移

2026-11-13，READY。原 28 场 MTAD 一轮 CPU，线程上限 2。按 [C49 分卷与上游资料位置](L2S-20261113-online-rejoin-witness.md) 读取原生量测和 C37、C42、C45，再用原 C49 冻结记录校准时钟与旧窗，最后连原匿名评价。 [事前设计](../docs/proposals/online_rejoin_fresh_displacement_20261113.md)

```sh
.venv/bin/python -m online_observation.rejoin_fresh_displacement_diagnostic \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT --shape-run PRIVATE_C45_OUTPUT \
  --witness-run PRIVATE_C49_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```

`PRIVATE_C49_OUTPUT` 为资料库 `docs/evidence/20261113_online_mtad/rejoin_witness_attempt_1`。

v1.1 一次修订轮用同命令和新输出目录。首轮在匿名连接退出 1，28 份已冻结文件与日志保全。硬删新报保留作覆盖，新增同括点遮蔽的批前等速坐标对照。 [修订依据](../docs/proposals/online_rejoin_fresh_displacement_20261113.md#v11-遮蔽修订)
