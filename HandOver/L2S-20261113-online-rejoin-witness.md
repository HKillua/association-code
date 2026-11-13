# MTAD 接回的严格批前对侧见证

2026-11-13，READY。只用原28场 MTAD 量测及 C37／C42／C45／C48 冻结证据。一轮本地 CPU，先冻结支撑和动作，后连匿名标签。零训练、选型、新场、外部数据。 [事前计划](../docs/proposals/online_rejoin_witness_20261113.md)

```sh
.venv/bin/python -m online_observation.rejoin_witness_diagnostic \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT --shape-run PRIVATE_C45_OUTPUT \
  --geometry-frozen PRIVATE_C48_FROZEN --geometry-run PRIVATE_C48_EVALUATION --output-dir NEW_PRIVATE_DIRECTORY
```

实际输入按 [Level1 分卷登记](../configs/data/README.md)，线程上限 2。资料库 `docs/evidence/` 下，`PRIVATE_C37_OUTPUT` 是 `20261112_online_mtad/hypothesis_gap_attempt_1`。其余在 `20261113_online_mtad/`，`PRIVATE_C42_OUTPUT` 是 `single_anchor_transfer_attempt_1`，`PRIVATE_C45_OUTPUT` 是 `past_motion_shape_attempt_1`，`PRIVATE_C48_FROZEN` 是 `rejoin_geometry_attempt_1/frozen`，`PRIVATE_C48_EVALUATION` 是 `rejoin_geometry_r1`。 [终态](S2L-20261113-online-rejoin-witness.md)
