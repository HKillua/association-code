# MTAD 接回当批的运动证据诊断

2026-11-13，READY。只读原 MTAD 28 场 C-ON45 保存的评分和匿名评价。先冻结各臂动作，再连接评价。零训练、新场或外部数据。 [事前计划](../docs/proposals/online_rejoin_batch_20261113.md)

```sh
.venv/bin/python -m online_observation.rejoin_batch_diagnostic \
  --shape-run PRIVATE_C45_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```
