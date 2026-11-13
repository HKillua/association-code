# MTAD 接回缓存证据诊断

2026-11-13，READY。只读原28场 C37、C45、C46已冻结输入，一次本地 CPU，先保存动作和特征后连已有评价。零训练、选型、新场或外部数据。 [事前计划](../docs/proposals/online_rejoin_cache_20261113.md)

```sh
.venv/bin/python -m online_observation.rejoin_cache_diagnostic \
  --gap-run PRIVATE_C37_OUTPUT --shape-run PRIVATE_C45_OUTPUT \
  --rejoin-run PRIVATE_C46_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```
