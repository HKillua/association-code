# MTAD 当批几何增量归因诊断

2026-11-13，READY。只读原28场 C37、C42、C45、C47 冻结件，先保存尺度、时距和动作，后连既有匿名评价。零训练、新场或外部数据。 [事前计划](../docs/proposals/online_rejoin_geometry_20261113.md)

```sh
.venv/bin/python -m online_observation.rejoin_geometry_diagnostic \
  --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT \
  --shape-run PRIVATE_C45_OUTPUT --cache-run PRIVATE_C47_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```
