# 在线双报端点和预测复用诊断

2026-11-13，READY。本地一次CPU，只读原28场C42冻结报文和既有匿名评价。逐批先保存关系和分差，再连接固定评价。零训练、选型或新场。 [事前计划](../docs/proposals/online_disjoint_report_closure_20261113.md)

```sh
.venv/bin/python -m online_observation.disjoint_report_closure_gap \
  --manifest configs/data/mtad-level1-scene3000.json \
  --anchor-run PRIVATE_C42_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```

接回前与当批、端点不相交与互指复用分开。新留场和同延迟发布由冻结判据决定。
