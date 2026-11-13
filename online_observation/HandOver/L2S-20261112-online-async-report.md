# 在线异步报告与预测不确定性诊断

2026-11-12，READY。本地一次CPU，零训练与选型。原28场按C28原候选和KF、C36/I22发布、C37风险窗、C39共时诊断复核。所有报文评分在读标签前冻结。严格原接回前与接回批分开，不改变发布。 [冻结计划](../docs/proposals/online_async_report_20261112.md)

```sh
.venv/bin/python -m online_observation.async_report_gap \
  --archive PART1 PART2 PART3 --manifest configs/data/mtad-level1-scene3000.json \
  --graph-run PRIVATE_C28_OUTPUT --gap-run PRIVATE_C37_CAPTURE \
  --timing-run PRIVATE_C39_CAPTURE --output-dir NEW_PRIVATE_DIRECTORY
```

以原56个稀疏焦点间隙和同形正确接回为主，保留全人口。核同刻实报与屏蔽、共同或候选方差、继续保留两种分支、配准敏感性与原生时距代理校准。异常保全，新场不在本合同内，后续由证据判据决定。
