# 在线跨源残差与历史参照诊断

2026-11-12，READY。本地一次 CPU，原28场，零训练、选型或新场。重放 C28、C37，逐批对账 C40 原实报及同刻屏蔽。用更早图资格生成历史参照，所有有符号残差、参照与分数落盘后才读评价身份。 [冻结计划](../docs/proposals/online_cross_source_calibration_20261112.md)

```sh
.venv/bin/python -m online_observation.cross_source_calibration_gap \
  --archive PART1 PART2 PART3 --manifest configs/data/mtad-level1-scene3000.json \
  --graph-run PRIVATE_C28_OUTPUT --gap-run PRIVATE_C37_CAPTURE \
  --async-run PRIVATE_C40_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```

主分析为20个接回批换源的稀疏错接组、同形正确接回及同组件。先分均值和中心化离散，再核严格提前的锚覆盖、留出报文、原报、屏蔽和保留。技术异常保全。新场与同延迟发布另由证据决定。
