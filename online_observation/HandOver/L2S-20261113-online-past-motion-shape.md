# 在线已到历史运动变化诊断

2026-11-13，READY。固定原28场和原候选图，本地一次CPU，先冻结无标签位置和运动读出，再连接既有匿名评价。零训练、选型及新场。 [事前计划](../docs/proposals/online_past_motion_shape_20261113.md)

```sh
.venv/bin/python -m online_observation.past_motion_shape_gap \
  --manifest configs/data/mtad-level1-scene3000.json --archive ARCHIVE_PART_1 ARCHIVE_PART_2 ARCHIVE_PART_3 \
  --gap-run PRIVATE_C37_OUTPUT --anchor-run PRIVATE_C42_OUTPUT \
  --closure-run PRIVATE_C43_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
.venv/bin/python -m online_observation.past_motion_shape_squeeze \
  --summary NEW_PRIVATE_DIRECTORY/summary.json --output NEW_PRIVATE_DIRECTORY/squeeze.json
```

净位移与两段历史比较用同一报告和支撑，当前坐标独立屏蔽。下一阶段由冻结条件决定。
