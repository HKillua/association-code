# 在线单锚跨目标迁移诊断

2026-11-13，READY。本地一次CPU。只读原28场C41冻结图与报文，零训练、选型、新场。先冻结跨目标单锚的评分和弃权，再用 A-B 核真实独立性、正确接回和锚对去重。 [事前计划](../docs/proposals/online_single_anchor_transfer_20261113.md)

```sh
.venv/bin/python -m online_observation.single_anchor_transfer_gap \
  --archive PART1 PART2 PART3 --manifest configs/data/mtad-level1-scene3000.json \
  --calibration-run PRIVATE_C41_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
```

保留全部固定窗，查两候选代价各自改变量。新留场及同延迟发布由证据判据决定，异常按原件保全。
