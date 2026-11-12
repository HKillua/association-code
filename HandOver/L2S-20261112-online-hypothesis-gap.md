# 在线短错段间隙诊断

2026-11-12，READY。本地一次 CPU 读取原 28 开发场的冻结候选与发布，因果读已到位置获取批前预测。保持原候选和发布，全部无标签风险组冻结后用 A-B 核真假和已有错链。零训练、零模型选择、零新场、参考、V 或封存测试。固定原始事件、片段键、原生发布及候选计数对账，发现异常则保全。 [冻结计划](../docs/proposals/online_hypothesis_gap_20261112.md)

```sh
.venv/bin/python -m online_observation.hypothesis_gap_diagnostic \
  --archive PART1 PART2 PART3 \
  --manifest configs/data/mtad-level1-scene3000.json \
  --graph-run PRIVATE_C28_OUTPUT --interval-run PRIVATE_C36_DEVELOPMENT_OUTPUT \
  --output-dir NEW_PRIVATE_DIRECTORY
```

用进程退出、完整汇总、Record、Squeeze 及 Mode C 回流收尾。新留场与多假设回放各自独立合同，仅在诊断满足证据条件时冻结。
