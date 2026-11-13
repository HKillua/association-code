# 原图区间指派的开发和新留场

2026-11-12，READY。本地CPU执行，开发及留场各一次，按[冻结Idea](../docs/proposals/online_interval_assignment_20261112.md) v1.0。28开发场／28组件，48新留场／31组件，当前开发与留场的共享A组件不相交。代码入口和[名单](../online_observation/interval_assignment_roster.json)冻结。固定原候选和空配，比较KF、等量右端点、等量整段、留空右端点、留空整段。0训练、0选型、0 V/封存测试/参考，按原生报文时刻输出。每阶段全部输出冻结后读相应A-B。首报伙伴变化、候选越界、旧开发基线不一致或其它技术异常即保全并停止。

```sh
.venv/bin/python -m online_observation.interval_assignment_experiment \
  --archive PART1 PART2 PART3 \
  --manifest configs/data/mtad-level1-scene3000.json \
  --phase development --output-dir NEW_PRIVATE_DEVELOPMENT_DIRECTORY
```

开发的原KF逐场配对及长段与上一轮对账后，用同一冻结入口替换为`--phase held --output-dir NEW_PRIVATE_HELD_DIRECTORY`。每阶段CPU线程上限2，留存进程退出、成本及匿名结果。独立完成Record、Squeeze与Mode C回流后关闭合同。服务器无需领取。
