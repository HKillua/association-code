# 超龄历史的首报关联资格

## Record

`EXPERIMENT_RECORD` v1.0，`REC-ONLINE-STALE-HISTORY-20261111`，2026-11-11，completed，partial。来源 `IDEA-ONLINE-STALE-HISTORY-20261111` v1、事前提交 `7f16ad3`。 [冻结计划](../proposals/online_stale_history_qualification_20261111.md) · [聚合原件](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261111_online_mtad/stale_history_qualification.json)

原三分卷的 T_core 0/3/7，固定 600 秒无标签校正、全部 279 个晚到出生、4 km 门及并列拒配。原 60 秒正／错／拒为 198/59/22，逐场对上 E-ON5。E-ON35 的超龄／未见真片段逐场为 1/5、13/23、0/0。各场预测和分数固定后才读 A-B。Apple M2，Python 3.13.15，标准库 CPU，真实一次退出 0，3.163 秒，聚合 199839 字节。无训练、V／测试／参考读取，无排除、重跑或参数变更。现有 15 项在线测试及独立合成检查通过。输入分卷见[数据约定](../../configs/data/README.md#数据文件与复现)。

```sh
python -S -m online_observation.stale_history_probe \
  --archive PART1 PART2 PART3 \
  --manifest configs/data/mtad-level1-scene3000.json \
  --output /Users/a1/Projects/biye-ronghe/docs/evidence/20261111_online_mtad/stale_history_qualification.json
```

|历史与位置|14 中有合格真键|该真键进门|胜原错误首选上界|实际救回|全体正／错／拒|原正确改错|候选对数|
|---|---:|---:|---:|---:|---|---:|---:|
|60 秒保持|0|0|0|0|198／59／22|0|30497|
|120 秒保持|2|2|2|2|201／58／20|1|31049|
|300 秒保持|8|8|7|7|205／54／20|2|32745|
|600 秒保持，主臂|13|13|11|11|211／51／17|3|35469|
|截至 t0 全历史保持|14|14|11|10|202／64／13|10|59259|
|600 秒，旧候选前推 60 秒|13|12|8|8|209／53／17|2|35469|

主臂的 11 个救回分场为 1/10/0。另把 5 个原拒配改对，3 个原正确改错，净正确 +13、错配 −8、拒配 −5。超过正确改错至多 1 的事前限制，状态 `not_qualified`。逐场、过龄候选数和全部转移见原件。比例为有限开发场描述，无独立出生假设。下游 squeeze-experiment。

## Squeeze

`SQUEEZE_REPORT` v1.0，`SQ-ONLINE-STALE-HISTORY-20261111`，来源 `REC-ONLINE-STALE-HISTORY-20261111`，partial。

- **历史可达性确有增量。** 600 秒内 13 个真旧键能进同一个 4 km 门，其中 11 个比原错误首选近，也都最终胜出。最年轻真旧报文在全 14 中为 76.304–648.131 秒，中位 246.689 秒。600 秒漏掉 1 个年龄超限的出生，另 2 个虽进门却输给原错边。不能靠盲目延长寿命消除排序歧义。
- **扩池风险不能被净收益盖掉。** 600 秒多 4972 对候选，增加的正确超过引入的错误，仍有 3 个正确首报改错。全历史多 28762 对，有 1 个原可救回的真边被新增假边截获，实际救回 11→10，原正确改错 3→10，另有 5 个原拒配变错。28 个从未见真片段的错配各臂都未救回。年龄较短的次要工作点用于报告风险曲线，未替代主臂。
- **外推未解释主要损失。** 在相同 35469 个候选对中，4972 个超龄对使用两点前推。目标组真边进门 13→12，救回 11→8；全体有 3 个错改对、5 个对改错，净少 2 正。结论限于这项 60 秒前推，曲率、各源误差和实际静止状态未知。

回流 F-ON39/40、E-ON38/39。C-ON18 支持旧身份候选确有可恢复性，600 秒直接最近覆盖原首报未达安全判据。后续可比较保留原首报的旧边独立资格与小规模冲突约束，同时保持状态新鲜度。先进入用户指定的 28 个无对侧历史的拒配检验。 [Mode C 回流](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v10-超龄历史实验回流)
