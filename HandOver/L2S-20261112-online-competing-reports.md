# 在线原有竞争边的后续报告核验

2026-11-12，READY，仅当前本地CPU会话一次。 [冻结Idea](../docs/proposals/online_competing_report_diagnostic_20261112.md)

按原28个Level1 T_core场回放KF。全部无标签轨迹保存后打开该批A-B。0训练、0参考、新验证及封存测试。原4 km、60秒、前缀配准与空配不变。输出原生段、起点候选和后续报告的相对排名/原决策及1秒配对核账。60／240秒为诊断统计窗口。新私有目录留原件，匿名汇总和运行记录归档。

```sh
.venv/bin/python -m online_observation.competing_report_diagnostic \
  --archive PART1 PART2 PART3 \
  --manifest configs/data/mtad-level1-scene3000.json \
  --output-dir NEW_PRIVATE_DIRECTORY
```

CPU两线程上限，依据原名单。一次完成或技术失败保全；不追加训练、留场或服务器作业。完成核账后关闭本合同并进入Record、Squeeze和Mode C回流。
