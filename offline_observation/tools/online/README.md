# 历史前缀计时入口

本目录保留离线方法在量测前缀上全量重算的历史计时合同和命令。请从离线观测工作根目录执行。对应报告见[11月5日整链计时](../../docs/reports/2026-11-05-pipeline-online-timing.md)。

```bash
python -m tools.online pipeline-timing --help
```

- [pipeline_timing.py](pipeline_timing.py) 保留原 Q64 历史输出和计时合同。旧脚本 `tools/analysis/deployment_pipeline_timing.py` 和模块命令 `python -m tools.analysis.deployment_pipeline_timing` 继续指向同一实现。
