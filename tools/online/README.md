# 历史在线计时入口与命令兼容

当前在线观测在 [online_observation](../../online_observation/README.md) 独立开发。本目录保留原计时合同的入口。历史计时仍运行当时的离线方法在量测前缀上的全量重算，对应报告见[11月5日整链计时](../../docs/reports/2026-11-05-pipeline-online-timing.md)。

```bash
python -m tools.online pipeline-timing --help
python -m tools.online causal-replay --help
```

- [pipeline_timing.py](pipeline_timing.py) 保留原 Q64 历史输出和计时合同。旧脚本 `tools/analysis/deployment_pipeline_timing.py` 和模块命令 `python -m tools.analysis.deployment_pipeline_timing` 继续指向同一实现。
- `python -m tools.online causal-replay` 转发到独立在线目录，保留参数和 JSONL 字段。
