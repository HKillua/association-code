# 原生数据审计工具

`audit_mtad_level2.py`保留用于读取官方ZIP目录和字段，已有原始审计与回执可追溯。旧环境探测脚本已经删除，日常实验不再经过一套环境验收流程。

数据和576场景装配审计已完成，不为新实验重复扫描。仅有新的具体数据故障时使用该工具；其历史哈希输出不是准入门槛。模型仍保留有效位置但vel/cou缺失的轨迹，与此工具的严格字段报告口径不同。

工具的15项合成ZIP检查可用 `python -m unittest discover -s tools/pilot -p 'test_audit_mtad_level2.py'` 单独运行。它们不读取真实数据、不构成实验效果。当前数据约定见[数据契约](../../docs/data_contract.md)。
