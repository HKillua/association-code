# 在线 MTAD 冻结配置

- `mtad-level1-scene3000.json` 保存已冻结的场景名单和共享 A 组件划分。
- `mtad-level1-alignment.json` 保存在线读取报文使用的场景时间变换。

两份配置随在线安装包分发。工程内的 `configs/data` 指向本目录，供已有命令定位。原始数据和参考轨分别由调用者按[数据约定](../../docs/data_contract.md)提供，评估参考轨只用于评分。
