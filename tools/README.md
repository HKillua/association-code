# 工具导航

|用途|入口|
|---|---|
|训练、数据准备、完整评价与一次性服务托管|[experiments](experiments/README.md)|
|冻结输出比较、科学pilot、诊断与独立核验|[analysis](analysis/README.md)|
|环境准备|[setup](setup/)|

共用模型/算法在 `association`，工具中也存在跨实验复用的编排和审计函数；不能因为原实验已结案就删除这些依赖。按[当前交接](../HandOver/Handover.md)定位具体入口和参数，不从目录或历史命令恢复队列。本轮整理保留所有Python模块原路径和数值实现。
