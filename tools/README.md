# 工具导航

先看[数据到输出的流程与实现](../docs/specs/evidence-association-fusion.md#从输入到输出)，再按用途查工具；历史实验编号用于定位来源，不表示待运行任务。

|用途|入口|
|---|---|
|在线观测与工程回放|[online_observation](../online_observation/README.md)|
|原整链计时合同和旧命令|[online](online/README.md)|
|离线全场系统评价与同行融合训练/评价|[offline](offline/README.md)|
|单组后端计时与历史计时入口|[工程计时入口](analysis/README.md#工程计时)|
|其他训练、数据准备、一次性服务托管及兼容入口|[experiments](experiments/README.md)|
|冻结输出比较、科学pilot、诊断与独立核验|[analysis](analysis/README.md)|
|环境准备|[setup](setup/)|

共用模型/算法在 `association`，工具中也存在跨实验复用的编排和审计函数；不能因为原实验已结案就删除这些依赖。按[当前交接](../HandOver/Handover.md)定位具体入口和参数，不从目录或历史命令恢复队列。退役专用工具及单文件恢复方法见[历史入口](../docs/history/README.md)；保留组件的数值实现不随导航整理改变。
