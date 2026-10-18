# 候选修正版：本地实现交付

2026-10-19。用户已批准本方案的本地实现及CPU验证，**已完成；没有启动GPU或真实MTAD训练，没有新READY**。方法规格仍以本目录README、representation、training为准，本页记录落实和验证。

## 实现行为

- `support_residual_v2` 与默认 `legacy` 分开。只用当前可见观测构造锚轨位置插值/端点底座，再学习残差；首尾留出后重新计算。没有使用reference、query伪状态或固定源优先级构造底座。
- T_core保留原生点一次拟合log1p速度统计；速度输出是直接softplus，不乘原生速度std。航向输入展开sin/cos，物理标签、原损失和合法query范围不变。
- 支撑查询进入补全decoder；局部身份头用双向排序摘要保持交换对称，非重叠仅末/首点，缺真实局部证据时回退全局身份分数；属性辅助摘要残差进入共享物理输出。
- A训练新公共模块，A末一次完整V确定身份阈值；B只训练效用头；C冻结共同编码/身份/query_pool/支撑查询/效用，四臂同初态、同采样、同身份分数。plain、scalar-u、attribute-u、learned-attribute均已接入训练和物理评价。
- 新检查点保存模型版本、训练侧统计与时间尺度，并直接核对持久buffer和声明统计。独立审查发现的物理mean/std加载缺口已修复并有回归测试；旧v1路径保留。
- 既有进度、失败原件、全人口评价及预测导出继续使用。没有另建训练框架，没有修改017执行文件、配置、原交接和历史产物。

## 验证结果及边界

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 全仓CPU unittest | 468项通过，退出0 | 旧流程与新实现的行为回归 |
| 原生数据审计测试 | 15项通过，退出0 | 原有数据契约检查仍通过；没有扫描真实576场景 |
| 新增针对性测试 | 24项，包含于468项中 | 遮挡隔离、圆周边界、速度零值/尾部、局部对称/回退、真实更新与冻结、四臂共享条件、检查点及CLI |
| 新版本合成训练CLI | A1+B1+四臂各1，共6更新；完整评价/预测导出 | 小规模CPU连通性，不是开发训练配置或MTAD效果 |
| 合成原生ZIP CLI | 四臂全流程，空输入但有reference场景仍保留分母 | 数据装配—训练—评价接口及覆盖语义 |
| 正式模型规模CPU前反向 | 2,190,735参数、Q64/cap128/d128/6层/8头，有限梯度 | 完整结构可运算；零优化器更新，不是耗时估计 |
| 旧main与新代码legacy对照 | 两种时间策略下225个初始化/编码/输出/梯度张量逐位相等 | 旧计算路径没有被新版模块改变 |
| 交叉独审 | 模型/表示、训练/评价、CLI分别复核；上述统计加载缺口已修 | 发现并修复确定实现问题，不证明科学假设成立 |

第一次全仓运行发现一处新增测试fixture写入了不合法原生坐标；修正fixture保留数据合法性后最终468项通过。没有放松生产数据校验或更改预期研究结果。另有独审的legacy训练对照：7条更新报告除耗时相同，348个checkpoint张量逐位相同。

小型证据：[核验汇总](verification.json)、[完整形状](cpu_shape_check.json)、[legacy逐值比较](legacy_equivalence.json)、[合成训练阶段](cpu_e2e_summary.json)。CPU设备路径已执行；CUDA设备选择/张量迁移路径已实现及静态复核，**本地无CUDA硬件，未进行CUDA实测**。不根据CPU秒数承诺服务器完成时间。四臂有额外效用预训练，不称总优化预算相等。

## 本地可复核命令

在仓库根目录运行，输出必须是新的目录。以下只构造合成数据：

```sh
python tools/experiments/cooperative_fit.py check --model-variant support_residual_v2 --device cpu --cpu-threads 1 --save-predictions --output /tmp/cooperative-v2-check-new
python tools/experiments/cooperative_fit.py shape-check --model-variant support_residual_v2 --device cpu --cpu-threads 4 --output /tmp/cooperative-v2-shape-new
python -m unittest discover -s tests
python -m unittest discover -s tools/pilot -p test_audit_mtad_level2.py
```

## 输入支撑评价收尾

用户随后要求立即完成代码。01f706b已将实际输入支撑/距离分层接入baseline与候选正式评价、场景汇总和预测导出，未改训练任务/原指标；见[规格与验证](../../specs/input-support-evaluation.md)。491全仓＋15审计CPU通过、两路独立复核通过；包含新增11项及合成原生ZIP正式CLI。v2正式模型入口有界复核没有新的可复现实现bug，不为凑改动重写已完成组件。

## 剩余事项

本次授权的本地工作已完成。[完整配置草案](config.draft.json)给出teacher1000、utility200、四臂各300，共2400更新，`max_seconds=null`；CLI会在打开archive之前拒绝，不能凭本地代码已完成直接开训。新预算由用户另行决定，再生成独立READY和累计时限。

018阶段机械外推3.11 GPU小时，先前提出4小时作为待决定上限，**不是新结构实测成本，也不是已批准额度或完成保证**。目前不另开成本测量。真实比较要看三阶段任务、四臂机制增量、覆盖、速度/航向退化与费用；尚不能说新方法优于baseline。
