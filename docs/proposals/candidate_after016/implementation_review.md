# 018时间特征缩放：独立实现复核

2026-10-18，独立只读审查`association/cooperative.py`、`association/cooperative_training.py`生产diff、规格修订、调用链及L2S-018草稿。未修改生产代码或017活动文件，未使用GPU、真实MTAD训练或原ZIP扫描。

## 结论

未发现需要阻止交付的正确性缺陷。核验范围内：

- `fit_support_time_scaling`只读取已装配的T_core FP32有效native tau与全部query tau，排除padding；不读取四属性、reference或标签。只在主fit的训练准备阶段调用一次，不在验证、endpoint mask视图或各臂重新拟合。
- 四个MLP的手工时间坐标先除S再组差/乘积；count/log-count保持。全局encoded times、queries、连续时间编码、支持判定及plain物理解码未修改；速度输出、loss与数据采样未修改。
- 冻结teacher与四臂由模型复制继承同一非训练S。run记录统计范围，检查点同时保存策略、元数据S和持久scalar buffer。
- 新策略要求显式有限S≥1；loader用声明值构造模型并检查state中的buffer一致，缺失或矛盾显式失败。旧策略默认legacy、S=1且不增加持久state键，历史缺字段检查点保留原行为。
- 生产diff仅涉及候选两个模块。014/017的PairFusion、TCM/TAM、配置与训练/评价入口未改。

## 实际执行的旧版等价实验

通过`git show HEAD:association/cooperative.py`读取未修改旧源码，动态装入独立模块，与工作区新legacy实现比较。审查时HEAD为`81b56cc4f3ef0e36450466a093e8044c1284ddf0`；仅作为Git历史记录，不是执行准入门槛。

- CPU、1线程、同一seed=50；小型合成原生场景，length4、cap4、d8、2头、1层、FF16、head8、dropout0。
- 比较原版与新legacy的全部state_dict键、参数/缓冲初值。
- 分别检查plain、identity、utility、support、learned五种routing前向，以及同一两对候选的身份分数。
- 对每种routing的`physical_output.square().mean() + identity_scores.mean()`反向，检查每个参数的梯度或None状态。
- **上述全部比较通过，rtol=0、atol=0；没有发现初始化、随机消耗、前向或梯度变化。**

本实验补充对历史实现的独立行为比较，不替代新增策略的训练侧拟合、checkpoint拒错、真实入口合成流程与全仓回归测试。那些测试由实现/测试代理分别记录；本文件不把未亲自执行的测试计入自身验证。CPU结果不证明CUDA数值完全一致、实际耗时或MTAD性能改善。

## L2S-018草稿命令复核

草稿仍为DRAFT_IMPLEMENTATION_PENDING，不能领取；须由根代理在验证完成后发布独立READY。

命令将主仓库路径捕获为task_repo，在同级独立`association-code-run018`创建detached worktree；执行入口、018配置与alignment都指向该worktree，manifest与独立018输出指向主仓库。没有进入或改动`association-code-run017`。动态选卡、CPU4、单GPU、一次无重试，允许与017并行且不要求其结束。

配置明确teacher100、utility100、四臂各adapt100，共600更新，内部7140秒，外层7170秒TERM加30秒清理；建立worktree、选卡及正式fit均在外层限时内。命令开启已有预测保存，无额外模型探测、prepare、重试或resume。独立输出与证据目录均拒绝覆盖已有尝试。按当前草稿未发现踩017、错用两步配置或追加训练预算的问题。

本审查不等于发布READY、服务器领取或实验已开始；600更新与新时间尺度一起改变，不能仅凭下一轮优于015就将全部改善归因于缩放。
