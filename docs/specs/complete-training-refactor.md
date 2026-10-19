# 完整参考训练与恢复规格

2026-10-19整理。此处记录已经实现的TMTAFF时间适配近似参考行为，合并早期重构和逐次CPU过程日志。代码、数据规则与已批准运行不因整理改变；实际任务/预算见[交接索引](../../HandOver/Handover.md)，历史失败和成本见[历史摘要](../history/README.md)。软件验证不等于作者精确复现或研究效果。

## 输入与四阶段

- 原生ZIP读取复用冻结划分和清洗，保全部有效短轨真实输入，长轨按cap选点保首尾。reference仅供训练监督/开发评价，T_test拒绝；完整目标人口包含无有效节点/参考身份。
- [真实时间TCM](tmtaff-time-adaptation.md)接收原观测时刻和下一预测时刻；teacher与双向rollout使用一致的因果encoder/decoder memory可见性，含增长history的矩形mask。共同query支持内用物理插值、支持外递推，再进入TAM/TFM。
- 阶段为TCM→冻结TCM训练TAM→冻结关联训练TFM→联合训练。配置显式给每阶段更新数、微批、预算和验证间隔，不将更新数写成论文epoch或自动保证收敛。
- TCM从具备时间转移的训练轨迹抽样；TAM/joint沿既有scene→pair平衡分布；TFM均匀正场景→正对。真实场景/轨迹/pair覆盖记录，不按监督好坏删身份或重抽。微批保持整batch监督分母，局部无监督可贡献零，全batch缺必需监督明确失败。
- 基线原Transformer规模为6层、d128、8头、FF256、Q64，TFM fc_hidden128是明确工程适配。位置连续SmoothL1 beta=.001；TFM/微批/递归的具体数学见[融合规格](tmtaff-fusion.md)。017采用[位置输出affine](tmtaff-output-conditioning.md)，旧raw检查点保留兼容。

## 监督、选择与全人口评价

`build_scene_fusion_targets`按显式目标顺序保全部身份，每行query_times等于输入物理网格。每场景offset与最大参考gap显式给定，不能自动按效果搜索。当前已批准cap128/time_unit20/offset/gap600，审计证据见[数据契约](../data_contract.md)和[冻结输入/监督审计](https://github.com/HKillua/association-code/blob/612afea/docs/results/frozen_input_audit_20261018/README.md)。

TAM/joint按原完整V macro-F1选快照及阈值；TFM保留末步，不称“关联最优”就是“融合最优”。选中association可以来自TAM某次验证，而fusion含后续TFM末步；累计训练更新不能全称为选中组成。终评重载selected并核同阈值确定性复算，不偷偷增加模型选择。

完整评价保留全部目标、输出组、单例、重复/漏目标和错并拆分。预测分组不接真实身份，oracle分组独立诊断；baseline singleton只输出已有观测，候选可模型补全，必须披露差别。单源按source固定首有效片段，简单融合先各source再均值，不按reference挑最好片段；经度/航向圆均值，抵消处无效。

位置球面米误差、native速度、圆周航向与覆盖共同报告。共同有效点误差不能替代全人口，零交集为null。新增[实际输入支撑/距离诊断](input-support-evaluation.md)不改主指标、阈值或选模，不回填旧运行或触发重跑。

## 一次训练侧CL margin校准

`margin_policy=fixed`或缺省保历史数值。显式`train_negative_median_v1`仅用于CL、正项一次距离及T_core训练池；在TCM结束、TAM首次更新前执行一次。独立BalancedScenePairSampler seed21，固定64批×64对，保留2048正/2048负呈现及重复；不去重/补抽，不读V/T_test。

TCM冻结eval，TAM train/dropout；no_grad并复用与TAM_FROZEN相同的端点分块/缓存语义。保存/恢复CPU与所选CUDA Torch RNG及模块模式，不触碰正式轨迹/pair/TFM采样器或其他GPU。2048负flatten欧氏距离取NumPy中位数，偶数取中央均值；非有限、负距离、数量/类别错误或非正margin直接失败，不退回1。之后TAM/joint固定此值，不在V重校准。

数学中位数float64保存，训练FP32不变；同时记录实际训练dtype margin表示和hinge/ties，避免中央值舍入时假称50%激活。按批保存原候选/标签/距离、重复和分位、分项loss及对distance的解析导数；解析导数不是参数梯度，校准不反传。缓存填充、前向和统计全部计时。

校准完成、首TAM更新前写完整安全checkpoint；有效margin与来源进入last/selected/加载器。INTERRUPTED若已完成校准则不重做，否则接续一次；失败或超时不把半校准当可恢复成功。旧fixed配置和旧checkpoint兼容不表示可混用不同模型/数据语义。

## checkpoint、恢复和运行边界

checkpoint包括association/fusion/loss状态、阶段optimizer、sampler/RNG、游标、normalizer、adapter/cap/time unit、原点/query和监督配置、有效margin及累计成本。阶段末/验证间隔/显式中断保存安全状态，原末步另存以便接续；恢复用新目录、接续原累计预算，FAILED不自动重试。

恢复核可解释配置、人口和实际状态，不引入SHA/hash门槛，不把整个原包复制到每份checkpoint。精确续训以同冻结输入和环境为前提；GPU换卡许可不等于已实际验证所有CUDA设备逐位等价。

验证期间临时统一association参数requires_grad=False，退出及异常恢复原阶段标志。本地PyTorch曾因requires_grad影响mm/bmm选择出现2.38e−7阈值漂移；修复是统一计算状态，没有放宽断言、改精度/阈值或重写历史结果。

## 成本记录的实际分母

`history[*].seconds`只统计训练更新；完整验证另记含评分、选阈值、必要selected CPU复制的耗时，不含checkpoint写盘。最终评价记selected恢复到完整评价和阈值一致性检查，不含summary写盘；未开始为null，失败保留已有耗时。恢复后的旧history保留，而当前run的调用计时不伪装跨调用完整累计。

成功CUDA计时在工作边界同步；异常路径不为补计时再次同步或覆盖原异常，失败秒数可能不含尚未完成CUDA工作。峰值allocated/reserved只来自选定设备本进程PyTorch allocator，建模前同步/reset一次，含当时仍存活分配/缓存；不是整卡用量、模型独占增量或跨resume峰值。CPU/未reset为null，采集失败保留原错误。

## 已完成验证与限制

原生小型合成ZIP→显式reference→四阶段CPU更新→完整V评价/选定重载已通过；含dropout中断恢复、冻结/解冻、监督分母、无效输入、预算和RNG/模式隔离。独立审查及既有回归完成，历次软件记录可在Git历史与[CPU原日志](https://github.com/HKillua/association-code/blob/612afea/docs/results/frozen_input_audit_20261018/local_cpu_regression.log)追溯。当前整仓累计检查以[实现交接](../../HandOver/Handover.md)为准，不把历史304/347/371项当仍待完成的新工作。

013/014已有真实短训/独立终评证据，017是修正后开发训练，均不能据CPU通过声称完整论文训练成功。实际效果只据运行原件解释；本规格没有新的训练授权。
