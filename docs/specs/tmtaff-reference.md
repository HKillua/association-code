# TMTAFF时间适配参考：维护契约

这是本项目 **time-adapted approximate TMTAFF**，不是作者代码或精确复现。代码路径具备TCM→TAM→TFM→JOINT及独立阶段能力，但有效完整baseline仍未复现。旧结果与退役诊断见[历史摘要](../history/README.md)，论文未公开项与评价歧义见[协议对照](../mtad_data_protocols.md)；本页不授权重训。

## 输入和TCM时间语义

四状态lat/lon/vel/cou，冻结T内normalizer；至少两个严格递增的合法原生点，长轨cap128保首尾，所有有效短轨保留。Q64取全场canonical观测最早／最晚时刻linspace，原点为最早观测，`tau=(t-origin)/20`。不能由reference决定输入、query或选点；20秒是数值单位，不是论文窗口或真实采样周期。

模型打包`[N,M+Q,5]`：真实原生前缀四个归一化状态＋tau，padding／query非observed；query支撑内四维来自物理插值，支撑外占位值完全忽略。query mask=false意为不参加原生teacher转移，不代表支撑内状态不可用。时间FP32舍入不得改变严格顺序／支撑判定，否则明确失败。

保持原四维输入投影、6层、d128、8头、FF256及L64参数形状；连续sinusoidal编码按`sin/cos(tau*10000^(-2r/d))`。原生embedding为`ReLU(W(normalize(x)+PE4(tau)))`，decoder history另加下一预测时刻的PEd，不把目标状态放入时间条件。

teacher对原生序列正反向的相邻transition做detach-target MSE，按全部有效embedding元素归一化。encoder、decoder self-attention及cross-attention均限制处理序因果可见性，不能经memory偷看下一状态。逆向处理使用更晚物理观测，整个任务仍为离线。

支撑内按原片段物理插值再编码；支撑外分别正／逆向endpoint rollout。纬度／速度线性，经度／航向最短弧；半周歧义按原始delta方向，不用真值挑选。每一方向encoder仅编码原生一次，memory固定，生成latent只追加到decoder history。正向原生`[2,5,9]`、query`[12,20]`时，第一步next-times为`[5,9,12]`，第二步`[5,9,12,20]`，不能重写历史槽时刻；逆向同时反转状态和时间，最终还原Q升序。

TCM输出为`[N,Q,d]`特征，latent MSE不是物理米误差。原生一步监督不保证长query外推有效。参考与候选输入预算不同须披露，不能把canonical_all多看观测的收益全算学习机制。

## TAM、分组与TFM

TAM与TFM消费同一Q上的TCM特征。TAM标签为全部合法i<j的A-B身份相同关系，允许同源中断同船；不偷偷改成跨源Hungarian任务。CE/CL及选择规则由配置声明。CL缺省保论文式；`contrastive_negative_scale=inverse_margin`为明确适配`0.5*[y*d+(1-y)*relu(m-d)^2/m]`，不是论文公式修bug，也未取得新的实验授权。

CL显式`train_negative_median_v1`时，仅在T_core、TCM结束后／TAM开始前一次校准：独立seed21、64批×64、2048正／2048负曝光；冻结TCM、TAM保训练dropout，保存恢复Torch RNG／模式，不消耗正式sampler。负flatten距离中位数为固定margin；非有限／非正则失败，不读V重校准。恢复已完成校准不重复。

预测分组为`anchor_direct/input`，首未分配节点收直接过阈值的剩余节点，含等号、不传递，身份不入预测。真身份组仅外部诊断。全部目标／节点／单例／重复／漏目标保留。

[PairFusion](../../association/models/fusion.py)左右encoder独立，拼接展开后两FC压缩／展开为memory；`fc_hidden=128`是本项目明确适配，论文中间宽度未披露。独立decoder用learned BOS和自生成latent因果递推，cross-attention看完整memory；BOS与递推解释、softplus速度、输出affine均需披露，不能冒称作者原实现。

多片段递归显式`feedback_policy=fc_memory`，左输入为累计FC memory、右为下一片段TCM输出，不能把四状态当128维特征。singleton记录missing，由上层既定单源策略处理，不自配对或丢弃。顺序影响与代表策略须说明。

TFM训练仅使用T身份真配对；预测错组仅进入完整评价。reference按显式offset与gap600对齐，不外推、不跨无效点、冲突属性屏蔽；无参考目标保留NaN/mask，不能制造零真值。完整目标registry和组对应由既有评价维护，详见[数据约定](../data_contract.md)。

## 损失与微批

位置已明确使用`standard_smooth_l1(beta=.001)`，纬经按原生度逐维；`paper_printed`只保历史兼容。经度短弧开关、航向圆周、输出conditioning需写入配置。速度为`log(v+1e-6)`差平方，航向为`1-cos(pi*delta/180)`，四项按有效属性平均，再用可学习log variance `sum(.5*exp(-s)*L+softplus(s))`组合；这不保证梯度自动平衡。

微批使用整个batch逐属性有效数作分母，uncertainty正则整batch只加一次；局部无监督贡献零，全batch缺必需监督失败，不重抽／删pair。TAM微批按pair数加权；TCM辅助对全batch唯一端点按`2*d*sum(max(n−1,0))`归一化。一次清梯度／裁剪／step，异常不保存部分更新。dropout=0下可检验整批梯度等价；启用dropout不能声称随机mask逐位等同。

## 阶段、选择与checkpoint

入口[tmtaff_complete_fit.py](../../tools/experiments/tmtaff_complete_fit.py)显式`--phase full/tcm/tam/tfm`，各phase与更新配置一致；单TCM不校准margin、不选关联阈值或跑下游，输出completion-evaluation为特征诊断。TCM有随机片段或`ordered_fragment`选项：有序按场景／节点遍历，尾批保留小批，每eligible片段一次才算一epoch；不等于已重建原文80epochs阶段分配或随机截短增强。

独立TAM/TFM只从COMPLETED TCM-only端点转移冻结补全参数与normalizer，核输入表示／数据／时间／cap；不继承源未训练下游、optimizer/RNG、历史或耗时。TAM dropout可单独设置，不能改全局dropout假装只改关联；resume必须保持目标和表示。

四阶段为TCM→冻结TCM训练TAM→冻结关联训练TFM→JOINT；记录真实场景／轨／pair曝光，不以总更新数代替各模块更新。TAM/joint按完整V Macro-F1选snapshot与阈值，TFM保末步，selected association可能早于末步fusion；不得把二者称为同一步全链端点。终评重载后须复核分数／阈值决策，训练与评价计算状态一致。

checkpoint保存模型／fusion／loss、阶段optimizer、sampler/RNG、游标、normalizer、data契约、有效margin及成本。只有同配置／同phase的INTERRUPTED可按明示方案恢复到新目录；FAILED或COMPLETED不自动续跑，不能在resume时改损失／数据。检查实际配置和数组，不设文件hash门槛。

## 验证与边界

保留真实小ZIP贯通、短轨与空场、时间平移／圆周边界、reference/padding隔离、双向next-time因果、微批分母、冻结／更新／中断恢复和全人口覆盖测试。软件通过只证明实现语义；历史033增训有特征收益、035/047关联仍弱、049/053B融合条件响应衰减，这些负结果不能因清理删除。

baseline下一次预算须针对已有证据提出可区分机制的改动；不自动恢复被删除的一次性诊断或继续加步。当前运行与去留统一看[研究状态](../research_status.md)和[交接索引](../../HandOver/Handover.md)；不能以战胜退化参考宣称超过原文。
