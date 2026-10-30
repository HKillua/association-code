# 151/152：全T2100训练与固定WS全V300公平比较

2026-10-30，IDEA_PACKAGE/research-pipeline-contract v1.0，系统/任务适配验证。承接Mode C1.62/F247/E147/E148/E149及公平合同§9；不是新网络创新，不依赖149/150结果。来源PTFM完整AP与DeepAF原P/V审查。此次重新提取用户本地DeepAF23页PDF，**仅定向核pp11–12 §4.1训练设置及§4.2开头**：原文给Adam/6+6层/d32/4头，但未在所读设置列公开训练epoch/batch/lr；不冒称再次全文精读，不增104计数。出版社页面429，仍可依原PDF继续。

## 151：完整、单位一致的可流式数据

idea_id `IDEA-FUSION-PEER-FULL-DATA-20261030`，plan1.0。固定Level1 T2100/V300/Q64/原offset-gap600/shortest_arc，T_test不读；逐场cache、RSS6GiB/CPU4，10800s advisory。原127 T32数据复用；其余T2068显式读原生量测与训练reference；V300量测用于canonical_all及cap128匹配控制，V reference只来自原056冻结registry/arrays，不重新生成/挑reference。原127/056不改。

新cache状态速度统一m/s，原量测kn乘1852/3600一次、reference按已定mps保持。全部合法输入/无输入目标/缺参考保留；额外训练索引仅选择有参考位置、恰双源有共同支撑的真身份组及全部连续共同支撑段。训练可用性与完整V人口分开，记录所有T无可训练组场景，不按效果排除。推理无truth/身份字段；真实数据仍K2，不称K≥3异构验证。

验收为2400场、T/V完整名单/node/query、旧127/056输入逐项绑定、T监督/单位一致、V冻结参考/完整目标/空场、cap128与all节点一致、新index与实际文件齐全；不以准备成功认证baseline准确。小ZIP实际reader检查V raw reference/test拒读、T reference允许、kn→mps仅一次、reference不变与无输入目标保留。

## 152：从头完整人口训练，固定末步比较

idea_id `IDEA-FUSION-PEER-FULL-TRAINING-20261030`，plan1.0；151成功后才冻结实际整数样本/更新数进L2S152并发布，不把本设计当立即运行许可。两模型均fresh seed20，不续148，不继承旧072/074C/099权重或optimizer。遍历全部T2100中既定可训练组，每epoch各组一次、保末批；按场/场内组shuffle。相同训练组池，记录各自有效监督/呈现和全部训练/评价成本。

- **PTFM**保148新双源支路/LSTM/特征门控/任务loss，训练覆盖每组**全部**连续共同支持段（区别于148仅最长段），段内等权后组等权；normalizer只用全T可见双支局部输入、流式累积，无reference/V拟合。原文最大150epoch作为固定150epoch端点；不执行原20轮V早停/10轮降LR，明确为共同任务固定预算适配，非作者优化流程复现。Adam .001/clip1/LSTM L2=1e−4、batch32，0 V选模/超参搜索。
- **DeepAF**保原独立P/V、6encoder+6decoder/d32、距离cross-attention/自身历史，实际dt、Anchor-OLS初始化/虚拟V/全Q64离线候选和TBPTT1，P desired与位置差分V；speed/course Mean。固定20epoch、batch16/Adam .001/clip1、每批梯度按全部组query归一，分64query replay微批累积后一次更新。20epoch是本项目适配预算，原文未公开的额度不补猜；与PTFM不同计算/监督，不叫同更新数公平。

全部T训练记录每epoch实际更新/组呈现/各属性监督、LR/目标/梯度/成本，检查point有效性/非有限/预算，保存固定末端及每epoch可定位进度；无自动resume/加步/seed。固定第一训练batch只作u0/末端T探针，0 V训练监测/选择；独立正式重复20/21/23仍以后具体合同，不因本次首seed单跑宣稳定。

末端一次固定114 WS全V300，与147 Qhard及同数据普通Mean/Anchor/Nearest、cap128 Mean/Nearest匹配控制一起经`fusion_benchmark`评价。PTFM分别报告共同支持原核心（域外missing）和核心＋预定公共Mean回退，不能偷偷补齐叫原模型；DeepAF离线恢复不叫在线原完整系统。原单源/错误组/漏目标/重复保留，raw前端分数不变。预测固定保存后开V参考；全域与双源共同支撑域、完整人口/覆盖/四属性物理与论文ME/SD/位置MAE/RMSE/P95/P99、场/38组件/去最大/新增>1km及成本完整保留。作者ESD仍未知，不能和有符号ME作提升率；不会与147不同人口混排名。

每路单GPU/CPU4/RSS建议12GiB，PTFM86400s、DeepAF172800s advisory，实际总成本未知，可能长于建议；超过只记录并完成固定额度，不加训练。最多四单卡/FIFO/兼容共卡，不触碰他人；新冻结目录/一次attempt，异常保全不缩batch/删样本/换配方。原107/132与149/150独立，不因新比较覆盖其现场。

Record→Squeeze核训练实际曝光/收敛与V完整反例，再回Mode C；差结果不认证同行净胜，需根据真实失配另立改进而非不断追V。当前“全量”指整个正式训练人口和完整评价域，预算及适配与作者原文差别如上明确，不宣作者精确复现或充分收敛。
