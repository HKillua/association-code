# 147：冻结Qhard的全V300基准与共同评价入口

IDEA_PACKAGE / research-pipeline-contract v1.0，idea_id `IDEA-SOURCE-RISK-FULL-BENCHMARK-20261030`，plan_version1.0。主模式为系统/范围审查，不主张新的评分或混合创新。消费Mode C1.61、MR-C1/F231/E129、F243–245/E143–145及133/138/143/144/146的Record/Squeeze；原矩阵/来源保持原位。用户本轮强调先把自己的方法把握好，故先明确现有基准的全量效果，再同步接可信同行。

## 问题与假设

原133在T32拟合的10维/66系数共享QUADRATIC、固定hard读出，在原V32相对均值/最近源有位置信号；其余268场是否延续、少数高害/零支持/组件依赖如何，是UN。竞争解释：V32选择与共享身份、MTAD生成角色、错误组/目标对应、支持域及相对几何特征可能限制范围。扩量不等独立测试或新的神经网络贡献，近邻不能由内部均值控制代替。

## 固定输入与操作

- 原Level1 T2100/V300/测试600名单；本项只读V300量测和既有056缓存参考，不打开T payload/reference/test，不新拟合任何参数。沿114 WS、056 final1000分数/阈值、全Q64和原节点顺序，0重构组/阈值选择。
- 读取原133 full QUADRATIC mean/std/beta，10特征/裁剪/分数及hard/soft原式不变；0新head/rho/注册/神经/teacher/搜索。原32的3个源策略直接导入133原预测，核原pack/score一致；其余268补合法原生canonical_all运动特征，逐项绑定056输入，再计算源等权、Qsoft、Qhard。原32native-motion复用127，不重新读其CSV；新268逐场读量测，不读原reference CSV，不重建新reference。
- 六臂：WS_MEAN、WS_ANCHOR、WS_NEAREST_SUPPORTED、WS_SOURCE_EQUAL、WS_SHARED_Q_SOFT、WS_SHARED_Q_HARD。普通三控按原state重放，Mean/NEAR与116/117原artifact核对；保留错误组、单例、漏目标、重复。没有固定源编号或FixedLOW策略，没有146状态修复拼接。
- 原Q特征/参数保持原输入语义；只在**输出评价边界**将原控制航速kn转m/s，reference按数据约定解释为m/s。位置/航向/预测有效性不变。公共评价函数要求每方法显式声明输出速度单位，保留缺预测；作者ME/ESD边界不改。Q策略速度/航向继承Mean，逐项声明，不称四属性学习。
- 所有新预测先固定再读评价字段；新private保存输入/pack/score/贡献和全部canonical m/s artifact。原32位置与原133、普通三控位置与历史对照核同点/人口。对比完整全300、原32、其余268、38组件、最大组件外、原32未见组件；V仍为历史开发集，共享身份不消除。

## 指标、资源、解释

主要估计量：其余268和全300的Qhard−NEAREST_SUPPORTED、Qhard−MEAN共同位置MAE米差；全部15对MAE/RMSE/P95/P99、覆盖/重复、新增>1km损害、0/1/多源支持层、场/组件方向和四属性物理/论文统计完整保留。2000次seed20组件重抽样给描述区间，不作确认性显著结论；不按V决定新权重/阈值或删难点。

一次attempt-1、CPU4/RSS6GiB/AS12GiB、3600s advisory，固定300终点；原32三源策略导入96臂，新268三策略804臂，普通三控900次解析重放，无训练。新目录，原数据/参数/失败现场不改；缺件/非有限/绑定错误/OOM保存现场失败，不自动重试或替代配方。参数/坐标/身份和大数组仅server；必要脱敏summary/逐场及原launcher回传。实际输入/命令/退出/成本及计划偏差独立记账。

共同评价入口的行为规格：显式各方法group输出/速度单位和统一目标/支撑，复制换算不改原数组，复用既有evaluate_full_scene；支持各方法不同预测有效性，无输出/重复/漏目标不消失。位置pair比较仍同点；物理m/s与论文kn分名，ME不当MAE。用缺目标/重复/部分输出与kn↔m/s手算检查；不建模拟训练资格链。

完成后独立NumPy审计重算保存artifact的位置/人口/尾部/组件及0拟合/输入绑定，注明审计范围；Record→Squeeze→Mode C回流。若Qhard优势局限于小范围或严重尾部，保留失败边界进入新Idea，不追温度/加seed，也不把该基准当最终模型或可信同行胜负。
