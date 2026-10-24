# 087：完整 helper 输入下的显式时间对应

本地先依据用户设计独立实现，随后收到外部 `3b7f34c` 源码并对照审查；机械对齐GRU初始化和输入通道约定，保留本地独立接口及验证。外部测试成绩不替代本地实测，不声称整体逐值复现。083表明整片段删除的总效果为负，不能从中断言缺少时间对应是根因。新问题只问：在完整物理路径和逐查询可见集合不变时，显式保留同一helper的跨查询对应是否有增量。

## 冻结结构与干预

- 原072C Transformer、fixed.95 base、Mean、76维输入、T_core/V_select、seed20261023、4000成功更新、batch8、Adam .001、clip1和线性Haversine位置loss不变。无V选模。
- 新支路使用现有27维可见pair特征，经`27→32→16` MLP。每个helper沿Q64送入共享单层双向GRU（input20、单向hidden8）。20维通道顺序为latent16＋interval support1＋原76维中71:74的elapsed/spacing/span三种`log1p(seconds/20)`时间。GRU增加1440参数；总参数109042。
- support=false时先把27维置零，再将phi输出置零；GRU仍看到时间和support，以允许间隔内的隐藏状态演化。池化只包含当前support=true的helper，使用float64均值再转float32。当前无helper支撑时分支精确为零。双向GRU和全序列Transformer均为离线恢复，不宣称在线因果。
- A（shuffled）每个查询对helper行独立置换，特征/support一起置换。B（preserved）消费同样的置换计划但实际使用恒等映射。原76维、Mean、base和目标mask不置换。
- A/B共同初参数逐张量相同；公共块取072C的step0，新增输入投影列为零。phi/GRU独立seed20348223，GRU每个weight张量使用Xavier uniform、bias全部为零；这两项机械对齐外部实现，不声称整条训练逐值复现外部代码。T路由seed20348023，V路由seed20348123，均与原采样RNG隔离。
- V按固定场景和组顺序消费计划，initial/final各自重置；全部组包括重复/无有效真值组先预测，再评价真值。保存每组planned和actual-applied路由及原helper成员，缓存比较逐项重放计划并核对实际应用。initial/final路由曝光与最终RNG必须相同，preserved实际变化槽位始终为零。
- H指整个组的helper行数。H=0/1时两个模式同参数等价；不是每q支撑数0/1就等价。B对整条helper重排保持集合不变。H≥2时A的路由也可能偶然恒等，不强制重抽。

## 配对、产物与判定

原始训练batch（场景、anchor、mode、成员、有效点mask计数）与072C严格一致；新增路由只在成功optimizer更新后记曝光，逐batch同时保存计划和实际应用，汇总sampled_anchors、multihelper_anchors、helper_query_slots、planned_changed_slots、applied_changed_slots；缓存比较独立重放并闭合这些计数。V端记录group_count、multihelper_groups和后三项slot计数，不把有计划但未应用的置换误报成实际干预。初/末checkpoint保存完整原采样和独立路由RNG，检查全参数Adam更新次数；记录phi与GRU真正非零梯度更新。零残差初值不证明分支已经获得更新。

每臂一次训练与CPU初末完整V评价，可独立回传；不因为另一臂先返回取消它。`compare_helper_temporal.py`支持一臂或双臂与072C、078B的只读缓存比较，每臂runner末尾保存自身对强线的reference-comparison，双臂均完成后另运行一次整表；另一臂失败不阻止已经成功的单臂screen。所有完整数组与权重留服务器，只提交必要汇总。

比较保留完整人口、普通控制、四种MAE加权、MAE/RMSE/P95/P99、H/I和固定伤害阈值、逐场、38组件与最大组件外。新增支撑区域×当前helper数0/1/≥2交叉表是互斥分层；每格误差和/平方和、H/I、相对072C的净差及全人口贡献必须闭合。净差份额可负或超过1，分母为0写null；它不是因果贡献。

主要筛选：对强072C点MAE至少改善3%，RMSE/P99不增（1e−6m数值guard），其他三种MAE加权及最大组件外不退化，超过半数场景/组件MAE改善。单seed过线仅支持必要匹配确认，不等于最终方法晋级。B−A单独报告，不能用赢破坏性控制替代强线要求。

次级预测：保住/增加anchor无支撑而helper有支撑区的I，并减少anchor已支撑区H。主指标通过但该预测不符时，不宣称时序可靠性机制得到证明。B胜A但不胜072C则没有竞争增量；两臂均胜但相近则公共容量/时间编码仍是解释；A胜B保留反证；两臂均未过线关闭固定配方，不自动扩网络、加步或搜路由种子。

T_test封存；066保持REJECTED。当前反复使用V、共享身份、单训练seed与破坏性A控制的局限全部保留。软件测试只是代码检查，真实实验成绩待服务器回传。
