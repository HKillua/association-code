# 134诊断实现规格

[事前Idea1.0](../proposals/mixture_opportunity_20261030.md)规定信息权限、全部边界、指标和预算；不修改133任何预测/API/缓存。

`association/evaluation/mixture_opportunity.py`纯numpy：

- `source_position_moments(*,states,support,groups,pack)`：合法有限[N,Q,4]、bool support及133源结构，重核node→source、counts/组绑定。返回[S,Q,3]，内容为源支持成员平均latitude度、mean sin(lon)、mean cos(lon)。无支持行置有限0。
- `mixture_positions(moment_pairs,alpha,anchors)`：pairs[M,2,3]、alpha标量或[M]∈[0,1]、anchor[M,2]；alpha为第一源质量，位置[M,2]；纬度线性、经度atan2，模长<=1e−12用anchor经度。预测函数不接reference。
- `grid_mixture_oracle(moment_pairs,anchors,truth,grid_size=4097,chunk_size=128)`：显式reference-only诊断，有限输入；小样例可用整数2..4097网格，正式固定4097。输出endpoint_errors_m[M,2]、best_single_error_m[M]、grid_error_m[M]、alpha[M]、gain_m[M]、gap_bound_m[M]（未定义为NaN）、min_resultant[M]、position[M,2]及必要计数。算法/界按Idea，分块不产生全M×4097常驻数组。空M合法。

runner `tools/analysis/mixture_opportunity_pilot.py`：新目录exist_okFalse；T名单/133completed/127cached lineage绑定；保存每场所有eligible位置的moment_pairs/anchors/truth/group_indices/query_indices/features[M,2,10]/source_errors_m[M,2]/role_indices[M,2]/group_pure[M]、原三head OOF scores和真实hard error、oracle全部数组到private/scene-{sid}.npz。公开只有场景/组件/匿名聚合；不下载原位置/身份/权重。全T scope包含未进入主人口的统计。

公开`scene-summary.json.gz`为全部32行；`summary.json`固定`all_components/by_component/without_largest_component/largest_component_id/primary_bootstrap`，聚合oracle增益/误差/界、OOF的signed excess/损害/错选及预定分层和匿名角色混杂。原有OOF score不重拟合；0 V数组/新teacher/神经更新，CLI固定CPU1..4/advisory。

验收：反向误差抵消有正收益、同向共偏无增益、端点包含、日期线、源角色交换同最优误差、多个本源成员下圆周非结合对照、连续界高密度手算检查、空/退化/非法值；本地小fixture全32链路和独立审计/破坏输入拒绝。软件例不计科学成绩。真实后独立审计不导入新core/生产恢复器，重算原节点moment绑定、4097网格、界、OOF和主要聚合/组件bootstrap，未核项明确登记。
