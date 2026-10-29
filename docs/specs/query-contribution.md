# 可替换的查询贡献与物理恢复接口

本页只规定新增软件接口，不发布实验或改变056、114、115及旧Mean配方。新接口不读取文件、不构组、不选对应，不接受identity、reference、目标有效mask或完整fragment字典。

- `supported_query_weights(nearest_seconds, support, sources, groups, *, method)` 接收有限非负时距和布尔支撑`[N,Q]`、非空source字符串`[N]`及覆盖全部节点一次、按原index排列的partition。返回`weights[N,Q]`和仅含计数的audit。组外节点不能影响本组。
- `source_balanced_supported` 在每组每query内，设有支撑source数为`Ns`、该source有支撑片段数为`ms`，每片段权重为`1/(Ns*ms)`。source只定义等价类，不预定质量；均匀复制某source的全部片段不改变恢复值（允许浮点舍入）。支撑不是量测准确性或校准协方差。
- `nearest_supported` 仅在有支撑成员内选最近原生观测时距，平局选最小原index，得到one-hot权重。没有支撑时全部权重为0，不能选无支撑的近成员。
- `recover_group_states(states, weights, groups)` 接收有限`states[N,Q,4]`（纬度、经度、速度、航向）及有限非负贡献质量，按组query总质量归一化。纬度/速度线性，经度/航向圆周均值；归一化圆周合量`<=1e-12`时对应角度回退组首anchor。总质量为0时整行回退首anchor；仅一个正权成员时完整复制该节点值，避免改变角度表示。返回`group_states[G,Q,4]`与回退计数audit。
- `query_contribution(states, nearest_seconds, support, sources, groups, *, method)` 组合以上两步，返回`group_states`、`weights`及`audit`。所有组先得到输出，每个组只有首anchor回退规则；该输出不含真实身份，也不决定最终对应。

输入不得被修改。Q必须大于0，N=0时允许空partition。未知方法、非法shape/非数值/非有限输入、负时距或权重、非布尔support、source长度/名称错误、重复/遗漏/越界/非整数或乱序partition均抛`ValueError`。audit只含方法名、各组query支持成员数、支持source数、source片段数最小/最大值及是否不均、回退计数；不包含source名、节点名或坐标。

| 验收行为 | 软件核验 |
| --- | --- |
| 两种W遵守source平衡/支持内nearest与原index tie | 手算小例、one-hot检查 |
| 无support、圆周包络/零合量、单成员复制 | 独立预期值与首anchor回退 |
| R/W/恢复可独立替换且无组外影响 | 固定W替换states、隔离组输入、独立调用两步 |
| 复制source、source重命名不影响质量假设 | 配对输入不变性检查 |
| 拒绝非法输入且不修改输入 | 参数边界及数组前后对照 |

仅运行新增单元测试；小合成数据是软件行为证据，不是MTAD成绩。旧`ordinary_fusion`、`evaluate_full_scene`及对应policy保持原实现和数值。
