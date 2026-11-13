# C128 · Autoferry双船GNSS外测资格

## Record

`EXPERIMENT_RECORD REC-AUTOFERRY-DUAL-GNSS-ELIGIBILITY-20261113`，2026-11-13，verified。消费 C128/M186/F410/F412 和事前Idea。固定Autoferry作者CC0公开9场、18份JSON、7,291,887 B，两船GNSS以作者逐行对应且各自与检测差不超过0.2秒的条件评价。先在LiDAR恰好两检、雷达至少两检的帧中，冻结互相最近且时差不超过0.5秒的213个扫描对。雷达选取和定向匹配只用LiDAR/雷达观测，GNSS在候选和等权均值冻结后，用来事后绑定LiDAR的有向身份次序、核雷达自身时刻，以及对固定位置输出评价。全程0训练、模型前向和MTAD读入。 [事前设计](../proposals/autoferry_dual_gnss_eligibility_20261113.md) · [实际终态](../../HandOver/S2L-20261113-c128.md)

|人口|数量|雷达在其原时刻|LiDAR位置 MAE/P99 m|等权均值位置 MAE/P99 m|LiDAR→均值相对向量 MAE m|
|---|---:|---|---:|---:|---:|
|共同评价|154|137两船正确，17至少一船门外|3.466 / 11.701|9.228 / 156.240|5.076→15.769|
|GNSS间距≤50m|70|53两船正确，17至少一船门外|2.678 / 8.965|15.082 / 175.328|3.853→27.172|
|≤50m且雷达原时刻两船正确|53|8场|2.316 / 6.073|2.842 / 10.728|3.355→4.177|

213扫描对中209个满足双方GNSS与各自扫描时差，154个经作者10/20m位置门及正指派余量确定LiDAR的两目标顺序。70个≤50m的评价对中，LiDAR与均值的非平方距离余量正号皆为70，雷达冻结顺序在LiDAR时标上为67正、3负。LiDAR排序事后用GNSS定义，因而这些符号只衡量固定身份候选的动作会否翻转。它们不能代表跟踪身份准确率。完整LiDAR/雷达消息2,060/620个，各自有两船参照2,052/613个，20/10m门下双船可匹配上界752/430个。其≤50m子集分母584/177，覆盖258/85。 [全9场结果](</Users/a1/Projects/biye-ronghe/docs/evidence/20261113_autoferry_dual_gnss_eligibility/science/summary.json>)

## Squeeze

`SQUEEZE_REPORT SQ-AUTOFERRY-DUAL-GNSS-ELIGIBILITY-20261113`，verified。 [2,425项独立复算及53点分层](</Users/a1/Projects/biye-ronghe/docs/evidence/20261113_autoferry_dual_gnss_eligibility/public-recomputation.json>)

- **DE。** 8场通过“两源均可评、近距、独立双GNSS”的资格。最差的等权融合位置和向量被17个雷达门外点推高。这些点的LiDAR→均值位置MAE 3.809→53.241m、向量MAE 5.407→98.864m。全部17个的双船误差和及向量误差都增加。53个雷达原时刻两船正确的点仍有等权位置和向量均值退化，23点两指标都退化、14点两者都改善、16点指标异向。对源共同有效的前提只能限制错误类型。
- **DE。** 近距且两源正确的扫描配对时差中位0.154s、最大0.490s。时间差、质心与GNSS天线、雷达合并及遮挡均可能改变方向量。此轮只测两传感器发布检测，未补偿不同扫描时刻的运动或拟合偏差。匹配时差与实际可见性需显式纳入将来的消融。
- **AI/UN。** 可用这9场作为后续开发外测资产，保留逐场固定候选和双船有向几何评测。后续独立确认仍需要未用场或另一个真实数据集，且要能建立预测前身份和对误检/并点/漏检的分母。当前小样本、传感器和输出对象与MTAD不同，数值仅用于外部机制诊断。 [外测资产与作者协议](</Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/offline_c128_dual_gnss_external_assets_20261113.md>)
