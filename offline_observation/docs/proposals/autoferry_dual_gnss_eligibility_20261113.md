# C128 · 双船独立GNSS外测资格

`IDEA_PACKAGE IDEA-AUTOFERRY-DUAL-GNSS-ELIGIBILITY-20261113`，2026-11-13，diagnostic。消费Mode C2.49/C128、M186、P02/P04/P05、F410/F412与E477/E479。C125两路原生相对方向分歧大，且重用了预测输入。MUT有两船各自的GNSS，但官方目录只提供整个双船日期所在的大归档。转查Helgesen等真实港测公开逐场检测及船载独立GNSS。目的为检查独立外测是否具备所需的同刻可见性、几何与指派可评价性。已有作者JIPDA结果，新增价值为在固定同刻候选上同时计单点、两船有向误差和身份次序，明确适用覆盖。 [上游](</Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-249>) · [作者数据](https://autoferry.github.io/sensor_fusion_dataset/) · [作者§5/6.2](https://www.researchgate.net/publication/359787436_Heterogeneous_multi-sensor_tracking_for_an_autonomous_surface_vehicle_in_a_littoral_environment)

- 固定作者仓提交 `319ed9a587be01ac68e5b5a8233316c5f5ac79bf` 的全部9场，仅读检测及GNSS两份JSON。以作者发布的配对行绑定消息和GNSS。检测转 PIREN NED；按两船各自的GNSS时间与消息差绝对值≤0.2秒作为参照时间资格。环境1/2各用作者检测评价的20/10m空间关联门。全部消息报告覆盖，真距≤50m为预置近距层，另报每场。
- 等输出的几何控制。选择LiDAR恰有两个检测、雷达至少两个检测，找互为最近且差≤0.5秒的扫描对。仅由已公开观测位置求雷达不同检测到两LiDAR检测的最低距离和匹配，确定候选后冻结三臂LiDAR、所选雷达、两者等权均值。不用GNSS选时刻、雷达候选或任何权重，不改传感器、坐标或时间。事后在LiDAR消息时刻用该帧GNSS确定两输出的锚身份，并独立对雷达原时刻GNSS核其被选顺序。报告全有效及近距的覆盖、每目标位置MAE/P99、两船有向向量误差/夹角、保持/交换的非平方距离余量符号、雷达两正确/交换/门外的数目。不以真值标签作为跟踪输出。
- 对所有LiDAR和雷达消息另用GNSS事后核双目标检测覆盖上界，区分门外和不共视。外测资格至少要求两场存在近距、两GNSS和两传感器皆能评价的扫描对，且有可复算的共同点。均值方法是否有用分别看绝对位置尾部、向量与指派，不以资格通过替代。无训练、调参或正式MTAD数据访问。
- 竞争解释为目标中心与检测质心偏差、配对时差、遮挡或雷达合并、仅双检成功后的人群偏差，以及真值锚身份只允许事后评估。结果为外部小样本机制诊断。不能合并进MTAD指标或宣称与TMTAFF等价。所有重复为场内报告，仅作描述。
