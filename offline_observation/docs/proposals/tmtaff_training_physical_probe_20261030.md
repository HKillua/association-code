# 153：原107冻结端点的训练侧物理恢复定位

IDEA_PACKAGE/research-pipeline-contract v1.0，2026-10-30；`IDEA-TMTAFF-T-PHYSICAL-PROBE-20261030`、plan1.0，系统审查/诊断。消费Mode C1.62 P02/M02/E149及原113 Record/Squeeze，原107训练T痕迹非零、V位置失败，保存normalizer/affine一致。候选原因包括TFM本身未拟好、TCM外推、全Q窗口/尺度、递归或域差；此前没有固定T末端物理预测，因此不能认定只过拟合/只错组/只单位。

固定原107/tfm1000末checkpoint，selected=False，0optimizer/TAM/新归一化/CSV/V/test，不续107。只读原127首T8原生事件和评价cache，复建同cap128/Q64/time20/新单位timed输入，节点/query及选点规则绑定；完整场景输入范围决定Q，不因仅选少数节点缩时间窗。按原节点i<j顺序每场首2个同身份跨源正对，共至多16对（数量不足如实记），不按真值/误差/支持层筛pair。只对所需unique节点做TCM，原T身份仅用于T诊断配对，物理reference在预测后读入评价。

四冻结TFM干预：原complete(left,right)、左右换序、左输入重复(left,left)、complete_state_hold(left,right)。后者保支持内实际观测嵌入，只以原生端点hold替换TCM支持外latent；若双源共同支持内也大错，不能仅由TCM外推解释。换序/自配对为接口敏感性诊断，可能偏离原训练分布，不能单独定位物理根因或将其选为新方法。普通cap128 Mean及canonical_all OLS Mean作同点物理强控，所有原生speed在输入转换一次mps、原reference保持mps。

所有输出固定并保存后再接原参考，完整Q64、共同/单源/零支持层、位置MAE/RMSE/P95/P99与速度/航向有效点、原特征latent差/左右干预位移；不能把latent MSE换成米、把T少量正对冒充全T拟合或V公平成绩。有限观测对照不按好坏删除，缺参考/缺覆盖保留。

新冻结checkout153/run/launcher一次，现场兼容GPU3/CPU4/taskset0–3、RSS6GiB/1800s advisory；不影响151/152。软件验证复建原生canonical/cap选点/真实query/一次kn→mps与归一化pack逐值一致，无TAM/reference进入输入。异常保全无自动重试、改batch或重训。真实命令/退出/成本/对数/节点数/各域结果Record→Squeeze→Mode C；没有效果前不预定修正方案或称机制已证。
