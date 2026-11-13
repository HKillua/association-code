# 三个场景诊断的实现核验

日期2026-10-27；对应`IDEA-STATE-DIAG-20261027` plan 1.3。**这是软件/输入路径检查，不是三个方法的实验结果，不进入科学效果DE，也不冒充record-experiment或squeeze-experiment已完成。**

实现为[diagnose_three_scenes.py](../../tools/analysis/diagnose_three_scenes.py)。独立只读审阅核了真实缓存写入schema、fixed first-anchor对应、duplicates、全query保留、仅三场原始记录访问、原生/query区分及101阈值字段。审阅提出完整090C配置/source_system绑定不足，已修。实现中还修正了`configuration`字段、reference generator关闭方式，以及090C本来没有069的`group_eligible`字段：只在内存从原support派生共享校验器所需mask，原region仍独立核对，未冒称缓存含该字段。

## 实际核验

命令：`.venv/bin/python -m unittest tests.evaluation.test_three_scene_diagnosis -v`。实际6项通过（0.035秒）：原评价器生成的synthetic cache包含缺预测、缺参考、重复组；原参考重放不同即拒绝；原生无共同时间不冒称同时观测；零向量方向为unknown及经度短弧；固定阈值重合分母；源对目标等权且oracle标注；完整配置/原system绑定。上述小数组仅软件排错，没有训练或算法有效性结论。

首次尝试bundled Python因测试fixture用原评价器、其类型导入需要torch而未执行测试；改用主仓已有`.venv`成功，没有安装依赖/修改生产模块。生产诊断脚本本身不导入torch，不创建模型。

真实输入路径另以主仓`.venv/bin/python`单次调用脚本的`validate_request → prepare_scene → scene_motion_evidence(canonical_all) → iter_reference_states → align_reference_states`，固定三个V场景、原config和原三个ZIP分卷。实际输出原件为`runs/THREE-SCENE-INPUT-CHECK-20261027/result.json`（本地，未推送原数据）；exit 0，1.370443秒，0训练/模型前向/权重读取/测试访问。读取6个量测/标签成员，另各读1个该场reference成员。未读其余场景payload；ZIP目录清单只用于原manifest一致性检查。

| 场景 | 原目标 | 保留片段 | query/目标 | canonical原生点 | reference原始行 | 位置reference可用query |
|---|---:|---:|---:|---:|---:|---:|
| 3035 | 67 | 186 | 64 | 55,653 | 15,565 | 3,644 |
| 2819 | 13 | 39 | 64 | 9,748 | 811 | 627 |
| 1043 | 6 | 17 | 64 | 4,958 | 416 | 307 |

三场原alignment均为offset=0秒、max_gap=600秒，reference经度shortest_arc，native位置canonical_all。`reference可用`不是`预测可评`：例如3035已有101的3614可评点不能拿来替代本次3644参考可用点，缺预测必须保留。

尚未验证服务器原NPZ逐值一致、三方法同点同方向、健康来源可辨识性、相对偏差或根因。对应实际结果应由独立L2S-103回传，保留真实argv/exit及中断记录；不能把本页“通过”写成科学诊断完成。
