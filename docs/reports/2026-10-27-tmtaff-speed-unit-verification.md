# TMTAFF显式速度单位链：本地软件核验

日期：2026-10-27。行为/兼容规范在[TMTAFF spec](../specs/tmtaff-reference.md)，研究理由见[候选与比较接口核验](2026-10-27-candidate-feasibility.md)。本次不是训练或MTAD效果实验，不生成EXPERIMENT_RECORD/SQUEEZE_REPORT。

## 改动与边界

显式 `data.speed_unit_policy="measurement_kn_reference_mps_v1"` 才启用。量测kn乘1852/3600转换一次，reference依据已核8个训练场景解释为m/s、不二次转换；原CSV/native对象不改。timed数组、normalizer、监督、TCM/TFM输出、observed singleton和cap128普通控制用m/s，物理速度误差为m/s，既有论文指标接口转为kn。

新策略绑定adapter/normalizer/target元数据，TCM和TFM的state_dict保存整数版本marker；新旧权重双向混用、错误目标声明和混合数据策略均拒绝，包括`strict=False`。独立TCM初始化先核normalizer/数据/marker并转移标记；缓存仍使用已校验的同运行数据。新prepared结果不保留原kn tracks，避免混用；调用者原对象保留。

缺省旧路径不加新单位字段，旧checkpoint继续恢复。仅支持timed TMTAFF；motion/native_events组合明确拒绝。没有改变架构/递归反馈/loss公式/数据人口/模型预算，没有修改102/103配置或新增READY。单位改变会影响损失尺度及优化，必须用新的合同/训练评价，不能修补旧神经输出便声称公平比较完成。

## 实际验收

新[7项测试](../../tests/data/test_timed_speed_units.py)覆盖真实小ZIP读取、normalizer、reference保持、全目标evaluator、TCM learned前向、多成员TFM、冻结完成缓存、单例/普通控制、NPZ及元数据、CLI inspect-only、原对象不变、重复转换拒绝和新旧权重错误边界。极小数值fixture用于确定转换行为，不能用于认证论文复现或数据成绩。

其中固定TFM输出为5 m/s，实际多成员oracle结果及其kn报告一致；observed singleton/Anchor/Mean为量测5 kn转换后的值。二者明确区分，检查不会只走旁路而遗漏融合分支。米制位置、圆周航向、完整目标计数均保持。

最终执行命令：

```sh
.venv/bin/python -m unittest \
  tests.data.test_timed_speed_units \
  tests.data.test_timed_inputs \
  tests.models.test_timed_models \
  tests.models.test_fusion \
  tests.evaluation.test_full_scene_evaluation \
  tests.evaluation.test_paper_fusion_metrics \
  tests.evaluation.test_prediction_artifacts \
  tests.evaluation.test_complete_evaluation
```

结果：**75项通过，1.690秒**；`git diff --check`通过。主代理检查生产diff、测试覆盖及阶段转移逻辑，另由未实施本改动的子代理只读独立复核，未发现有证据的P1/P2问题；独立复核未重复跑测试。未运行优化更新、作者程序或真实MTAD扫描，未访问T_test；本地fixture前向不冒称服务器诊断已完成。

## 尚未证明或完成的事项

固定056与统一canonical_all普通控制尚未形成新同协议端点；本轮只是修好单位前提，不使当前cap128控制自动等价090C控制。不证明单位是历史失效唯一原因，不证明TMTAFF原表重现，不解决作者ESD/反馈/预算歧义，不构成任何新方法收益。当前研究安排与执行条件只见[研究状态](../research_status.md)和[交接](../../HandOver/Handover.md)。
