# 013后续本地实现与核验

2026-10-18。013与012均已终态后整合；本次未执行新的真实数据模型或GPU作业。新评价预算仍待批准，命令方案在[后续方案](../../proposals/tmtaff_after013/README.md)。

## 改动

- `association/fusion_training.py`：只对TFM_FROZEN加入每步独立、CPU detached补全缓存；同轨在该步只补全一次，设备batch仍≤2×pair_microbatch。融合调用顺序/dropout、有效batch分母、optimizer及JOINT梯度路径保持；正常/异常均清理缓存。它改变补全batch布局，允许通常浮点舍入误差，不承诺跨布局GPU逐位相等。
- `association/complete_evaluation.py`与trainer：增加关联评分/物理融合逐场景进度和融合forward累计数；聚合与预算检查完成后才发completed。失败仍由run终态判定，progress不能冒充成功。
- `tools/experiments/tmtaff_complete_evaluate.py`：独立恢复已完成训练更新的selected检查点，原normalizer/阈值/适配器/验证batch，完整V评价；零optimizer、零校准、零重选。新目录，准备至写盘均计时，中断保留阶段；不读T_test量测或参考。

缓存的独立未缓存对照覆盖原生/历史输入、FP32/FP64、microbatch1/2，比较loss、梯度、Adam更新及状态、RNG。FP32最大梯度差1.19e-7、参数差6.73e-7；FP64差在1e-15量级，RNG相同。重复3对调用3→2，含共享端点的5对完成轨数10→4、调用5→3。这里只证明合成CPU行为与去重，**未测真实CUDA总耗时收益**。

## 验证

在仓库根运行，解释器为既有本地venv：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /Users/a1/Projects/association-code/.venv/bin/python -m unittest discover -s tests -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /Users/a1/Projects/association-code/.venv/bin/python -m unittest discover \
  -s tools/pilot -p 'test_audit_mtad_level2.py' -q
```

结果：主测试427项、18.820秒、退出0；数据审计测试15项、0.036秒、退出0。日志中的预期FAILED分支来自拒绝覆盖、缺显式device等负面测试，不是测试失败。新增8项评价CLI测试真实走合成ZIP和selected模型完整评价；真实013 checkpoint仍留服务器，没有在本地执行。

独立复核分别覆盖冻结缓存、评价CLI、科学与预算方案。确认并修复了“聚合前发completed”的进度问题，加入聚合超时回归；方案纠正selected只含TFM2而非JOINT、漏预测仅保留在覆盖分母的口径。未改损失权重、采样、阈值、模型规模或精度；未发布新READY。
