# association-code

MTAD不完整多源航迹的关联与状态恢复研究。与TMTAFF对齐输入、最终输出及评价任务，允许重设计中间流程；以完整系统误差、目标覆盖和重复输出判断方法价值。**当前尚未成功复现完整TMTAFF，也未证明超过原论文。**

正式数据为 **Level1，2100训练／300开发验证／600封存测试**。候选使用完整合法观测和Q64真实时间网格；测试集未用于本轮开发，跨场共享身份限制必须披露。[数据入口](configs/data/README.md) · [数据约定](docs/data_contract.md)

日常开发和服务器回传统一使用 **`main`**；本地工作目录为 `/Users/a1/Projects/association-code`。历史研究分支已合入main并退役，实验按提交保留，不再维护多条长期分支。

## 现在看哪里

| 入口 | 用途 |
| --- | --- |
| [当前研究状态](docs/research_status.md) | 已确认的结果、未证明的机制、下一步决策 |
| [交接索引](HandOver/Handover.md) | 唯一当前执行状态；本地AI写代码和决策，服务器AI执行并回传 |
| [062执行方案](HandOver/L2S-20261022-062.md) | 冻结056分组与三个057模型的七臂评价；已获CPU4预算，服务器已提交 |
| [当前方法](docs/specs/evidence-association-fusion.md) | 学习关联、属性融合及公平对照 |
| [TMTAFF参考实现](docs/specs/tmtaff-reference.md) | 参考模型实现边界和复现限制 |
| [结果证据](HandOver/results/README.md) | 小型聚合/逐场证据与服务器原件路径 |
| [历史摘要](docs/history/README.md) | 退役实验的结论、负结果与Git恢复位置 |

056学习关联已产生完整系统收益；057融合三臂、059采样、060增训和061受限Oracle均已完成审查。当前重点是062：检验冻结融合模型在学习分组上是否超过普通控制及可观测支撑规则。没有新增训练授权，不因空闲GPU自行扩展实验。

## 代码与验证

- `association/`：数据、当前关联/融合、TMTAFF参考、评价和训练。
- `baselines/`：TSA-cTFER独立关联对照。
- `tools/experiments/`：保留的[训练与执行入口](tools/experiments/README.md)。
- `tools/analysis/`：当前缓存复核与062冻结迁移。
- `configs/`：当前配置和固定数据名单；文件存在不代表可执行授权。
- `tests/`：数据划分、时间/单位/mask、参数更新/冻结、评价人口及小型贯通验证。

```sh
python tools/experiments/evidence_association_fit.py --help
python tools/experiments/evidence_fusion_fit.py --help
python tools/analysis/frozen_fusion_transfer.py --help
python -m unittest discover -s tests
```

本地不SSH值守，push不会唤醒服务器。最多四个**已获批**单卡任务；具体预算和一次尝试以当前交接为准。[协作规则](AGENTS.md) · [现场执行规则](docs/server_autonomy.md)

已结案的大预测包、重复日志和旧方案不常驻工作树；必要结论及小结果保留，原件从Git历史或服务器读取，不另建archive树、不改写Git历史、不动服务器数据/权重/runs。
