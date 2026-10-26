# MTAD航迹关联与状态恢复

正式场景：真实MTAD **Level1，T2100／V300／封存测试600**，不完整量测下的补全、关联与融合。**近期优先做出一项位置融合／恢复贡献，与可信文献同行比较，再接回完整系统。** 接受成熟跨领域机制的小幅海上适配，不要求三环节同时各发明新模块。[目标订正与讨论沿革](docs/reports/2026-10-26-goal-reconciliation.md)。当前**尚未建立稳定胜过可信领域同行的证据，也未完成有效TMTAFF复现。**

## 当前入口

| 文件 | 用途 |
|---|---|
| [研究状态](docs/research_status.md) | 当前结果、主要缺口和下一步 |
| [交接索引](HandOver/Handover.md) | 唯一当前执行名单与状态 |
| [数据约定](docs/data_contract.md) | 划分、时间、单位、mask与评价人口 |
| [方法实现](docs/specs/evidence-association-fusion.md) | 当前模型与评价职责 |
| [运行入口](tools/experiments/README.md) | 训练、评价和缓存比较代码 |
| [历史摘要](docs/history/README.md) | 已结案正负结果与Git恢复位置 |

100三臂与CPU整表已完成：原强模型MAE564.834m，继续训练564.968m，原生事件固定池化566.092m、attention汇总567.187m；当前候选不晋级，不自动追加训练。096完整系统也已结案为无效参考适配，099 DeepAF保持HOLD。**101经纬度补算已结案；当前102纯CPU完整指标补算READY**，复用090C/056原结果，零训练/前向；[本地分析与证据边界](docs/reports/2026-10-26-experiment-review.md)，实际执行以交接为准。

**DeepAF自建模拟只是排错记录，不是作者复现、MTAD成绩或同行资格证明。** 不再扩展合成训练，也不以它作为推进真实数据实验的前置门槛。保留生产实现及单位/梯度/评价检查；真实同任务比较须披露适配差异。[DeepAF边界](docs/reports/2026-10-25-deepaf-method-audit.md)

## 代码结构

- `association/data`：读取、真实时间装配和采样。
- `association/models`、`training`、`evaluation`：模型、训练与评价。
- `baselines`：独立文献方法相关实现。
- `tools/experiments`、`tools/analysis`：实际运行入口与仍被使用的比较工具。
- `configs`：冻结配置与数据名单；存在不代表可以领取。
- `tests/data`、`models`、`training`、`evaluation`、`integration`：必要科学契约和小型贯通；共享小样例统一在`tests/fixtures`，不再在根目录堆实验脚本。

本地主仓为`/Users/a1/Projects/association-code`，统一main协作；服务器运行独立冻结checkout。数据、权重、预测及runs不随文档清理删除。旧副本通过Git查看，不复制另一套archive。协作规则见[AGENTS](AGENTS.md)。

维护原则：新实验尽量复用现有训练/评价入口；结案后保留关键结论和必要汇总，清除逐臂重复回传与专属临时工具。旧代码按需从Git恢复，不把历史实验全部挂在日常工作树中。
