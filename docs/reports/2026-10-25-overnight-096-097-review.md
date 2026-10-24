# 096—097夜间实施与独立审查

2026-10-25，北京时间。起点为远程`7cc8252`，本次用户明确要求在睡前安排可接续实验。本报告只记录真实开发、检查与发布状态，不代表正式Level1新GPU任务已执行。

## 最新科学结果

090C和三臂CPU表完成。新增独立审计从300场重聚全体、38组件、最大组件外、四种加权、H/I与SSE等117486个数值等式，无失败，最大归一化差约3.38e-13。精确输入与边界见[审计记录](../../HandOver/results/LEVEL1-FROZEN-ADAPTER-20261024-090-summary/overnight-local-review.json)。本地没有加载服务器原权重、原始NPZ或完整训练JSONL；分位数引用原逐点统计，未平均场景分位数。

C位置MAE590.7436→564.8340m（−4.3859%），RMSE3289.2144→3322.5300m。A/B比C的MAE差9.4445/15.9751m，P95/P99也更差，只有RMSE更小约36m。当前adapter配方按原出口关闭，C不自动晋升默认。S2L的I/H反写、全部属性改写、screen与candidate混淆及泛化“尾部更优”等勘误写入研究状态，原S2L不改。

新097只问第二seed的同样fresh-Adam/原样本重放预算干预是否产生一致取舍；它不是给090adapter补seed，也不证明充分训练。新096只做两个预定provider的已训练TAM/TFM系统组合，零训练。完整规格见[设计](../specs/overnight-system-and-budget-096-097.md)。

## 实际修改

- 096评价入口：两个完成来源在任一权重打开前做终态检查；复用公开单臂验收，094关联终点＋093融合终点，禁止假联合checkpoint。原094完整分数与阈值保留，新TAM评分为0；实际补全/TFM有新前向。重新准备的完整V浮点输入及参考数组对原093 final NPZ逐值核验。
- `evaluate_complete`两个可选参数默认关闭，旧行为保持。新增同预测组的支撑内插值anchor/节点等权mean控制，经度/航向圆均值未定义时保持缺失。无支撑外外推、完整缺失/重复人口保留。
- 096 CPU表：只读完整预测原件，核源V名单防两侧共同截断，冻结原阈值重建组，实际数组重聚精确尾部/组件/全人口/共同点。072C跨schema桥接仍未实现，不混排不同人口。
- 097独立公开配置与入口：seed21/073C严格绑定，fresh Adam4000，step0来自source-final，重放该seed原批次。旧090默认合同不放宽；薄CPU表分别比较各seed自身变化。
- 有限队列：固定3GPU＋4CPU项，已有原CPU表观察不重复；全局四个GPU领取上限，active service在CPU阶段仍保留UUID；idle-only、一次提交、依赖失败阻断下游但独立项继续、整批结束退出。systemd提交不确定时观察同一unit，失败终态释放slot，未知SUBMITTING保守保留。

## 实际验证

主代理执行：

```text
python -m unittest discover -s tests -q
655 tests / 144.348 s / OK
```

全仓运行后最后修订队列inactive/no-worker-status释放和缺失request归属保守处理边界，针对最终队列版本再次执行：

```text
python -m unittest tests.test_post088_overnight_queue -q
11 tests / 0.022 s / OK
```

不能把655+11简单相加声称666项不同测试；后者是必要边界修订后的定向复核。完整日志留本地测试输出，不作为正式实验结果。

实施代理实际完成096真实tiny链：TCM→两个TAM/两个TFM→两套完整系统→新CPU联表；包含无输入目标、缺参考、错policy/冻结权重、改变真实原生浮点输入、保护原088/093/094目录、无新增TAM评分/Adam、原checkpoint字节保持、超advisory继续。新入口及旧complete评价/CLI回归21项通过（7.324s）。

097实际tiny链覆盖两个seed各自原模型训练→追加训练→完整初末V→各自缓存表→双seed预算表；新097与旧090共8项通过（6.347s）。独立审查者另亲跑同组8项通过（5.851s），核真实073C回传的seed/profile与来源路径。

独立审查者复跑096入口＋比较器10项通过（7.206s）；新比较器7项通过（0.866s），包括不同覆盖/零交集、完整分位数、候选/阈值/分组/共同截断拒绝，以及torch.load强制失败仍可完成CPU比较。

两个独立审查发现并已修复：原TCM输出目录保护、普通控制支撑内插值语义、CPU准备阶段UUID保留、systemd提交不确定的同unit观察、inactive缺worker状态的slot释放。最终独立结论无剩余阻断；最终队列10项独立复跑也通过（0.038s）。新CLI解析、代码编译与diff whitespace检查通过。

## 未在本地验证的部分

本地CPU开发环境未执行正式Level1的096/097训练/完整评价，也无CUDA、服务器原卷/权重或真实systemd用户服务。队列逻辑测试与真实tiny算法检查不能证明服务器已启动；服务器须在一次托管后记录Linger、服务状态、实际argv和最终exit。原093/094活动checkout保持e718fca，不修改或重跑。

## 发布的实际限制

本次GitHub读main成功；`GIT_TERMINAL_PROMPT=0 git push --dry-run origin HEAD:main`因缺少HTTPS用户名凭据失败。GitHub插件以实际设计文档创建tree返回 **403 Resource not accessible by integration**。这是接口访问限制，**不是自动审批审查拒绝，也不是需要重新批准科研预算**。未尝试绕过访问控制、寻找私密凭据或SSH。

实现与交接将以增量git bundle交付。有写权限的本地环境合入并push后，服务器领取一次性夜间队列即可按固定依赖自动接续。没有真正的main发布及队列领取回执之前，096/097只记本地READY，不写RUNNING。


实施冻结为`21ce9dc908075cb9556d16fc4596d1c79d6af7b3`，新L2S与队列使用该版本；交接发布记录另提交，不修改已冻结实现。

最终追加的缺失request归属处理已独立只读复审，无剩余阻断；根代理最终11项队列检查通过，未扩大重跑整个模型测试。
