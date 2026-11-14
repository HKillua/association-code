# 离线观测基线重启审计

2026-11-14。验证限 MTAD。依据用户指定的 `research-workflow`、项目保存的飞书第一章与第三章核读记录，以及[每次实验的完整约定](../research_workflow.md#2-必须遵守的研究顺序)。本轮做来源重筛和既有记录审计。新的浏览器会话需要飞书登录，使用仓内已留的全文核读映射，未确认网页是否有新版本。 [飞书原文](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4WC7ctFPXnxd)

## 最早需要重来的步骤

| 时间与证据 | 当时完成的工作 | 对当前研究的限制 |
|---|---|---|
| 10 月 17–18 日，TSA-cTFER 与 TMTAFF 实现 | TSA-cTFER 按本地场景、时间词表和选模重建了离线编码器及成对分数。TMTAFF 写出 TCM、TAM、TFM、联合训练入口，实际原生时间与共同网格仍待闭合。工程规格已写明适配差异。 [TSA 初始规格](https://github.com/HKillua/association-code/blob/8176ca3/docs/tsactfer_adaptation.md) · [TMTAFF 初始规格](https://github.com/HKillua/association-code/blob/35e808d/docs/specs/complete-training-refactor.md) | 已找到最早的证据断点。代码可用和局部适配先于多个可信基线的共同协议整链实测。首要缺口在基线选择、原作者协议和可复建材料。 |
| 10 月下旬，TMTAFF 096A、107、113 | 096A 关联输出退化。107 的 TFM 实际训练 1,000 步，TAM 与联合训练更新均为零。113 用本地 056 产生关联。 [原件及口径](2026-11-12-tmtaff-protocol-gap-audit.md) · [实际快照](2026-10-29-fixed-endpoint-review.md) | 这些运行可定位本地退化，无法作为已成功复现的作者整链。相同分组和输入的小样本也只能定位这套快照的融合误差。 [C129](2026-11-13-tmtaff-same-group-support-review.md) |
| 10 月 30 日，PTFM 与 DeepAF 的 152 对照 | 共享 WS 分组和 V300，147、PTFM＋Mean、DeepAF 的位置 MAE 为 590.760／738.682／1,361.695 米，PTFM 的 RMSE、P99 和速航更低。 [实测记录](2026-11-10-offline-peer-route-review.md) | 保留条件融合实测。PTFM 原任务从已关联的双源航迹开始，DeepAF 原任务从含杂波雷达点迹开始。本地 DeepAF 的输入、输出及速航读出均有适配。 |

将 C143、C144 和依赖旧工作点的后继局部探索退出当前队列。旧分数、代码、结果、失败和反例仍可按原协议复查。当前状态的旧版本可从[冻结记录](https://github.com/HKillua/association-code/blob/8ced9f2e3ae918853ed157a7524f7e8965150b59/offline_observation/docs/research_status.md)查找。

## 重新检索与来源初筛

AnySearch 匿名查询三组，分别返回 7／8／8 条。检索式为 `MTAD Multisource Track Association Dataset offline full system track association fusion benchmark TMTAFF 2025 2026`、`MTAD Global AIS cross sensor track association 2025 2026 MIS-GNN Zhan BiFormer available code track association`、`marine multisensor track to track fusion 2025 2026 reproducible trajectory association fusion radar AIS source code non causal`。按 DOI 和题名去重，回查作者论文、数据声明及代码入口。近两年、引用、后继、原任务效果、材料可用性及 MTAD 输入适配分别核验。

| 来源和已核范围 | 作者任务与直接证据 | 去留及待补材料 |
|---|---|---|
| Zeng 等，TMTAFF，2026，§§2–3、表1–3、数据声明 | 离线整链覆盖 ITSA、MSTSA 和四属性融合。报告随机抽 3,000 场按 7:1:2 划分、长度 64、80 epochs 和验证选模。公开声明指向 MTAD 数据。 | 保留最贴近的整链复现对象。实际 Level 与场名单、窗、配准、分阶段训练预算及表3 mESD 程序仍缺。印刷式与残差标准差口径不同。 [原刊](https://www.mdpi.com/2076-3417/16/14/7251) · [逐项协议审计](2026-11-12-tmtaff-protocol-gap-audit.md) |
| 陈等，多源航迹关联融合，2025，§§3–5、式17–21、表2–4 | MTAD 上先用端点关联同源中断，再以 DTW 跨源匹配并加权融合。文中跨源 PCA/PFA 从80.83%／28.09%变为84.58%／14.89%，一源的同源 PFA 变差。§5.2 的融合证据为图示。 | 传统整链候选。点数比筛选的正文与图示方向有歧义，权重方差公式和执行阶段待核。复建须显式选择并声明规则，区分自动输出及人工修正。 [原刊](https://cjournal.hep.com.cn/1671-0673/CN/10.3969/j.issn.1671-0673.2025.05.015) · [已有全文分析](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/mtad_all_fusion_20261028.md) |
| Zhao 等，MIS-GNN，2026，正式刊版 pp.10–13 | MTAD 跨源关联，场景级 80:10:10，报告抽样配对 202,735／26,415／22,822。图由32个100秒节点构成，测试时跨源候选经 Hungarian 指派，报告 AF1 .9264。 | 进入 MSTSA 候选。待核场名单、空配规则、抽样测试对与全候选输出关系、图窗对应时间和代码。 [原刊](https://doi.org/10.1088/1361-6501/ae8929) · [本地原文](/Users/a1/Projects/biye-ronghe/docs/evidence/literature_scout_20261015/direct_sources/misgnn_2026.pdf) |
| Yuan 等，MVW-GNN，2025；Xu 等，MH-T2TA，2025 | 前者使用 MTAD Level1 场景切分，并另测 Level2。后者公开正文及算法附件，其比较含五个固定场景与 50 次 Monte Carlo。两者目标均为跨源航迹关联。 | 保留作独立结构的 MSTSA 候选。逐篇补预处理、空配、选模和输出差异；MH-T2TA 的生成口径不能直接当作已下载的原 Level1 名单。 [MVW-GNN](https://doi.org/10.1088/1361-6501/ae0497) · [MH-T2TA](https://jzus.zju.edu.cn/article.php?doi=10.1631%2FFITEE.2400340) |
| TSA-cTFER、PTFM、DeepAF，核对原任务与已有实现 | TSA-cTFER 用两阶段递推指派，本地有离线适配。PTFM 用已关联双源，DeepAF 关联和滤波杂波点迹。 | TSA-cTFER 适配留作关联开发对照，152 留作条件融合对照。作者原报值与我们的共同协议结果分别引用。 [TSA-cTFER](https://www.mdpi.com/2072-4292/16/18/3380) · [PTFM](https://www.mdpi.com/2079-9292/14/17/3461) · [DeepAF](https://www.mdpi.com/2226-4310/12/3/194) |
| 单源中断补查，TSAGLHA，2026，§4、表1和5 | 原文以 300–600 m/s 的三维模拟轨迹、1秒采样和3–7步中断建样本，完整模型 F1 .9012，去 Transformer .8799，去孪生结构 .8145。 | 仅作 ITSA 方法线索。现阶段的 ITSA 对照优先核陈等同源分支、TSA-cTFER 离线适配与原刊传统 MF 2-D TSA，避免候选清单被跨源论文占满。 [TSAGLHA 原文](https://journals.sagepub.com/doi/full/10.3233/ATDE260137) · [MF 2-D TSA 任务与原引用](https://www.mdpi.com/2076-3417/16/14/7251) |
| AnySearch 新线索，DRC-TCN、MSTrack、SCTA-Net | DRC-TCN 有作者代码，原文评估单源 AIS 船舶身份分类；MSTrack 需要雷达图像、可见光和 AIS，原数据是 WHUT-MSFVessel；SCTA-Net 本轮只找到题录。 | 前两项不进入 MTAD 同任务整链名单。SCTA-Net 待取原文后再判范围。 [DRC-TCN](https://www.mdpi.com/2077-1312/13/11/2129) · [作者代码](https://github.com/hoyon-lab/AIS_tracking) · [MSTrack](https://www.mdpi.com/1424-8220/25/7/2179) · [SCTA-Net](https://doi.org/10.1049/icp.2026.0995) |

2026-11-14 同源 Crossref `is-referenced-by-count` 返回 TMTAFF 0、MIS-GNN 0、MVW-GNN 2、PTFM 4、TSA-cTFER 12。其余查询遭遇限流或题录未覆盖。计数仅描述该索引，近作的后继和代码维护仍待核。 [Crossref 字段](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)

## 当前阶段和下一道比较

本轮来源初筛为 `partial`。论文原报值、按作者协议复现、按项目共同协议适配各自保存。主比较首先核每个方法能承担整链、ITSA、MSTSA、还是仅给定组融合。候选需要可定位的实现或充分算法描述、输入与输出人口、选模／阈值来源、空配／单例、关联分组及融合参考细则。由实际用途决定是否入表。

共同开发评价沿用已冻结的 [MTAD Level1 数据约定](../data_contract.md)和[指标协议](../mtad_data_protocols.md)，封存测试保持关闭。固定场景、可见量测、查询时刻与参考，整链各用自己的关联、分组和四属性输出，报告 ITSA/MSTSA、覆盖、重复和四属性误差与位置尾部。旧模型和透明传统方法可作为工程对照。只在适用模块下比较 TSA-cTFER、MIS-GNN、MVW-GNN、MH-T2TA、PTFM 与 DeepAF。按实际分层对齐开发场和训练曝光，已有 V300 暴露不得称新留场。

下一步先把 TMTAFF 整链、自动传统链路以及 ITSA、MSTSA 各自的对照核到可复建级别。随后用来源与已有实测建立 Mode C，冻结第一份基线复建 Idea 和执行合同。当前无新研究运行、基线定案或同行胜负。

## 来源复核与首轮立项

续轮用 AnySearch 运行8组通用、5组学术及2组代码查询，去重前收回119条。按同源中断、跨源关联、条件融合、完整自动链分别判断。查到的新基线优先采取以下定位。

| 用途 | 当前选型 |
|---|---|
| MTAD 作者整链 | TMTAFF、陈等自动分支保留。TMTAFF 已确认 Campus 存在有自身 TAM 的旧096A，300场中290场只剩一组，边接受率99.973%，位置覆盖3.36%。旧107没有完整联合链。陈等原表无数值四属性融合，公开材料的参数与评估场分离也未闭合。两者的原表仅作文献参照。 [TMTAFF核读](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_baseline_tmtaff_20261114.md) · [陈等核读](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_baseline_chen_20261114.md) |
| ITSA 同源中断 | 选 Chen Yiming 等2025真实岸基雷达的双向中点机制，对照陈等 MTAD 端点距离。先以双端 CV 单变量适配。学习参考为已有 TSA-cTFER 082 原 V300 的全节点对分数及本地 anchor_direct 分区；带保存阈值的 raw／decoded 宏 F1 为.4212／.3729，训练1,000步，选阈用同一 V，历史弱对照。 [HFSWR原文](https://www.mdpi.com/2072-4292/17/3/500) · [TSA-cTFER原文](https://www.mdpi.com/2072-4292/16/18/3380) |
| MSTSA 跨源关联 | 第一份合同以 DTW 与同刻距离建可归因工程对照。MH-T2TA 所用 LCSS 是后选独立传统基线。学习方法优先 IAGM-TAN 的显式空位与图匹配，原文的5,000／1,000场和80/20并不一致，作者代码及名单尚未找到。MIS-GNN 为另一评分骨架；GMvA 有公开代码但依赖 CCTV，只保留可移植的图机制。 [MH-T2TA](https://jzus.zju.edu.cn/article.php?doi=10.1631%2FFITEE.2400340) · [IAGM-TAN](https://doi.org/10.1016/j.neucom.2026.134264) · [MIS-GNN](https://doi.org/10.1088/1361-6501/ae8929) · [近两年分任务筛选](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_baseline_peers_20261114.md) |

Mode C2.60 已冻结。C145 在原 V300 用陈等的端点→跨源对齐结构，三臂各用自身分组，四臂用共同支持 Mean。记录四属性、覆盖、重复、全节点边及尾部。该合同的传统适配不包含作者插值、加权融合和人工步骤。 [Idea](../proposals/offline_baseline_restart_20261114.md) · [READY](../../HandOver/L2S-20261114-c145.md)
