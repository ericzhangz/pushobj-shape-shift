# 共享响应规律反演：研究记录、代码与数值结果

2026-09-28 发布。主问题保持为：如何利用已完成的视觉作用经历，修订同一个预测器对未执行动作链的完整后果与相对后果？本目录收录数学分析、一次原生形成验证，以及必要的前期过程记录。

**本轮裁定：当前“稠密共享响应模型＋CSI 式源分拆”不晋升为主论文机制。** 普通多步适配已改善候选差分与固定池选择；源分拆尚未显示独立的优化或计算优势，支持拟合改善也未稳定迁移到完整未来预测。负面结果、审计修正和未通过的门槛均保留。

## 阅读入口

| 内容 | 文件 |
| --- | --- |
| 最终实验报告、对照及研究裁定 | [形成验证报告](shared_response_formation_20260928/REPORT.md) |
| 冻结数据边界、方法和验收条件 | [实验协议](shared_response_formation_20260928/RUN_PROTOCOL.md) |
| 数学建模与 CSI 等一手文献边界 | [反演分析报告](shared_response_inverse_analysis_20260928/REPORT.md) · [文献审查](shared_response_inverse_analysis_20260928/PRIOR_REVIEW.md) |
| 核心拟合与原生推演代码 | [formation.py](shared_response_formation_20260928/formation.py) |
| 独立评分与算术复核 | [score_formation.py](shared_response_formation_20260928/score_formation.py) · [review_score_cpu.py](shared_response_formation_20260928/review_score_cpu.py) |
| 核实后的数值结果 | [SUMMARY.csv](shared_response_formation_20260928/scored_verified/SUMMARY.csv) · [RESULTS.json](shared_response_formation_20260928/scored_verified/RESULTS.json) · [FINAL_METRICS.json](shared_response_formation_20260928/FINAL_METRICS.json) |
| 优化过程、算子检查及审计 | [TRAINING.json](shared_response_formation_20260928/TRAINING.json) · [OPERATOR_CHECKS.json](shared_response_formation_20260928/OPERATOR_CHECKS.json) · [完整性报告](shared_response_formation_20260928/INTEGRITY_REVIEW.md) |

六种方法共用四个开发案例、32条候选和合法已完成经历。以下为四案等权平均，数值越低越好：

| 方法 | 全未来视觉 MSE | 末端候选差分 MSE | 固定池 regret |
| --- | ---: | ---: | ---: |
| NATIVE | 0.063376 | 0.121884 | 0.013871 |
| PRIOR | 0.058716 | 0.114000 | 0.003955 |
| STEP_RIDGE | 0.061742 | 0.108389 | 0.002898 |
| DIRECT | 0.064909 | 0.105147 | 0.001663 |
| SOURCE μ=1 | 0.060912 | 0.107319 | 0.003610 |
| SOURCE μ=10 | 0.063202 | 0.105009 | 0.002376 |

这些是旧开发资产上的形成验证，不是新盲测或闭环成功率。严格旧代价重放门槛有4/32条未通过；敏感性检查中候选排序及方法平均 regret 顺序不变，详见完整性报告。没有新增环境交互、重训或修改 backbone。

## 过程记录

下列历史目录保留当时的 Markdown 论证和裁定；其文字反映各阶段认识，后续结论以形成验证报告为准。除当前评分必需的辅助代码外，不重新发布历史阶段的原始输入与完整实验包。

- [早期进展](research_progress_20260922/)
- [方向审查](research_direction_20260927/)与[重新立项讨论](cvpr_oral_reframe_20260927/)
- [原生交互验证](interaction_validation_20260927/)
- [数学重建](mathematical_reconstruction_20260927/)与[响应约束检验](object_response_constraints_20260927/)
- [开放问题与 ChordEdit 链路](research_discussion_20260927/)及[研究汇总](research_synthesis_20260927/)
- [共享响应反演分析](shared_response_inverse_analysis_20260928/)与[原生形成验证](shared_response_formation_20260928/)

## 发布范围与复现边界

本次上传 Markdown、Python、JSON 和标量评分 CSV。**不上传结果图、模型权重、张量缓存、实验场景、原始图像／动作轨迹、物体坐标轨迹或第三方论文全文。** 数据审计与准备清单使用公开摘要；第三方论文保留官方链接。历史链接所指文件未上传时已在正文标明。

代码依赖本仓库基线提交 `94fb16376261f204320fcc0aff5028e9dfff8e93`。[DEPENDENCIES.json](DEPENDENCIES.json) 标出依赖是否与基线一致；[runtime_snapshot/](runtime_snapshot/) 保存原执行环境中相对基线有变化的源码，路径对应仓库根目录。它是执行依赖快照，不是另一套新算法，也未覆盖仓库原有主路径。

完整重跑需要另行具备原模型权重、合法实验输入和原运行环境（Python 3.9.19、PyTorch 2.3.0+cu121）。请在独立执行副本中按依赖清单还原运行代码，并将源码里的原本地输入／输出路径映射到自己的环境。当前发布包用于阅读、数值复核和审查，不能只凭本目录重建未上传的实验场景。执行顺序和数据隔离要求见冻结协议。

`scored/` 保存初次评分；`scored_verified/` 保存修正调用计数与 terminal step 后的评分，`score_formation_executed.py` 保留初版代码。两者核心指标一致，少量像素读出差异在报告中披露。审计脚本有的仍需未公开张量缓存；不将“源码已公开”等同于“全部检查无需外部输入即可复跑”。

[PUBLISH_MANIFEST.json](PUBLISH_MANIFEST.json) 区分原执行文件与公开副本的 SHA-256。公开副本仅进行路径、链接、排除范围说明与换行整理；历史报告中的执行散列仍指原文件。数值结果保留原值。清单自身不包含自身散列。
