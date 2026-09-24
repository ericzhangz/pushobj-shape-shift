# Round 5：B1 factual update 与任务选择误差归因

本目录只包含 2026-09-24 这一轮的核心代码、可审计实验结果和结论报告。前几轮仍分别位于仓库根目录的 `round_2_reframe_v3/`、`round_3_use_matched_reference/` 和 `round_4_effect_geometry/`，本轮不改写它们。

## 这轮回答的问题

在已经执行的经历上，AdaJEPA predictor-only factual update 能降低 support loss，但是否也能稳定降低未执行候选之间的任务代价误差，并保住实际选择？归因对象是固定 B 计划池：同一 12 个锚点、每池 10 条五步候选、固定编码器和原始 B1 更新顺序。它不是 C 的首 chunk＋反馈续行，也不是带 warm-start 的原生闭环 MPC。

## 目录

- `code/`：B1 更新复现、真实前缀任务代价望远镜和候选对更新归因，以及合成恒等式测试。
- `results/`：12 锚点×10 候选的 Frozen/B1 分数、五步项、540 个候选对和汇总 JSON/CSV。未上传模型权重、latent sidecar 或环境缓存。
- `reports/`：实验合同与本轮结果分析。
- `prior_stage_context/`：前一阶段历史残差迁移与积分诊断的边界摘要；它们不被当作 B1 归因的根因。

## 结论入口

- [B1 任务误差归因报告](reports/TASK_ERROR_ATTRIBUTION_REPORT.md)
- [B1 归因协议](reports/B1_UPDATE_ATTRIBUTION_PROTOCOL.md)
- [完整运行摘要](results/RUN_SUMMARY.json)
- [按锚点摘要](results/ANCHOR_SUMMARY.csv)

## 核心结果

- 12/12 锚点 factual support loss 下降；原始 Frozen/B1 候选分数、真代价与选择记录全部复现。
- MPC2：regret 改善/恶化/不变为 0/4/2；MPC4 为 2/2/2。
- 全池候选对 MAE 的平均值在 MPC2 从 0.011167 降到 0.010013，在 MPC4 从 0.008700 降到 0.007266；这不保证决定性候选对的排序。
- 一个 MPC4 池中，全池 MAE 从 0.006621 降到 0.001128，但模型分差翻转，改选真代价高 0.000345 的候选。

本轮只定位“事实拟合—新动作相对后果—选择”的连接缺口，未提出或冻结新的处理器、能量原则、最优传输或 Jensen 机制。B 池已多轮查看，仅作开发性证据。

运行脚本依赖主仓库已有的 AdaJEPA 模型/数据接口和前几轮证据路径；本目录是可审计发布分区，不声称脱离这些依赖即可独立执行。
