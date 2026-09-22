# Round 4：Effect Geometry and Processing Interface

本目录只包含 2026-09-22 本轮新增的代码、实验结果和分析汇报。前两轮仍分别位于仓库根目录的 `round_2_reframe_v3/` 与 `round_3_use_matched_reference/`，本轮不改写它们。

## 目录

- `code/`：G0 来源/真值合同、G1 向量传播/几何/hybrid、G2 固定动作处理接口及本轮测试。
- `results/`：本轮的 JSON、CSV、日志和失败记录；原始 `.pt`/`.npz` checkpoint/latent sidecar 不上传，避免把权重和二进制缓存混入代码结果提交。
- `reports/`：实验计划、执行跟踪、代码审查、Round 4 结果报告、处理接口报告和当前未冻结的机制草案。

## 结论入口

- [Round 4 向量效应报告](reports/ROUND4_REPORT.md)
- [固定动作处理接口报告](reports/PROCESSING_INTERFACE_REPORT.md)
- [完整实验结果摘要](reports/EXPERIMENT_RESULTS.md)
- [机制位置说明（未冻结）](reports/MECHANISM_BRIEF.md)

G0：120/120 B 真值重编码通过。G1：17 条物理 C 分支、255 次传播 transition，并按预先写明必要性追加 68 次 hybrid；几何恒等式闭合但成因是 mixed。G2：两条真实 checkpoint 观察的固定动作 identity/非 identity 接口通过，完整 100 步 GD parity 明确为 `NOT_RUN`。G3 处理器训练和新环境确认不在本次提交中。

运行代码依赖仓库已有的模型/数据接口以及前几轮保存的证据路径；本目录是本轮可审计提交分区，不声称脱离这些依赖即可独立执行。
