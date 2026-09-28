# 本轮原生交互诊断独立复核

2026-09-27。未使用 skills，未运行模型、GPU、训练或环境；使用独立 CPU 脚本读取已封存张量与 CSV 重算。审查范围：初始预测/评分、评价修订，以及后验 conditional runner 的静态索引和全部追加数值。

## 裁定

**初始原生预测与评分通过完整性复核。结果只支持四个开发 case 的原生后果/上下文诊断，尚不足以判定共享事件边界错误，更不构成修订机制成功。** 可继续报告可靠的大尺度视觉异常，但精细事件归因受 decoder 误差、宏时间分辨率和缺乏因果干预共同限制。

复算代码：independent_cpu_review.py（历史本地产物，未纳入本次上传）；机器结果：INDEPENDENT_RECOMPUTATION.json（历史本地产物，未纳入本次上传）。未修改 root 的 runner 或 scorer。

## 已独立验证

- 32/32 候选动作张量逐元素相等，32/32 sidecar 与缓存评价目标相等，32/32 起点 RGB/proprio 与合法 query 相等。每个 case 均为八个不同动作，没有完全重复动作冒充独立候选。
- 输入 manifest、评价 provenance 与封存预测 SHA256 全部匹配。
- 320 条逐步误差从 latent 张量重新计算一致；224 条 pair 数据均来自每 case/arm 的 28 个无序对，没有反向对重复扩样；八条选择/遗憾记录重新计算一致。
- 时间对齐正确：环境 control_hz=10，旧 oracle 以 prefix_env_len::frameskip 保存；frameskip=5，所以预测/sidecar 第 k 点对应 query 后 5k 控制步，即 0.5k 秒。单宏区间内的接触新增/消失仍无法从端点排除。
- 原始代价复算差最大约 8.4e-5；四个 case 的最优真值候选不变，所有真实 pair 排序均不变。独立按两个浮点分量相加所得最大值 8.3804e-5 与 scorer 的 float32 total 最大值 8.3923e-5 相差约 1.2e-7，属于求和舍入。

## 代价差与上下文结论

CSV 的 gap_error 定义为 predicted_gap−true_gap，因此等于总 proposal 中 E0 的负值；CSV 没有把它误标 E0，反序条件 true_gap*predicted_gap<0 正确。后续若画 K/E 图需沿用这一符号区别。

| case | SINGLE pair MAE | FULL pair MAE | SINGLE→FULL 反序对数 | SINGLE→FULL 遗憾 |
|---|---:|---:|---:|---:|
| T0 | 0.019553 | 0.018295 | 4→9 / 28 | 0.003113→0.011898 |
| T1 | 0.050492 | 0.047886 | 8→6 / 28 | 0.001012→0.001012 |
| L0 | 0.080759 | 0.088127 | 6→6 / 28 | 0.002494→0.002494 |
| L1 | 0.712728 | 0.709346 | 12→12 / 28 | 1.318108→1.318108 |

FULL 是三帧合法历史输入，SINGLE 是单帧当前输入，两者均为冻结 F0。FULL 没有形成新的 F_D；箭头只能标为上下文对照。这里的遗憾是**候选被实际执行后、按原生 latent objective 计算的候选池遗憾**，不是物理状态距离遗憾，也不是最终任务成功率差。selected_recorded_success 必须分报。

## 读出类别与相关样本

排除每条候选重复出现的 query 起点后，每个 arm 有 160 个未来宏采样：

| arm | near | separated | ambiguous | unreadable |
|---|---:|---:|---:|---:|
| REAL | 113 | 37 | 10 | 0 |
| RECON_REAL | 114 | 36 | 3 | 7 |
| SINGLE | 116 | 39 | 5 | 0 |
| FULL | 124 | 35 | 1 | 0 |

含重复起点的 192 个真实/重构配对中，185 个双侧可读，175/185 类别相同，没有 near↔separated 直接翻转；可见 gap 误差均值 1.5166 px，最大 14.3474 px。读出失败的七点保留在分母里，不能删掉后报告总体准确率。175/185 也不是通用事件分类准确率：类别不均衡、样本相关，且原生预测 latent 分布上的读出不由真实重构保证。

以上类别阈值只表示可见邻接，不表示物理接触力/guard。存在大尺度分离异常时可逐例报告，但不能由这些类别数量直接推导事件偏移、接触时长或共享边界位移。

四个 case 是反复检查过的开发资产。32 个候选、各 horizon、224 个 pair、重复 query 起点均非独立实验。报告描述性逐 case 结果合理；按 pair 或 frame 做独立样本显著性检验不合理。

## 两处执行异常的处理

1. 初版 scorer 在 inverse_proprio 的 Identity 断言失败。代码顺序与保留日志一致：此 scorer 尚未打开 ORACLE_COSTS 或 branch sidecar。实际 LayerNorm 破坏所提简单仿射逆，移除该辅助逆而继续原 RGB/latent 诊断是正确修复；不能称该逆通过了验证。原脚本、失败日志及空输出目录保留，原预测未重跑。
2. 预测封存 JSON 的 encoder_calls=0 是计数 hook 错误。encode_obs 直接调用 encoder.forward，跳过 module hook。根据执行结构重建为 76 encoder forwards / 176 frames；这是静态调用核算而非 hook 实测。320 predictor calls 和 72 decoder calls / 428 frames 的 hook 路径正常。原错误计数应保留并在报告中明确更正。

此外，预测前资产盘点已打开 88 个 sidecar 来读取 shape/dtype 和核对共同起点；不应笼统声称整个研究过程从未打开 query 文件。准确表述为：预测 runner 不读取未来结果；资产盘点只导出 schema/共同起点，未来 outcome 未用于挑选候选、拟合阈值或训练。当前四 case 本来也不是盲测。

## 后验 conditional 追加：静态索引与数值复核通过

CONDITIONAL_DIAGNOSTIC_PROTOCOL.md 明确记录追加源于已见 L1/fact 异常；这不是原冻结协议的确认结果。索引正确：joined=[D宏0,D宏1,query0,query1,...,query5]；k=1 的 [0:3] 配 [D0,D1,U0] 预测 query1，k=5 的 [4:7] 配 [U2,U3,U4] 预测 query5。被预测的 truth[k] 未进入其输入。

这种 actual-prefix teacher forcing 使用原查询时未知的分支真实过去，适合作局部响应诊断，不能生成可部署整段计划性能、修订后 regret 或 event oracle。它同时刷新全状态证据，不能单独区分时序、强度、几何和前缀偏差。

已运行独立 CPU independent_conditional_review.py（历史本地产物，未纳入本次上传）。结果 INDEPENDENT_CONDITIONAL_RECOMPUTATION.json（历史本地产物，未纳入本次上传） 为 PASS：160 条 visual/proprio 误差从保存张量重算一致，输入和输出 SHA256 全部匹配；32 个 k=1 的 visual/proprio latent 与原 FULL 第一预测严格逐元素一致，最大差为 0。

| 统计 | FULL 整段递推 | TF 条件一步 |
|---|---:|---:|
| 全部四 case 的未来 visual MSE 均值 | 0.0960039024 | 0.0137364609 |
| L1 未来 visual MSE 均值 | 0.2279312739 | 0.0188106361 |
| L1/fact 末步可见 centroid 误差 | 45.89493 px | 5.89888 px |
| L1/fact 末步可见 gap | 7.82599 px | −0.51629 px |

TF 的 L1/fact 第四步可见 centroid 误差仍为 8.58041 px。这说明更准确的输入前缀能消除该个案大部分后期异常，同时仍有局部预测/读出误差；不能声称纯事件边界病因已被验证，或把表格称作方法性能收益。原来的事件缩放关系依然未检验。

审查过程另有一次过强校验：最初同时要求不同 decoder batch 大小（FULL 六帧、TF 一帧）的 RGB 浮点输出 bitwise 相同。该断言失败，已保留初版 reviewer 脚本。拆开核算后，latent 差严格为 0，decoded 浮点最大差为 1.1116e-5；该读出差异独立报告，不将 latent parity 扩张成整个 decoder 输出完全相同。该检查没有重跑模型或改动实验产物。
