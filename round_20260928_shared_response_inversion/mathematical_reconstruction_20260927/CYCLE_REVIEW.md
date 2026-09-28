# Cycle 原生干预独立复核

2026-09-27。审查 `cycle_probe.py`、协议、原生 rollout/预处理源码、封存 PT/JSON 与实际 CSV，并执行独立 CPU 复算。未运行 GPU 或模型，未修改原封存脚本、预测或评分结果，未使用本地 skills。

**裁定：主要执行与指标复算通过；有两项记录不足和一项表述边界。** 当前结果足以限制 `encoder(decoder(z))` 直接作为纠错重traction的解释。T0、L1 的候选选择改善必须与所有 case 的视觉误差、动作 contrast 误差恶化并列报告，不能用 regret 阳性替代机制验证。

## 1. 原生路径与信息边界

- FEEDBACK 使用既有 `model.rollout(..., observation_transition=transition)`。callback 调用同一个 `model.predict(h)`，只将新预测 visual 384 维经过 cycle 后替回，proprio 10 维直接取本次原生输出。后续动作插入、最近三帧截取和缓存 append 全在既有 `VWorldModel.rollout` 内完成；没有另造递推路径。
- FULL_REPLAY 共 32 条、160 predictor calls；FEEDBACK 同量；READOUT 在 FULL_REPLAY 完成后处理其预测 visual，不再调用 predictor。320 次 predictor、200 次 cycle forward、408 帧 cycle 的计数与循环结构一致。
- callback 的输入动作虽然未显式使用，但当前动作已经是 h 最后一帧的 action embedding；`model.predict(h)` 与原生路径同样获得它。原生未来 action overwrite 保留。因此不存在因为 callback 不引用参数 a 就漏掉动作的问题。
- 预测进程读取已有 native cache 中的合法 D、目标编码、候选动作和旧模型预测，并重新读取已完成的两段经历。代码没有读取 `EVALUATION_TARGETS.pt` 或 conditional 真历史缓存；未来真值只出现在独立 score 阶段。校准没有写入预测参数，实际 FEEDBACK 也是固定的无 D 拟合基线。
- 这四案仍是旧开发数据；“预测进程没有读未来目标”不能提升为整个研究从未见过这些结果。

## 2. 归一化核对与协议记录缺口

`Preprocessor.preprocess_obs_visual` 执行 RGB/255，所加载配置使用 `datasets.img_transforms.default_transform(img_size=224)`，之后 Normalize(mean=.5,std=.5)。输入已经 224×224，resize/crop 不改变坐标与大小。模型重构训练直接比较 decoder 输出和这一规范化图像。因此 cycle 将 decoder 浮点输出直接送入 encoder_transform/encoder.forward，**不进行第二次 RGB/255 或 Normalize，是正确的坐标约定**。

协议要求报告输入与重构范围，但原封存输出没有记录该项。独立审查补充读取原生缓存中的 `source_rgb` 和未量化 `source_decoded`，得到：

| case | 已完成输入归一化范围 | 旧未量化重构范围 | 重构超出 [-1,1] 比例 |
|---|---|---|---:|
| T0 | [-0.490196, 1] | [-0.537920, 1.103261] | 1.941% |
| T1 | [-0.490196, 1] | [-0.495947, 1.079799] | 1.647% |
| L0 | [-0.490196, 1] | [-0.506190, 1.072261] | 1.442% |
| L1 | [-0.490196, 1] | [-0.530423, 1.119163] | 1.596% |

这些是**旧事实重构缓存的范围**，不是本次未保存的每一个 cycle decoder 输出的范围，不能追溯声称原协议该记录已经完整执行。该补充支持坐标约定正确；decoder 输出并非被硬限制在 [-1,1]，不能把 sealed 中的 `normalized [-1,1]` 理解为输出值域保证。协议预先指定不 clipping，本次没有根据结果改变它。

第二项记录不足：CALIBRATION 保存了 C(z)-z、C(C(z))-C(z) 的 RMS，但没有保存 C(z)、C(C(z)) 原始 latent。独立 CPU 审查能验证校准 CSV 聚合，无法不调用模型地重算这两组原始差值。预测路径本身的原始 latent 已保存并可完全复核。

## 3. 独立复算

脚本：cycle_independent_review.py（本地附件或输入，未上传）。输出：CYCLE_REVIEW_NUMERICS.json（本地附件或输入，未上传）。

- 62 个受保护输入文件 SHA-256 全部一致；预测 PT hash 与 sealed 一致；评价 target hash 与评分 seal 一致。
- FULL_REPLAY 对旧 FULL 全部 visual/proprio 最大逐值差 **0**。三个 arm 的共同初始观测差 **0**。READOUT 的全路径 proprio 相对 FULL 差 **0**。
- 480 条逐步误差、336 个候选差距、336 个终点 latent contrast、12 个选择和全部 summary 重新计算。相同数值运算约定下，误差/contrast 最大差 `2.22e-16`，pair 最大差 0，summary 最大差 `1.11e-16`。
- 直接使用终点 latent 与旧合法目标编码，按 visual MSE + proprio MSE 独立双精度重算预测代价，最大差 `1.45e-7`；属于原 GPU float32 归约与 CPU double 归约差异。
- 评分沿用旧 `recorded_truth_cost`。以保存的实际编码重新计算真实目标代价，与旧值最大差 `8.37e-5`；四案最佳候选保持一致。T0、L1 的 FEEDBACK 选择均为对应真实最佳，因此用新计算真值时 regret 也仍为 0。
- FEEDBACK 第一预测 visual 与 READOUT 第一预测 visual 最大差 `3.34e-6`。二者的 cycle 分别按单帧和五帧 batch 执行，代码语义相同但不应声称逐值一致；第一预测 proprio 仍完全一致。

复核脚本首次运行曾错误地把 full-double subtraction 与原 scorer 的 float32 subtraction→double squared mean 按 `1e-12` 精度要求比较，导致审查断言失败。没有修改原结果或放松 `1e-12` 检查；修改审查以分别重算原运算与 full-double 运算，并保留两者差异：逐步 MSE 最大 `2.23e-8`、contrast MSE 最大 `1.75e-8`。原因与修订记录在审查 JSON。

## 4. “proprio 保留”的正确含义

cycle 干预不直接修改每次原生预测的 proprio，但视觉被回写后，未来原生 proprio 预测也会改变。FEEDBACK 相对 FULL 的全路径 proprio 最大绝对差依次为 T0 `0.04676`、T1 `0.13791`、L0 `0.08424`、L1 `0.08394`，第一步均为 0。

所以可以报告“只直接干预 visual，actions 按原生规则保持”；不能报告“FEEDBACK 的整条 proprio 轨迹保持不变”。这也是归因边界：原生目标 regret 的变化不能全部视作纯视觉 readout 变化。READOUT 才提供完整 proprio 轨迹不变的控制。

## 5. 校准与效能结论

校准 CSV 按 case 的均值：

| case | C(z)-z RMS | C(C(z))-C(z) RMS | 真实相邻帧差 RMS |
|---|---:|---:|---:|
| T0 | 0.05209 | 0.02656 | 0.05436 |
| T1 | 0.06366 | 0.04601 | 0.09685 |
| L0 | 0.04371 | 0.02772 | 0.04695 |
| L1 | 0.07477 | 0.05297 | 0.06313 |

第一、二列说明该 cycle 不严格保持真实编码，也不幂等。与真实帧差比较只能提供尺度背景；它不自动给出“足够小”的阈值或物理流形距离。不能利用重新居中、阻尼参数或挑选帧把此次固定规则改成已通过的投影。

| case | FULL→FEEDBACK visual MSE | FULL→FEEDBACK contrast MSE | FULL→FEEDBACK regret |
|---|---|---|---|
| T0 | 0.02543 → 0.03052 | 0.02967 → 0.05285 | 0.01190 → 0 |
| T1 | 0.12092 → 0.22927 | 0.04194 → 0.11632 | 0.001012 → 0.05722 |
| L0 | 0.009732 → 0.01657 | 0.01517 → 0.03794 | 0.002494 → 0.002494 |
| L1 | 0.22793 → 0.23268 | 0.71343 → 0.73393 | 1.31811 → 0 |

全部四案的 visual 和 contrast MSE 均恶化；两个选择改善、一个恶化、一个不变。pair-gap MAE 也须独立报告：T1/L1 有小幅改善，T0/L0 变差。这些指标评价对象不同，不能相互替代。

因此，此次不是“几何机制已经修复分支后果”的证据。它否定的是未经校准的当前 C 被直接当作合格重traction的实现方式，并支持把真实编码保持、作用方向保留及共享原生反馈作为后续候选必须解释的条件。它没有否定学习得到的其他重traction，也没有证明所有循环纠错无效。
