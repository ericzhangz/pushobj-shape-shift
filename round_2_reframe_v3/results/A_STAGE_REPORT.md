# Reframe V3 — A 阶段审计报告

## 阶段判定

**SELECTION_GAP_SHOWN；进入 B，但尚未证明 DECISION_VALUE。**

当前证据显示：在每个真实 replan 的同一 10-endpoint 记录池中，模型最优候选经常不是环境最优候选，而且差距不是“没有找到这十个已记录候选中模型最偏好的方案”造成的。它不证明连续动作域或全部迭代上的优化器已经收敛，也不证明某个适应方法能修复这个缺口，或 pool oracle 最优在闭环共同续行下仍占优；后两点分别留给 B、C。

## A0：插桩工程资格

冻结、官方 AdaJEPA（predictor+encoder）和 predictor-only 三条臂各完成一对 `ON/OFF` 短运行。每条运行包含一次真实反馈更新并进入下一次 replan（冻结臂保持零更新作为阴性控制）。比较覆盖：

- 返回动作及长度；
- 适应参数和模块 buffers；
- observation/action/score 经验 buffers；
- Python、NumPy、torch CPU/CUDA RNG 状态；
- 两个 post-feedback 时间点的元数据。

三组比较均逐位通过，差异列表为空：

- `qualification/frozen_comparison_rngfixed.json`
- `qualification/official_comparison.json`
- `qualification/predictor_comparison.json`

初次冻结 ON 运行暴露 NumPy RNG 被 fresh replay environment 构造消耗。该运行保留为失败证据；修复是在 replay oracle 边界保存并恢复所有全局 RNG 状态，随后重跑通过。没有通过放宽比较准则规避问题。

这三组新资格运行证明的是**修复后插桩路径**的 ON/OFF 等价性。下面的 800 个历史 pools 来自修复前完成的 15 条 runs，不能把新 RNG 合同追溯为旧数据已经验证过的属性。旧失败只观察到 NumPy RNG 状态差异，而动作、参数与经验均一致；因此当前没有旧数据已失效的证据，但也不把它写成通过了新的 RNG 精确合同。

有效六次资格运行的已知预算为 150 official-evaluation environment steps；三次 ON 运行另有 1,665 oracle environment steps。修复前的完整失败 ON 运行另耗 25 official-evaluation steps 和 555 oracle steps。更早两次启动失败没有完成预算记录，也没有作为有效证据计入。

## 原始资产与代数资格

- 输入：4,000 个 paired GD records，组成 800 个 replans、每个 10 个 before/after endpoints。
- 21 个 matched initial/goal cases 全部通过跨臂映射检查。
- 800 个实际返回 prefix 全部与 `g99_after` 的首个执行 chunk 精确一致。
- 800 个 `g99_after` oracle 分支的起点及首个 model chunk 结束后的 RGB、proprio、state，全部与真实已执行片段的 frame 0 / frame 5 逐元素一致。oracle sidecar 只保存 model-stride 帧，因此不声称比较了中间四个环境子步的图像。
- 800 个 pool 均有 10 个不同的 action tensors；精确 tensor 去重不改变任何 selection regret。
- 缺失/重复 GD、混合版本、非有限值、delta closure 均通过严格校验。
- 收益恒等式最大绝对残差为 0；两个 regret bound 的最小 slack 都为 0.000239891000092。

## 主要结果

原始 `epsilon=1e-6` 主分析保持不变：

| 指标 | 结果 |
|---|---:|
| replans | 800 |
| `R_selected > epsilon` | 688 / 800 |
| 考虑模型数值并列后 `R_selected_tie_min > epsilon` | 682 / 800 |
| 模型 final-record 优化 gap `eta_m > epsilon` | 0 / 800 |
| `eta_m` 最大值 | 7.45e-9 |
| 模型预测整次优化改善 | 800 / 800 |
| 真实后果改善 | 658 / 800 |
| 真实后果恶化 | 142 / 800 |

为避免混合 terminal 与 full-horizon 的绝对量纲，使用每个 pool 自身环境代价跨度归一化后，tie-aware selection regret 的均值为 0.1807，中位数为 0.0747，90 分位为 0.5450。21/21 个 matched cases 都出现过 tie-robust selection loss；case 内发生比例的中位数为 0.85。

`R_selected` 的绝对值必须按 objective stage 分开看：

| objective stage | n | mean | median | max | mean / pool span | median / pool span |
|---|---:|---:|---:|---:|---:|---:|
| full-horizon | 488 | 0.002137 | 0.000377 | 0.116505 | 0.1722 | 0.0792 |
| terminal | 312 | 0.083853 | 0.005964 | 2.096156 | 0.1939 | 0.0712 |

完整优化通常确实改善真实后果，因此不能把结果概括为“越优化越差”：

- 540 个 replan：最终真实后果改善，但仍放弃了同池内更好的候选；
- 142 个 replan：模型预测改善，最终真实后果却恶化，且存在 tie-robust pool regret；
- 118 个 replan：最终真实后果改善，模型并列集合内包含环境最优或差异不超过 epsilon。

模型最优全部落在最后记录点：675 个为 `g99_after`，125 个为与其数值并列的 `g99_before`。环境最优则分散在全部十个 endpoints。这和 `eta_m` 近零共同排除了“只是没有走到这十个已记录候选中模型自己偏好的方案”这一解释；不外推为连续动作域的全局收敛。

## 与旧命题和后续阶段的关系

旧的数学反例仍成立，但本轮不再用局部单步反转直接推出适应失败。A 阶段的新支持是有限记录池上的真实排序/选择损失；不支持的是：

- 不支持把局部 response calibration 当成已经确定的主矛盾；
- 不支持把三条 on-policy arm 的差异解释成适应的因果效果；
- 不支持把 full-open pool oracle best 当成闭环 oracle；
- 不支持任何新算法优势或一般化结论。

B 阶段因此必须使用 Frozen donor 的共同 anchors、共同候选池和共同事实经历，比较同一新 query 上 predictor-only shadow update 前后的排序与 `R_selected`。C 再检验这种 pool 选择差异能否转成共同 continuation 下的实际决策价值。B 若失败，只否决本轮所测学习器、数据构造或规模，不构成禁止其他最小机制推导的总门槛。

## 可复核产物

- `offline/path_metrics.csv`：4,000 个 after-node 路径快照及 best-so-far。
- `offline/selection_metrics.csv`：800 个 pool 的 selection/optimization 指标。
- `offline/case_metrics.csv`：120 个 arm-stage-case 聚合散点行。
- `offline/strata_summary.csv`：按 phase/split/arm/stage 的 case-cluster summaries。
- `offline/validation.json`：结构与代数校验。
- `asset_validation_continuation/validation.json` 及对应 CSV：case、prefix、首 chunk 续行、action tensor 资产验证。
