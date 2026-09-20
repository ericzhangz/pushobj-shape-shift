# Reframe V3 — C 阶段共同续行报告

## 阶段判定

**`DECISION_VALUE_NOT_SHOWN`。**

A 阶段的有限池选择损失没有在本轮 C 设计中稳定转化为实际执行决策价值；B1 predictor-only shadow adaptation 的第一次选择也没有优于 Frozen。该判定只针对 6 个 late anchors、当前固定 pool 和共同续行协议。

## 因果设计与资格

每个 late anchor 在执行环境分支前先冻结 manifest，三条逻辑臂为：

1. Frozen 在固定 pool 中的模型最优计划；
2. B1 predictor-only shadow adaptation 在同一 pool 中的模型最优计划；
3. 同一 pool 的环境 oracle-best 整条计划，作为诊断权限而非部署算法。

每臂只执行所选计划的第一个 model chunk；之后统一切回相同 Frozen checkpoint、相同分支无关 RNG、零 action warmstart，并执行 4 次新 replan。完整前缀从原初态重放。

- 18 条逻辑分支中有 1 对第一次物理干预完全相同，去重为 17 条物理分支。
- 17 条环境 rollouts；340 个 prefix replay steps、85 个 first-decision steps、340 个 tail steps，共 765 个 environment steps。
- 68 次 tail replans，共 6,800 个 world-model rollout/backward steps。
- 全部 anchor 前缀精确恢复；对应 tail RNG 起点一致；Frozen 模型张量版本保持不变；无非有限值。

## B1-adapted 第一次选择相对 Frozen

第一次 chunk 后，6/6 anchors 的几何距离和固定参照终端代价数值都下降，其中若干差异接近数值尺度。这种即时差异没有在共同闭环续行后形成稳定收益：

| 最终指标 | 改善 | 不变 | 恶化 | mean delta | median delta |
|---|---:|---:|---:|---:|---:|
| environment state distance | 2 | 0 | 4 | +0.312898 | +0.00274086 |
| fixed-reference terminal cost | 0 | 3 | 3 | +0.000296614 | +0.000155052 |
| success | — | 6 | — | 0 | 0 |

正值表示 adapted 第一次选择最终更差。故“更新改变了选择”不等于“更新获得了决策价值”。

## Pool-oracle 第一次选择相对 Frozen

即便使用整条记录计划的环境 oracle-best，第一次 chunk 在 5 个 anchors 上即时改善、1 个完全相同，最终结果仍然混合：

| 最终指标 | 改善 | 不变 | 恶化 | mean delta | median delta |
|---|---:|---:|---:|---:|---:|
| environment state distance | 2 | 1 | 3 | -0.715592 | +0.155066 |
| fixed-reference terminal cost | 2 | 1 | 3 | -0.000565058 | +0.000307272 |
| success | — | 6 | — | 0 | 0 |

均值由少数大改善影响，而中位数方向更差；不能据此声称稳定决策价值。

## 定位到的问题

C 暴露的是一个测量层次错位：A 的 `R_selected` 比较的是**整条开环计划在固定记录池中的总后果**，实际 MPC 只执行其第一个 chunk，随后重新规划。一个计划可以在完整开环执行下是 pool oracle-best，但其第一个 chunk 不一定在统一闭环续行下最好。后续 replan 还可能吸收、覆盖或反转即时差异。

所以当前最强问题陈述应分两层：

- 已证实：AdaJEPA 的模型目标在有限开环候选池上经常误排真实后果，且 factual support 拟合改善不稳定修复这种误排。
- 未证实：这种整计划误排是 PushObj 闭环控制的任务瓶颈，或 pool-oracle 计划的首 chunk 具有稳定可执行价值。

共同续行无稳定收益，因此协议中的 native-tail warmstart 补充条件没有满足，本轮未运行该可选分支。这样也避免把候选自身尾部/controller memory 重新带入首次决策的因果比较。
