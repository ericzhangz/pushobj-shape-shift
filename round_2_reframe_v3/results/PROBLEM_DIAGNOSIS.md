# AdaJEPA PushObj：问题定位

## Introduction-style：先说真实困难

这个系统里最直接的困难不是“世界模型预测误差很大”，而是下面这个可观察事实：

> 梯度规划几乎总能把模型内部目标继续降下去；真实经历更新也确实能把 factual support loss 降下去；但最终选出的候选并没有因此更可靠地对应真实环境中的更好后果。

800 个 replans 中，模型优化目标 800/800 改善，记录池内 optimizer gap 在 `1e-6` 下为 0/800；然而 682/800 仍有排除数值并列后的环境选择 regret。另一方面，完整优化又在 658/800 上改善真实后果。因此真实困难不是“optimizer 没工作”，也不是“越优化越差”，而是：

**optimizer 在忠实地优化一个不能稳定保持真实候选次序的目标。**

## Structural diagnosis：问题断在两个接口

令固定候选池为 `P`，模型和环境对整条开环计划的代价分别为 `c_m(p)`、`c_e^open(p)`：

```text
p_m = argmin c_m(p)
p_e = argmin c_e^open(p)
R_plan = c_e^open(p_m) - c_e^open(p_e)
```

A 阶段证明 `R_plan` 经常显著非零。它定位了第一个断点：

### 断点一：factual 拟合不是 counterfactual 排序

官方更新最小化已经执行经历 `E` 上的 `L_fact(theta; E)`；planner 需要的却是同一新 query 中多个未执行候选之间的**相对次序**。平均 factual loss 下降既不约束各候选误差的差值，也不保证 argmin 不被翻转。

B 阶段把这个区分变成了同锚点因果检查：12/12 support loss 下降，11/12 候选选择改变，但 tie-aware regret 只有 2 改善、6 恶化、4 不变。也就是说，经历已经写入模型，问题不是“没有适应”；问题是**写入方向与选择所需的相对误差没有稳定对齐**。

但实际 MPC 并不执行整条 `p`。它只执行首 chunk `q(p)`，随后重新规划。令共同续行下的最终真实结果为：

```text
J_cont(q) = outcome after executing q, then using the same continuation controller
```

实际决策需要比较 `J_cont(q(p))`，而 A 比较的是 `c_e^open(p)`。这定位了第二个断点：

### 断点二：整计划 oracle 不是首 chunk 的闭环 oracle

整条计划的优势可能来自后四个将被 MPC 丢弃的 chunks；后续 replan 也可能吸收、覆盖或反转首 chunk 的差异。因此 `R_plan > 0` 不推出执行该 pool-oracle 计划的首 chunk 会改善最终任务。

C 阶段直接验证了这一点。adapted 和 pool-oracle 的首 chunk 都产生了即时几何改善，但共同续行后结果混合：adapted 相对 Frozen 的最终 state distance 为 2 改善、4 恶化；连 pool-oracle 也只有 2 改善、3 恶化、1 相同，success 净变化均为 0。

## 当前最准确的问题表述

因此，这不是一个已经闭合的“模型误差导致任务失败”故事，而是一个两段式代理错位：

```text
真实经历
  -> factual support loss          （确实下降）
  -> 新候选的相对模型排序           （不稳定改善）
  -> 整条开环计划的环境排序          （存在广泛失配）
  -> 被实际执行的首 chunk            （只保留计划前缀）
  -> 共同闭环续行后的最终任务结果     （未显示稳定价值）
```

已摸清的核心问题是前半段：**训练目标改善与规划所需的 counterfactual 排序不对齐。**

尚未闭合的是后半段：**整计划排序失配是否构成 receding-horizon 的 actionable bottleneck。** 当前 C 证据是否定“已经证明”的，而不是证明这种瓶颈永远不存在。

## Appendix-style：因素排查

| 因素 | 检查 | 结论 |
|---|---|---|
| 插桩改变主路径 | 三臂 ON/OFF 比较动作、参数/buffers、经验与全部 RNG | 修复后的路径逐位一致；不是当前结果来源 |
| 历史数据资格 | 新资格运行发生在旧 800 pools 之后 | 旧数据无失效证据，但不追溯声称通过新 RNG 合同 |
| 候选重复 | 800 pools 的 action tensor 精确去重 | 每池 10 个均不同；regret 不由重复制造 |
| 记录池 optimizer 未收敛 | 计算 `eta_m` | 0/800 超阈值；只排除池内搜索解释，不证明连续域全局收敛 |
| 数值并列 | tie-aware regret | 682/800 仍为正；不是浮点并列假象 |
| objective 量纲混合 | terminal/full-horizon 分层并做 pool-span 归一化 | 失配在两层都存在；不比较跨层原始绝对值 |
| 经历没有写入 | 共同 support loss | 12/12 下降；该解释被排除 |
| 只是更新次数更多 | `E1->E2` 对 `E1->E1`、`E2->E2` | 净收益只在 1/6、同一 anchor；不可复用 |
| encoder 表示漂移 | 主分析固定 encoder；完整适应单列 | 主 B1 可同版本比较；完整适应的跨表示 contrast 不作解释 |
| 样本伪独立 | A 按 21 cases、B 按 6 cases 复核 | A 为 21/21 cases；B 仍是小样本混合结果 |
| 环境重放不精确 | prefix、首 chunk 起止帧、action 精确检查 | 已保存层级全部一致；不声称检查未保存的 4 个子步图像 |
| 即时动作没有物理影响 | C 的 first-chunk native metrics | adapted 6/6 数值下降，oracle 5/6 下降；动作差异确实进入环境 |
| 闭环重规划 | 首 chunk 后统一模型、RNG、warmstart 规则 | 最终优势消失/反转；这是 plan-regret 不能直连任务的关键因素 |
| success 指标灵敏度 | C 的 success delta | 6 anchors 全为 0；当前样本不能证明成功率后果 |
| controller memory | C 使用零 action warmstart | 因果隔离成立，但不是 native deployment 成绩；native-tail 条件未满足，未扩跑 |
| pool 覆盖 | `P` 由 Frozen 优化路径产生 | 结论只对该有限池成立，不是全动作空间 oracle |
| 第二宿主 | 本地 LeWM 代码、配置、checkpoint 清点 | 资产缺失，NOT_RUN；没有跨宿主一般性 |

## 停止线

本轮应停止把 `R_plan` 直接称为任务瓶颈，也不应从 support loss 下降推出适应有效。问题已经定位到两个代理接口；在后半段的 decision value 被证明之前，不进入修正讨论。
