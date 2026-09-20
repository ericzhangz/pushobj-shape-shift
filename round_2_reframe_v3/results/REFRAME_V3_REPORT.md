# AdaJEPA PushObj Reframe V3 最终报告

日期：2026-09-20

## 总判定

本轮最终层级为：

**`SELECTION_GAP_SHOWN -> DECISION_VALUE_NOT_SHOWN`**

`REUSABLE_CORRECTION_CANDIDATE` 未达到；第二宿主为 `NOT_RUN_ASSETS_UNAVAILABLE`。

本轮找到的问题不是“适应一定有害”，也不是“模型优化越久一定越差”。证据支持一个更窄但真实的诊断：

1. AdaJEPA 规划器在 800 个真实 replans 的有限 10-endpoint 记录池中，经常用模型代价选错真实后果更好的候选；
2. 官方 predictor-only factual 更新能降低已经历片段上的 support loss，却不能在同一新 query 上可靠降低这种排序损失；
3. 但整条开环计划的 pool regret 与 MPC 实际只执行的第一个 chunk 之间存在层次错位。共同闭环续行没有显示稳定任务收益，连整计划 pool oracle 的首 chunk 也没有稳定胜出。

因此，当前已定位的是**有限池模型排序问题及 factual objective 与排序目标的不对齐**；尚未定位到一个已证明影响闭环任务结果的决策瓶颈。

若只阅读问题论证，见 `PROBLEM_DIAGNOSIS.md`：它按 Introduction-style 的现象、structural diagnosis 的双接口断点、以及 Appendix-style 因素排查组织，不涉及修正方法。

## 1. 旧数学反例仍成立；实际新支持/不支持了什么？

旧结论仍成立：只控制点值或函数值误差，不能无条件控制动作导数。V3 没有推翻该数学事实。

新的真实数据支持：

- 在固定的有限候选池上，模型偏好与环境偏好经常不同；
- 这种差异不是简单的“记录池内 optimizer 还没找到模型自己偏好的候选”；
- factual support loss 的下降与候选排序改善不是同一个目标；
- 整计划排序与 receding-horizon 的首 chunk 决策价值也不是同一个对象。

新的真实数据不支持：

- 不支持从局部 response/Jacobian 反例直接推出实际适应失败；
- 不支持把三条原 on-policy arms 的差异解释为适应因果效果；
- 不支持把有限池的整计划 oracle 当作闭环 oracle；
- 不支持任何修正算法、跨任务规律或跨宿主一般性。

## 2. 完整优化和模型选择是否产生有意义的 oracle 池损失？

是，在**有限记录池**含义下有清晰、非数值噪声的选择损失。

| A 阶段指标 | 结果 |
|---|---:|
| replans | 800 |
| `R_selected > 1e-6` | 688 / 800 |
| tie-aware `R_selected_tie_min > 1e-6` | 682 / 800 |
| recorded-pool `eta_m > 1e-6` | 0 / 800 |
| `eta_m` 最大值 | 7.45e-9 |
| 模型预测整次优化改善 | 800 / 800 |
| 真实后果改善 | 658 / 800 |
| 真实后果恶化 | 142 / 800 |

按每个 pool 的环境代价跨度归一化后，tie-aware regret 均值为 0.1807、中位数 0.0747、90 分位 0.5450；21/21 matched cases 都出现过 robust selection loss，case 内比例中位数 0.85。

但“有选择损失”不等于“完整优化通常伤害任务”。540 个 replans 是真实后果改善但放弃了同池更好的候选；142 个是模型预测改善、真实后果恶化且存在 robust regret；另有 118 个改善且没有 robust regret。问题是经常未选到 pool 内更好的方案，而不是普遍越优化越坏。

## 3. 与 optimizer 未收敛、native metric、closed-loop correction 的关系

### Optimizer

`eta_m` 在 800/800 上不超过主阈值，且模型最优只落在最后的 `g99_after` 或与其数值并列的 `g99_before`。这排除了“只是没找到十个已记录 endpoints 中模型最偏好的方案”。它不证明连续动作域全局收敛，也不证明 100 步之外没有更低模型代价。

### Native metric

terminal 与 full-horizon 环境代价量纲不同，绝对 regret 分开报告；跨阶段只使用 pool-span 归一化统计。C 另外直接报告官方环境几何 state distance、success 和固定参照 terminal cost。

完整官方适应补充臂更新了 encoder，其模型代价不能与 Frozen encoder 下的 `c_env_ref` 解释为同版本 contrast calibration；因此只保留固定环境后果的选择比较，不用跨表示误差宣称 calibration 改善。

### Closed-loop correction

C 隔离第一次选择后统一 Frozen 模型、RNG 和零 action warmstart，执行 4 次共同 replan。B1-adapted 相对 Frozen 的最终 state distance 为 2 改善、4 恶化，固定参照 terminal cost 为 0 改善、3 恶化、3 在阈值内不变，success 净变化为 0。

更强的诊断是：整计划 pool-oracle 的首 chunk 相对 Frozen，最终 state distance 也只有 2 改善、3 恶化、1 相同；固定参照 cost 为 2 改善、3 恶化、1 相同。共同 replan 可以吸收或反转第一 chunk 的即时差异。因此 A 的整计划 pool regret 不能直接作为闭环任务瓶颈。

## 4. 同一真实经历在同一新 query 上改变了什么？

主 B1 结果如下：

- 12/12 anchors 的共同 support loss 下降；
- 11/12 top selections 改变；
- 固定 encoder 下 mean absolute contrast error 平均下降 `0.000496830`；
- tie-aware selection regret 仅 2/12 改善、6/12 恶化、4/12 不变；均值变化 `+0.000154397`，即总体略差；
- 按 6 个 case 聚合为 2 改善、3 恶化、1 不变；
- support-loss delta 与 regret delta 的相关为 `0.0832`。

这说明经历确实改变了预测器和候选选择，但改变方向没有与真实排序稳定对齐。

在 6 个 late anchors 上，`E1->E2` 相对相同两次更新预算的 `E1->E1`、`E2->E2` 各只改善 1/6，其净收益都来自同一个 `val_L|L|s0|m4` anchor。它不是可复用经历效应。

## 5. 第二宿主是否支持一般性？

不支持。本机没有可执行的 LeWM 代码、原生配置和官方 checkpoint，按协议记为 `NOT_RUN_ASSETS_UNAVAILABLE`。没有下载、从零训练或用随机替代模型补齐。因此所有实证结论限于当前 AdaJEPA PushObj 宿主、固定候选池和已测 anchors。

## 6. 唯一停止项

按照“只找问题、暂不讨论修正”的要求，本轮不推导新机制。应明确停止下面这一假设：

> **停止把有限池的整条开环计划 regret 直接当作 PushObj receding-horizon 控制的 actionable task bottleneck。**

原因不是 A 的误排不存在，而是 C 已显示：整计划 oracle-best 的第一 chunk 在共同续行下也没有稳定价值。若后续还要继续问题验证，唯一应先测量的对象是“相同 first-chunk 干预在统一 continuation 下的结果”，而不是继续放大整计划 pool regret、继续做局部导数扫描，或先设计修正算法。这是一项测量边界，不是方法提案。

## 分阶段判定

| 阶段 | 判定 | 核心含义 |
|---|---|---|
| A0 | QUALIFIED_ON_REPAIRED_PATH | 三条 ON/OFF 资格比较逐位通过；修复前失败被保留 |
| A | SELECTION_GAP_SHOWN | 有限池存在广泛、tie-robust 的模型选择损失 |
| B | REUSABLE_CORRECTION_CANDIDATE_NOT_REACHED | support 拟合改善未可靠转成排序改善 |
| C | DECISION_VALUE_NOT_SHOWN | 第一次选择在共同闭环续行下无稳定最终价值 |
| 第二宿主 | NOT_RUN_ASSETS_UNAVAILABLE | 不支持一般性 |

## 工程资格与证据边界

插桩 ON/OFF 比较覆盖动作、适应参数及 buffers、经验 buffers、Python/NumPy/torch CPU/CUDA RNG 与两个 post-feedback 快照。Frozen、完整官方适应、predictor-only 三组修复后比较均逐位通过。

首次 Frozen ON 资格运行发现 fresh replay environment 构造扰动 NumPy RNG；修复是在 replay oracle 边界保存并恢复全部全局 RNG，随后重跑通过。新的资格结论只适用于修复后插桩路径。800 个历史 pools 来自更早 runs；旧失败中动作、参数和经验一致，当前没有证明旧数据失效，但也不追溯宣称它们已通过新的 RNG 精确合同。

资产验证覆盖 21 个 case mappings、800 个真实返回 prefixes、800 个 `g99_after` branch 起点与首 model-chunk 末端，以及 800 个各含 10 个不同 action tensors 的 pools。oracle sidecar 只保存 model-stride 帧，因此不声称验证了中间 4 个 environment substeps 的图像。

## 实际计算与交互预算

| 项目 | 实际消耗 |
|---|---|
| A 离线审计 | 读取 4,000 paired records；0 新环境、0 新模型调用 |
| A0 有效资格运行 | 150 official-evaluation environment steps；1,665 oracle environment steps |
| A0 修复前保留失败运行 | 25 official-evaluation steps；555 oracle steps |
| B 最终证据运行 | 0 环境调用；600 candidate model rollouts；108 adaptation steps |
| B 为加入等预算重复控制而保留的前一版运行 | 0 环境调用；另 600 candidate model rollouts；另 108 adaptation steps |
| C 主共同续行 | 17 physical rollouts；765 environment steps；68 tail replans；6,800 model rollout/backward steps |
| C native-tail 可选补充 | 未运行，因为共同续行未显示稳定价值 |
| 第二宿主 | 0 环境、0 模型调用 |

## 复核入口

- `A_STAGE_REPORT.md` 与 `offline/validation.json`
- `B_STAGE_REPORT.md` 与 `common_anchors_v2/{summary,validation}.json`
- `C_STAGE_REPORT.md` 与 `common_continuation/{summary,validation}.json`
- `asset_validation_continuation/validation.json`
- `qualification/{frozen_comparison_rngfixed,official_comparison,predictor_comparison}.json`
- `second_host/NOT_RUN.md`

本轮独立复核重点检查了旧数据资格边界、recorded-pool 收敛措辞、B3 单 anchor 驱动、完整适应跨 encoder 指标、以及 C 主臂是否严格使用 B1；这些边界已写入最终判定。
