# Reframe V3 — B 阶段共同锚点报告

## 阶段判定

**未达到 `REUSABLE_CORRECTION_CANDIDATE`。**

B 阶段定位到一个明确的问题：同一批过去真实经历足以降低官方 predictor-only 更新的 support loss，也经常改变同一新 query 上的候选选择，但这种拟合改善没有可靠地转化为更低的有限池选择损失。这里报告的是问题诊断，不提出修正方法。

## 设计与资格

- donor 固定为 Frozen 的 `val_T`、`val_L` 各 3 个 case，MPC index 2 和 4，共 12 个按时间固定、无缺失 anchors。
- 所有版本在相同 10-endpoint pool 上重算模型代价，环境后果完全复用 A 阶段记录，不新增环境调用。
- 主臂为相同 Frozen checkpoint、固定 encoder、官方配置 predictor-only shadow adaptation。
- 每个 anchor/variant 使用相同受控 shadow RNG 起点；模型 reset 精确，固定 encoder 无变化。
- Frozen 重算代价与原记录逐项一致，最大绝对误差为 0。
- 资格结果：66 个 model-version pools、600 个候选模型 rollout、108 个 adaptation backward steps；全部有限值，验证通过。

这些 RNG 是受控 shadow 试验 RNG，不冒充原 on-policy 运行当时的精确 RNG；shadow adaptation 也不冒充各适应臂自己的 on-policy 轨迹。

## B1：官方 predictor-only factual 更新

12/12 anchors 的共同 support loss 都下降，但同一 pool 上的 tie-aware selection regret：

| 结果 | 数量 |
|---|---:|
| 改善 | 2 / 12 |
| 恶化 | 6 / 12 |
| `epsilon=1e-6` 内不变 | 4 / 12 |
| top selection 改变 | 11 / 12 |

均值 `delta R_selected_tie_min = +0.000154397`，中位数 `+0.0000265746`；正值表示恶化。固定 encoder 下可比较的 mean absolute contrast error 均值变化为 `-0.000496830`，但 selection regret 反而变差。support-loss delta 与 regret delta 的 Pearson 相关只有 `0.0832`。

按 6 个 case 聚合后，2 个改善、3 个恶化、1 个不变。结果不是由把 MPC2/MPC4 当成 12 个完全独立 case 才产生，但样本仍然很小，不作总体发生率推断。

因此，问题不是“过去真实经历完全没有写入模型”，而是**当前 factual support objective 的改善与新候选排序质量没有稳定对齐**。

## B3：时间分离经历与重复控制

在 6 个 MPC4 anchors 上，`E1 -> E2` 相对 Frozen 表面上有 4 改善、1 恶化、1 不变，均值 `-0.00188051`。但主比较必须是相同两次更新预算的重复控制：

| 比较 | 改善 | 不变 | 恶化 | mean delta regret |
|---|---:|---:|---:|---:|
| `E1->E2` 减 `E1->E1` | 1 | 5 | 0 | -0.00101979 |
| `E1->E2` 减 `E2->E2` | 1 | 5 | 0 | -0.00101979 |

两项净收益都只来自 `val_L|L|s0|m4`；其余 anchors 在 tie-aware regret 上没有复现。故不能把这一单点效应写成“新增时间分离经历产生了可复用修正”。

## 完整官方适应补充臂

6 个 late anchors 的固定参照 selection regret 为 3 改善、2 恶化、1 不变，且受单个 `val_L` case 明显影响。由于该臂更新了 encoder，模型预测代价与 Frozen encoder 下保存的 `c_env_ref` 不属于同一表示版本；因此这里不把 contrast-error 变化解释成同版本 calibration 变化。固定环境后果上的候选选择比较仍可读，但不足以改变阶段判定。

## 证据边界

B 阶段支持：官方 factual 更新的训练目标改善，不保证同一 held-out pool 上的决策排序改善。

B 阶段不支持：

- 不支持“任意真实经历都无法改善模型选择”；
- 不支持“任何适应算法都不可能复用经历”；
- 不支持把一个 anchor 的 E1/E2 效应外推为机制；
- 不支持把 pool ranking 变化直接等同于闭环任务收益。

后一个边界由 C 阶段直接检验。
