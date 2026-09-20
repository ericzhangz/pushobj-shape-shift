# AdaJEPA PushObj 真实 Checkpoint 决定性 Pilot

## 结论先行

**判定：REDEFINE。** 测量基座通过，真实新查询 contrast mismatch 明确存在，而且 Official AdaJEPA 的 factual support loss 改善没有稳定消除它；但本轮尚未执行多尺度 response 稳定性、response-only/value-only 因果 oracle 和按时间顺序的 experience-reuse 检验。因此证据支持继续做**限定的因果诊断**，不支持现在设计新主算子，也不足以 KILL response-mismatch 研究对象。

本轮共有 63 个正式 episode。contrast 是每个 episode/replan 内固定 GD 步 `0,24,49,74,99` 的重复诊断，不是独立样本；所有成功率以官方环境返回的 `success` 为准。

## A. Reproduction

- 官方 `val_T, n=1, max_iter=1` smoke 成功运行；原始日志：`artifacts/baseline/smoke_max_iter1_attempt2/baseline_stdout.log`。
- 官方 `val_T, n=1, max_iter=20` 在 MPC iter 10 成功，最终 `state_dist=58.35865`；原始日志：`artifacts/baseline/full_max_iter20/baseline_stdout.log`。
- instrumentation OFF 的完整 episode 返回 11 个 model-action blocks，与捕获的官方输出逐元素完全相等，最大绝对误差 `0.0`：`artifacts/regression/full_episode_action_regression.json`。

## B. Measurement integrity

- Fresh-env full-prefix zero-action replay：**800/800 全部逐项精确通过**；RGB、公开状态、native same-version cost 和官方环境指标均无差异。汇总：`D:\EV-TTT\adajepa_official_51d8665\artifacts\metrics\replay_errors.csv`。
- 动作单位合同通过：model pack/unpack 误差 `0`，normalize/denormalize 最大误差 `1.49e-8`，完整 pixel round-trip 最大误差 `1.91e-6`。
- objective 合同通过：MPC step 0–4 使用 terminal，step 5 起使用 full horizon；四个 closure 检查最大误差均为 `0`。
- 三臂的 sample id、seed、初始状态与目标 evidence 已逐数组精确配对；失败/半成品运行与重复 run 路径被汇总入口拒绝。
- Astra 审计修正了两个离线统计错误：(1) Frozen 的空 finetune 调用不再误计为真实 adaptation；(2) 官方 final evaluator 的 padded-action 环境步数按实际整批执行量派生，原记录值同时保留。原始日志和原始 evidence 未修改。

## 原始任务表

| split | arm | success | rate ± episode SD | Δrate vs Frozen | final state distance mean ± SD |
| --- | --- | --- | --- | --- | --- |
| val_T | Frozen | 2/3 | 0.667 ± 0.577 | +0.000 | 30.00 ± 24.37 |
| val_T | Official | 3/3 | 1.000 ± 0.000 | +0.333 | 34.96 ± 22.69 |
| val_T | Predictor-only | 3/3 | 1.000 ± 0.000 | +0.333 | 45.80 ± 14.20 |
| val_L | Frozen | 2/3 | 0.667 ± 0.577 | +0.000 | 27.39 ± 6.65 |
| val_L | Official | 3/3 | 1.000 ± 0.000 | +0.333 | 28.81 ± 7.68 |
| val_L | Predictor-only | 3/3 | 1.000 ± 0.000 | +0.333 | 23.10 ± 6.07 |
| val_I | Frozen | 0/5 | 0.000 ± 0.000 | +0.000 | 78.38 ± 23.41 |
| val_I | Official | 1/5 | 0.200 ± 0.447 | +0.200 | 113.60 ± 70.33 |
| val_I | Predictor-only | 1/5 | 0.200 ± 0.447 | +0.200 | 130.11 ± 118.93 |
| val_small_tee | Frozen | 5/5 | 1.000 ± 0.000 | +0.000 | 69.22 ± 38.04 |
| val_small_tee | Official | 4/5 | 0.800 ± 0.447 | -0.200 | 76.78 ± 32.25 |
| val_small_tee | Predictor-only | 5/5 | 1.000 ± 0.000 | +0.000 | 82.39 ± 28.93 |
| val_square | Frozen | 1/5 | 0.200 ± 0.447 | +0.000 | 650.04 ± 1188.89 |
| val_square | Official | 2/5 | 0.400 ± 0.548 | +0.200 | 135.61 ± 99.14 |
| val_square | Predictor-only | 0/5 | 0.000 ± 0.000 | -0.200 | 168.91 ± 63.81 |

说明：`state_dist` 与官方几何成功判据不是同一个量，二者并列报告，不能互相替代。95% Wilson 区间和逐 episode 原始值见 `D:\EV-TTT\adajepa_official_51d8665\artifacts\metrics\task_metrics.csv`；本 pilot 的 `n=3/5` 不支持显著性外推。

## C. Real failure

`false improvement` 使用预注册的数值阈值：`delta_pred < -1e-6` 且 `delta_env_samever > 1e-6`。

| split | arm | contrasts | MAE | Spearman | sign acc | false / predicted-improve | max real worsening | Δfalse-rate vs Frozen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val_T | Frozen | 160 | 0.0028 | 0.382 | 62.5% | 22/48 (45.8%) | 0.0025 | +0.000 |
| val_T | Official | 120 | 0.0039 | 0.469 | 64.2% | 19/47 (40.4%) | 0.0167 | -0.054 |
| val_T | Predictor-only | 95 | 0.0034 | 0.382 | 64.2% | 17/42 (40.5%) | 0.0055 | -0.054 |
| val_L | Frozen | 160 | 0.0068 | 0.316 | 56.9% | 26/49 (53.1%) | 0.0057 | +0.000 |
| val_L | Official | 105 | 0.0258 | 0.267 | 63.8% | 21/45 (46.7%) | 0.1296 | -0.064 |
| val_L | Predictor-only | 100 | 0.0331 | 0.235 | 62.0% | 26/50 (52.0%) | 0.1167 | -0.011 |
| val_I | Frozen | 500 | 0.0071 | 0.201 | 61.0% | 89/208 (42.8%) | 0.0527 | +0.000 |
| val_I | Official | 435 | 0.0083 | 0.338 | 64.4% | 108/231 (46.8%) | 0.0527 | +0.040 |
| val_I | Predictor-only | 435 | 0.0090 | 0.239 | 63.7% | 104/235 (44.3%) | 0.0809 | +0.015 |
| val_small_tee | Frozen | 175 | 0.0216 | 0.187 | 64.6% | 33/88 (37.5%) | 0.0414 | +0.000 |
| val_small_tee | Official | 220 | 0.0158 | 0.167 | 64.1% | 53/119 (44.5%) | 0.0714 | +0.070 |
| val_small_tee | Predictor-only | 210 | 0.0182 | 0.210 | 64.3% | 47/107 (43.9%) | 0.0351 | +0.064 |
| val_square | Frozen | 430 | 0.0119 | -0.133 | 51.6% | 104/227 (45.8%) | 0.1833 | +0.000 |
| val_square | Official | 355 | 0.0158 | 0.093 | 56.6% | 110/217 (50.7%) | 0.6951 | +0.049 |
| val_square | Predictor-only | 500 | 0.0118 | 0.286 | 62.6% | 125/275 (45.5%) | 0.6975 | -0.004 |

结论：所有 15 个 split/arm 单元都出现了 predicted-improvement / real-worsening。条件伪改善率约为 37%–53%，最大真实恶化出现在 `val_square`（Official `0.6951`，Predictor-only `0.6975`）。该错误在第一次 MPC replan 和后续 replans 都存在；对两个适配臂，后续 replans 已消费此前 factual updates，而 Frozen 始终没有参数更新：

| arm | stage | contrasts | thresholded false / predicted-improve |
| --- | --- | --- | --- |
| Frozen | first_mpc_replan_pre_update | 105 | 22/74 (29.7%) |
| Frozen | later_mpc_replans_no_updates | 1320 | 252/546 (46.2%) |
| Official | after_prior_factual_updates | 1130 | 289/585 (49.4%) |
| Official | first_mpc_replan_pre_update | 105 | 22/74 (29.7%) |
| Predictor-only | after_prior_factual_updates | 1235 | 297/635 (46.8%) |
| Predictor-only | first_mpc_replan_pre_update | 105 | 22/74 (29.7%) |

精确到每个 MPC iter 和固定 GD step 的表分别在 `D:\EV-TTT\adajepa_official_51d8665\artifacts\metrics\contrast_by_mpc_iter.csv` 与 `D:\EV-TTT\adajepa_official_51d8665\artifacts\metrics\contrast_by_gd_step.csv`。

任务层联系目前是描述性关联而非因果证明：

| episode outcome | n episodes | episode contrast MAE mean ± SD | episode false rate mean ± SD |
| --- | --- | --- | --- |
| failed | 28 | 0.0086 ± 0.0076 | 23.0% ± 8.8% |
| successful | 35 | 0.0198 ± 0.0182 | 21.6% ± 8.1% |

逐 episode 结果见 `D:\EV-TTT\adajepa_official_51d8665\artifacts\metrics\episode_outcomes.csv`。难 split 会产生更多 replans，从而贡献更多 pooled contrasts；因此不把 pooled contrast 数量当作独立样本数。

## D. Effect of AdaJEPA

| split | arm | real updates | support improved | independent false rate | contrast MAE | task success |
| --- | --- | --- | --- | --- | --- | --- |
| val_T | Frozen | 0 | — | 45.8% | 0.0028 | 66.7% |
| val_T | Official | 24 | 91.7% | 40.4% | 0.0039 | 100.0% |
| val_T | Predictor-only | 19 | 89.5% | 40.5% | 0.0034 | 100.0% |
| val_L | Frozen | 0 | — | 53.1% | 0.0068 | 66.7% |
| val_L | Official | 21 | 100.0% | 46.7% | 0.0258 | 100.0% |
| val_L | Predictor-only | 20 | 100.0% | 52.0% | 0.0331 | 100.0% |
| val_I | Frozen | 0 | — | 42.8% | 0.0071 | 0.0% |
| val_I | Official | 87 | 89.7% | 46.8% | 0.0083 | 20.0% |
| val_I | Predictor-only | 87 | 93.1% | 44.3% | 0.0090 | 20.0% |
| val_small_tee | Frozen | 0 | — | 37.5% | 0.0216 | 100.0% |
| val_small_tee | Official | 44 | 93.2% | 44.5% | 0.0158 | 80.0% |
| val_small_tee | Predictor-only | 42 | 92.9% | 43.9% | 0.0182 | 100.0% |
| val_square | Frozen | 0 | — | 45.8% | 0.0119 | 20.0% |
| val_square | Official | 71 | 98.6% | 50.7% | 0.0158 | 40.0% |
| val_square | Predictor-only | 100 | 94.0% | 45.5% | 0.0118 | 0.0% |

Official 与 Predictor-only 的 deterministic support loss 在绝大多数真实更新中下降，但新查询伪改善率仍高，且 Official 相对 Frozen 只在 `val_T/val_L` 下降、在三个 OOD split 均上升。OOD 任务成功合计为 Frozen `6/15`、Official `7/15`、Predictor-only `6/15`；这只是一个 episode 的差异，不能解释为稳定收益。

把一次 factual update 与**下一次** MPC 新查询按时间对齐后的描述性检查如下。相关系数不是因果量，且每个 next-query 内仍只有五个固定 probe：

| split | arm | aligned pairs | support improved | next-query MAE | next-query false rate | rho(support delta, next MAE) |
| --- | --- | --- | --- | --- | --- | --- |
| val_T | Official | 21 | 90.5% | 0.0034 | 15.2% | -0.451 |
| val_T | Predictor-only | 16 | 93.8% | 0.0026 | 17.5% | -0.403 |
| val_L | Official | 18 | 100.0% | 0.0277 | 21.1% | 0.011 |
| val_L | Predictor-only | 17 | 100.0% | 0.0364 | 28.2% | -0.164 |
| val_I | Official | 82 | 89.0% | 0.0068 | 24.6% | -0.457 |
| val_I | Predictor-only | 82 | 93.9% | 0.0075 | 23.7% | -0.485 |
| val_small_tee | Official | 39 | 92.3% | 0.0161 | 25.1% | -0.242 |
| val_small_tee | Predictor-only | 37 | 91.9% | 0.0188 | 23.2% | -0.254 |
| val_square | Official | 66 | 98.5% | 0.0154 | 31.5% | -0.316 |
| val_square | Predictor-only | 95 | 95.8% | 0.0112 | 25.1% | -0.499 |

除 `val_L/Official` 接近 0 外，`rho(support delta, next MAE)` 多为负：更大的 support-loss 下降并没有对应更低的下一查询 MAE，描述性方向反而相反。该现象仍可能受 episode 难度、replan 选择和少量固定 probes 混杂，不能作因果结论。完整配对记录在 `artifacts/metrics/support_next_query_pairs.csv`，分组汇总见 `D:\EV-TTT\adajepa_official_51d8665\artifacts\metrics\support_next_query_summary.csv`。

## E. Encoder confound

| split | arm | |samever-ref| MAE | independent false rate | task success |
| --- | --- | --- | --- | --- |
| val_T | Official | 0.0000175 | 40.4% | 100.0% |
| val_T | Predictor-only | 0.0000000 | 40.5% | 100.0% |
| val_L | Official | 0.0000500 | 46.7% | 100.0% |
| val_L | Predictor-only | 0.0000000 | 52.0% | 100.0% |
| val_I | Official | 0.0000494 | 46.8% | 20.0% |
| val_I | Predictor-only | 0.0000000 | 44.3% | 20.0% |
| val_small_tee | Official | 0.0000554 | 44.5% | 80.0% |
| val_small_tee | Predictor-only | 0.0000000 | 43.9% | 100.0% |
| val_square | Official | 0.0000793 | 50.7% | 40.0% |
| val_square | Predictor-only | 0.0000000 | 45.5% | 0.0% |

Predictor-only 的 evaluator 坐标漂移按构造为 0；Official 的 drift 非零但量级较小。关闭 encoder 更新并未稳定降低独立 contrast error：它在部分 split 改善、部分 split 恶化，并在 `val_square` 得到 `0/5`。因此 encoder drift 是可测 confound，但不是本轮 failure 的充分解释。

## F. Response diagnostics

本轮只测了 native Adam 实际 action update 的**有限步 task contrast**，没有运行预注册的多尺度方向 probe；不能声称恢复了完整 Jacobian，也不能回答是否存在 scale-stable response 区域。该问题保持 **unresolved**，不是负结果。

## G. Causal evidence

本轮没有运行 response-only oracle 或 value-only oracle，因此不能量化两者各自修复多少 planning failure。当前结果证明 mismatch 与规划轨迹共现，不证明 response error 单独因果控制失败。结论：**unresolved**。

## H. Experience reuse

本轮 adaptation 使用官方 recent-5 factual history，但没有构造两个按时间分离的 experience 组，也没有在相同预算下测试 later unseen action/goal query 的可重复增量。support-loss 下降不等于 experience reuse。结论：**unresolved**。

## I. Go / No-Go

**REDEFINE**，理由如下：

1. 测量基座通过，不触发 measurement NO-GO。
2. 真实新查询 contrast failure 在 Frozen、Official 和 Predictor-only 上均存在，不触发“failure 不存在”的 NO-GO。
3. AdaJEPA 没有稳定自然消除 OOD contrast failure。
4. 但 GO 的后半条件——因果层定位、past experience 的 later-query 复用价值、以及同预算简单控制不能解释收益——尚未建立。

因此下一阶段只能做以下冻结诊断，不进入新方法设计：

1. 在事先固定的失败 anchor 上运行 response-only 与 value-only oracle；
2. 用事先固定的尺度/方向检查 finite response 的稳定区间；
3. 用两个按时间分离的 experience 组测试 later unseen query，并匹配真实环境、forward/backward 与墙钟预算；
4. 只有前三项支持 GO 时，才加入简单 finite-scale readout、replay/TTA、GRASP 和 CEM/零阶规划对照。

## 预算与可复核路径

- 逐 run 的真实环境、官方 replay/final eval、world-model rollout、forward/backward、oracle、墙钟与峰值显存：`D:\EV-TTT\adajepa_official_51d8665\artifacts\metrics\run_budgets.csv`。
- 原始 planner/adaptation/version 记录：`artifacts/traces/*.parquet`。
- 原始 oracle branches 与 query budget：`artifacts/oracle/branch_records.parquet`、`artifacts/oracle/query_budget.csv`。
- 任务、contrast、episode、stage 与 support-next-query 表：`artifacts/metrics/`。
- 图：`artifacts/plots/task_success_rate.png`、`artifacts/plots/conditional_false_improvement_rate.png`、`artifacts/plots/predicted_vs_environment_contrast.png`。
- 每个正式 run 的原始日志、sidecar tensor/NPZ 与 Hydra 配置保留在 `artifacts/development/` 和 `artifacts/ood/`；启动失败的 `val_T_official_n3` 与 `val_I_frozen_n5` 也保留，但不会进入任何分母。

## 与说明文本不同、以源码为准的实现事实

- 5 个 model actions 的 `VWorldModel.rollout` 返回 initial + 5 个预测 observation，共 6 帧。
- staged objective 在 MPC step 0–4 取 terminal，从 step 5 起取 full horizon；full objective 的归一化指数权重后仍执行外层 mean。
- native GD 配置实际使用 Adam；诊断记录的是 optimizer/scheduler 后的真实 `u_after-u_before`，不是假设的 `-lr*grad`。
- 官方 final evaluator 会先执行整批 padded actions，再用 `action_len` 选择评价帧；预算已按真实执行量校正。

## 审计边界

本报告未执行任何 SHA 或其他哈希检查。资产身份仅沿用用户提供路径、文件大小/时间、Git 分支和说明中给定的提交标识；这不是密码学完整性证明。
