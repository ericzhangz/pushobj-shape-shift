# Round 4：已保存 C 分支的向量效应审查

日期：2026-09-22。性质：已查看 development 六个 anchor 的诊断，不是盲确认、方法效果或闭环部署实验。

## 结论

在这 17 条物理 C 分支中，自由反馈预测的终点候选效应向量通常比真实效应小：全部 16 个物理 pair 的 depth 1→5 finite-pair log-gain 为负（范围 −4.77 至 −0.10）。这是真实的向量差异，而不只是标量 cost 压缩。不过它不是唯一机制：真实动作 teacher 路径有 7/16 个正 gain；带符号的共同中点和效应项可同向也可抵消。可选 hybrid 又显示自由反馈误差中的输入、动作与真实输入残差三项及其交叉项均不可忽略。**不能据此冻结“应统一增加增益”、动作导数高频不稳、缺少激励或近共轭可处理为主因。**

旧 C 合法 Frozen/Adapted 子集的真实终点收益仍为 0 改善／3 tie／3 恶化（tie ε=1e−6）。因此本报告没有显示一个当前可兑现的闭环决策修复空间；旧 oracle 标签仅是回顾性诊断。

## 数据及数值合同

- G0 的 [来源与时间合同](../results/g0_20260922_105708/INPUT_CONTRACT.json)、[120 条 B 真值重编码](../results/b_truth_full_20260922_110025/TRUTH_ENCODING_VALIDATION.json)与 [G0 汇总](../results/G0_FINAL.json)通过。B 六帧同批编码和本报告 C 的逐时刻单帧编码是不同真值语义，未混用；C 未来向量／真实动作仅用于诊断。
- [G1 传播审计](../results/g1_propagation_20260922_111000/propagation_audit_summary.json)：6 anchors、17 物理分支、depth 0–5，冻结模型 255 次一步 transition、119 次真实帧/目标编码；初史和目标重编码最大差均为 0，已保存 Qenv 完全复现，Qhat 最大差 2.42e−8，模型参数/buffers 未变。原生 chunk 边界每步保留单帧；不以整条动作一次 rollout 替代。
- [G1 几何闭合](../results/g1_geometry_20260922_111000/GEOMETRY_CLOSURE.json)：16 物理 pair ×5 个非初始深度×4 模式=320 行；W 为 visual 块 2/n_v、proprio 块 2/n_p，令终点原生代价为 0.5‖z−g‖²_W。旧 scalar cost 和 visual/proprio MSE 的最大绝对复现差 2.30e−8；signed、径向/方向和强制误差的最大闭合残差 8.88e−16。除 persistence 在 depth1 因同锚点前一真实帧完全相同而 mask 外，各模式/深度 16/16 pair 的 effect norm 可定义。没有用 oracle 调整 W。
- 原材料提到的 `geometry_contract.py` 与 `check_processing.py` 未在本机找到；下述控制是按公式独立重建的测试，**不是**声称原 20 项测试已重跑。

## 六个 anchor 全表

下表每个 anchor 对其物理 pairs 求均值；`T|s1` 只有 2 个物理分支，因此只有 1 pair，其余各有 3 pair。`norm 比`为终点预测/真实 effect W 范数的 pair 均值；`|ΔC误差|`是终点有符号候选 cost contrast 误差的绝对值均值。各 pair/depth 不是独立样本。

| anchor | pairs | free norm 比 | free 平均 |ΔC误差| | teacher norm 比 | teacher 平均 |ΔC误差| | reset 平均 |ΔC误差| | persistence 平均 |ΔC误差| |
|---|---:|---:|---:|---:|---:|---:|---:|
| val_L·s0·m4 | 3 | 0.006 | 5.52e−4 | 0.357 | 6.27e−4 | 4.90e−4 | 3.50e−4 |
| val_L·s1·m4 | 3 | 0.500 | 1.47e−3 | 1.065 | 4.27e−3 | 6.72e−4 | 2.07e−3 |
| val_L·s2·m4 | 3 | 0.495 | 4.92e−4 | 0.845 | 7.02e−4 | 2.65e−4 | 1.39e−3 |
| val_T·s0·m4 | 3 | 0.281 | 2.64e−4 | 0.581 | 7.11e−4 | 8.19e−4 | 1.70e−3 |
| val_T·s1·m4 | 1 | 0.946 | 2.06e−6 | 2.462 | 7.36e−6 | 4.89e−7 | 2.81e−7 |
| val_T·s2·m4 | 3 | 0.193 | 4.07e−3 | 0.879 | 3.83e−3 | 1.91e−3 | 3.27e−4 |

三种预定 pair 子集终点的逐 pair 均值如下；这是描述量，不是以 16 或 11 为独立样本的显著性检验。

| 子集 | pair | free |ΔC误差| | free effect 范数比 | teacher |ΔC误差| | reset |ΔC误差| | persistence |ΔC误差| |
|---|---:|---:|---:|---:|---:|---:|
| 全部物理 pair | 16 | 1.283e−3 | 0.336 | 1.900e−3 | 7.785e−4 | 1.096e−3 |
| Frozen–非 Frozen | 11 | 1.001e−3 | 0.430 | 1.475e−3 | 5.987e−4 | 8.087e−4 |
| 合法 Frozen/Adapted | 6 | 2.797e−4 | 0.659 | 3.091e−4 | 1.878e−4 | 2.100e−4 |

合法六对中，真实 `C_Frozen−C_Adapted` 分别为 val_L s0 −8.30e−4、s1 +2.30e−8、s2 +7.53e−9、val_T s0 −3.10e−4、s1 +3.60e−7、s2 −6.40e−4；后三个近零正差按既定 ε 计 tie，而非收益。val_L s0 的 free contrast 误差 +8.29e−4 = midpoint +2.09e−4 + effect +6.20e−4；val_T s0 的 +1.22e−4 = midpoint +5.47e−4 + effect −4.25e−4，展示明显抵消。不能把绝对分量百分比解释为因果占比。

## 有限传播、强制输入与反馈动作

[FINITE_PAIR_GAIN.csv](../results/g1_geometry_20260922_111000/FINITE_PAIR_GAIN.csv) 从 depth1 定义固定 pair 的 log-gain；free 16/16 终点为负，teacher 为 7 正／9 负。它只描述已执行的两条轨迹，不估计完整 Jacobian 奇异值、Lyapunov 指数或可控性谱。正确预测也可以随真实系统同步收缩；因此不追求 gain 单调增大。reset/persistence 只报告逐时刻快照比，不能称可部署传播。

[OBSERVED_FORCED_ERROR.csv](../results/g1_geometry_20260922_111000/OBSERVED_FORCED_ERROR.csv) 中，teacher 的下一步误差按同真实动作拆为「不同输入」p 和「真实输入上的模型残差」η，并保留交叉内积；pair 级数据见 [FORCED_PAIR_ERROR.csv](../results/g1_geometry_20260922_111000/FORCED_PAIR_ERROR.csv)。终点全部 pair 的平均平方范数为 p 1.363e−3、η 3.519e−3、二者内积 −1.121e−3，实际总误差 2.640e−3，不能将 MSE 差额解释为误差占比。

由于 free/teacher 差别影响结构定位，先写 [hybrid 执行裁定](HYBRID_DECISION.md)，再执行 [68 次同 checkpoint hybrid](../results/g1_hybrid_20260922_112600/FEEDBACK_SPLIT.json)。自由反馈的有序三项为输入差、动作差、真实输入/真实动作模型残差；全部 pair 终点的平均平方范数依次为 2.830e−3、3.163e−3、3.519e−3，实际总误差为 3.238e−3。三种交叉内积均值为 −2.003e−3、−2.059e−3、+0.925e−3。各项量级相近且显著抵消；不同 anchor 的主导项不一致。该拆法依赖加减顺序，给真实 z 喂未执行动作是模型查询，不是真实反事实后果。最大闭合差 1.39e−17，68 次 predictor、batch1，零环境/GD/训练。旧 G1 报告没有改写；[额外来源绑定](../results/G1_SOURCE_BINDING.json)将原执行日志、向量目录和 checkpoint 对齐。

## 负控制、范围与裁定

独立算术测试验证：共同平移能改变/降低 cost 而不改变候选分离，旋转能保持分离范数而翻转任务排序，真实与预测同步收缩的 gain 应为零；mask 断裂时不伪报 telescoping。全套 `research/reframe_v3/test_*.py` 为 41/41 通过。原始逐 pair、深度、mode 数据在 [PAIR_GEOMETRY.csv](../results/g1_geometry_20260922_111000/PAIR_GEOMETRY.csv)、[PAIR_CONTRAST_DECOMPOSITION.csv](../results/g1_geometry_20260922_111000/PAIR_CONTRAST_DECOMPOSITION.csv) 与 [MASK_COVERAGE.csv](../results/g1_geometry_20260922_111000/MASK_COVERAGE.csv)。

| 判断 | 本轮状态及限定 |
|---|---|
| PAIR_VECTOR_DISTORTION | **SUPPORTED**：所测 free 分支的终点效应向量误差及 norm 收缩；不推及新 checkpoint。 |
| MIDPOINT_PROJECTION_DISTORTION | **SUPPORTED**：signed 中点项改变相对 cost，且可与效应项抵消；不等于唯一原因。 |
| INPUT_RECURSION_EFFECT | **SUPPORTED**：同动作 teacher 的 p 项非零；不是独立因果份额。 |
| FEEDBACK_DEPENDENCE | **SUPPORTED**：free/teacher 与有序 hybrid 不同，动作差项非零；不能推出 planner 设计本身错误。 |
| RECOVERABLE_NEW_QUERY_VALUE | **UNAVAILABLE**：尚未训练共享处理器，也没有可兑现的旧合法正 margin。 |

总体裁定为 **MIXED、可进入受控开发集可行性试验，而非机制已冻结**。模型在真实输入上也有残差；反馈动作和自生成输入均参与，且任务投影会抵消。下一步若检验一个受控 P，必须仅用查询截止前事实经历训练、固定外部 W 和目标，并与同信息的单端修正及原 AdaJEPA 更新比较。不能用本报告的 oracle 分支训练或选择超参数。

本阶段实际成本：G1 必做 255 次 F transition，额外 hybrid 68 次；119 次真实/目标编码，G1 传播墙钟 5.91 秒、其中编码 0.656 秒、向量写出 0.029 秒；名义输入 sidecar 479,821 字节、输出向量 747,490 字节。hybrid 墙钟 4.20 秒。环境 0、GD 0、参数更新 0。没有运行新 case、完整 Jacobian、SVD、反馈规划重算或最终方法训练。
