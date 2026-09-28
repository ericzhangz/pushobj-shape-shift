# 决策与经历写入：直接近邻及剩余机制空间

日期：2026-09-27。未读取或使用任何本地 skill。方法级核查限于下述六篇一手论文；另检索了主动辨识的经典先例。没有运行新实验，没有以文献中的自述代替独立验证，也没有完成全领域穷尽查新。

## 结论

**旧主问题值得保留：已经发生的经历如何可靠改善尚未执行的行动后果。当前有充分理由否决的是几种过宽的解法表述，不是这个问题。** “预测准确不等于决策正确”已经不是空白；“跨 episode 预训练一个可在线适配的上下文先验”同样不是空白。加入主动探索，也不能自动产生新颖性。

读取的本地历史依据：`<RUNS_ROOT>/research_progress_20260922/RESEARCH_POSITION.md`、`MATHEMATICAL_BRIDGE.md`、`<RUNS_ROOT>/cvpr_oral_reframe_20260927/DECISION_REPORT.md`。这些记录已经把事实拟合与查询修正区分为 A/B，并明确通用伪逆、信赖域、普通回归不是独立机制。上一轮原生观察刷新改善预测，也没有检验并推翻“参数／规律适配无法传到新动作”的旧问题。

## 六篇方法级核查

| 一手论文与阅读位置 | 实际计算和训练权限 | 对我们有约束力的边界 |
|---|---|---|
| [D-JEPA, 2609.24749](https://arxiv.org/html/2609.24749)，§1–4、§5、A.1–A.7、B.1；截图写成 ID-JEPA，页面题名实际为 D-JEPA | 对同起点候选的 goal-relative descriptor 和候选内 rank 作 Transformer set-wise 编码；有界残差修正分数。用恢复同一起点执行各分支所得成功／失败标签学习。另适配 predictor 尾部，用原预测保留项和排名项约束。terminal radial realization 把最终排序写回 goal distance。 | 已覆盖局部排序失真、候选关系学习、有限改动、native distance 读出。部署不读分支真值；训练读。其任务局部监督与候选池条件必须如实比较，不能称为纯 factual TTA。rank 编进 radius 是排序实现，不自动等于物理后果校正。 |
| [CLAW, 2609.12278](https://arxiv.org/html/2609.12278)，§2–5 | 随机策略收集 context trajectories；transition MLP 加平均池化得到 context；分块 hypernet 输出各层 LoRA。base 与 hypernet 在多环境联合训练，更新损失来自另一 planner rollout buffer 的 TD-MPC2 目标。测试重新生成 adapter，不需目标候选全分支标签。 | “跨经历先验→当前经历→模型适配”已有直接实现，且允许换世界模型。已与 in-context baseline 比较。作者还明确提出优化 context collection policy；因此仅增加探测策略不足够。实证以 state-based TD-MPC2 为主，不等于视觉分支传播已解决。 |
| [Counterfactual Quotient Models, 2608.22092](https://arxiv.org/html/2608.22092)，§1–4、§6 | 学习未来 successor feature 的候选中心化值，模型输出也强制中心化。训练同起点、共同随机数、相同后续策略，仅改变首动作的配对分支；差分回归与完整 pairwise 回归等价。给出 ordinary-trajectory density-ratio 形式，但实验实现的是 branched 版本。 | 已占据“先去共同模态、直接学动作差别”的宽主张。有限 reward feature、固定 continuation policy、有限首动作集；state-vector 加自主扰动实验不是视觉连续动作整链 MPC。不能误称它原则上只能有 simulator cloning。 |
| [GRASP, 2602.00475](https://arxiv.org/html/2602.00475)，§1–4 | 同时优化中间 state 与 action，使一步动态残差小；state 加噪，切断 predictor 的 state-input 梯度，保留 action 梯度；加逐步 goal 项，间隔执行完整 rollout 梯度同步。使用现有可微世界模型，无在线候选真值监督。 | 已明确演示视觉状态 Jacobian 可被优化器利用，且 grad-cut 后不再是真正 Langevin 场。它修改规划求解，不学习 factual 证据对未执行行动的作用；仅说“梯度不好／planner exploits errors”不再构成新切口。 |
| [A Control Theory of Predictability, 2607.10362](https://arxiv.org/html/2607.10362)，§1–5 | 区分训练分布和 planner-reachable 分布；基本 regret 界使用候选集上的真实与预测 cost supremum gap。线性控制条件下分析外推放大；实验干预给预测 latent 加线性低维状态 readout loss，需要训练期状态目标。Jacobian probe 是代理量。 | 已覆盖数据平均 MSE 不约束规划外推的宽论点。论文自己说明有限 seeds、probe 不是 safety certificate。不可把无一般上界推成统计相关必为零，亦不可把线性条件理论直接套到当前非线性 checkpoint。 |
| [Poke and Strike, 2509.00178](https://arxiv.org/html/2509.00178)，§1–4、§5、limitations | simulator 中用真实物性训练 task policy；逐物性扰动测 task success sensitivity，确定估计阈值。联合训练 exploration RL 与 LSTM 物性估计；以估计 uncertainty 决定何时结束探索。部署只需观测，任务执行无需重训练或 simulator 查询。 | 已覆盖任务定向 probe、只精确识别重要物性、自适应停止。训练依赖物性标签和任务成功测量，采用 kinematics、固定形状与参数化模型；未建立视觉表征中未知相互作用的组合适配能力。 |

经典直接先例还包括 [Task-Optimal Exploration in Linear Dynamical Systems, ICML 2021](https://proceedings.mlr.press/v139/wagenmaker21a.html) 和 [Learning Active Task-Oriented Exploration Policies, RSS 2020](https://roboticsconference.org/2020/program/papers/85.html)。本轮仅据原始摘要／正式入口确认其任务导向辨识与实验设计范围，没有把它们记作额外方法级精读。泛化的信息增益、任务 Hessian/Fisher 加权或 B-covariance 准则应作为基线。

## 哪一处尚值得明确验证

以下是**本轮推断，不是已确认的文献空白或新方法**：视觉模型的当前适配经常把一份经历压成全局模型／context 修正，却没有显式辨认这份经历到底支持哪一类相互作用，以及哪些未来分支会重用它。例如一次推动揭示桌面局部滑动规律；另一候选在不同时间、位置发生接触，第三候选根本不接触。有效修订不能只是把共同 residual 写给所有分支，也不能只根据 query latent 接近程度传播。真正需要复用的是相互作用规律，复用范围取决于未来分支上的交互结构。

这与现有方法的剩余差别只能落在**可学习结构与组合泛化**，不能落在“RNN 不能表示”或“meta-learning 不支持”：CLAW 能表达这种映射，D-JEPA 也能拟合候选关系。要证明价值，需要出现强普通方法在同训练信息下无法以合理样本有效学会的、可复现的特定结构规律；本轮没有此证据。

一个可暂时冻结的机制假说是：**跨 episode 学习局部相互作用的修正规律，在线证据只识别其系数；沿候选的交互发生位置与时间传播同一个规律。** 数学上可以用下式记账，而不能立刻命名为算法：

\[
\Delta F(h,a_{0:H-1};D)
=\sum_{e\in\mathcal E(h,a)} T_{e\to H}(h,a)\,U_\theta(x_e,a_e;D).
\]

这里 \(\mathcal E\) 是模型从合法图像／动作历史推断的交互事件，\(U\) 是同种相互作用共享的局部修正，\(T\) 把该修正传播至未来。公式的线性相加只是局部近似；事件变化、接触新增／消失和误差反馈都可能破坏它。若 \(\mathcal E\) 依赖真接触标签或 \(T\) 依赖真实动力学而部署没有这些信息，机制不成立。普通 object-centric residual、graph network、meta-conditioned dynamics、Jacobian influence 可能已经覆盖它，需针对最终结构进一步查新；当前公式本身不新。

这是假说进入实证的价值所在：它具体预测**相同 online 证据的修订效应，应该随未来交互发生而出现，而不会按整个动作序列相似度均匀扩散**；并预测以未见过的次序组合已学交互时，显式局部作用共享应比全局 adapter 更省样本。反过来，若 context-conditioned Transformer／CLAW 适配器用相同数据已经同样成功，或事件对齐不比普通多步监督有益，就没有独立机制。

该假说是否足以承载 CVPR oral，取决于它在广泛重要视觉操作中的结果；仅在一个为事件模型设计的 toy task 上成功远远不够。它不能现在替代主线，也不能成为扩大训练预算的理由。

## 主动 probe 的真正附加条件

如果被动经历尚未辨认某个竞争分支涉及的相互作用，可以允许一次合法短探测；但必须把探测改变当前状态的后果纳入后续规划。不能从探测后的观察判断，再暗中用探测前保存状态的候选真值选择原动作。所有方法共享总环境动作、wall-clock 及训练权限。

有意义的检验不是“B 加权比参数 entropy 更合理”，而是：在当前两条真实可执行路线竞争时，模型是否知道哪一段未试过的接触规律决定排序，能否用一段可恢复、低代价的交互检验它，随后把结果正确传到新的候选后续动作。若只是参数估计后重规划，这是已有 task-oriented system identification；若学到了可跨形状、空间布局与动作组合复用的视觉相互作用修订算子，才可能形成实质区别。

## 建议的下一步边界

先对已有失败池做作用层面的归因，保持原 query 和全部 case：把 online 修正与真实需要修正的后果画在同一候选／时间轴，检查错误是否集中于接触发生时序、被交互对象及分支特定传播。此处隐藏物理量仅可作诊断标签，不能传给部署机制。

若没有这种可重复结构，就关闭该机制假说，保留原研究问题。若存在，再比较三个同信息控制：全局 CLAW/context adapter、直接 multi-step conditional residual、局部共享并传播的修正。公平加入相同的多环境预训练和被动／主动数据。第一轮只检验作用传播与 held-out interaction composition，不急于全系统优化；最终升级必须证明真实行动 regret 或成功率收益。

这条建议保留旧 A/B 问题，也放开了过去未经证明的“只靠冻结 F0 与极少 D 就能外推”限制。它仍是一条需要发现实验支持的机制方向；不能报告为已找到 oral 级 idea。
