# 共同响应状态路线：核心机制查新与投入裁定

核查日期：2026-09-28。对象为上一轮确定的“真实作用响应 → 共同预测状态 → 共享动作移位 → 连续视觉后果”候选。未使用本地 skills，未启动新采集或训练。

## 1. 结论先行

**当前版本没有通过以新核心机制为目标的新颖性检查。建议将标准响应表谱实现降为数学参照/基线，不直接围绕它扩大主论文实验。**

这一裁定有三层依据：

1. 核心响应表分解与动作算子恢复，不只是“受经典理论启发”，而是在相应有限维假设下与已发表的向量加权自动机/预测状态谱实现直接对应。
2. 从视觉学习可推进的线性或结构化潜在动力学，也已有正式发表的现代控制/世界模型研究。
3. 近期JEPA及相关预印本已覆盖反事实分支监督、自由推演中保留隐状态、动作后果差分等相邻主张，进一步压缩了仅靠改叙事形成贡献的空间。

**没有找到一篇经本轮核对、将我们全部设定原样组合的现代JEPA论文；这不构成“尚有新颖性”的正面证据。** 已知求解器换成视觉目标，再加常规历史估计器与分支采集，整体仍可能只是已有技术在新资产上的实现。

上一轮“可作为形成原型”的判断属于可执行性。它不足以推出“值得作为CVPR oral主机制投入”；本轮将这两个判断明确分开。保留动作条件连续推演这一主任务，调整的是候选机制的地位。

## 2. 冻结查新的实际机制，避免只按名字比较

待检查版本包含：

- 用多个未来动作试验的真实视觉响应区分历史，而不把当前图像直接当成完整动态状态。
- 重放同一根历史，采集后续动作词v与前置动作后的av。
- 形成响应表H及移位表H^a，求H≈OX、H^a≈OA_aX。
- 用SVD/伪逆恢复共同坐标、共享动作作用和输出读出。
- 部署只从合法过去初始化，并通过动作矩阵的有序乘积产生未来输出。
- 用完整后果、分支差分及动作选择验收，而非只看辅助拟合。

比较分为：**求解机制直接对应、部分机制重合、相同研究问题、任务不同的邻近工作。** 不把“都预测未来”称为撞机制，也不因数据集或输出维度不同就宣布机制独立。

## 3. 已正式发表的直接先例

| 工作及发表状态 | 已有核心内容 | 对当前候选的影响 |
|---|---|---|
| [Predictive Representations of State，NeurIPS 2001](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) | 以未来可观测试验的预测定义状态 | “当前观测不是状态、未来作用定义状态”不能作为新的建模原则 |
| [Learning and Discovery of Predictive State Representations in Dynamical Systems with Reset，ICML 2004](https://icml.cc/Conferences/2004/proceedings/papers/117.pdf) | reset后重放历史并执行tests，发现和学习PSR | 同历史交叉试验不是新的数据机制 |
| [Multitask Spectral Learning of Weighted Automata，NeurIPS 2017](https://papers.neurips.cc/paper/6852-multitask-spectral-learning-of-weighted-automata.pdf) | §2推论2、§3推论4给出低秩Hankel因子、双侧伪逆恢复符号算子和向量读出 | **公式级直接对应**，比仅引用PSR概念更强 |
| [Connecting Weighted Automata and RNNs through Spectral Learning，AISTATS 2019](https://proceedings.mlr.press/v89/rabusseau19a.html) | 向量输出、Hankel分解、共享符号矩阵、连续输入线性二阶RNN的谱学习 | 当前最强的求解器重合；视觉token作为向量输出不改变代数身份 |
| [PSIM，ICML 2016](https://proceedings.mlr.press/v48/sun16.html)；[PSRNN，NeurIPS 2017](https://arxiv.org/abs/1705.09353) | 学习预测状态的递归推断；谱/回归初始化与可训练网络 | 常规神经初始化、非线性递归或后续梯度优化也不能自然补成新机制；过滤与无观测推演仍需区分 |
| [Recurrent Predictive State Policy Networks，ICML 2018](https://proceedings.mlr.press/v80/hefny18a/hefny18a.pdf) | 动作条件预测状态、过去特征初始化、连续动作特征与BPTT | 加入合法历史推断及动作预测接口本身已有强先例 |
| [Efficient Dynamics Modeling in Interactive Environments with Koopman Theory，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f074a994e062146561db9cdc63999efa-Abstract-Conference.html) | 从像素/状态学习潜表示，在动作条件线性系统中连续推进并训练多步后果 | 直接限制“将视觉世界模型改为算子动力学以改善长程”的宽泛贡献；它不是我们的响应Hankel求解器 |
| [RoboKoop，CoRL 2024，论文集2025](https://proceedings.mlr.press/v270/kumawat25a.html) | 视觉控制表征、Koopman线性动力学及控制 | 不能仅凭视觉输入、控制用途和低维线性演化区分；其任务条件控制与本项目完整响应合同不同 |
| [TD-JEPA，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html) | 策略条件的长期预测及successor features，低秩动力学表征 | JEPA中“超越一步、由未来结构塑造状态”已有强工作；它不等于任意给定动作词的逐步视觉后果实现 |
| [Learning Bilinear Models of Actuated Koopman Generators from Partially Observed Trajectories，SIAM 2024](https://epubs.siam.org/doi/10.1137/22M1523601) | 双线性隐状态模型、EM、从过去观测估计状态后预测新控制下的输出 | 部分可观测受控实现不是空白；其估计路线与交叉响应表不同 |

PSR与WFA工作决定数学骨架的归属，神经预测状态及现代受控动力学工作约束常见扩展。不是把所有方法说成同一个算法。

### 最直接的代数对应

本候选部署形式为

\[
\widehat R_h(a_0\cdots a_{k-1})
= C A_{a_{k-1}}\cdots A_{a_0}x(h).
\]

在确定性向量输出、有限线性实现设定下，这就是共享符号转移的向量序列函数实现。历史—后续表的低秩因子给出状态，移位块给出转移。采用

\[
A_a=O^\dagger H^a X^\dagger
\]

不会新增一个不同的求解原则。连续动作若写成`A(a)=Σ b_l(a)A_l`，则是所选动作特征上的双线性递推，与线性二阶RNN族相连。一般随机PSR还需动作—观察条件化；不能将它与确定响应矩阵不加条件地完全等同。

新增图像编码器改变输出特征，新增observer改变状态估计途径；这些部件可能对性能重要，但现版本没有给出区别于已有实现和推断方法的专有规则。详见[响应状态支线](RESPONSE_STATE_COLLISIONS.md)。

## 4. 近期JEPA/世界模型：哪些会撞，哪些只是相近

下表除特别说明外均按预印本处理；未核实正式录用，不把arXiv公开称为会议发表。预印本仍应纳入研究重合判断。

| 工作 | 机制/问题重合程度 | 实质区别 |
|---|---|---|
| [UWM-JEPA，2026-05](https://arxiv.org/html/2605.25313v1) | **高：**真实反事实动作目标、改变内部状态及动作算子、无新观察的连续推演 | 密度矩阵与酉共轭保持联合谱；不从H/H^a辨识一般响应实现。不能把其特定实验解释扩成JEPA必然遗忘 |
| [Koopman Dreamer，2026-07](https://arxiv.org/abs/2607.19719) | **高层结构较高：**可控潜在动力学、双线性动作交互和长程稳定性 | 学习结构化随机世界模型，并非交叉响应表的谱恢复 |
| [CQM，2026-08](https://arxiv.org/html/2608.22092v1) | **分支与差分机制较高：**同步反事实数据，直接学习动作后果对比 | 输出是固定继续策略下首动作的累计特征差分；不是任意动作串的完整逐步视觉状态。若我们转成“去掉共同未来，只保留动作差异”，重合会明显增加 |
| [SG-JEPA，2026-09](https://arxiv.org/abs/2609.10464) | 多步作用、共享递推和表示训练的叙事相近 | 方法并非一个新的Hankel实现；不能误报其增加了独立半群一致性正则 |
| [ACPC，2026-08](https://arxiv.org/abs/2608.12939) | JEPA状态的动作条件后果一致性、推演后的区分性诊断相近 | 研究视觉扰动下的诊断，不是共同响应状态的学习算法；也不是我们跨动作分支收缩现象的逐项复现 |
| [ARPS，2026-09](https://arxiv.org/html/2609.23369v1) | 多未来监督形成紧凑预测状态 | 状态供动作专家生成动作，部署不推进任意给定动作词；不能只按“predictive state”名称判同构 |
| [VJEPA，2026-01](https://arxiv.org/abs/2601.14354) | JEPA与预测状态/贝叶斯过滤之间已有明确连接 | 概率预测与条件化路径不同，非响应表谱辨识；“把JEPA理解为PSR”不能单独宣布新颖 |
| [Spectral-Target Physical Latent Structuring，2026-09](https://arxiv.org/html/2609.04264v3) | 非坍塌潜表示仍忽略物理信息的动机相近 | Fourier对象几何辅助目标；这里的spectral不是Hankel实现 |

更多方法公式、理论条件、发表核对见[JEPA支线](JEPA_COLLISIONS.md)、[世界模型支线](WORLD_MODEL_COLLISIONS.md)、[新增重点比较](SUPPLEMENTARY_COLLISIONS.md)。这些是边界检查，并非每篇都必须作为完全同构的实验基线。

## 5. 不能再作为独立贡献的表述

- 用未来作用定义状态，而不只重建当前图像。
- 在共享潜在空间学习动作算子并递归组合。
- 用SVD/Hankel/伪逆提供可解释求解。
- 对相同历史采集多个真实分支，使模型使用动作。
- 加多步损失、历史observer、神经预测头或非线性动作特征。
- 避免状态坍塌，保持不同动作的后果差异。

以上各项可成为完整新方法的组成部分，但目前不足以单独支撑新机制。将它们组合也并非自动无价值；需要证明组合中的新关系实际解决了此前方法未解决的困难，而不是依靠命名或更换评测域。

## 6. 对研究投入顺序的具体调整

**现阶段不把“完成经典谱实现＋扩充视觉实验”作为主论文推进计划。** 标准实现可以保留为受控数学基线；如果小原型用于排除某个明确新假说，它仍有价值。但实验成功本身只能先证明路线适用，不自动证明方法独立。

借用成熟应用数学并非问题。以用户要求的ChordEdit式研究强度衡量，关键是：针对重要接口中的具体矛盾，是否重新定义了需要求解的对象，并推导出对现成方法有实质区别的计算规则。当前骨架已有直接实现先例；缺的是**为什么现成实现不够，以及新规则改变了什么**。

下一份机制草案如要恢复优先投入，至少需在动手采集前写明：

1. **确切新对象或新求解关系。** 删除本文已覆盖的PSR/谱法/Koopman/分支监督部分后，还剩哪条不可被直接替换的规则？
2. **一项任务所需的能力差异。** 它与最近方法的区别，应导致未见动作后果的不同预测或不同可辨识范围，而不只是实现形式不同。
3. **对应的最强普通替代。** 可以是同信息PSR/双线性辨识、视觉Koopman、多步递归训练，具体选取应随所主张的差异决定。不能只击败冻结原生模型。

这三项是投入条件，不是已经识别出一个新的研究空白。有限数据、非线性闭合、连续动作、部分可观测性都已有长期研究，不能把这些关键词直接填进“剩余创新点”。若没有形成实质差异，应更换数学实现候选，保留“动作条件连续视觉后果”主问题。

## 7. 检索范围、证据与限制

本轮采取三支并行定向检索加根任务交叉核对：PSR/WFA谱实现及reset发现；现代视觉Koopman/世界模型；JEPA及预测状态相关2024–2026工作。原始来源为PMLR、ICLR/NeurIPS/ICML官方论文、arXiv原文与官方项目资料。高风险候选读取方法章节/公式，不以摘要定机制。通过关键词及相关工作回溯检索；不是形式上的穷尽证明。

主要检索词族包括：`JEPA predictive state representations`、`JEPA Hankel/spectral`、`world model spectral learning`、`visual Koopman action conditioned`、`future tests reset predictive states`、`action-conditioned predictive consistency`、`counterfactual quotient`、`bilinear state transitions`。搜索中出现的疑似本项目公开仓库材料不算独立prior art；AI摘要和综述仅用于找原文。

未发现完全相同现代JEPA实现，只能表述为**本轮未找到**，不能表述为不存在。反过来，已有正式发表的骨架先例足以否定“当前数学骨架本身是新贡献”，无需等到找到名字完全相同的JEPA。

本轮产物是研究判断及来源对照，没有新模型性能结果，也没有对所有引用论文的定理进行逐条独立证明。
