# 响应状态路线的直接机制碰撞核查

核查日期：2026-09-28。范围：用户附件提出的真实交叉响应表、共同状态、动作移位算子、历史初始化及视觉后果链路。读取上一轮 PRIOR_ART.md，并重新查询原始论文正文。未使用本地 skills；未采集数据或运行模型。本笔记不负责所有 JEPA、Koopman、现代 foundation world model 近邻的穷尽检索。

## 裁定

**当前裸方案的核心机制新颖性不足，而且已经可以给出公式级碰撞；不是仅仅和旧工作“理念相似”。** `H=OX`、`H^a=OA_aX`、SVD/伪逆恢复共享算子、按动作字连续相乘、向量输出，这一主体属于谱实现/向量值加权自动机（vv-WFA）/线性二阶 RNN 已有机制。可复位后重放历史、执行不同测试来构造真实响应，也已有 PSR 原始工作。

把输出换成 V-JEPA/DINO token，可以形成新应用或新实验系统，但本身没有改变上述数学算法。添加一个从过去估计状态的神经 observer、谱初始化后的 BPTT、非线性状态更新或未来预测监督，也各有直接先例。**因此不建议以“真实响应→共同状态→动作算子→视觉后果”本身作为拟议主论文的独立新机制。** 它仍然值得作为机制透明的基线/科学探针；是否投入主线，需要先提出额外且必要的技术差异。

这不是“经典数学已经用过，所以永远不能借用”的结论。阻挡当前立项的是：现有描述暂时没有指出把经典构造搬到这里时，哪个既有算法不能解决的困难导致了怎样的新求解规则。

## 1. 最强碰撞：2017/2019 谱加权自动机，而非必须找到一篇 JEPA

[Multitask Spectral Learning of Weighted Automata，NeurIPS 2017](https://papers.neurips.cc/paper/6852-multitask-spectral-learning-of-weighted-automata.pdf)，§2 Corollary 2、§3 Corollary 4，明确构造输入字的 Hankel 表及在中间插入一个符号的移位表，由低秩分解和双侧伪逆恢复共享转移矩阵。其向量值版本让多个输出共享同一状态和算子，有限块要求完整基；有噪声时使用截断 SVD。论文 venue 可由[官方条目](https://papers.nips.cc/paper_files/paper/2017/hash/e655c7716a4b3ea67f48c6322fc42ed6-Abstract.html)确认。

以下是本候选与该机制的变量对应（转置/动作字方向约定不影响等价）：

| 附件对象 | 既有谱实现对象 | 判定 |
| --- | --- | --- |
| 历史根节点 h、未来动作测试 v、输出分量 j | prefix、suffix、vector-output index | 输入/输出语义特化 |
| `H[(v,j),h] = R_h(v)_j` | 向量值 Hankel tensor 的展开 | 同一可观测函数表形式 |
| `H^a[(v,j),h] = R_h(av)_j` | 在 prefix/suffix 间插入 symbol 的 Hankel block | 同一移位操作 |
| `H=OX`、共同状态 `x_h` | Hankel 低秩因子与前向状态特征 | 同一实现对象 |
| `A_a=O†H^aX†` | Corollary 2/4 的谱算子恢复 | 公式级直接碰撞 |
| `C` 取空测试对应行；`C A(U)x_h` | 空后缀读出与 word-product evaluation | 同一递推与读出 |
| 多个视觉 token 分量共享状态 | vv-WFA 的向量输出共享实现 | token 维数大不构成新的数学机制 |

适用边界：随机视觉系统的条件概率/均值表，不能不加条件地等同于单根确定性动作字函数；多根历史需具有兼容的状态实现。这个区别会增加算法要求，但尚未给出一种新的实现算法。附件的确定性、可重复根节点版本正是最容易落入上述同构的版本。

[Connecting Weighted Automata and Recurrent Neural Networks through Spectral Learning，AISTATS 2019](https://proceedings.mlr.press/v89/rabusseau19a/rabusseau19a.pdf)，§3 Theorem 2 证明 vv-WFA 与线性二阶 RNN 的表示等价，§4 将谱恢复扩展到连续输入向量。其递推为状态与输入的双线性张量收缩，输出是线性读出。**因此把离散动作换成 `b(a)`，令 `A(a)=Σ_l b_l(a)A_l`，也没有自然避开近邻。** 该工作还试验谱估计后梯度微调。[正式发表条目](https://proceedings.mlr.press/v89/rabusseau19a.html)。

## 2. “真实反事实交叉采集”也已有明确机制

[Learning and Discovery of Predictive State Representations in Dynamical Systems with Reset，ICML 2004](https://icml.cc/Conferences/2004/proceedings/papers/117.pdf)，§3 从不可见物理状态—测试矩阵转向可观测历史—测试矩阵 `Z_ij=p(t_j|h_i)`；§3.2 通过 reset、重复历史、执行未来测试估计条目，并据此发现 core tests 与模型参数。

它不是高分辨率视觉交互实验，但已经覆盖“同一历史后执行不同作用，收集真实响应并发现共同预测状态”的采集原理。故不能将新增分支数据本身称为新的因果辨识机制。要主张采集创新，需要新的测试选择/数据利用规则及相对传统实验设计的实质收益。原文同时承认有限测试发现可能提前停止；小块秩不再增长不代表全系统秩已识别。

## 3. 过去初始化、神经 observer 和非线性扩展并非空白

| 已发表原始论文 | 核心机制证据 | 与候选的关系/必须保留的区别 |
| --- | --- | --- |
| [Learning to Filter with Predictive State Inference Machines，ICML 2016](https://proceedings.mlr.press/v48/sun16.pdf)，§3–4、Algorithm 1 | 用未来观测统计定义预测状态，训练递归过滤器，并处理自身推断状态导致的训练分布差异 | 覆盖“未来给状态监督、过去/新观察递归估计”。它主要做过滤，不等于任意动作串的开放环物体后果生成 |
| [Predictive State Recurrent Neural Networks，NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/file/2bb0502c80b7432eee4c5847a5fd077b-Paper.pdf)，§4 Eq.5 | 观测和状态双线性交互、二范数归一化、深层堆叠、2SR 初始化后 BPTT | 覆盖把谱/预测状态转成可训练递归模块的路径；不是本候选固定 action-only 算子的逐字同构 |
| [Recurrent Predictive State Policy Networks，ICML 2018](https://proceedings.mlr.press/v80/hefny18a/hefny18a.pdf)，§3 Eq.1–3、§5 | `p_t=W_ext q_t`，按实际动作与观察条件化；`W_pred(q_t⊗φ(a_t))` 输出预测。过去特征两阶段回归初始化，然后 BPTT/策略训练 | 已结合动作条件预测状态、历史推断、连续动作特征及机器人控制。文中明确动作是 intervention，不只是观察相关性。不能靠冠名“因果”拉开区别 |
| [Predictive-State Decoders: Encoding the Future into Recurrent Networks，NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/file/61b4a64be663682e8cb037d9719ad8cd-Paper.pdf)，§3 | 给普通 RNN 内态增加对未来统计的预测监督 | 覆盖“让 hidden state 显式预测未来”这种最小修改；不提供共同 Hankel 算子实现 |

对本候选的判断：岭回归 observer 是标准线性可观测性逆问题。改成神经 observer 并不自动形成新贡献；预测状态从真实未来训练、部署由过去得到，也不是新的信息使用范式。若我们需要一个比这些方法不同的 observer，必须讲清现有推断为何在具体视觉作用任务中失效，以及新增约束如何解决它。

## 4. “连续动作、连续输出、机器人视觉”也不能作为空白声明

[Hilbert Space Embeddings of Predictive State Representations，UAI 2013](https://homes.cs.washington.edu/~bboots/files/HSE-PSRs.pdf) 用 RKHS 的条件嵌入算子表示受控未来分布并执行更新，以处理连续动作/观察；[UAI 官方 accepted list](https://www.auai.org/uai2013/acceptedPapers.shtml)确认发表状态。这削弱了“PSR 只有离散小玩具、我们首次扩展连续世界”的说法。它的核方法、条件化和本候选的有限向量确定性谱实现并不相同。

[Closing the Learning–Planning Loop with Predictive State Representations，IJRR 2011](https://journals.sagepub.com/doi/10.1177/0278364911404092)，[可读作者稿](https://www.cs.cmu.edu/~ggordon/boots-siddiqi-gordon-closing-loop-psrs.pdf)的§III–IV，已经从视觉动作轨迹估计 transformed PSR 并规划。示例机器人使用 16×16 彩色视觉；共 10,000 条短轨迹，其中 2,000 条选核中心、8,000 条估计模型。它是小型模拟导航，不能据此声称现代复杂物体接触已解决；但足以否定“视觉历史→PSR→预测→规划”首次完成的主张。

[Revisiting Bi-Linear State Transitions in Recurrent Neural Networks，NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/64d71dc33dbd217e3752c9ff1a41798d-Paper-Conference.pdf)，§2 Eq.2 与 state-machine/group 分析，将输入依赖的乘法状态转移用于状态追踪和长序列外推。因此“动作作为变换作用于内部状态”“矩阵不交换所以保留动作顺序”也已有当代明确机制。其主要任务是抽象状态追踪，不是视觉物体响应。

本次没有确认一篇已经把**现代冻结视觉 token + 同历史交叉试验表 + 谱移位恢复 + 合法历史 observer + 复杂接触自由推演**这整个具体组合全部实现的已发表论文。但这只能说明暂未找到同一包装组合，不能提供核心机制新颖性的正证据。直接谱实现已经同构，其余组件也多有先例；审稿人无需找到一篇一模一样的 JEPA 才能质疑贡献。

## 5. 同时出现的状态抽象近邻，不应误判成精确同构

[Observable quotient world models for knowledge-state abstraction under partial observability，Knowledge-Based Systems，2026](https://www.sciencedirect.com/science/article/pii/S0950705126014942)，作者 Yan Jiao、Pin-Han Ho、Limei Peng。出版商页面可查摘要、Introduction 与 methods snippets；完整正文未完整取得。本次仅能确认其从 sensing-action 条件观测似然定义 observable equivalence、quotient belief、约束下的 dynamics/filtering，并有神经 belief/dynamics/active-sensing 实现。

它会碰撞“按可作用区分性重新定义世界状态”这层动机，但不是已经核实的 `H^a=OA_aX` 谱算法碰撞。观察等价的动作族、时间长度、conditioning、干扰规范化都必须逐项比较。不能把摘要中相同的 quotient/state 字样当成逐式等价。

## 6. 建议给主代理的决策

1. **暂停把当前谱响应状态当作新核心机制立项。** 它是可信的既有数学基线，不是尚无人做过的底层重建。
2. 保留原主问题：视觉 world model 对动作后果的区分与连续推演。此次查新没有推翻该任务。
3. 在新数据采集前，先要求候选写出一句可审查的差异：对哪类现有 PSR/WFA/递归视觉 world model，增加或改变了哪个必要数学对象/约束/求解步骤，并产生哪项它们没有的能力。不能用“高维视觉”“真实响应”“连续动作”“BPTT”“多步预测”代替这句话。
4. 若差异尚写不出来，下一步应为机制形成，而不是将谱原型扩成大规模实验。小原型仍可用于验证某个具体困难，但要把目的明示为研究探针，不能在成功后再补 novelty。

最强拒稿论证可被简洁表述为：该方法将已知 vv-WFA/PSR 谱实现接到视觉特征上，以标准复位试验估计 Hankel 表，再用标准 observer 初始化；现有材料尚无超越这一组合的新算法原则。下一步必须正面推翻这条论证，而不是避开旧文献名称。

