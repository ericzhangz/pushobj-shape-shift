# 视觉世界模型与受控 Koopman 路线的机制碰撞审查

日期：2026-09-28。未使用本地 skills，未运行实验。本文件只审查世界模型/受控动力学近邻；PSR、OOM、WFA 的直接谱实现先例由另一份审查覆盖。阅读范围包括下列一手论文的 Introduction/Methods，出版状态以会议 proceedings、期刊或 arXiv 页面为依据。

## 裁定

**“视觉历史 → 共同潜在状态 → 共享动作算子 → 连续未来”已经有相当多的正式发表先例。** 将 MLP/Transformer 递推替换成线性、双线性或谱约束的潜在受控动力学，不能成为当前候选的独立核心贡献。

目前候选与多数现代 Koopman 世界模型仍有一个真实区别：它从交叉作用响应表 `H=OX, H^a=OA_aX` 识别坐标和算子，而非只对相邻轨迹训练视觉 lifting。但这个差别需要面对更早的 PSR/WFA/系统实现文献，不能因为现代论文没有写 `H` 就视为创新空白。跨过去两类文献，候选的各核心部件均有来源。

因此，保留它作为数学参照和可检验原型是合理的；将它直接作为新机制推进大规模实验，目前不合适。首先必须说明经典响应实现进入视觉交互后遇到的具体计算障碍，以及新求解规则为什么不能被普通谱实现或直接多步训练代替。

## 1. Efficient Dynamics Modeling in Interactive Environments with Koopman Theory

**状态：ICLR 2024 正式发表。** [会议页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f074a994e062146561db9cdc63999efa-Abstract-Conference.html) · [正文，§3.1–3.2](https://proceedings.iclr.cc/paper_files/paper/2024/file/f074a994e062146561db9cdc63999efa-Paper-Conference.pdf)

方法核验：视觉输入使用 CNN，状态和动作各自编码为 `x=g(s), u=f(a)`。§3.1 Eq. (3)–(6) 定义连续潜在受控系统 `dx/dt=Kx+Lu`，离散后 `x_next=Kbar x+Lbar u`，跨多步预测潜在量及解码状态。§3.2 将动力学组织成复数谱坐标，支持并行长序列计算并分析梯度与谱的关系。

**碰撞：** 从视觉表征转入具有可计算结构的动力学、共享状态递推、每一步接入动作、连续后果、多步潜在/输出监督，这些层次均已存在。因而“用应用数学重新建模 world model 的计算”只是研究方式，不能作为差异点。

**区别：** 其特定模型采用解耦的加性控制，当前每动作矩阵 `A_a x` 可包含更一般的状态—动作作用；其训练未使用当前提议的交叉响应表谱实现。但“更一般的动作相关矩阵”也须面对后文的双线性模型和 PSR/WFA。

## 2. RoboKoop

**状态：CoRL 2024 正式发表；PMLR 卷的出版年份为 2025。** [论文页](https://proceedings.mlr.press/v270/kumawat25a.html) · [会议卷说明](https://proceedings.mlr.press/v270/) · [正文 §3](https://arxiv.org/html/2409.03107)

方法核验：§3.1–3.2 由像素学习对比式 Koopman 表征，显式使用 `z_next=Az+Bu`；复数谱坐标与控制学习结合，学习 LQR 控制及 SAC 相关目标。表示受下游控制需求约束。

**碰撞：** 视觉表征不应仅服务重建、应保存行动相关信息；结构化潜在演化应改善控制。这两类高层动机已有正式视觉机器人论文覆盖。

**区别：** 它主要学习任务控制表征，不以真实后续试验集合定义状态，不从 `H/H^a` 求实现，也不把恢复任意动作的完整视觉后果作为唯一目标。区别是具体训练对象和求解路线，不能被扩大成“此前没有视觉作用状态”。

## 3. Koopman Dreamer

**状态：2026-07-22 arXiv 预印本；本次没有核实到正式会议/期刊发表，不能写成已发表主会论文。** [元数据](https://arxiv.org/abs/2607.19719) · [正文 §IV](https://arxiv.org/html/2607.19719)

§IV-C Eq. (2)–(3) 的确定性骨架为

\[
\phi_{t+1}=\operatorname{clip}\left(A_K\phi_t+B_a a_t+
\beta W_o[(W_\phi\phi_t)\odot(W_a a_t)]+B_z z_t\right).
\]

`A_K` 使用带界半径的二维旋转—缩放块；双线性项使动作效应依赖状态；随机状态提供局部信息。§IV-G 联合单步、多步及无未来观测的自由预测目标。其报告实验输入为本体感知和 LiDAR/任务向量，不能表述成已经完成大规模视觉交互验证。

**碰撞：** 若下一版只是增加共享双线性动作算子、谱约束、多步训练和合法初始化，核心很接近这类已有世界模型。普通 `A(a)=A_0+sum_j a_j A_j` 已是双线性控制，不能重新命名作为贡献。

**区别：** 当前提议的响应表识别与输出语义约束不等于其 Dreamer/EMA 教师训练。我们也没有理由照搬其稳定谱约束：稳定性本身不确保动作后果的区分度。

## 4. Learning Bilinear Models of Actuated Koopman Generators from Partially Observed Trajectories

**状态：SIAM Journal on Applied Dynamical Systems 23(1):885–923，2024-03-14 正式发表。** [期刊/DOI](https://epubs.siam.org/doi/10.1137/22M1523601) · [作者全文](https://arxiv.org/html/2209.09977v2)

虽然不是视觉大模型论文，它对“从不完整观测学出共享响应规律”的机制边界更直接。论文将受控可观测函数演化写成双线性隐马尔可夫模型，通过 EM 学习状态与参数；E 步使用 Kalman 过滤/平滑，M 步类似受控生成元的 DMD。部署实验从过去观测估计状态，再预测后续未见输出。

**碰撞：** 部分可观测、隐状态推断、共享控制算子、仅依据过去初始化、预测新的动作作用，本身均不是空白。它还明确讨论有限维潜在空间同时满足动力学封闭及线性输出重建的困难，这接近我们的实际门槛。

**区别：** 当前原型走交叉响应表分解，不是 EM；视觉高维输出也不同。但换估计器和模态不足以自动构成高水平新机制。

## 5. Course Correcting Koopman Representations：防止回到旧的重编码修补

**状态：ICLR 2024 正式发表。** [会议页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0429ececfb199efc93182990169e73bb-Abstract-Conference.html) · [正文](https://openreview.net/pdf?id=A18gWgc5mi)

这篇的具体机制是周期性解码再编码，再继续潜在线性推演。它讨论 encoder/decoder 不是全空间互逆、长程潜在轨迹漂移及局部线性化限制；周期性重编码不读取未来真实数据。

它不是当前响应表候选的直接等价物。但若下一步看到自由递推失败后，又提出“将解码结果投回编码流形”的修补，应明确承认已有对应机制，且项目过去的负结果不能被新名称覆盖。

## DINO-WM：相关任务基线，不是谱实现撞车

[DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) 是 **ICML 2025** 正式发表；不能因早期 ICLR 投稿稿而误写成 ICLR 2025。其[方法](https://arxiv.org/html/2411.04983)使用冻结 DINOv2 patch 特征和动作条件预测器，以自回归预测支持 MPC；解码器主要用于可视化。

它覆盖任务与评价接口，但没有当前响应表/SVD/线性动作实现。单独与它比较有价值，却不能代替与受控谱实现及同信息多步递归模型的机制比较。

## 参数化差别需要精确，不能靠措辞制造空白

以下是本次审查的代数判断，不是上述论文的新增结论：

1. 对加性模型 `x_next=Kx+Lf(a)`，增加常数坐标后即可写成 `x_aug_next=A(a)x_aug`。所以写成动作矩阵乘状态并不能自动区分于加性潜在模型。
2. `A(a)=A_0+sum_j b_j(a)A_j` 是标准状态—动作双线性/输入基展开；其一般性比纯加性大，但并非新结构。
3. `C A_b A_a x` 和 `C A_a A_b x` 可以不同，依赖矩阵非交换；这不说明普通 RNN 缺乏动作顺序能力。
4. 训练使用真实未来是监督学习的常态；当前交叉分支使同一历史下的不同动作可比较，是具体数据结构优势，不等于自动获得因果理论贡献。
5. 新方法若主要赢在交叉分支数据，而同信息普通递归模型也获得收益，就应把贡献归于数据/监督结构，不能归于谱动力学机制。

## 补充检索与限制

- 搜索还命中 *Spectral World Models: Provably Consistent Long-Horizon Video Generation via Koopman Operator Decomposition*，出现在[CVPR 2026 VideoWorldModel workshop 官方列表](https://videoworldmodel-workshop.github.io/)，标记为未公开 camera-ready 或请求撤下论文文件之一；[OpenReview 稿件](https://openreview.net/pdf?id=c7UEXxYkLp)检索片段给出 `z_next=K_a z` 的 unitary 动作矩阵。直接打开遭遇验证，未完整核正文。因此不把其理论、实验数字当作可靠事实，也不把它当 CVPR 主会论文；列为需保留的先行公开线索。
- 检索出现 `submissionrepo/world-model-psr`，内容疑似本项目同源记录，本次剔除，不作为独立先例。
- UWM-JEPA、SG-JEPA 等更接近的 JEPA 路线由主报告/另一审查核验，不在此重复。
- 本轮检索能证明“已有重大机制交集”，不能证明没有遗漏，也没有给出新颖性认证。结论不依赖任何一篇未正式发表的新稿：ICLR 2024、CoRL 2024 和 SIAM 2024 已足以排除宽泛的共享受控潜在动力学首创主张。

## 对下一步的约束

暂不把“响应状态 + 共享动作矩阵”升级成主机制。先用一页推导回答：**交叉视觉响应中，经典有限维实现具体失效在何处；新增未知量或约束如何改变可求解的问题，而非只替换估计器？**

若只能给出“预测更准、跨动作更好”的期待，没有新的结构性求解差异，现方案应作为基线使用。若能形成这种差异，再开展小规模验证，并确保所有有竞争力的基线获得相同真实分支信息。这个门槛保留原任务，不要求为了避撞车更换场景。
