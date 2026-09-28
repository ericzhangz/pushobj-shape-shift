# 文献边界：局部视觉证据如何修订行动未来

日期：2026-09-27。状态：有限范围的一手原文审查；不是全领域新颖性证明，也不是复现结果。

## 裁定

“保留历史、吸收新观察、修订未来”已经有成熟的状态估计定义和多个神经实现。进一步加入“无像素解码器”“多个行动分支”“区分观察与想象”，仍不足以成为独立贡献。我们应检验**既有视觉世界模型在允许合理历史、训练和推断之后，是否仍不能从局部证据得到正确的、跨行动查询一致的修订**；若能，当前切口结束。

不能把所有近邻简化成记忆模块。尤其：UWM-JEPA 已含观测更新；3D-Belief 已在线保留已观测场景并替换旧想象；MuDreamer 已有不靠重建学习的 posterior 更新；PSR 的状态本来就是一组可共同更新的行动条件未来预测。

## 八篇机制矩阵

“边界”指所读原文建立的范围，不能据此断言作者方法在其他任务必然失败。

| 原文与阅读范围 | 实际输入／信息权限 | 状态与更新方法 | 训练监督 | 任务及与本方向的边界 |
|---|---|---|---|---|
| [HERA, 2608.05523](https://arxiv.org/html/2608.05523)：Method 全部、Experiments setup/results、Discussion | 当前视频 context 内的历史 patch；冻结 V-JEPA 2 编码器与主预测器；不使用物理标签 | 按早／中／近时刻组织 memory，通过 gated cross-attention 写入 Memory Registers，再经冻结 self-attention 影响预测；逐窗口重新构建 | Physion 视频未来 latent L1 | IntPhys2 违反物理规律的 surprise 配对；已覆盖历史证据选择及路由。没有建立动作分支修订或校准后验的证据。不能把“冻结 WM + evidence routing”作为新点。 |
| [UWM-JEPA, 2605.25313](https://arxiv.org/html/2605.25313)：§2–4、§7、Appendix S1/S4 | 部分观测历史与动作；**反事实训练目标调用模拟器隐藏状态**后分支 rollout | 联合 density matrix；观测以正测量算子 sandwich 后归一化；动作以 unitary conjugation 推进；读出先 partial trace 再 projection | EMA latent target；反事实分支目标；冻结 probe | 受控隐藏速度与 blind rollout。联合谱保持不等于读出可用信息不丢失；belief 未证明为真实隐藏状态的校准后验。不能称其没有观测纠正接口，也不能把其权限直接照搬到普通视频。 |
| [State-space JEPA, 2608.13621](https://arxiv.org/html/2608.13621)：§4.4–6、§7.2/7.4、§8；其余方法段定向查读 | 观测历史；受控合成 HMM；部分实验知道生成模型以构造诊断 | categorical belief + shared Markov transition；明确区别 local encoding、predictive prior、filtering posterior；混合训练可蒸馏 exact filter | JEPA latent 对齐；对照加入序列 likelihood 和 filtering KL | 已讨论 predictive sufficiency、历史 residual 诊断、共享转移的路径一致性。一般 JEPA 与 HMM 的等价需要额外条件；有限合成实验不能外推为视觉 WM 通用结论。 |
| [DreamerV3, 2301.04104v2](https://arxiv.org/html/2301.04104v2)：Learning algorithm / World model learning、Critic learning、Results | 图像或向量、动作；训练读取 reward 和 continuation | RSSM：recurrent history h；prior p(z|h) 与收到图像后的 q(z|h,x)；同一状态出发想象动作后果 | 像素重建、reward/continuation、prior/posterior KL；端到端训练 | 多领域控制；已具备“保留历史 + 新证据 posterior + 多动作想象”的主路径。它不是给冻结 JEPA 加接口；也未因此获得任意新观察下的校准修订保证。 |
| [3D-Belief, 2605.11367v2](https://arxiv.org/html/2605.11367)：§3、§4、§9.1–9.2、§10 | RGB **及 camera poses**；实机还用 sensed depth 标定尺度；场景多视图数据 | explicit 3DGS 分 observed / imagined；新观察扩充 observed，旧 imagined 由新采样替换；渲染候选导航路径，执行短前缀后更新 | 多视图图像／特征重建、semantic distillation；主文列 optional depth，细节应按实际配置核对 | 场景记忆、补全、物体导航；已直接覆盖在线 belief 更新和行动比较。已读实验未建立物体动力学介入后、多个动作后果的选择性修订。不能把“观察与想象分离”重新包装。 |
| [MuDreamer, 2405.15083](https://arxiv.org/html/2405.15083)：§3–4、§5、Appendix limitations | 图像、动作；训练有 reward、continuation、value targets | 继承 RSSM 的 recurrent state、prior、observation posterior；辅助 decoder 的梯度不回传主模型 | reward/value/continuation/action prediction + KL；batch norm 防 collapse | Visual Control Suite、自然视频干扰、Atari100k；已证明不依赖重建也可学 posterior / dynamics。任务相关学习与冻结通用视觉表征的目标不同，但“decoder-free belief update”本身不新。 |
| [TRACE, 2608.26219](https://arxiv.org/html/2608.26219)：§3–5、Appendix A.3 | 带位置／时间的稀疏物理测量；预训练有完整 field trajectories；已知 latent-to-measurement operator | frozen Tucker decoder + diffusion prior；观测生成 latent posterior samples，Gaussian summary 与 Matérn state-space prior 融合；RTS backward smoothing | field reconstruction、离线 diffusion；推断时 likelihood guidance | 局部、缺测、非均匀时间的 field 重建；已覆盖冻结 latent 空间的证据积累及后验传播。没有动作；测量映射比真实视觉观测显式得多。其 α=β=1 融合退化为标准 Kalman 信息形式。 |
| [Predictive Representations of State, NIPS 2001](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)：§1–3，尤其 Eq.3 与 Theorem 1 | 离散动作／观测历史；理论构造可从已知有限 POMDP 获得 | 状态为一组 core tests 的条件概率；每收到 (action, observation)，按条件概率比共同更新，再读出任意 future test | 此文重点是表示与存在性，不是大规模视觉训练配方 | 已建立无需显式物理隐状态而表示、更新行动条件未来的原则；尚不能直接解决高维图像、连续动作下如何可辨识地学更新器。不能声称首创“以未来查询定义状态”。 |

## 经典状态估计已经解决到哪里

以下为结合上表的数学推论，并非新定理：若状态足够、转移与观测 likelihood 正确，那么 Bayesian filtering 给出唯一共享 posterior，所有动作未来都由它积分得到。新观测究竟影响哪些远期查询，取决于 posterior 的依赖关系，而非图像或 latent 坐标上的距离。PSR 则说明这种思想不要求显式物理状态。[PSR 原文](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)

因此，以下形式单独出现时不能作为新核心机制：

- 用 prior regularizer 加 observation-fit 修正 latent：这是 MAP／变分同化的基本结构；跨时间累积误差对应经典 4D-Var 思路。本轮未精读额外 4D-Var 论文，故不作其具体方法的优劣裁定。
- 按 observation 可见性或置信度融合：首先要和 learned filter、Kalman/ensemble 同化及 RSSM posterior 比较。
- 把修订局限到某个 latent 子空间：欧氏局部性不等于未来后果的语义局部性；非线性重参数化可改变该子空间。必须证明目标在所用表征中有意义。
- 将新 evidence 与已含相同历史的 posterior 相乘：可能重复计数。是否需要除去训练 prior／旧证据，必须由各概率对象定义推导，不能凭一个融合公式解决。

TRACE 提供了直接近邻：新颖性若只剩“在 latent 空间进行带 uncertainty 的局部证据融合”，已被该类方法覆盖。[TRACE §3、Appendix A.3](https://arxiv.org/html/2608.26219)

## 最多两个仍可检验的缺口

### G1：冻结视觉预测器中的修订可辨识性

**问题候选，不是已成立缺口：**只有图像／动作轨迹、没有真实物理状态与可计算像素 likelihood 时，能否学习一次证据更新，使同一个更新后状态对未参与本次拟合的动作未来也给出正确修订？

必要条件：先证明原有视觉表示和历史足以区分所需因素；训练数据覆盖相关动作，并明确何种观测／干预保证识别。若单条轨迹根本不能识别未执行动作后果，应改变数据契约或缩小主张，不能用架构弥补信息缺失。

最强反证：完整历史重新编码、合理训练的 RSSM / reconstruction-free posterior、或简单 learned observation head + latent MAP 已能通过相同预算的测试。若仅新机制拥有 simulator-state 分支监督，其优势不能归因于更新规则。

潜在贡献必须是可验证的识别条件、实际可学习的证据接口，以及它改善跨动作修订的因果证据。“冻结”仅有工程价值，不能单独撑起研究动机。

### G2：动态场景中保留与推翻的范围能否泛化

**问题候选，不是已成立缺口：**面对组合外局部观察，是否能纠正受影响的多步行动后果，同时保留依然有效的其他历史约束，并在旧证据确已失效时撤销它？

必要条件：任务必须同时含“保留”“纠正”“推翻”，且不同因素会经接触／遮挡／动作在未来相互作用；当前画面相同的两段历史不足以单独建立该问题。选择性应按未来查询的改变评估，不能把“未直接可见”当成“永不受影响”。

最强反证：共享 posterior 的常规模型、显式状态估计或 3D 场景 belief 的合理动态扩展已经解决；或者所谓差距只来自少见事件训练不足。若成立，也必须定位缺口是状态、观测模型、转移还是数据覆盖，而非笼统称世界模型不会纠错。

G1 与 G2 高度相关，不能自然算作两项贡献。立项时优先把 G1 作为机制问题、G2 作为最严格行为检验，除非实验发现两者确实独立。

## 对决定性实验的约束

1. 所有方法得到同一历史、同一新证据、同一训练轨迹和动作权限；GT state 只用于评分或单列 oracle，绝不混入主路径。
2. 必须测更新前后各行动查询的变化；平均预测误差降低不能证明修订正确。单独报告应变而未变、不应变却变、该推翻却固守。
3. 测试保留历史的方法也必须被迫接受反证；测试纠错的方法也必须保留其他仍有效证据。
4. 先将长历史和标准 filter 训练到合理水平；若同等信息下已解决，停止该问题版本。
5. 泛化需改变证据位置、到达时刻、因素组合与动作序列；不能只改变随机种子。

## 检索与阅读审计

- 搜索覆盖 frozen visual WM、decoder-free inference、latent data assimilation、belief correction、counterfactual evidence updating、PSR。截止到本轮日期。
- 定向阅读上表八篇。额外检索命中包括 DF³、World in World、Frozen Flows Forget、Generalised Latent Assimilation；本轮未深入阅读，不用于机制裁定，也不据标题宣称重复或差异。
- arXiv HTML 经浏览工具访问；对方法段使用同一官方 arXiv HTML 的 section 文本核读。Dreamer Nature 页面及 Tensor-Var OpenReview PDF 的访问未成功，未据其摘要作方法判断。
- 文中所有“已覆盖”仅指对应数学原则或报告能力；未执行任何论文代码、未独立验证报告指标、未完成穷尽式 prior-art 检索。
