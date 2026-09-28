# 原生算子上的数学重建：可计算对象、等价退化与几何风险

2026-09-27。本审查独立于主线程的候选推导。未读取或使用本地 skills；未调用模型、GPU、decoder 或环境；未修改原生算子、checkpoint 或旧结果。执行了一项只读旧缓存的 CPU 几何检查。本文区分完整可实现的问题、尚缺先验的问题，以及只是普通拟合换坐标的问题。

## 结论

原生接口已经足以实现一个所有候选共享、作用于自身预测历史的修订预测器，无需替换 backbone，也无需建立另一个 world model。全路径最小作用量、占用分布加权算子距离、multiple shooting 都能写成完整可计算优化；但这些名称本身没有产生新的跨动作信息。在线性局部模型中，前两者通常退化为带不同度量的最小二乘；硬约束 multiple shooting 与直接全路径拟合是同一个问题，软约束版本则可能靠不部署的节点缺陷降低训练损失。

联合历史重traction是一个不同的候选建模对象：它试图限制模型在自身预测历史上的响应，而非仅拟合某次输出残差。但当前每案 11 帧 D 只有一条完整原生三帧历史，不能据此辨认联合历史流形及其法向；把 D 的仿射包络直接当流形会压缩已有动作差异。这条方向需要额外、明确的合法几何先验，而不是再给 D 加一个 PCA 名称。

## 1. 已核对的原生路径

- 模型：[visual_world_model.py:410](../runtime_snapshot/models/visual_world_model.py)。`rollout` 的 `observation_transition(history, action)` 入口接收同一个原生缓存的最近 `num_hist=3` 帧，以及当前未经 action encoder 的 packed 动作。这里的动作已经过原有标准化，不能再解释为物理原始坐标。
- 在当前 checkpoint 布局，完整每帧 token 为 `[visual 384, proprio 10, action 10]`，共 404 维；callback 返回 394 维观测 token。`next_observation` 检查形状、dtype、device 和有限性；后续 action 插入仍由原生 `replace_actions_from_z` 负责。[入口代码:474](../runtime_snapshot/models/visual_world_model.py)
- 既有 `anchored_transition` 已实现 `native(history) + after(history,action) - before(history,action)`，然后把结果写回同一缓存；它证明该入口可以放置共享残差，并不证明旧增量本身可迁移。native_preserved_increment.py:36（本地附件或输入，未上传）
- 原生编码、拼接与动作绑定的已有帮助函数为 [_join_encoded_observation_action](../runtime_snapshot/research/reframe_v3/matched_feedback_forecast.py)。不应再建一套学生 rollout 来近似宿主的历史行为。

因此，一个完整候选可以保持

\[
f_\theta(h,a)=\operatorname{obs}\bigl(F_0(h)\bigr)+r_\theta(h,a),
\quad h_{t+1}=\operatorname{native\_append}(h_t,f_\theta(h_t,a_t),a_{t+1}),
\]

其中同一个参数 θ 在一次查询的所有候选和所有时间步共享。θ 的函数类与离线先验尚需具体确定；上述写法不是认领新算法。

**联合历史修订的边界：**现 callback 只返回下一个观测，不能持久改写已有历史槽。它直接支持 `F0(Rθ(h))` 这种输入映射，也支持最后预测 token 的修订；如果候选要求修订整段 history 并让旧槽的修改持续存在，就应明确扩展当前原生 rollout 的缓存 API。不能在 callback 内对收到的 view 偷偷原地写入，制造两种不同的状态语义。

## 2. 数据能提供什么

源数据根为 `<ADAJEPA_REPO>/artifacts/round_6_shared_revision/results/`。每案此前完成的两块在 `donor_{T,L}_seed101_n12_capture/real_evidence/s{sample}_executed_mpc{0,1}.npz`。`load_completed` 严格读取已完成图像、proprio、规范化动作，合并为 11 帧、10 个 10Hz 动作；每五个动作构成一个原生 0.5s 宏步。[读取函数](../runtime_snapshot/research/reframe_v3/rgb_geometry_probe.py)

旧封存缓存 native/PREDICTIONS.pt（本地附件或输入，未上传） 包含：

| 对象 | 每案形状/数量 | 本轮用途与边界 |
|---|---|---|
| `source_z.visual` | `(1,11,1,384)` | 全部是合法已完成 D，可用于拟合或分析 |
| `source_z.proprio` | `(1,11,10)` | 编码 proprio；不能假设可仿射反解为物理状态 |
| 8 候选动作 | 每条 5 个宏步，10 维 packed action | 查询时可见，可形成不带真实标签的模型占用分布 |
| `plans[*].arms.FULL.z` | 6 帧，初始帧 + 5 个预测 | 是模型产生的路径，可用来定义部署度量；不是正确路径标签 |
| conditional cache | 真实未来历史条件下的一步预测 | 相对原查询使用未来，不能用于该查询的 D 校准 |
| evaluation targets | 实际分支未来 | 只能隔离评价，不能用于选 θ、投影维数或修订方向 |

11 帧 D 在原生 stride=5 下只有 `[0,5,10]` 一个完整三帧输入 history；没有后续第 15 帧目标。它允许两次从较短历史启动的已完成宏步拟合，也允许把两个宏步合成一条全路径事实；不能把 10Hz 相邻三帧当成大量原生 2Hz 三帧训练样本。这一差异尤其限制在线 joint-history geometry 的估计。

## 3. 全路径最小作用量的完整版本与退化情形

令 \(\Phi_\theta^t(h,U)\) 为上述原生递推到第 t 步的观测。一个合法的完整目标可以是

\[
\min_\theta L_D(\theta)+\lambda\,\mathbb E_{U\sim\nu}
\sum_t\|f_\theta(h^\theta_t,a_t)-f_0(h^\theta_t,a_t)\|_W^2,
\qquad h^\theta_{t+1}=\operatorname{native\_append}(h^\theta_t,f_\theta(h^\theta_t,a_t),a_{t+1}).
\]

其中 \(L_D\) 可以对合法完整事实路径评分；\(\nu\) 是预先固定的可见候选分布或离线动作先验，不含查询真实后果。所有路径用同一个 θ。W 可以先沿用外部度量 `visual MSE + proprio MSE`，即每部分按自身维数归一化；把这个度量称为物理动能需要额外证明。

此问题可通过原生 autograd、JVP/VJP 或普通反向传播求解。它和普通 factual fitting 的实际差异是：使用了自身预测前缀上的算子作用，而不是只惩罚参数距离或真实历史单步误差。但是：

1. 若优化的是每条查询路径各自的控制修正，而无查询端点/路径约束，原生路径自身就是零作用量解。它不会因为目标被叫作 least action 自动改善。
2. 若事实与查询各有独立自由控制，事实拟合也不能耦合到查询；必须使用共享函数或明确的跨路径先验。
3. 若把源路径和目标路径分布设为待执行动作的真实未来，就使用了部署时没有的答案。当前不存在与 ChordEdit 相同的可查询目标条件场。
4. 对 θ=0 线性化，事实作用写成 Aθ，占用历史上局部输出变化写成 Gθ，目标变为 \(\|A\theta-r_D\|^2+\lambda\theta^TH\theta\)，其中 \(H=\mathbb E G^TWG\)。这就是广义 ridge，不是越过旧 A/B 问题的新信息来源。

因此可保留它作为强普通参照或候选的求解层；尚不能把“从拟合改成最小作用量”本身当作主贡献。

## 4. Occupancy metric 与真正的全路径 metric 不能混淆

冻结原生候选路径 \(h^0_t\)，令 \(G_t=\partial_\theta f_\theta(h^0_t,a_t)|_0\)。局部占用度量是

\[
H_{\rm local}=\sum_{U,t}G_t^TWG_t.
\]

它可抑制访问区域上的单步算子变化，但不包含修订反馈后的传播。令 \(J_t=\partial_\theta\Phi^t_\theta(h,U)|_0\)，则

\[
H_{\rm path}=\sum_{U,t}J_t^TWJ_t
\]

包括整个原生缓存的导数传播；可以用矩阵无关乘法，没必要显式保存巨大 Jacobian。后者确实在度量“同一更新对整个部署路径的作用”，与参数 L2 不同，也比只对单步占用求和更符合本任务。

但在 H 正定的简化情形，事实硬约束下的最小变化解仍然是

\[
\hat\theta=H^{-1}A^T(AH^{-1}A^T)^\dagger r_D.
\]

即换了坐标/度量的伪逆。H 奇异时还要限制到可识别商空间或加入有意义的正定先验，不能把数值 regularizer 描述为新机制。Hpath 的保守方向取决于原生模型，原生模型已将某些动作后果压扁时，该度量不会自动指出应当扩展哪一个被压扁方向。**保护已有预测的变化小，与预测更正确是两件事。**

若最终候选只有以上变化，公平比较应直接包括全路径事实拟合 + 参数 ridge、局部 occupancy ridge、完整 path metric；允许它们使用同样 D、候选动作和计算预算。不能通过只比较弱单步拟合制造新颖性。

## 5. Multiple shooting 的作用与不能承担的主张

在已完成 D 上增加若干 history 节点 s_j，求解

\[
\min_{\theta,\{s_j\}}\sum_j\|\operatorname{obs}(s_j)-z^D_j\|_W^2,
\quad s_{j+1}=\Psi_\theta^{\ell_j}(s_j,A^D_j).
\]

硬约束满足时，消去 s_j 就是同一原生算子的全路径拟合。它可能改善长路径优化条件、梯度传播和并行求解，但不会增加 D 的信息，也不会自动解决跨动作可识别性。

若将等式换成软缺陷惩罚，节点可以通过偏离真实动力学链来拟合观测。部署时没有这些未来观测来放置节点，因此必须单独报告缺陷是否降到可忽略，以及用同一 θ 自由推演是否仍获得改进。若把节点缺陷解释为过程噪声，则需要过程噪声模型和预测时的边缘化规则；这是新增建模承诺。逐步置入真实未来 history 的旧 TF 诊断不能作为它的部署算法。

## 6. Joint-history retraction：几何承诺必须是什么

令 \(\mathcal M\) 为给定动作语义下可实现的 history 集合，\(R\) 为其附近的重traction。理想局部性质包括合法 history 上 \(R(h)=h\)，以及真实可实现扰动方向 v 上 \(DR(h)v=v\)；希望被抑制的是可辨认为不合法的误差方向。

问题不在是否能写这个算子，而在如何知道这些方向：

- 一条 D 的时间差分只显示这一条执行路径上的变化；它没有覆盖其他接触方式、动作次序或隐藏状态方向。
- 一致应用于所有帧的亮度/颜色变换能给出图像生成或观测噪声方向，但“图像 nuisance tangent”与“物理状态 tangent”不同。前者可能应被表示忽略，不能直接作为所有应保留动力学方向。
- 空间图像变换若要解释为物理等变，必须同时处理 proprio、动作单位、几何边界等；只平移 RGB 通常不是一个合法受控 history。
- 无标签高斯 latent 扰动可以定义人工去噪任务，却不能证明这些方向就是原生递推误差的法向。噪声分布是新增先验。
- 保留原生模型的动作灵敏度 \(DR\,\partial_U\Phi_0\approx\partial_U\Phi_0\) 是可计算的防压缩约束，不使用查询真值；但如果原生灵敏度已经退化，它不能恢复被遗漏的真实动作方向。
- 离线多动作、多状态、同机制的合法视觉 history 可以提供更广的流形/响应先验。它应与普通 adapter/denoising/多步训练参照公平共享；在线 D 负责选择或校准其中的 episode-specific 几何，不能承担凭一条轨迹恢复全流形的任务。

`E(decoder(z))` 的固定点、幂等性和真实编码保持性是有用必要条件检查，但它不是自动成立的正交投影，更不保证保留动作响应。它也没有约束编码 proprio；如果用纯视觉重编码解释 joint-history 几何，必须补上这一缺口。

## 7. 已执行的 CPU 小验证

代码：operator_geometry_check.py（本地附件或输入，未上传）。结果：OPERATOR_GEOMETRY_CHECK.json（本地附件或输入，未上传）。

只读取 native 封存缓存。每案用 **11 帧合法 source_z** 的均值与双精度 SVD 构造仿射投影，秩均为 10。对原生 FULL 八候选的终点差分计算投影后的平方范数保留率；没有加载未来真值、conditional cache 或模型。分别使用 visual MSE 与 visual+proprio MSE 两个已解释的度量，不按结果选择维数。

| case | visual：全部 28 pair 能量保留 | visual：fact–g99 能量保留 | visual+proprio：全部 pair 保留 |
|---|---:|---:|---:|
| T0 | 80.00% | 65.27% | 79.98% |
| T1 | 50.45% | 21.14% | 41.46% |
| L0 | 48.83% | 60.82% | 57.91% |
| L1 | 39.45% | 26.55% | 40.06% |

这项检查说明 D-only span 会删除大量现有模型分支差异，因而“不伤动作后果”不能从它的正交投影性质推出。它**没有证明被删掉的分量是真实信号，也没有证明投影一定恶化决策**；原生预测自身已经有误。这里的能量是 latent endpoint pair 平方距离，不是候选代价差，更不是修订效果。

模型调用 0，GPU 调用 0，训练步 0；源缓存 SHA-256 前后一致。该小验证只约束直接把 D-PCA 当合法流形的解释，不否定拥有独立、丰富几何先验的重traction。

## 8. 建议裁定

1. 全路径变化度量保留为完整可实现的强参照；它给出了新旧拟合方式的真实区别，但单独不足以形成主机制。
2. Multiple shooting 保留为优化选择；不将节点优化收益误报为新增知识或跨动作规律。
3. 几何修订候选可以推进必要条件实验。其研究承诺应是从可得离线/在线证据识别应保留与应抑制的 history 响应，并且解释为什么不会压扁未执行动作差异。当前 D-PCA 和简单 E∘decoder 都没有自动满足这一承诺。
4. 这些候选都应进入同一个 `observation_transition`/原生缓存路径；只有真正需要持久联合 history 修订时才扩展当前 API，写清旧槽语义并保留零修订原生一致性回归。

审查依据为既有 [MATHEMATICAL_BRIDGE](../research_progress_20260922/MATHEMATICAL_BRIDGE.md)、[HISTORY_AND_INTENT](../research_synthesis_20260927/HISTORY_AND_INTENT.md)、[原生验证报告](../interaction_validation_20260927/REPORT.md) 及本文列出的源码；本文没有执行新文献检索，经典优化等价性为显式代数分析，不据此宣称新颖性。
