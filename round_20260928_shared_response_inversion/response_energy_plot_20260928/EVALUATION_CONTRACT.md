# 冻结的响应评估与 energy plot 合同

2026-09-28。仅评估设计与既有实现核查；本文件作者未读取新测试结果、运行环境或训练模型，未使用 skills。`metric_contract.py` 是独立 NumPy 评估器，输入已经封存的预测与评分真值，不参与状态辨识或训练。

## 1. 核心图的对象与坐标

将“energy plot”理解为问题证据图，不声称这里存在物理能量或我们已提出能量求解原理。每个根历史 r、固定候选对 i<j、未来步 t 定义完整 384 维视觉差分

\[
d=z_{rit}-z_{rjt},\qquad \hat d=\hat z_{rit}-\hat z_{rjt},\qquad D=384.
\]

核心坐标为

\[
x=\|d\|/\sqrt D,\qquad y=\frac{d^T\hat d}{\sqrt D\|d\|}.
\]

- x 表示真实动作后果差异；y 表示预测在真实后果方向上保留了多少差异。
- y=x 是正确平行分量，y=0 是无该方向差异，y<0 是反向。散点不能单独证明整个预测正确。
- 必须同时记录正交误差 \(e_\perp=\|\hat d\|^2/D-y^2\)，因为 \(\|\hat d-d\|^2/D=(y-x)^2+e_\perp\)。光看预测差分模长会把错误方向误当成成功分化。
- 真实差异恰为零时方向不存在。投影和增益记录为 null；单列伪造差异 \(\|\hat d\|^2/D\)。不靠添加 epsilon 伪造方向，不删去零差异样本。
- 主图固定终点；逐步曲线报告全部未来步。用同一组真实差分比较各方法。箭头只是同候选对的预测变化，不表示在线修订。
- 全部候选对进入评分。根历史是聚合单位；候选对和时间步不作为独立统计样本。共享同一事实 episode 的多个根仍相关，后续置信区间应按 episode，而非仅按根重采样。本次小型形成试验以描述性结果为主。

## 2. 不能被核心图替代的完整误差

完整视觉 MSE 是所有未来帧、候选与维度的均值。另单列 proprio embedding MSE，不能把两个模态拼接后隐含改变权重。维度均方和终点均方均报告。

候选对差分 MSE 为所有无序 pair 的平均。若 e_i=预测_i-真值_i，则精确有

\[
\frac1{\binom C2}\sum_{i<j}\|e_i-e_j\|^2/D
=\frac{2C}{C-1}\frac1C\sum_i\|e_i-\bar e\|^2/D.
\]

因此

\[
\text{完整MSE}=\|\bar e\|^2/D+\frac{C-1}{2C}\text{差分MSE}.
\]

这同时给出回归校验和科学解释：所有候选一起偏移不会影响差分图，却会破坏完整后果。不能只报差分改善。

## 3. 原生三帧 rollout 的复用边界

已核查现有 `shared_response_formation_20260928/formation.py::cached_rollout` 和原仓库 `models/visual_world_model.py::rollout`：

- 三帧是模型时间帧，相距 5 个环境微步；`initial['visual']` 为 `[B,3,1,384]`，`initial['proprio']` 为 `[B,3,10]`。`encode` 内部才为 proprio 添加 patch 维，`rollout` 返回时再去掉该维。
- `actions` 使用归一化模型动作，打包每 5 个二维微动作为 10 维。前两块是三帧之间实际发生的历史动作，随后 H 块为未来候选。总形状 `[B,H+2,10]`。
- 结果时间长度是 `3+H`。评分完整未来必须切 `[:,3:]`；若为兼容旧目标而保留根帧则切 `[:,2:]`，但不能把这个根帧计入 H 帧预测误差。
- 旧 `load_completed` 只提取 RGB、proprio、normalized_model_actions；不能用 sidecar 隐藏状态作为新模型输入。
- 普通原生基线调用 `cached_rollout(..., theta=None, sources=None)`。新辨识基线若采用不同状态参数化，应清楚说明是对照算法，不能说它已经写入同一原生 backbone。
- 必查共同动作前缀得到共同输出；候选未来后缀不能影响过去预测。

## 4. 原生目标与选择评分

原仓库 `research/contrast_probe.py::native_objective_breakdown` 复现原 staged 目标。terminal 情况为

\[
c(U)=\operatorname{MSE}(z^v_H,z^v_{goal})+
\alpha\operatorname{MSE}(z^p_H,z^p_{goal}),\quad\alpha=1.
\]

对于旧 6 帧（根+5未来）路径，step=2 和 step=4 均走 terminal；不能给一个只剩 H 帧的新数组再机械沿用 step=4，否则条件 `step < horizon-1` 可能改变成 full_horizon。新实验若要求端点目标，直接明文固定 terminal，或保留根帧并断言返回 stage='terminal'。

full_horizon 代码对指数权重归一化后仍另做时间 mean，这是原实现事实，不应擅自修正。goal 只在预测封存后用于评分。新辨识模型若只预测视觉而不预测 proprio，不能拿它的视觉-only代价和原生 visual+proprio总成本当同一目标；需要共同目标或完整两模态输出。

选择比较使用实际真值代价、候选间代价差与 regret，完整轨迹与差分误差仍为主。旧池曾有重编码目标漂移，禁止把旧 ORACLE_COSTS 当未经核验的当前 encoder 真值；新评分必须统一 encoder、目标和公式，并把历史成本仅作为另一条有明确来源的审查值。

## 5. RGB 几何是辅助读出

checkpoint 自带 decoder 接收 visual `[B,T,1,384]`，输出 `[B,T,3,224,224]`。归一化逆变换 RGB=(decoded+1)/2，导出 uint8 前保留原始浮点与越界比例。`decode_obs` 返回的 proprio 仍是 embedding，不能当位置/速度。

旧 `rgb_rigid_observer.observe_rgb` 和 `contact_kinematic_reference.contact_geometry` 可用于可见物体形心、推杆中心与轮廓间隙；这些不是物理质心、接触力或精确接触标签。

旧44帧真实编码重构校准：形心误差均值1.93px、最大3.45px；gap误差均值0.93px、最大4.18px；object-mask IoU均值0.83。可支持明显分离诊断，不能支持2–4px接触边界推断。旧校准也不保证新分支、离开真实编码分布的预测 latent 解码有效。

每次新测试解封后，应对真实未来 encode→decode 走同一个observer作为读出参考；保留所有读出失败及分母，不能只在读出成功部分宣布主机制成立。若预测 latent 本身完整误差大、几何读出异常，则不能单凭 decoder归因。

## 6. 本模块验证

运行 `python metric_contract.py` 通过8项纯合成代数核验：精确预测、共同偏移、后果塌缩、方向反转、零真值差分、pair/中心化恒等式、平行/正交分解、完整误差/共同偏移/差分恒等式。随机pair恒等式最大绝对误差8.88e-16。这只验证评分实现，不是实验结果。

已复核文件：本分区 `shared_response_formation_20260928/formation.py`；`interaction_validation_20260927/score_interaction_probe.py`、`READOUT_AUDIT.md`；仓库根目录 `research/contrast_probe.py`、`research/reframe_v3/rgb_geometry_probe.py`、`models/visual_world_model.py`。
