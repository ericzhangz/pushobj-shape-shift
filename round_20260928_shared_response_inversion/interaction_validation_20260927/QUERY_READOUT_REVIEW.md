# 原生 query 交互读出独立复核

2026-09-27；只用 CPU 读取封存预测、评估图像及 VISIBLE_GEOMETRY.csv。未加载模型、未做注册拟合、未训练或运行环境。完整四 case × 八候选 × 五未来宏步全部纳入；L1 的两条分支图片仅供展示。

## 判定

**L1 存在经真实图像重构校准支持的“原生预测提前出现可见分离”现象，SINGLE 与 FULL 均未解决。** 但“共享接触边界位置估计错误是该现象的原因”仍未得到证明。L1 fact 在尚未读出分离的 step2/3 已有明显物体位置/姿态错误；后来的可见分离可能是此前响应错误的结果。此处物理接触、guard 与作用强度不是从 global latent 直接读出的变量。

因此当前可保留“这组原生分支没有预测出真实的交互持续结果”的诊断；不能写成“已证明边界偏移或接触时钟错误，故逆速度修正成立”。

## 真实 query 重构校准

去除重复初始 anchor 后，每个 arm 有 160 张未来图像。真实 RGB 全部可读；RECON_REAL 153/160 可读，7 个失败全部属于 T1 pusher 检测：g0 step4/5、g24 step5、g49 step5、g99 step4/5、fact step4。失败及未读出类别保留在 JSON，不用预测图像可读来掩盖重构失败。

| case | 重构可读/总数 | 重构 gap MAE / max(px) | 重构 object centroid MAE/max(px) | RECON 的明确 near→separated |
|---|---:|---:|---:|---:|
| T0 | 40/40 | 2.505 / 8.957 | 1.527 / 2.303 | 0 |
| T1 | 33/40 | 3.124 / 14.347 | 1.549 / 2.669 | 0 |
| L0 | 40/40 | 0.578 / 1.211 | 1.637 / 4.764 | 0 |
| L1 | 40/40 | 0.764 / 4.269 | 1.544 / 3.085 | 0 |

全量重构 object IoU 均值 0.819；pusher IoU 0.579。最大的重构 gap 错误并不小，不能套用一个四 case 共用的小像素误差界。L1 的 4.269px 极端来自 g99 最末帧（真实 ambiguous，重构 near）；fact 自身五个未来点 gap 重构误差均小于 0.767px。

这些是实际读出质量，不是 predicted latent 的保真保证。decoder 在真实编码上正确、在 off-manifold 预测 latent 上错误，仍是无法完全排除的解释。

## 全量 native 邻接差异

使用协议冻结的 near<=2px、separated>=5px、中间 ambiguous，不更改阈值。

| case | REAL=near 且 RECON=near，SINGLE=separated | 同条件 FULL=separated | REAL=separated，SINGLE/FULL=near |
|---|---:|---:|---:|
| T0 | 0 | 0 | 7 / 11 |
| T1 | 0 | 0 | 0 / 0 |
| L0 | 2 | 0 | 0 / 0 |
| L1 | 5 | 5 | 1 / 1 |

每 case/arm 的分母均为 40 个未来宏步；同分支连续点相关，不能当独立样本作显著性声明。

L0 的两个点均属于 zero_environment_action 的 step4/5，完整历史消除了明确分离。这是普通历史控制解决具体异常的例子。T0 更常见的是实际分离、预测仍邻接，反对笼统说原生模型总是“提前脱离”。T1 有很大的物体位置误差，却没有该类明确分离；事件分类本身不覆盖全部预测错误。

L1 两个 arm 的五个明确差异点属于：g0_before step4/5、official_pred_return step5、fact_matched_return step4/5。均在真实图像及其重构保持 near 时，预测变为 separated。更强的 g0 例子 gap 误差约15–17px，物体 centroid 误差约49–52px，而 pusher center 误差约1.4–3.6px；不支持“全是小幅 pusher 重构误差”的解释。

## L1 fact 与 g99 的每个宏步

表中是可见 gap，单位 px；t=0 是共同 query 初始帧。负值来自可见 mask/等效半径估计，不意味着物理穿透。

### fact_matched_return

| t(s) | REAL | RECON_REAL | SINGLE | FULL |
|---:|---:|---:|---:|---:|
| 0.0 | 0.044 | -0.696 | -0.696 | -0.696 |
| 0.5 | -0.125 | -0.415 | -0.102 | -0.171 |
| 1.0 | 0.178 | -0.589 | -0.446 | -0.263 |
| 1.5 | 0.065 | -0.480 | -0.536 | -0.240 |
| 2.0 | 0.034 | -0.723 | **6.661** | **6.399** |
| 2.5 | 0.238 | -0.480 | **6.722** | **7.826** |

这里的明确可见分离不能仅用 fact 的重构误差解释。step4/5 的 object centroid 预测误差是 SINGLE 37.20/44.60px、FULL 38.75/45.89px；相同真实帧重构误差仅1.87/2.15px。pusher center 预测误差 SINGLE 2.83/3.05px、FULL 2.28/2.57px。

但在 step2/3，SINGLE 的 object centroid 误差已为5.21/10.43px，FULL 为5.53/11.10px，此时 gap 全部仍 near。真实/预测图像也呈现物体姿态差异。因此数据支持“运动响应逐步偏离，后续出现不同可见邻接”；暂不支持把“分离提前”唯一放在因果链最前端。

### g99_after

| t(s) | REAL | RECON_REAL | SINGLE | FULL |
|---:|---:|---:|---:|---:|
| 0.0 | 0.044 | -0.696 | -0.696 | -0.696 |
| 0.5 | 0.026 | -0.624 | -0.356 | -0.156 |
| 1.0 | 0.023 | -0.144 | -0.353 | -0.192 |
| 1.5 | 0.514 | -0.815 | -0.707 | -0.244 |
| 2.0 | 0.178 | -0.327 | -0.365 | -0.532 |
| 2.5 | **3.601 ambiguous** | -0.668 near | 8.797 separated | 8.135 separated |

g99 的最后一点不应计为可靠“真实接触而原生分离”：真实 gap 已进入 ambiguous，decoder 重构又错误地缩到 near。这条 branch 的末端 gap 不适合给出明确接触结论。

fact 与 g99 的真实 object endpoint centroid 相距42.05px，RECON_REAL 保留39.37px，而 SINGLE/FULL 只预测0.655/0.709px；这与 root 的候选差距误差证据可以组合，但不能据此倒推我们已经识别了一条共享边界修正规则。

## gap 几何误差分解

CPU 代码额外计算 `g(M,c,r)=min_distance(M,c)-r`，其中 M 为 object mask、c 为 pusher center、r 为半径。对 M/c 的替换按两种顺序求平均，半径项严格为 `r_real-r_pred`。三项精确加回预测 gap 减真实 gap；它是读出函数的代数分解，**不是对世界动力学的因果干预**。

L1 fact 的 SINGLE step4 gap 误差6.628px，分量为 object mask 6.416、pusher center -0.169、radius0.381。FULL step4误差6.365px，分量为5.968、-0.117、0.513。因此可见分离主要跟预测物体的位置/轮廓有关，单纯校正解码 pusher center 不能消除它。但“物体 mask 项”包含位置、姿态、形变及 decoder 失真，不能等同接触边界参数。

## 研究决策边界

可以进入下一步的是：对 L1 已出现的原生分支响应分化失败，寻找实际可辨识、可干预的物体作用变量，并检查与更早响应误差的关系。不能直接进入的是：从五个宏步 gap 拟合精确 delta-tau、引用 inverse-speed 关系、或宣称执行轨迹的边界估计已能迁移。

宏步分离被观测到的时间为2.0s，但其发生区间最多定位在1.5–2.0s；区间内不保证只有一个事件。修改 guard 可以同时改变模式和强度，并非当前图像证明的唯一解。

## 产物与复现

- `QUERY_READOUT_REVIEW.json`：全量分层、7次失败、每点测量和输入 hashes。
- `QUERY_READOUT_ERRORS.csv`：473个可读的未来真实/重构/预测图像对；480个潜在对中的7个失败显式在 JSON 中。
- `query_L1_fact_g99_montage.png`：两分支 × REAL/RECON_REAL/SINGLE/FULL × 六宏步，已实际查看。
- `query_readout_review.py`：CPU 脚本，无模型、环境或优化调用。

初次脚本将“全部几何可读”写作核查断言而失败；随后将实际7个失败保留为报告内容，未改任何读出规则或预测文件。全部三个输入文件执行前后 SHA256 保持一致。
