# 端点几何约束审查

2026-09-27。独立阅读 `GEOMETRY_PROTOCOL.md`、`geometry_constraint_audit.py` 及调用的 observer、fusion、scene 实现；不改脚本，不重复注册或全量图像测量。运行了保存结果的 CPU 算术复核和四个当前帧的 mask 坐标核对。未使用 skills。

## 裁定

**作为“由 D 观察到的已知材料支撑是否排除旧错误”的有限读出审查，本实现与协议相符；未发现源坐标、SE(2) 方向或统计实现的实质 bug。结果不支持把当前已知材料端点非穿透条件当成足以修复 L1 后果的约束。** 这是当前估计支撑与读出器上的结论，不是全部接触几何或连续路径约束无效的证明。

不能把任何负 gap 或超过 D 最大值自动认证为真实物理违规：REAL 和 RECON_REAL 也有负 gap 及越界；D 参考范围不是已校准物理噪声界。

## 1. 源 mask、参考点与变换方向

历史 `COMPLETED_GEOMETRY.npz` 的生成路径是 `prepare_completed_contact → fuse_completed_object_assets`：各过去帧向最后完成帧注册，当前可见对象像素优先，历史只在当前 pusher 遮挡区补充一致观察到的对象像素，冲突和未观测部分留空。保存的 reference 是当前可见对象 centroid，不是质心。

四案逐一读取新审查使用的 `native/PREDICTIONS.pt` 当前 RGB 与几何 NPZ，得到：

| 检查 | T0 | T1 | L0 | L1 |
|---|---:|---:|---:|---:|
| reference 与当前可见 centroid 差 | 0 | 0 | 0 | 0 |
| radius 与当前可见 pusher 面积半径差 | 0 | 0 | 0 | 0 |
| mask 遗漏当前可见对象像素 | 0 | 0 | 0 | 0 |
| 新增像素落在当前 pusher 区外 | 0 | 0 | 0 | 0 |
| unknown 与 known mask 交集 | 0 | 0 | 0 | 0 |
| unknown 像素数 | 4 | 7 | 3 | 103 |

四案源序列均为 11×224×224×3。主脚本还断言所有已评估分支的初始 REAL RGB 与该源最后帧逐值相同。

`register_pair(source,target)` 返回 source→target 的 affine A=[R,t]。脚本令 pose translation=R·reference+t−reference，正好使 `SingleContactScene` 的绕 reference 旋转再平移等于 A；由保存的 pose 还原 affine，全行最大误差 2.84e−14。没有把 target→source 与 source→target 混用。

## 2. 注册与真值使用边界

注册直接看目标图像的对象轮廓，pusher 中心也由该目标图像提取。这是一项事后几何读出，不能称为从 D 前向预测了目标物体姿态。FULL、REAL、RECON_REAL 使用相同读出；原生 forecast 没有受到 REAL/RECON 的反馈或重拟合，协议亦明确这一点。

源支撑在 D 中形成，但目标刚体姿态是拟合量。低 IoU、大 Chamfer、物体变形或近似对称均可能影响 gap。因此 negative gap 应称“估计已知支撑的端点穿透读数”，不能省略“估计”。保存收敛标志和多起点 objective gap 是有用诊断，但收敛不是姿态唯一性证明；多个起点也可能收敛到同一解，极小 alternative gap 不等于发现两个不同的物理解。

## 3. 未知区域和时间尺度

本审查故意不传 unknown mask 到 `SingleContactScene`，只计算 known-only 支撑；这与协议一致，不是误删的物体补全。该方向可以发现与估计已知材料的重叠，但非负结果不能证明与完整物体无重叠。尤其 L1 仍有 103 个 unknown 像素。

本审查没有计算相邻端点之间的对象姿态、推杆轨迹或中间接触。0.5 s 一帧的正 gap 不能证明区间内无穿透，也不能称为 swept-path viability、持续接触正确或作用传递正确。仅能说在这几个被读出的端点上，该条件是否触发。

## 4. 保存结果独立复核

保存 524 行：44 个 COMPLETED，160 个 FULL，160 个 REAL，160 个 RECON_REAL。其中 517 行 measured，7 行 unreadable，均为 **T1 的 RECON_REAL pusher 不可见或可见像素不足**。失败记录完整保留，没有转成零 gap 或从总行数隐藏。

未重新运行 `register_pair`。用每行保存的 affine 独立求解 source-coordinate pusher 位置，再从 NPZ mask 重新生成 EDT signed distance 并手写双线性插值：517 行 gap 最大差 **6.04e−14 px**。另用保存 pose 经过原 scene 接口复核，差为 0。全部汇总的均值、最大值、计数、收敛数及 D envelope 重算，最大差为 0。10 个 INPUT_HASHES 文件重新 SHA-256 校验全部一致。

| 案例 | D 最大穿透读数(px) | FULL 超范围 | REAL 超范围 | RECON_REAL 超范围/可读数 |
|---|---:|---:|---:|---:|
| T0 | 1.403954 | 10/40 | 6/40 | 1/40 |
| T1 | 10.115905 | 0/40 | 0/40 | 0/33（另7行失败） |
| L0 | 0.798075 | 23/40 | 25/40 | 11/40 |
| L1 | 1.024973 | 6/40 | 1/40 | 10/40 |

T1 的 D 最大值约 10.12 px，所涉读出存在明显配准偏离（D 最低 IoU 0.6063、最大 Chamfer 4.4957 px）。这个最大值会让 T1 的 empirical envelope 很宽。应原样报告，不能叫真实物理容差或为提高检出率删除该帧。REAL 的 L0 越界数多于 FULL，L1 的 RECON_REAL 越界数亦多于 FULL，说明这个范围不能用作已验证的“错误预测分类器”。

## 5. L1 关键失败是否被排除

`fact_matched_return` 第4、5宏步的读出复算如下。centroid error 是同一步相对 REAL 的可见对象 centroid 欧氏距离，受可见支撑变化影响；不是物理质心误差。

| 步 | arm | gap(px) | penetration(px) | 可见centroid error(px) |
|---|---|---:|---:|---:|
| 4 | FULL | 7.360253 | 0 | 38.753156 |
| 4 | REAL | −0.382429 | 0.382429 | 0 |
| 4 | RECON_REAL | −1.115016 | 1.115016 | 1.874194 |
| 5 | FULL | 7.659389 | 0 | 45.894926 |
| 5 | REAL | −0.081526 | 0.081526 | 0 |
| 5 | RECON_REAL | −0.470780 | 0.470780 | 2.148376 |

因此，存在物体后果明显错误而 known-support endpoint gap 显著为正的原生预测。当前必要条件不会排除它们，也不给出如何恢复作用的方向。REAL/RECON 控制支持这不是仅由重构丢失数十像素位置差所造成，但不消除注册、栅格和遮挡限制。

可以据此停止把**仅满足这项端点非穿透**当作足够机制；不能据此停止所有几何建模，更不能声称未检查的连续运动、作用冲量或共享响应关系也无法区分这些预测。未来若构建额外约束，应说明新增信息与适用条件，而不能把这里的非负 gap 重新解释为“物理已经正确”。

## 审查限制

未重复全量图像注册，也未独立确认每一帧的视觉分割；本次独立性是源坐标/实现审查及保存 affine 的几何算术复核。geometry assets 的历史融合本身存在测量误差，不能当精确物体模板。候选与时间步存在相关性，本审查未将 524 行视为独立样本计算显著性。结果仅来自旧开发池。
