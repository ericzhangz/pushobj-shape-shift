# G2：真实 checkpoint 固定动作处理接口验收

日期：2026-09-22。结论：**固定动作接口 PASS；完整 100 步 GD 动作 parity NOT_RUN。** 这是实现等价性，不是 P 学习有效、预测改善或 planner 完整集成的证据。

输入为 G0 冻结的 `D:\EV-TTT\pushobj_shape_shift` checkpoint 与 `val_T` Frozen donor 的预定 sample0/mcp4、sample0/mpc5；源配置、donor metadata 与 G0 来源/真值合同在运行前绑定。候选均为保存的 `g99_after`、形状 `(1,5,10)`。第一个点是 terminal、第二个点是 full-horizon 目标。

处理器为事先写定的可逆小 shear：`P(v,p)=(v+0.01·p_first,p)`，逆变换精确减去同项，作用于每个观测槽，不触碰动作通道；目标保持固定外部编码。它仅用于接口验收，不是 G3 的已选方法或训练结果。

| 检查 | mpc4 | mpc5 | 固定门槛 |
|---|---:|---:|---:|
| identity 对原生输出/cache/cost/动作梯度最大差 | 0 | 0 | rollout 1e−6、cost 1e−8、gradient 1e−6 |
| P 输入实际变化最大值 | 4.044e−3 | 4.809e−3 | >1e−5，非数值身份 |
| P 逆一致最大差 | 2.98e−8 | 2.98e−8 | 1e−7 |
| 非恒等逐步 vs 边界输出/cache 最大差 | 9.54e−7 | 4.77e−7 | 1e−6 |
| 外部目标 cost 最大差 | 9.31e−10 | 2.33e−10 | 1e−8 |
| 动作梯度最大差 | 8.52e−9 | 4.89e−10 | 1e−6 |
| 动作通道最大差 | 1.79e−7 | 1.86e−7 | 1e−6 |

RNG 在无梯度 forward 及有梯度两类路径的 before/after 均精确一致。运行时 hook 实测每条固定动作 forward/gradient 路径 5 次 predictor、batch 1；两观察、四路径、两种求值共 80 次调用和 80 个样本。模型参数与 buffers 未变，墙钟 4.07 秒；环境 0、GD 0、训练 0。逐观察原始数值见 [PROCESSING_INTERFACE_REPORT.json](../results/g2_interface_20260922_112100/PROCESSING_INTERFACE_REPORT.json)。

原附件所述 `check_processing.py` 不在本地，因此 CPU 测试为独立重建；真实权重验收由本地代码运行，不伪称原 20 项已通过。现在只可说“固定动作处理接口可在当前 checkpoint 上实现且未增加 F 调用/样本”。仍不可说完整 100 步规划器动作一致、训练后的 P 良态、新候选预测改善或闭环任务收益。
