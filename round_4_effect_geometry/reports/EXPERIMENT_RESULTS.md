# AdaJEPA Round 4 至处理接口：首轮实验结果

日期：2026-09-22。执行范围为 G0、G1（含预先论证的可选 hybrid）、G2 固定动作接口；G3 的学习/四臂对照尚未运行，G4 新环境未获本轮授权。

## 已通过的关口

1. **G0 来源与真值**：[G0 汇总](../results/G0_FINAL.json)；12 个 B anchor/6 case、120 个真实候选 outcome 均在严格 factual 截止下核对。使用真实冻结 checkpoint 按旧六帧编码合同重编码 120/120，total/visual/proprio 旧成本最大差均为 0，模型未更新。B oracle 结果只在 evaluation-only 目录。
2. **G1 向量审查**：[Round 4 完整报告](ROUND4_REPORT.md)。17 条物理 C 分支、16 个物理 pair、四种模式；旧 scalar/分块最大差 2.30e−8、几何闭合 8.88e−16。free 的终点 finite-pair gain 16/16 为负，但 teacher 7 正/9 负；signed midpoint/effect 可抵消。可选 68 次 hybrid 三项闭合 1.39e−17，且没有单一项解释全部 anchor。合法 Frozen/Adapted 实际 0 改善/3 tie/3 恶化。
3. **G2 固定动作接口**：[接口报告](PROCESSING_INTERFACE_REPORT.md)。两条真实 checkpoint 观察中，identity 与 native 全为 0 差；预定非恒等 shear 的逐步/边界输出最大差 9.54e−7，cost 9.31e−10，动作梯度 8.52e−9。实测每路径 5 个 predictor forward、batch1；所有 RNG 比对、参数不变及逆一致通过。完整 100 步 GD 动作仍 `NOT_RUN`。

## 研究裁定

本轮把“仅 scalar 压缩的错觉”排除，定位到当前已查看 C 分支中真实的候选效应向量及任务投影失真；与此同时，teacher、reset、persistence 和 hybrid 说明反馈动作、自生成输入、真实输入残差及交叉抵消共同参与。它没有证明误差主要是坐标失配、P 可修复、缺少信息是主因或 planner 本身出错。G2 只让后续 P 试验有可信的实现入口，不是方法成绩。

下一步 G3 必须先冻结一个共享、良态的处理器类及优化预算，仅由各 query 之前完成的事实经历训练；在旧 B 未用于更新的动作池上评估固定外部 W 向量/相对后果，并与 identity、同信息 output-only 和原 factual 更新比较。旧 C 未来绝不用于训练、调参或挑候选；旧合法 C 池又没有非平凡的正向实际 margin，所以不能把开发集 ranking 直接写成闭环收益。G4 的新 case/新环境仍需单独预注册和授权。

## 实际成本与复现范围

- G0 B 重编码：132 次 encoder 调用，120 候选；5.07 秒，PyTorch GPU 峰值分配 423,726,592 字节；环境/训练 0。
- G1：255 次 F transition＋必要性先写明的 68 次 hybrid；119 次真实/目标编码，传播 5.91 秒、hybrid 4.20 秒。环境/GD/训练 0。
- G2：两观察、四实现路径×forward/gradient，实测 80 次 batch1 predictor 调用；4.07 秒，环境/GD/训练 0。原生与被处理边界路径各占 20 次，未增加单条固定动作的大模型步数。
- 独立 CPU 几何/处理器测试及现有 suite 共 41/41 通过；原附件 `geometry_contract.py`、`check_processing.py` 在本机不可得，未称其原 20 项已重跑。
- 所有成功及失败的日志保留在 `refine-logs/`；输出均在独立 `artifacts/round_4_effect_geometry/`。未执行任何 SHA 或其他散列检查。
