# AdaJEPA 机制探索执行跟踪

更新：2026-09-22。判据见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。本轮零环境分支，禁止且未执行任何 SHA／其他散列检查。完整状态见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。

| ID | 关口 | 目的与产物 | 实际核验/预算 | 状态 |
|---|---|---|---|---|
| R001 | G0 | 冻结逐查询 factual 截止、分支与候选池 | 12 anchors/6 cases、120 B outcome、17 C physical；mpc2 不读 mpc4 未来 | PASS |
| R002 | G0 | 旧 B 120 个候选按六帧语义重编码 | 120/120，cost 最大差 0；132 encoder calls，环境/训练 0 | PASS |
| R003 | G0 | 原 20 项检查包或独立重建 | 原包未在本机找到；已独立验证几何代数/处理器逆及真实接口，41/41 本地测试 | PARTIAL_SOURCE_UNAVAILABLE |
| R004 | G1 | 六 anchor 向量、persistence 与有限 pair gain | 17 physical、16 pairs×5×4；255 F transitions＋条件 hybrid 68；119 编码，环境/GD/训练 0 | PASS_DIAGNOSTIC_MIXED |
| R005 | G2 | 真实 checkpoint identity/非 identity 固定动作接口 | 2 观察，实测 80 predictor calls/样本，零 GD；输出/成本/梯度/cache/RNG 通过 | PASS_FIXED_ACTION_ONLY |
| R006 | G2 扩展 | 最终 100 步 GD 动作 parity | 原计划 Scope A/预算冲突按零 GD 裁定；旧 R3 parity 不代替新 P | NOT_RUN_NOT_AUTHORIZED |
| R007 | G3 | 单一 exact-inverse additive-coupling P 的 factual 可行性 | G0/G2 已通过；须先预注册类、优化/时间/显存预算，不读 C future | READY_NOT_RUN |
| R008 | G3 | 同信息控制与旧 B 新动作向量评价 | P=I、paired P、output-only、既有 factual 更新；需先完成 R007 | PENDING |
| R009 | G4 | 若 G3 通过，具体经验写入机制与先例等价审查 | 当前仅有未冻结草案 | CONDITIONAL_NOT_RUN |
| R010 | G4 | 未见 case 的实际价值确认 | 最多 6 case×4 分支；当前无新环境授权 | NOT_AUTHORIZED |

保留全部失败尝试，尤其 G0 默认 Torch cache 导入失败和 CPU 1e−6 数值差；没有放宽阈值，随后使用文档化 Torch cache 的 CUDA 路径，通过了完整 120 条重编码。G1/G2 使用既定 FP32 门槛，未为结果改阈值或挑 anchor。后续不得把 C oracle 真值当训练数据，也不得把旧 C 固定续行真值当新策略闭环价值。
