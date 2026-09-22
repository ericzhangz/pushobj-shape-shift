# G0–G2 部署前独立审查记录

日期：2026-09-22。由本轮 secondary xhigh reviewer 对执行源做只读审查；审查本身没有修改文件、运行实验或执行散列检查。主代理按反馈修正后再运行对应 GPU 审计。该复核是代码层把关，不替代运行结果。

| 模块 | 审查发现 | 处置与运行门禁 |
|---|---|---|
| G0 来源/真值 | 冻结 donor 的 `adaptation_buffer_ids` 是既往事实段清单，不是空表；执行源导入路径须为本地 package；CPU 数值复现略超既定 1e−6。 | 修正为精确事实 ID、改 module 入口；不放宽阈值，使用原运行所需 Torch cache 和 CUDA，120/120 真值完全复现。保留四次失败输出。 |
| G1 vector export | 超额分支/动作长度可能突破 255；prediction/C 身份、初史/目标对齐未全门控；验证前发布向量可能造成假 PASS。 | 模型调用前锁定 17 ID、各 5×10-D 动作、候选/trajectory 映射；向量 staging 后仅在传播 PASS 发布；对齐/模型不变进入 gate。 |
| G1 geometry | 断裂 mask 后误报 telescoping；NaN 可被 `max` 掩盖；只核 total MSE 忽略 block；scalar/vector 标签可错配。 | endpoint/adjacent/telescope 三层 mask；拒绝非有限数，逐 depth 验证 visual/proprio 与 cost；要求上游 PASS 与同向量目录、核对 anchor/labels；最终 staging 发布。 |
| G2 processing | F 调用/样本数原先写死；非有限误差与广播 shape 可假通过；梯度路径 RNG 未对齐；非恒等幅度可能低于接口容差。 | predictor hook 实测 5 次/路径且 batch1，总 80；张量 shape/有限性先验；forward 与 gradient 分别重置/比对 RNG；非恒等输入和内部变化均大于 10× rollout 容差；绑定 G0 checkpoint/donor。 |
| G1 optional hybrid | 与旧 reset 向量可能不是同一 F；派生内积非有限；pair 标签类别未核；任意同 checkpoint 日志不足以绑定旧运行。 | 不改写原 G1 结果，另立来源绑定并同时核对原日志 checkpoint、vector output path 和 255 transition；绑定 C physical 17 ID/labels 及 11/6 分类；显式交叉项平方闭合、拒绝非有限。 |

复审结论：G1 与 G2 的部署前阻塞项已关闭；hybrid 最后一项日志路径绑定后通过实际 68-call 运行。全套本地测试 41/41 通过。仍未解决的是原附件中 `geometry_contract.py` / `check_processing.py` 源文件缺失，不能称其原 20 项测试已重跑；B outcome 无动作 trace，动作与后果仍依赖已审计的归档记录/精确 sidecar 路径，而非 outcome 内部自证。
