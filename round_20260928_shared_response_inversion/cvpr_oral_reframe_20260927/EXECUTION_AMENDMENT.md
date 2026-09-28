# 评分前的数值执行修订

本修订在任何未来误差评分之前冻结。不改变数据、切点、时域、三个比较条件、指标、聚合或原定 `atol=rtol=1e-5`。

首轮在 T/sample0/cut1 的共享历史一致性检查失败，未完成预测封存、未读取未来图像评分。原 manifest、代码快照 `native_probe/FAILED_SOURCE.py` 与 `native_predict.log` 保留。

单独诊断表明，同一真实旧帧用时间 batch 1 与 batch 2 编码，visual max abs 为 0.00027823448，proprio 为 0；action 为 1.788e-7。相同差异可直接由 `encode_obs` 重现，无需预测或未来真值。所有模块均为 eval。

统一关闭 `torch.backends.cudnn.allow_tf32` 与 `torch.backends.cuda.matmul.allow_tf32` 后，同帧 visual max abs 降为 4.768e-7、action/proprio 为 0，通过原定容差。这支持精度执行差异的解释，但没有单独区分两个后端开关的责任。正式三臂推断和评分均使用该 FP32 设置；不声称默认生产配置已经完成相同检验。

把原脚本额外的 action bit-exact 断言改为协议中相同的浮点容差；原始动作始终来自同一数组且按绝对时间对齐。每个 cut 另核对输入动作嵌入与明确动作切片。

正式完整结果写入 `native_probe/fp32/`，不覆盖初始失败。失败运行 13 次 predictor 调用；两次完成的数值诊断各 9 次；另一次诊断在读取数据前因 `_load_segments` 的导入缺失停止，随后修复且日志追加保留。这 31 次额外诊断调用不计作正式 780 次预算，也不作研究正面证据。

独立 probe reviewer 在评分前核对了时间索引、逐帧编码器和本修订，认为这是保持实验不变量的数值修复。详见最终 `PROBE_REVIEW.md`。
