# 原生参考执行记录

2026-09-28。复用既有 `formation.runtime` 和 `cached_rollout`，仅读取 `test_public_h2.npz` 与 `test_public_h5.npz` 的过去编码、过去动作、候选字及标识。没有读取目标、未来标签或隐藏状态，没有训练或环境调用。

首次执行在 `model.encode` 拼接阶段发生 shape 错误：原生 `encode_obs` 的 proprio 实际为 `[B,T,10]`，本次适配代码误添了 patch 维。异常发生于 predictor 调用之前。已按原仓库真实接口改为三维 proprio；返回值也按三维处理。未修改原仓库或旧 `formation.py`，未产生失败预测文件。评估合同同步更正。

修正后的完整执行成功：56 次 predictor forward，对应 1,200 个 batch 内样本转移；8 roots×25候选×2步，加8 roots×20候选×5步。重复候选共同动作前缀的视觉输出最大绝对差均为0。

预测与输入/输出散列封存于 `predictions/NATIVE_SEALED.json`。完整 rollout 自变量是三帧历史，前两块历史动作和 H 块未来动作；评分输出仅保存 H 个新帧，根帧不进入未来误差。

原生 proprio embedding 同时封存作接口记录，但本轮公平主比较仅使用全体384维visual，因为学习参考臂不预测proprio。不能把本轮visual-only目标叫作原发布规划器的visual+proprio总成本。

训练臂因仅依据训练/验证曲线发现未收敛而延长后，评分入口等待 `FINAL_MODEL_PREDICTIONS_SEALED.json`；旧 `MODEL_PREDICTIONS_SEALED.json` 不足以授权解封。
