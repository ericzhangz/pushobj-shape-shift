# 原生预测 runner 独立静态审查

2026-09-27。审查 native_interaction_probe.py、RUN_PROTOCOL.md 及其调用的原生 rollout、decode_obs、VQVAE、load_model、_sealed_candidates、load_completed、anchor 和 objective 实现。未使用 skills，未运行 GPU，未修改 runner。

**结论：未发现阻止执行的静态错误。** 标准库 AST 解析通过。

- 动作对齐正确：FULL 输入观测在已完成证据的第 0/5/10 子步，动作排列是两个对应历史块后接五个候选块。原生 rollout 以第三帧及其候选首块启动，递推五次；从位置 2 截取后是 query 到终点六帧。SINGLE 也生成同一候选的五步未来。
- 既有 decoder 直接来自 checkpoint。当前配置 decoder_path=null、quantize=false；没有新建随机 decoder 或更新量化 EMA 的路径。eval 与参数冻结在 _load_runtime 内落实。
- 候选导入函数只读取封存合同、摘要、FROZEN 行和行动张量，不打开环境候选结果。已完成两块证据、当前观察和初始 goal 属于合法输入。
- 原生 objective 在当前 step=2、horizon=6 时使用 terminal 分支，两个上下文臂的时域相同。
- 320 次 predictor 预算和参数/缓冲区版本检查合适。SINGLE→FULL 箭头须保持“上下文对照”的标签，不称新规律适配或因果事件干预。

已告知根代理的补强建议：

1. 评分读出须按当前训练变换反归一化：RGB=(decoded+1)/2，而非直接解释 raw decoder 值为 [0,1]；保留越界率，clamp 不能代替校准。
2. 输入保护哈希补充每个 source 的 RUN_CONTRACT/RUN_SUMMARY 及实际 helper、plan/preprocessor 文件。
3. native cost 加显式有限数检查，避免异常分数进入 .pt。

本审查只覆盖预测封存阶段；真实分支绑定、decoder 校准、统计与绘图需在 CPU 评分实现后独立检查。
