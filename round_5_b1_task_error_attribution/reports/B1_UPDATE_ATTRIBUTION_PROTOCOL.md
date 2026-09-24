# B1 update attribution protocol

本合同只复现已运行的 predictor-only factual AdaJEPA update，并在固定 B 计划池上做评价端归因。它不是新方法、额外环境实验或盲验证。

冻结对象：12 个锚点、每池十个五步动作、固定编码器、原始 `official_prefix_schedule(mpc_iter)`、原 checkpoint、`260920 + anchor_ordinal` 种子和原 B1 trainer。先复现原保存的 Frozen/B1 候选分数与 support loss，误差超过 `1e-6` 即停止归因；不调参、不放宽阈值。

对版本 `v`，`M^v(U)` 是查询锚点预测代价，`C_s^v(U)` 是插入 B 真实前缀至第 `s` 帧后、用同一原生终点目标续行的代价。使用

\[
E^v(U)=J^R(U)-M^v(U)=C_0^v(U)-M^v(U)+\sum_{s=0}^4(C_{s+1}^v(U)-C_s^v(U)).
\]

对候选对取差并比较版本前后，报告起点项和五个有序时间项。它们的和必须闭合，但不是互斥因果份额；B 的真实未来不能进入更新、候选选择或阈值。

验收：Frozen/B1 分数、support loss、真代价和选择复现；逐候选及候选对闭合不超过 `1e-5`；不产生新环境调用；输出仅限本目录核心 CSV/JSON。不得执行散列检查。
