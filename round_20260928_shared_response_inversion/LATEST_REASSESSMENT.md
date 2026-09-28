# 动作条件视觉推演：最新证据与开放问题

这份入口把本分区先后形成的实验与研究裁定放在同一边界下阅读。当前任务是：世界模型沿尚未执行的动作串自由推演时，究竟应该推进或求解什么数学对象，才能产生与动作相关的交互后果，并在连续推演中保持这些后果？**尚未确定新的主方法，也未证明某一类 JEPA 普遍存在同一缺陷。**

## 先看正反证据

旧四案中的 L1 `fact/g99`：实际可见物体终点分开约 **42.05 px**，原生完整历史预测仅分开 **0.709 px**。真实编码差分 RMS 为 **1.08935**，原生预测差分模长为 **0.02313**。使用每步真实历史的条件诊断让末步物体误差从 **45.89 px** 降到 **5.90 px**，但它使用未来真实观测，不能当作部署算法的收益。见[旧原生报告](interaction_validation_20260927/REPORT.md)、[结果摘要](interaction_validation_20260927/RESULT_SUMMARY.json)和[独立复核](interaction_validation_20260927/INDEPENDENT_RECOMPUTATION.json)。旧四案反复用于开发，不是盲测。

新交叉动作池中的**同一原生预训练模型**，五步全视觉 MSE 为 **0.016128**、候选差分 MSE 为 **0.011146**、末端真实差异方向增益为 **0.950536**。因此不能把旧 L1 的严重压缩推广到全部历史和动作。旧、新池同时改变历史、动作与候选构造，不构成某个因素的匹配因果对照。新池只有 **4 个独立测试 episode**；大量候选对不增加独立 episode 数。见[交叉响应报告](response_energy_plot_20260928/REPORT.md)、[数据协议](response_energy_plot_20260928/DATA_PROTOCOL.md)、[完整数值指标](response_energy_plot_20260928/scored/METRICS.json)及[旧、新池范围对照](response_energy_plot_20260928/scored/NATIVE_OLD_NEW_CONTEXT.json)。后者公开副本只把输入哈希的本地绝对路径改为逻辑名称，数值结果未改。

同信息小模型的标准谱响应实现，在 H5 的全视觉/候选差分 MSE 为 **0.338389/0.186708**；GRU 为 **0.328383/0.123121**，直接辨识为 **0.294093/0.168778**。谱法没有显示优势；原生模型有不同的预训练预算，不能把小模型失利当作它的结构性证明。见[实现代码](response_energy_plot_20260928/fit_models.py)、[评分合同](response_energy_plot_20260928/metric_contract.py)及[独立审计](response_energy_plot_20260928/INDEPENDENT_AUDIT.md)。

此前两轮在线延拓审查也有明确负结果：[有界延拓验证](response_extension_validation_20260928/REPORT.md)发现单纯投影当前响应参数几乎不改变预测，有限预算下等支持规律可产生不同查询后果，却没有选出正确规律；[RWM 高层思想迭代](rwm_task_iteration_20260928/REPORT.md)改善保留经历的一步结果，却恶化完整五步后果和代价差。其[代码](rwm_task_iteration_20260928/run_iteration.py)与[数值结果](rwm_task_iteration_20260928/RESULTS.json)保留作为反证，不作为当前主线。

## 当前研究裁定

主问题从“在线修订同一预测器”放宽到**重新审查状态演化及其计算过程**。已尝试的受控路径、记忆闭合和变分动力学只是数学搜索入口，[推导记录](dynamics_reformulation_20260928/REPORT.md)未构成算法。标准响应表谱实现已有公式级直接先例，见[查新报告](response_realization_novelty_20260928/REPORT.md)。我们借鉴 ChordEdit 的是“具体计算失败 → 重新定义理想待求对象 → 建立可观测桥梁 → 导出可算规则 → 同源机制验证”的研究链；**不预设 OT、源/目标状态、Jensen 收缩或在线校正**。见[ChordEdit 链路更正](chordedit_chain_correction_20260928/REPORT.md)。原先要求预先冻结候选差分 `energy plot` 的建议已撤回；旧报告中的图合同是历史记录。

新候选的最低门槛：使用合法视觉与动作信息；说明约束为何能排除错误响应且保留正确响应，以及欠定时凭什么选解；同一历史与动作前缀必须产生相同的预测前缀；在同数据、合理预算下对完整未来及候选关系产生区别于普通递归、多步训练、标准 PSR/谱法和视觉 Koopman 的可证伪预测。当前没有已经过这些门槛的单一机制。

## 代码、数据和复核边界

公开了交叉数据的[采集脚本](response_energy_plot_20260928/collect_responses.py)、[拟合脚本](response_energy_plot_20260928/fit_models.py)、[评分脚本](response_energy_plot_20260928/score_and_plot.py)、[结果复核](response_energy_plot_20260928/verify_results.py)、延拓及 RWM 形成验证代码，以及对应的标量 JSON/CSV。旧原生实验的额外数值摘要也已补齐。见[逐项发布清单](LATEST_PUBLICATION_MANIFEST.json)。

按研究者要求，**未公开结果图、权重、原始视觉/动作轨迹、场景资产、逐点坐标或张量缓存**。公开数值文件中含本地绝对路径的输入哈希键已改为逻辑名称；公开源码中的本地运行路径改为相对路径或显式环境变量 `PUSHOBJ_CHECKPOINT_DIR`。原始与公开文件的哈希同时记在发布清单中。部分源码和历史报告仍描述本地数据依赖；仅凭此公开包无法完整重跑模型或重建场景。报告中标为“本次未公开”的链接属于这些输入/附属产物；数值审计文件和代码可用于审查协议、公式与结果范围。历史主张以本页和[交叉响应报告顶部的更正](response_energy_plot_20260928/REPORT.md)为准。
