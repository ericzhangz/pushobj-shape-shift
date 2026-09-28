# 测试解封前的优化预算修订

写定于2026-09-28，尚未打开测试未来进行性能评分。保留所有初始1500步结果，不覆盖它们；最终解封必须等待FINAL_MODEL_PREDICTIONS_SEALED.json。

独立审查发现：DIRECT r8三个seed最后500步训练误差分别下降26.99%、23.41%、21.68%，验证误差仍下降2.88%、3.37%、1.34%。这不足以假定普通辨识已充分优化。为避免以弱优化基线制造结构性优势，在相同数据、模型、归一化、目标、步长和验证标准下，对所有端到端训练臂（DIRECT、SPECTRAL_FT、GRU、NONLINEAR_STATE）的两个状态维度及三个seed统一增加1500步。谱history头已平台，不延长。

追加阶段从各自验证集最优checkpoint开始，重新建立AdamW。由于原运行没有保存optimizer状态，这不是连续3000步Adam的精确恢复。追加阶段step0也参与验证选择，故可以保留初始最佳模型。最终rank仍按三个seed的平均验证误差选择，全部三个seed进入评分。未读取测试结果决定该修订，不增加模型类、损失、数据或超参数搜索。

DIRECT与GRU预算为1500+1500步；SPECTRAL_FT与NONLINEAR_STATE还继承1500步谱history初始化，总预算4500步；SPECTRAL仅1500步。共同的闭式SVD开销另记。若追加后仍明显改善或验证集不足以可靠选点，将在报告中保留优化不确定性，不能因耗尽本轮预算宣称模型类无能。

continue_optimization.py直接复用fit_models.py的Reference与optimize；没有创建另一个动力学实现或替代主模型路径。
