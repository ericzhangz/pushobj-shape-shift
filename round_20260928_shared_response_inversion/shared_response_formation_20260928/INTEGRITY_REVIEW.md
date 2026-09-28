# 数据与评价完整性独立复核

2026-09-28。本复核只读代码、合法 prepared cache、控制元数据及随后生成的评价表，不打开候选 outcome NPZ，不启动 GPU、不调用环境、不使用 skills。

## 初审结论

**没有发现需要阻断当前预测的数据泄漏或宏步时间错位；评分汇总通过独立复算，但旧真值代价的严格重放门槛未通过。** 离线先验按整 episode 留出交叉验证，在线支持只使用 mpc0..3，查询确实移到 mpc4。初版评价器有两项清晰问题：`native_objective_breakdown` 两处沿用了 step2，以及真值 `c_env_ref` 重放误差只记录未硬断言。前者已在32个缓存真实轨迹上确认与step4逐值恒等；后者4/32超过预定2e-5，不能宣称完整数值重现。独立敏感性检验表明这项drift没有改变任何候选真实排序和本轮方法的平均regret顺序。运行期间未修改源码，以免破坏其 hash 终检。

这不是算法有效性裁定。优化病态性、物体响应能否跨动作迁移，以及公平求解预算，需要结合封存训练日志和最终评价另外判断。

## 1. prepared cache 与先验

已运行 [review_prepared_cpu.py](review_prepared_cpu.py)，结果见 [INTEGRITY_COUNTS.json](INTEGRITY_COUNTS.json)。

- `PREPARED.pt` SHA256与`PREPARE.json`一致。
- 24条episode的帧数、动作长度与独立数据清点逐条一致；visual/proprio latent布局为`[1,N,1,384]`/`[1,N,10]`，均为有限CPU张量。
- `fit_prior`只取sample2..11；完整历史起点`start=0..N-16`，历史帧`start+[0,5,10]`、目标`start+15`，动作恰好覆盖`start:start+15`，没有跨episode或越界。
- 实际T=105、L=100，总205个完整历史窗口；这是重叠窗口数，不是独立episode数。
- leave-one-episode-out用`owners != sample`训练、`owners == sample`验证；同episode的所有相位一并留出，不会把相邻相位泄漏到训练。每折误差先在该episode内部平均，再十折等权平均。三个预定alpha比较以及最终选择与日志一致。
- 各形状分别拟合先验，避免把T/L误称为同一物性。交叉验证误差既用于选alpha，就不是独立测试误差；最终先验重拟合十条episode，与协议一致。

先验标签是合法事实上的一步visual residual；完整缓存和actions构造与原生模型的action插入路径一致。它没有使用candidate未来、物体state或几何标签。

## 2. 支持与query索引

`support_view`读取历史`[0,5,10]`、四个packed actions即20个微动作、targets`[15,20]`。原生rollout由前三帧和前三个动作开始，执行一次循环再执行一次末步预测，恰好生成两帧；`data_error`只比较结果索引3以后，对应15、20。辅助源回写visual，proprio保留原生传播，没有接入未来真实proprio。

query历史`[10,15,20]`，历史actions为微步10..19，加五个候选packed actions，合计七个。原生rollout产生三历史帧加五未来帧，再裁去前两帧`[:,2:]`，得到mpc4锚点及后续五帧，恰好与outcome的六个macro端点对应。没有额外stride5二次下采样。

`_sealed_candidates`验证split/sample/mpc/donor及已封存无真值来源，只读取动作；`_load_anchor_observations`用于读取已知goal图像/proprio，目标不是未来响应标签。各arm使用同一候选动作和同一真实初始历史。goal信息用于最后评价，不进入拟合。

当前common-prefix数值检验只在DIRECT上运行；所有arm的接口都有相同非预知结构，但报告时不应声称逐arm都运行了前缀数值检验。

## 3. 评价器读数与对象

- 先检查预测SHA256，才加载评价CSV和NPZ；评价过程安装predictor前置hook，禁止重跑或更新预测器。
- 候选ID八个全保留，读取record中的动作tensor并与封存动作逐值相等；未来sidecar仅访问visual/proprio字段。没有用物体state选动作或计算修订。
- 逐arm验证预测latent锚点与真实latent锚点一致（2e-5）；未来visual/proprio MSE只取step1..5，不把真实锚点计入均值。
- 全部28个无序候选对计算terminal visual差分MSE和代价差误差；差分前转float64，避免uint8/低精度减法问题。
- regret用封存模型代价选出的候选，再在同固定八候选的`c_env_ref`上扣除真实最优；旧`success`只是该固定分支的已存结果，不是本轮闭环执行成功率。
- centroid是固定RGB分割器得到的可见物体形心；不是物理质心、位姿或动力学状态。`REAL`与`RECON_REAL`均保留，预测读出失败记为缺失并计数。汇总的形心误差是成功读出条件下均值，不能在失败率不同情况下单凭这个均值宣称改善。
- 总均值按四案等权；每案latent/pair数量固定，形心有效帧数可能不同。

## 4. 必须说明的评分代码问题

初版`score_formation.py`重放真实与预测代价时调用`native_objective_breakdown(..., step=2)`，而形成预测调用step4。当前`check_z`固定六帧；原目标函数定义`terminal = step is not None and step < horizon-1`，所以2和4均小于5，两者执行完全相同的terminal MSE分支。**这是应修正的阶段标记错误，当前数据下不是分数错误，也无需重跑GPU解码。** 后续应将两处改成4并断言返回stage为terminal。

初版只把真实重放代价与旧`c_env_ref`的差写入`REPLAY.csv`，没有assert；预测代价重放已有2e-5断言。采用旧真值代价进行regret之前，必须独立检查32个真实重放误差均小于2e-5；修改后的评分器也应直接fail-fast，并验证record的`objective_stage`为terminal。

`PREPARE.json`绑定模型、runner和支持文件；新query动作和goal原文件没有同时全部写入该manifest。当前预测bundle保留实际动作、goal latent，评价器动作相等与真值代价重放可验证本次语义一致性。若继续做可发布实验，应扩大输入manifest范围，绑定动作source manifest与goal文件；这不是本次已观察到的数据变化。

## 5. 评分后独立数值汇总

执行 [review_score_cpu.py](review_score_cpu.py)，从六张CSV与RESULTS进行4460项检查。960条未来误差、672条候选对、1536条几何记录、224条重放记录数量一致；逐案及全局均值、缺失计数、候选对差、选中候选argmin与regret身份关系全部通过。完整机器结果：[INTEGRITY_SCORE_CHECKS.json](INTEGRITY_SCORE_CHECKS.json)。

| 检查 | 最大绝对差 | 裁定 |
|---|---:|---|
| 预测封存代价与重新计算 | 4.76837158e-7 | 通过2e-5 |
| 预测latent锚点与真实latent锚点 | 2.14576721e-6 | 通过2e-5 |
| 旧`c_env_ref`与本轮重编码真实代价 | 6.12139702e-5 | **4/32不通过2e-5** |
| 实际32条缓存轨迹使用step2/step4 | 0 | 全部terminal、逐值相同 |

第一次审计脚本在真值重放assert处实际失败。随后保留同一2e-5门槛，把结果写为`AGGREGATES_PASS_REFERENCE_COST_DRIFT`并明确`truth_replay_gate_2e_5_passed=false`；没有通过放宽阈值隐藏失败。drift的精确底层来源本复核未识别，不能无证据归因于浮点精度、TF32或编码批次。

## 6. 代价漂移的敏感性边界

按照root后续授权，仅读取已评价的`TARGETS.pt`与`PREDICTIONS.pt`缓存，未重新打开任何outcome NPZ。用缓存中本轮重算`truth_cost`替换历史`recorded_truth_cost`，重算候选排序、best ID、各arm regret和代价差误差。脚本：[review_cost_drift_cpu.py](review_cost_drift_cpu.py)；逐候选/逐pair/逐arm数据：[COST_DRIFT_SENSITIVITY.json](COST_DRIFT_SENSITIVITY.json)。

- 全部112对真实候选的符号排序不变，包括精确平局的状态。
- 四案真实best ID均不变：T0=`g24_after`，T1=`fact_matched_return`，L0=`fact_matched_return`，L1=`g24_after`。
- 所有24个case/arm组合中，regret最大绝对变化1.22897327e-5，pair-gap MAE最大绝对变化1.64526116e-5。

| arm | 历史真值平均regret | 本轮重编码真值平均regret |
|---|---:|---:|
| NATIVE | 0.0138712563 | 0.0138742097 |
| PRIOR | 0.0039551924 | 0.0039588320 |
| STEP_RIDGE | 0.0028975233 | 0.0028979043 |
| DIRECT | 0.0016629603 | 0.0016604709 |
| SOURCE_MU1 | 0.0036104443 | 0.0036118105 |
| SOURCE_MU10 | 0.0023758812 | 0.0023743771 |

因此，使用旧或本轮真值代价不会改变本轮平均regret的方法顺序。仍应同时说明严格重放失败，而不把敏感性稳健等同于旧代价逐值复现。latent/形心误差本来就以本轮重编码/真实图像为参照；这一敏感性只涉及使用历史`c_env_ref`的选择与代价差指标。

## 7. 调用计数的审计说明

初版`RESULTS.json`中的`encoder_calls=0`、`encoder_frames=0`不是实际没有编码；原生`encode_obs`调用`self.encoder.forward(...)`，绕过注册在module `__call__`上的forward hook。评分逻辑对32条候选各调用一次`reconstruct`，每条6帧小于chunk8，因此实际32次`encode_obs`、192帧。该计数缺陷不改变结果，但报告必须修正实际逻辑调用数，并保留初版hook读数。预测器禁止hook与不调用predictor的代码路径一致；本复核没有使用GPU验证新的计数instrumentation。

最终审计结论：**数据和时间边界、评分算术通过；真值代价存在已显式保留且已做排序/选择敏感性检验的重放漂移，初版编码器hook计数无效。** 这些边界应进入最终报告，不宜只写“独立复核全部通过”。

## 8. 修正版复评分补记

root已保留实际执行的初版源码 [score_formation_executed.py](score_formation_executed.py)，将现行 [score_formation.py](score_formation.py) 的两处阶段参数改为4、增加terminal断言，并改用明确的逻辑编码调用计数。修正版已完成单独复评分，结果在 [scored_verified/RESULTS.json](scored_verified/RESULTS.json)，初版`scored/`保留。

本次仅读取上述完成结果与 [FINAL_METRICS.json](FINAL_METRICS.json)，没有重跑4460项审计或GPU。汇总器报告所有非形心指标与初版完全相同，几何读出失败计数也相同；重复解码后的逐案形心汇总最大差为0.0007773744像素，已显式披露，不能声称所有几何读数逐位相同。

修正版一次评分的实际记录为：32次编码、192帧，224次解码、1344帧，predictor调用0。T/L各两案、每案八候选，故每种形状贡献16条真实候选、96个真实编码帧；两次评分合计64次编码、384帧以及448次解码、2688帧。这里的“帧”按实际计算次数计数，包含重复的各候选锚点，不是独立观测数量。

各臂未来几何尝试均为160帧；失败计数依次为REAL=3、RECON_REAL=5、NATIVE=3、PRIOR=4、STEP_RIDGE=4、DIRECT=4、SOURCE_MU1=2、SOURCE_MU10=4。这些失败仍保留，形心误差依然是成功读出条件下统计。

修正版没有使历史`c_env_ref`与本轮重编码真值的差消失：最大差仍为6.12139702e-5。第5—6节的门槛失败与排序/选择敏感性边界继续有效；这项限制不应因评分器阶段与计数已修正而撤销。
