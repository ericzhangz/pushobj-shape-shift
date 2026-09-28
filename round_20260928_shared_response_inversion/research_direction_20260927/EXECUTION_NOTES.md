# 本轮执行记录

2026-09-27。附件报告与粘贴答复为待分析资料；未执行其中历史流程指令。应用 result-to-claim 技能，独立 reviewer 为 `/root/claim_judgment`；另由 `/root/source_boundary` 限定核验一手来源。

- Python：`<BASE_PYTHON_ENV>/python`，3.11.5。
- PyTorch：2.1.0+cu121；评价在 CPU，1 thread。附件报告为 2.10.0。未安装或升级依赖。
- 设置 `PYTHONDONTWRITEBYTECODE=1`、`PYTHONUTF8=1`。原附件、checkpoint 与项目实现未编辑；新文件位于本目录。
- 运行附件 `code/test_audit.py`，18 项通过；日志 `test_audit.log`。
- 运行附件 `code/replay_frozen.py --out <RUNS_ROOT>/research_direction_20260927/frozen_replay`，加载 18 个既有模型，0 训练，72 行评价；日志 `frozen_replay.log`。生成的数据使用附件固定种子，不是新独立测试集。
- 按行键对比原包 `primary_replay/PRIMARY_ROWS.csv` 和 `PRIMARY_AGGREGATE.csv`。键集合、样本数和全部辨别率相同。数值预设容差 atol=1e−6、rtol=1e−5；4 个单模型字段与 2 个聚合字段超出，逐项存于 `VALIDATION.json`，未调整容差。
- 独立数值定位：复用 `trajectories(256,927410,lengths=(9,),paired=True,candidate=0.)` 和 `fixeddata(ae,256,927720,(9,),False)`；读取冻结 motion fullAR seed11。在原数据上检查帧 7..9 编码、第9帧写完状态、第11帧状态。随后仅在诊断副本将配对的帧 7..9 编码设为完全相同，再检查状态及第12帧预测。记录见 `NUMERICAL_PROBE.json`；该诊断模型在 view 数据上的用途只是数值定位，不计为 view 性能结果，也不改写主重放表。
- 独立审查全文见 `CLAIM_REVIEW.md`；完整初始 prompt、后续提供的信息、response、结构化裁定存于 `.aris/traces/result-to-claim/2026-09-27_run01/`。
- 没有新训练、GPU 实验、环境交互、官方大模型权重行为验证、全套实验完整性审计或完整前沿查新。

## 读取并保留的项目连续性材料

- `<RUNS_ROOT>/research_progress_20260922/RESEARCH_POSITION.md`
- `<ADAJEPA_REPO>/artifacts/processing_math_bridge/SECOND_STAGE_PROGRESS.md`
- `<ADAJEPA_REPO>/findings.md`
- `<ADAJEPA_REPO>/artifacts/controlled_generator_formation_20260926/REPORT.md`
- `<ADAJEPA_REPO>/round_20260926_native_preserved_increment/DECISION.md`
- `<ADAJEPA_REPO>/round_20260926_native_preserved_increment/ISOLATION_DECISION.md`

上述旧结果仅用于解释方向衔接；没有重新运行，也没有追溯重判。
