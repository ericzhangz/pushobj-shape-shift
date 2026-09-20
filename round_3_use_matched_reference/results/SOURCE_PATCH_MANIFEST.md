# Round 3 执行源与路径清单

日期：2026-09-20。

## 执行环境

- 工作目录：`D:\EV-TTT\adajepa_official_51d8665`
- Python：`C:\Users\ASUS\anaconda3\envs\adajepa-pilot\python.exe`
- 设备：`cuda:0`，本机 RTX 4060 8GB
- Torch cache：`D:\EV-TTT\adajepa_runtime\torch`
- checkpoint 根：`D:\EV-TTT\pushobj_shape_shift`
- Frozen donor：`artifacts\development\val_T_frozen_n3`、`artifacts\development\val_L_frozen_n3`
- C 真值：`artifacts\reframe_v3\common_continuation\physical_branch_results.csv` 及其 trajectory sidecars

遵照用户约束，本轮没有执行任何 SHA 或其他散列校验；修订身份不通过散列声明。所有对应关系均基于显式路径、导入、数值复现和测试。

## 本轮直接执行源

| 文件 | 作用 |
|---|---|
| `research/reframe_v3/existing_data_reanalysis.py` | 从原 CSV 重算 A/B/C 与目标一致小表 |
| `research/reframe_v3/matched_feedback_forecast.py` | 原生 latent rollout/planner 验收与 S3 盲预测 |
| `research/reframe_v3/propagation_audit.py` | 自由递推、真实动作递推、真实状态一步重启诊断 |
| `research/reframe_v3/shadow_selection_audit.py` | checkpoint/runtime、锚点和候选加载依赖 |
| `research/reframe_v3/common_continuation.py` | 已完成 C 的续行定义与真实结果来源 |
| `research/replay_oracle.py` | RNG 保存/恢复依赖 |
| `conf/__init__.py` | 本地配置包导入入口 |

对应测试为 `test_existing_data_reanalysis.py`、`test_matched_feedback_forecast.py`、`test_propagation_audit.py`，并与 `research/reframe_v3/test_*.py` 全套共同运行。

## 发布快照边界

当前本地工作树包含本轮实际导入和执行所需的 `research.reframe_v3.shadow_selection_audit`、`research.replay_oracle.preserve_global_rng_state` 及相关测试，导入 smoke 已通过。因此，本报告对应的是这份本地执行树。

此前审查指出的外部公开快照依赖缺失，本轮没有通过远端 revision 或散列重新核验，也没有声称已经修复公开归档的独立 clone 可复现性。若后续发布，应将上表实际执行源作为一个完整 package 一并导出，而不能只复制结果 CSV 或 `common_continuation.py`。

## 验证记录

- `python -m unittest discover -s research/reframe_v3 -p test_*.py`：34/34 通过。
- `python -m compileall -q research/reframe_v3`：通过。
- 无资产 import smoke：`existing_data_reanalysis`、`matched_feedback_forecast`、`propagation_audit`、`common_continuation`、`replay_oracle` 全部通过。
- Latent interface：两个预固定真实观察全部通过，rollout/cost/gradient/GD action 最大差均为 0，模型参数和 buffers 未变。
- S3：17 条物理预测、6800 次内部 GD、0 环境调用；预测先于真值读取保存。
- 最终传播审计：17 条物理分支、255 个一步模型 transition、0 GD、0 环境调用；模型参数和 buffers 未变。

## 保留的失败与过渡记录

没有删除失败输出；其命名直接说明失败口径。

- `s3_forecast_attempt1_cuda_stats_failure.log`
- `s3_forecast_attempt2_precontext_stats_failure.log`
- `propagation_audit_attempt1_wrong_full_sequence_semantics.log` 及同名前缀 CSV/JSON
- `propagation_audit_attempt2_wrong_donor_path.log`
- `propagation_audit_attempt3_reencoded_replay_initial.log` 及同名前缀 CSV/JSON
- `propagation_audit_attempt4_batched_real_encoding.log` 及同名前缀 CSV/JSON
- `propagation_audit_attempt5_without_real_state_reset.log` 及同名前缀 CSV/JSON

只有无 `attempt` 后缀的文件是本轮最终主结果。
