# 源码快照与运行条件

这里按原模块路径保存本轮执行代码及必要研究 helper 快照，避免覆盖仓库已有版本。`generator_formation.py`、`controlled_generator.py` 和新增测试为本轮核心；`completed_phase_evidence.py` 等为共享依赖。`models/` 保存执行时的模型模块，包含此前已存在的 callback；不将其中每一行都归为本轮新增。

这不是新建的第二套生产算子：实际实验中共享观察修订是在原生 VWorldModel.rollout 内执行，原生动作插入与 cache 仍唯一。目录是冻结源码发布快照，不是不同版本互相叠加的运行接口。

## 完整重跑前提

需在独立、准备好的 AdaJEPA checkout 中将本快照按相对路径放回对应模块，并保留原项目的 `conf/` YAML、`planning/`、环境/数据与预处理代码及依赖。不要直接覆盖自己的未提交改动。仅下载这份 code 子目录不能完成端到端重跑。

实际运行环境为 Windows、Python 3.10 的 adajepa-pilot 环境、CUDA；本轮使用 RTX 4060 Laptop。还需要：

- 匹配的 PushObj shape-shift checkpoint 与本地 DINOv2 代码/权重；不随本次上传。
- T/L donor 的 seed101、sample0/1、MPC2 前两段完整已执行证据与初始/锚点观察；不随本次上传。
- 依赖库由原项目环境提供；CPU 测试使用 unittest，不要求 pytest。

`round6_reference._local_hub_loader` 保留当次执行的 DINO cache 本机路径，禁止隐式联网下载。其他机器必须显式配置等价的本地 cache；只改 `--checkpoint-dir` 不足以跨机器复跑。此限制是公开的可移植性边界，不伪装成一键复现。

## 实际阶段命令

下面命令应从准备好的实验 checkout 根目录运行，每个 `--out` 必须为不存在的新目录。日志、CSV、JSON 和权重由 runner 写出，禁止覆盖已保存结果。

```powershell
python -m research.reframe_v3.generator_formation --stage lift --out artifacts/controlled_generator_formation_20260926/formation_run
python -m research.reframe_v3.generator_formation --stage prefix --out artifacts/controlled_generator_formation_20260926/prefix_formation_run
python -m research.reframe_v3.generator_formation --stage attribution --source artifacts/controlled_generator_formation_20260926/prefix_formation_run --out artifacts/controlled_generator_formation_20260926/post_seal_attribution
```

按实际机器指定 `--checkpoint-dir`、`--donor-root` 和 `--device`，并在运行前设置本地 TORCH_HOME。前两阶段默认 CUDA；归因不训练、不改变既有门。

CPU 回归入口：

```powershell
python -m unittest research.reframe_v3.test_generator_formation research.reframe_v3.test_generator_rollout research.reframe_v3.test_controlled_generator research.reframe_v3.test_actuator_rollout research.reframe_v3.test_actuator_response research.reframe_v3.test_completed_phase_evidence -q
```

原执行 36/36 通过。发布时另检查源码/CSV/PT 的逐字节一致性、JSON 可解析性、公开链接和阶段状态；不运行 SHA 或其他哈希检查。
