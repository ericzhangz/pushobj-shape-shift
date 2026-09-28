# 原生资产与最小证伪探针审计

日期：2026-09-27。范围：只读审查本地代码、配置、文件清单与环境；没有加载 checkpoint、模型推理、训练或环境交互。附件和旧报告只作证据，不作为本轮行动指令。此报告不能充当新的实验结果。

## 结论

现有资产足以做一个零训练、固定真实动作的原生接口开发探针：新观察到达时，比较继续想象、官方单帧重启、保留完整原生历史的观察更新。**它只能判断这个 PushObj 模型与已有轨迹上的接口现象，不能证明长历史、持久状态和常规融合都无法解决问题。**

关键事实：训练配置允许 3 个模型时间步历史，但实际 MPC 每次反馈后只把最后一帧交给下一次规划。两者必须分开评价。若完整 3 帧更新消除了劣化，应停止把结果包装成深层证据修订困难。

## 代码、模型和环境

- 仓库：`<ADAJEPA_REPO>`，HEAD 为 `51d8665b7978824bd218decab9e05ddb6eb1f47b`。
- 未发现 `D:/AGENTS.md`、`D:/EV-TTT/AGENTS.md`、仓库根目录 `AGENTS.md` 或仓库内检索到的同名文件。本审计遵从用户提供的全局规范。
- 仓库已有改动：`models/visual_world_model.py` 为 tracked modified，diff 统计 84 additions / 6 deletions；`.aris/`、`MANIFEST.md`、`artifacts/`、`conf/__init__.py`、`findings.md`、`refine-logs/`、`research/`、`round_20260926_native_preserved_increment/` 未跟踪。本审计未修改这些文件。
- checkpoint：`<CHECKPOINT_DIR>/checkpoints/model_latest.pth`，418,103,400 bytes，约 398.7 MiB。只验证路径、大小和配置，没有动态核实其内部对象类型、参数数目或散列。
- 配置：`<CHECKPOINT_DIR>/hydra.yaml`。声明 `SmallResNetGeM(dim=384)`、全局单 token，`frameskip=5`、`num_hist=3`、`num_pred=1`、`concat_dim=1`；proprio 和 action 嵌入各 10D，repeat 均为 1。完整输入缓存应为 `[B,T,1,404]`，预测观察为 394D（visual 384D + proprio 10D），另外 10D 为明确写入的动作。它不是官方大规模 V-JEPA 资产；这里的“官方”指该 AdaJEPA 仓库/权重来源语境。
- `models/encoder/resnet.py:325` 定义 `SmallResNetGeM`，五个下采样块后 GeM 池化及投影为一个全局 token。空间局部归因无法直接依赖原生 patch token。checkpoint 采用整对象加载，运行前仍必须断言真实 encoder 类与该配置一致。
- 现有运行环境：`<PYTHON_ENV>/python`，Python 3.9.19，PyTorch 2.3.0+cu121，NumPy 1.26.4，Hydra 1.2.0，OmegaConf 2.3.0。`torch.cuda.is_available()` 为 True。
- GPU 只读查询：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 596.36。没有运行 GPU 任务。
- 系统默认 Python 是另一套环境：`<BASE_PYTHON_ENV>/python`，Python 3.11.5、torch 2.1.0+cu121；后续复现应明确指定 pilot 环境。

## 原生路径与更新边界

所有相对代码路径均以 `<ADAJEPA_REPO>` 为根。

| 路径 | 已核实行为 |
|---|---|
| `models/visual_world_model.py:140` | `encode(obs,act)` 联合视觉、proprio、动作；不能把 visual-only 测试称为完整 state。 |
| `models/visual_world_model.py:169` | `encode_obs` 逐图像编码，proprio 经独立 encoder；返回两项。 |
| `models/visual_world_model.py:185` | `predict` 将时间和 token 维展开送入 predictor，再恢复时间维。 |
| `models/visual_world_model.py:399` | `replace_actions_from_z` 用真实传入的计划动作嵌入覆写动作部分；模型预测动作不是 authoritative action。 |
| `models/visual_world_model.py:410` | `rollout(obs_0, act, ...)` 先编码初始观测历史；动作序列前部必须与历史帧对齐。返回初始历史及随后每一步预测。 |
| `models/visual_world_model.py:475` | 每次预测只用 `z[:, -self.num_hist:]`。数组中保存更长历史不等于 predictor 消费全部历史。 |
| `models/vit.py:100` | `ViTPredictor` 使用固定 `num_frames * num_patches` 位置嵌入及同窗口因果 mask。配置的 3 帧窗口不能直接无限加长而称公平长历史基线。 |
| `planning/mpc.py:149` | 反馈后 `slice_trajdict_with_t(e_obses,start_idx=-1)`；下一步 `cur_obs_0=e_final_obs`，实际只给最新一帧。 |
| `planning/adajepa_mpc.py:115` | 单样本规划调用 `super().plan(...)`，继承上面的单帧反馈入口；adapt replay buffer 另有用途，不等同于 planner history。 |
| `planning/gd.py:105`、`:122` | 直接变换 `obs_0` 后送入 `wm.rollout`，没有补成三帧。 |

当前 `rollout` 的本地扩展提供 proprio transition、visual revision、observation transition 回调，仍共享原生缓存。它们不是一个已经验证的通用真实观察滤波器。本次最小探针可直接调用原生 `rollout`，无需再添加主路径或 operator 版本。

## 加载器和本地 DINO 依赖

- `research/reframe_v3/shadow_selection_audit.py:162` 的 `_load_runtime(checkpoint_dir,device)` 读取上述 YAML、调用 `plan.load_model`、设 eval、冻结参数并构造相同数据归一化 preprocessor。
- `plan.py:534` 的 `load_ckpt` **先无条件实例化一次 DinoV2Encoder，再读取 checkpoint 整对象**。因此，即使实际编码器配置为 SmallResNet，也不能忽略这项加载时依赖。
- `research/reframe_v3/round6_reference.py:159` 的 `_local_hub_loader()` 把预期的 DINO hub 请求导向 `D:/EV-TTT/adajepa_runtime/torch/hub/facebookresearch_dinov2_main`，并检查本地权重存在。
- 本地 DINO 权重：`D:/EV-TTT/adajepa_runtime/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`，88,283,115 bytes。
- 已有 `research/reframe_v3/native_preserved_increment.py:651` 使用 `with patch.object(torch.hub,'load',_local_hub_loader()): _load_runtime(...)`。复现说明还设置 `TORCH_HOME=D:/EV-TTT/adajepa_runtime/torch`。应复用这个入口并断言布局，不另建兼容加载器。
- 未动态验证当前完整加载成功、torch.hub 的所有内部行为、模型 mask 是否随 CPU/GPU迁移正确。`models/vit.py:58` 构造 mask 时直接 `.to('cuda')`，故不能把 CPU import 检查当成 CPU 模型运行已经可用。

## 真实证据资产

根目录：`artifacts/round_6_shared_revision/results/`。

| donor | 已执行段文件数 | 样本分布 | 初始文件 | 候选张量 |
|---|---:|---|---:|---:|
| `donor_T_seed101_n12_capture` | 59 | s8 有 4 段，其余 11 样本各 5 段 | 12 | 236 |
| `donor_L_seed101_n12_capture` | 58 | s4 有 3 段，其余 11 样本各 5 段 | 12 | 232 |

这些计数来自文件名清单，尚未逐段解码验证连续性/形状。不得静默补齐早停样本。

`research/reframe_v3/shadow_selection_audit.py:315` 的 `_load_segments` 已实现合法读取：每个 `real_evidence/s{sample}_executed_mpc{index}.npz` 只取 `visual`、`proprio`、`normalized_model_actions`；要求一段为 6 个真实帧、5 个环境子步，归一化 packed action 为 `[1,1,10]`。`full_resolution=False` 取首尾帧，恰对应模型 frameskip；True 保留全部帧，不能因此把每个子步错误地当成一个模型时间步。

`_load_anchor_observations`（同文件 :288）可读 `current_real_observation` 及初始 goal，但新探针只需事实动作和 observation，不需 goal/隐藏 simulator state。

**候选动作反事实真值缺口：** 两份 n12 donor 的 `oracle/branch_records.jsonl` 均不存在。虽然有候选张量，现成 `_load_candidate_pool` 要求的 `c_env_ref` / `c_env_samever` 分支真值不在这些目录内。因此不能直接报告候选排序准确率、真实 regret 或声称比较了动作后果；最多先完成事实预测。须另行核验可合法复用的分支真值才能扩展。

旧 `round_20260926_native_preserved_increment/README.md` 明确其四 case 已是 development。读取旧说明不构成本轮重跑，也不构成新的独立验证。这批历史总体应按已使用开发资产对待。

## 最小可执行 probe 建议（尚未执行）

目标仅为决定：**在实际原生输入条件下，新观察是否比继续想象稳定更差；这种差异是否由单帧重启丢弃上下文造成。**

记每条 donor 的真实模型时间帧为 `x_0,...,x_n`，动作块为 `a_0,...,a_{n-1}`。预先固定所有可用样本、所有 `1 <= k < n`，及全部后续 `h=1,...,n-k`。对同一 cut，用截至 `k-1` 的真实前缀最后至多 3 帧，沿真实 `a_{k-1}` 得到 `xhat_k`。三臂共享冻结模型和后续动作：

1. **IMAGINE**：保留 `xhat_k` 的 visual/proprio，保留同一缓存其余历史及明确的动作，继续预测。
2. **FULL-UPDATE**：在同一历史长度和对齐位置，用 `encode_obs(x_k)` 替代最新预测观察；保留其他历史、全部已执行动作及后续动作。可直接从真实 `x[max(0,k-2):k+1]` 调原生 `rollout`，获得同样的原生历史窗口，无需改模型。
3. **SINGLE-RESTART**：仅从 `x_k` 调 `rollout`，复现官方 MPC 的反馈入口。

IMAGINE 可从真实 `x[max(0,k-3):k]` 连续 rollout 到记录末尾，再按绝对时间取 `k+h`。FULL 的 `act` 前部必须含历史帧对应的已执行动作，再接 `a_k...`；SINGLE 只从 `a_k` 开始。断言每次 predictor 实际看到的动作和绝对时间一致，不能把三帧 obs 配上仅未来动作导致错位。

输入和输出检查：

- 所有相邻段边界真实帧/动作连续；3 帧槽分别对应 5 子步的间隔，不把重叠端点重复计时。
- 运行时确认 checkpoint 内 encoder、state shape、num_hist=3；eval 且参数全程不变。
- 在 cut 处，IMAGINE/FULL 只有最新观察 token 不同（缓存更早项、动作、查询集合相同）；前两个暖启动 cut 的历史长度也要匹配。
- 固定全 24 样本，T 的 s8、L 的 s4 按记录早停，无后果的末尾 cut 不进入指标。
- 预测阶段仅取已经可见的 observation，未来 observation 必须单独评分后读取；事先封存动作和预测。真实未来动作作为离线固定 query，不能作为当时规划可知事实声称在线有效。

评价优先分开报告 visual MSE 和 proprio MSE；若同时给出两者之和须保留原单位/归一化约定。主要看 `FULL-UPDATE - IMAGINE` 的每 trajectory 配对差值、不同 horizon 和 T/L 的一致性；SINGLE 对 FULL 是接口诊断。229 个 cut×horizon 不独立，不能当 229 个统计独立样本。不要只挑极端个例或用负相关例子支持整体命题。

按当前文件长度，全部资产共有 **93 个合法 cut、229 个 cut×horizon 对**。一次最长 rollout 同时提供所有 horizon，三臂约 **780 个 batch=1 predictor 调用**；真实 model-time 帧去重后最多 141 个。若为坚持原生调用而重复编码，仍只需数百帧；可先测一次加载及一个固定配对，再报告实际耗时，当前没有实测时间。无需 optimizer、训练、环境、候选搜索或 oracle 分支。

这项 probe 的去留解释：

- FULL 优于/不劣于 IMAGINE，但 SINGLE 劣化：优先裁定有限历史接口问题；不足以支持新的核心机制。
- FULL 稳定劣化：保留现象候选，下一步仍须排除训练分布/编码噪声/时间对齐和条件化问题。此结果单独不能证明缺少选择性修订机制。
- 各样本混合且效应小：当前切口缺乏立项证据，不通过选样增强叙事。

## 本资产不能支持的论文级主张

- 不能把最多 3 帧条件说成“允许完整长期历史”的强基线；要检验后者需合适的架构和训练比较，而非把长数组送进窗口模型。
- 不能从该小型 PushObj 全局 token 模型外推官方 V-JEPA 或广泛视觉世界模型。
- 没有物体级证据作用范围标签或视觉局部 token，不能直接验证局部语义影响传播规律。
- 缺少同一锚点的完整分支真值时，事实 forecast 不等于动作选择能力。
- 旧开发轨迹不提供新 blind validation，也不能以大量相关 cut 冒充独立样本。

文件阅读记录：初始审计读取 10 个相关源码/配置/说明文件；因追加明确要求核实实际 planner 入口，另读取 `planning/mpc.py`、`planning/adajepa_mpc.py`、`planning/gd.py` 的相关片段。没有解码大型 checkpoint 或运行旧实验脚本。

## 最小入口示例（给执行者；本审计未运行）

工作目录必须为 `<ADAJEPA_REPO>`。建议将新 runner 放到本轮输出工作区，运行时显式把该仓库加入 `sys.path`，不修改主仓库。

```powershell
$env:TORCH_HOME = 'D:/EV-TTT/adajepa_runtime/torch'
& '<PYTHON_ENV>/python' '<NEW_PROBE_SCRIPT_ABSOLUTE_PATH>'
```

```python
from pathlib import Path
from unittest.mock import patch
import sys
import torch

REPO = Path('<ADAJEPA_REPO>')
sys.path.insert(0, str(REPO))
from research.reframe_v3.round6_reference import _local_hub_loader
from research.reframe_v3.shadow_selection_audit import _load_runtime, _load_segments

DONORS = {
    'T': REPO / 'artifacts/round_6_shared_revision/results/donor_T_seed101_n12_capture',
    'L': REPO / 'artifacts/round_6_shared_revision/results/donor_L_seed101_n12_capture',
}
device = torch.device('cuda')
with patch.object(torch.hub, 'load', _local_hub_loader()):
    model, preprocessor, cfg = _load_runtime(
        Path('<CHECKPOINT_DIR>'), device)
assert cfg.frameskip == 5 and model.num_hist == 3
assert model.concat_dim == 1
assert (model.proprio_dim, model.action_dim) == (10, 10)
assert all(not p.requires_grad for p in model.parameters())
model.eval()

def load_model_time_facts(donor, sample, n):
    # This convenience function reads future observations too. For sealed
    # prediction/evaluation separation, replace it with action-only reads plus
    # per-cut visible-prefix reads, and keep future target reads in evaluation.
    segments = _load_segments(donor, sample, n, 5, preprocessor,
                              full_resolution=False)
    observations = {
        key: torch.cat([segments[0][0][key][:, :1]] +
                       [obs[key][:, 1:2] for obs, _ in segments], dim=1)
        for key in ('visual', 'proprio')
    }
    actions = torch.cat([act for _, act in segments], dim=1)
    assert actions.shape == (1, n, 10)
    return observations, actions

@torch.no_grad()
def three_arms_at_cut(observations, actions, k):
    # Input and output times are absolute model steps, each five env substeps.
    # No pretrained embeddings are cached, no predictor/callback is replaced.
    n = actions.shape[1]
    assert 1 <= k < n
    specs = {
        'IMAGINE': (max(0, k - 3), k),
        'FULL_UPDATE': (max(0, k - 2), k + 1),
        'SINGLE_RESTART': (k, k + 1),
    }
    predictions = {}
    for arm, (start, stop) in specs.items():
        raw_prefix = {key: value[:, start:stop].to(device)
                      for key, value in observations.items()}
        # Crucial: leading act entries describe observed context frames.
        # Passing only actions[:, k:] with a multi-frame prefix is misaligned.
        aligned_actions = actions[:, start:n].to(device)
        predicted_obs, joined = model.rollout(raw_prefix, aligned_actions)
        assert joined.shape == (1, n - start + 1, 1, 404)
        predictions[arm] = {
            h: {key: value[:, k + h - start:k + h - start + 1].cpu()
                for key, value in predicted_obs.items()}
            for h in range(1, n - k + 1)
        }
    return predictions
```

上述简例只固定原生调用及时间索引；它不是已经封存的信息权限实现。正式 runner 应把 `_load_segments` 的读取范围限制为当下已到达前缀，未来段只单独取 `normalized_model_actions`，最后再读取 future visual/proprio 评分。每个输入验证失败应停止，不跳过样本。只用样本文件清单确定段数，另核验 `mpc0...mpc(n-1)` 连续存在；T s8=4、L s4=3，其余为5。
