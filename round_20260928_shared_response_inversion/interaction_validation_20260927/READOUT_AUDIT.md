# 原生 F0 交互读出核查

2026-09-27。独立只读核查；未使用本地 skills，未加载 GPU、训练、运行环境或修改主算子。当前 checkpoint 配置、原生代码及既有诊断产物是本报告依据；本报告不把旧接触参照当成原生 F0。

## 结论

现有模型有原生 RGB decoder，因此应先验证它是否支持事件测量，而非立刻训练新读出头。逆归一化 `rgb01 = decoded / 2 + .5` 正确。但存在三个不能绕过的限制：

1. 输入是 SmallResNetGeM 的单个 384 维全局 token；VQVAE 从 1×1 latent 插值到 14×14，再卷积上采样到 224×224。空间读出能力不是接口本身保证的。
2. 当前 `VWorldModel.forward` 对真实编码直接优化重构；计算的 `decoder_loss_pred` 没有加入最终 `loss`。真实编码的重构通过，仍不足以保证原生预测 latent 的事件读出可靠。这是当前代码事实，不据此断言未知历史训练代码完全一致。
3. 原生一步是五个低层控制步，即此环境下 0.5 秒。能够测量宏步边界的视觉邻接/运动变化，不能从五个未来点确定精确接触时刻、微步响应速度或法向穿越速度。

只要重构校准失败，事件归因与反速度传播检验就应记为**未可测/未识别**，不能把 decoder 异常写成 F0 接触时序失败，也不能记成事件假说被证伪。

## 原生接口及归一化

- 配置：`<CHECKPOINT_DIR>/hydra.yaml` 指定 `has_decoder: true`、`train_decoder: true`、`decoder_start_epoch: 3`、`epochs: 5`、`frameskip: 5`、`num_hist: 3`。decoder 为 `models.vqvae.VQVAE`，`quantize: false`；因此虽使用 VQVAE 类名，实际此配置不做向量量化。
- `Preprocessor.preprocess_obs_visual` 将 uint8 RGB 转到 `[0,1]`；`datasets/img_transforms.py` 再执行三个通道均为 `.5/.5` 的 Normalize。decoder 的训练 target 就是这个变换后的 RGB。
- `VWorldModel.decode_obs(z_obs)` 接受 `visual: [B,T,1,384]`，返回 `visual: [B,T,3,224,224]`。它返回的 `proprio` **仍是预测 embedding**，代码明确写着没有 proprio decoder，不能当真实位置/速度。
- 物理 proprio 经过 Conv1d(4→10) 与 LayerNorm 编码。不能对这十维直接乘原始四维均值方差以获得物理状态；本轮不建议临时创建未经校准的逆函数。
- 正确 RGB 导出：

```python
decoded, _ = model.decode_obs(z_obs)
raw_rgb01 = decoded['visual'] * .5 + .5
# 先记录 raw_rgb01 的范围和越界比例，避免 clipping 隐藏失真。
rgb = (raw_rgb01.clamp(0, 1) * 255).round().to(torch.uint8)
rgb = rgb.permute(0, 1, 3, 4, 2).cpu().numpy()
```

checkpoint 文件夹内只有 hydra.yaml 和 model_latest.pth，本核查未找到其训练重构质量日志。`has_decoder` 和训练配置不等于 decoder 已达到几何测量精度。

## 现成 RGB 几何读出：能做什么

`research/reframe_v3/rgb_rigid_observer.py` 提供 `observe_rgb` 和 `register_pair`，`contact_kinematic_reference.py` 提供 `contact_geometry`。

`observe_rgb` 固定按 PushObj 调色规则选最大连通区域：object 至少 40 像素、pusher 至少 8 像素。它返回可见 object mask/centroid、pusher mask/center、由面积计算的圆等效半径。检测不到会抛错，不会制造 pose。这些规则只验证过真实渲染帧，不应在看过预测结果后调色阈值。

`contact_geometry` 的 `gap = min_visible_object_pixel_distance - pusher_area_radius` 是**可见几何间隙**，不是接触力、接触时刻或模拟器物理接触标签。前景 pusher 遮掉真实轮廓，半径偏差和 raster 误差都进入 gap。因此 gap 误差应与 center、radius、object silhouette 的分量误差一同报告。

`register_pair` 在两个已知图像之间用可见边界拟合 SE(2)，排除 pusher 遮挡周围部分边界，使用四个旋转初始化。返回 IoU、完整/trimmed Chamfer、边界覆盖、多个初值目标差。它读取两端图像，是回顾性运动测量，不是预测器；拟合参考点也不是物体质心或响应中心。

已有 `rgb_geometry_probe_audited/summary.json` 的真实帧微步注册：80 个已完成区间，平均 silhouette IoU 0.96744，最小 0.94608，最大完整 Chamfer 1.69088 px。这支持在此旧池使用真实 RGB 的可见刚体运动诊断，**没有验证 decoder 图像的同等质量**。

旧刚体重绘用真实两端注册后，重编码 visual MSE 为 0.001079；这是表示闭合，不是未来 pose 可解码的证据。不能拿它代替 native decoder 校准。

## 最小诊断及校准顺序

1. 对所有四个 case、每个已完成前缀的 11 张唯一 RGB（共 44 张），执行同一 frozen encoder→existing decoder。
2. 原图与重构图分别使用**同一** `observe_rgb`。逐图保存检测成功/错误原因，object/pusher mask IoU，pusher center、radius 和 gap 的误差，object 可见面积、RGB 误差、解码越界比例。所有失败都保留分母。
3. 对真实 RGB 的实际低层相邻帧及其重构分别量运动。除了平均误差，必须看“近/远”类别是否在重构后翻转，以及静止物体是否被读成移动。原始 source montage 与 reconstruction montage 应人工查看。
4. 再按 SINGLE/FULL 原生 rollout 保存预测 latent 与解码浮点值，在任何 branch truth 读取前封存。
5. 封存后，branch truth 只供评估。将真实 branch RGB 也 encode→decode，观察 held branch recon 的同样几何误差。这样能区分“预测有偏差”和“当前 decoder 根本不能读这类图像”。
6. 若校准与预测图像均支持可见几何测量，原生预测的 gap/pose/运动误差才能被报告为**解码后的预测几何误差**。仍不能直接宣称是原生动态内的接触 guard 错误；需要独立对 latent 几何方向的可干预性证据。

校准不会使 off-manifold 风险消失。对预测 latent，即使解码看起来有物体，也可能有失真或多解。因此不应通过平滑、最近真值检索、真实运动插值，把失败读出补成事件轨迹。

## 能否分离自运动时钟、边界、响应强度

现有接口只能先做有条件的分解：

| 检查 | 可用观测 | 能支持的判断 | 不能支持的判断 |
|---|---|---|---|
| 自运动 | 解码 pusher center 与真实 center 的宏步差；已有独立 actuator 的 raw proprio 预测 | pusher 轨迹是否先产生明显误差；解码自运动误差是否超过校准误差 | 把 proprio embedding 误差直接当物理时间延迟 |
| 边界邻接 | object mask 与 pusher 的可见 gap、两者位置误差分量 | 宏步时刻是否预测分离，而实际仍邻接 | 精确接触时刻、不可见真实轮廓或接触力 |
| 响应 | 对可靠图像注册所得 object 相对位移/转角，以及 pusher 位移 | 在可信邻接区间，物体运动是否明显过强/过弱 | 把响应误差唯一归因于摩擦、接触时长或边界位置 |

原生 F0 没有独立 guard、event variable 或 post-contact gain 接口。三种误差会在全局 latent 中耦合，观测诊断不能自动提供一条忠实干预通道。

环境 `env/__init__.py` 将 PushObj 注册到 PushTWrapper。底层 `pusht_env.py` 设 `control_hz=10`、`sim_hz=100`；帧间为 0.1s，而一步 native prediction 为 0.5s。真实事件最多由相邻 0.1s 帧形成区间；原生事件最多由相邻 0.5s 输出形成区间，且区间内可出现接触又脱离，端点完全漏掉。`completed_phase_evidence.phase_chains` 改变已完成数据采样相位，**不改变原生预测步长**。

因此本轮旧四 case × 八 candidate × 五宏步轨迹只适合查可见宏步现象。若接触时差位于 0.5s 内，不能用插值“发现” `delta_tau ∝ 1/v_perp`。该规律还需要同局部几何、相同模式序列、可辨认的穿越和来自合法 D 的边界估计；分支真值速度只能评估，不能倒灌估计。

## 其他既有读出不应被误用

- `contact_readout.ContactReadout`：由独立 actuator 路径加 classical contact integrator 产生 pose，再重绘并重新编码。它是候选替代路径，**不是从 F0 latent 解出 native 接触行为**。旧“提前分离”属于这个模型。
- `rigid_orbit_projection.project_orbit`：在 source-only RGB renderer 的 SE(2) orbit 上优化 encoder 距离；不使用未来图像/goal 来选解，但要求已有 pusher 位置且不保证全局/唯一解。旧两初始化、20/60 步试验并未把它验证成可靠 decoder。
- 旧 `orbit_alignment_with_pixel_diagnostics/OVERALL_SUMMARY.csv`：all8 的 ACTUATOR-FEEDBACK 平均 endpoint visual MSE 0.15823，ORBIT-60 为 0.41486；投影到更像某个可渲染 pose 的点并不等于更接近真实后果。这不排除未来改进读出，但反对把此函数当现成真值。
- nearest-neighbor 从真实 query branch 库检索图像会泄露 branch outcome；仅从 D 检索也只给经验模板、不保证新动作的状态可辨识。本次未发现已验证的独立 pose-head checkpoint。

## 已封存输出上的独立 CPU 校准

主代理完成原生预测后，本代理使用 `calibration_check.py` 读取已封存的 `native/PREDICTIONS.pt`，对 44 张 source RGB 进行上述同规则读出。没有再加载模型或调用 encoder/decoder，未读取 query branch truth。文件 SHA256 前后均为 `3f20e82e001ff6a38c2ee4945d05d4214d5c3e7a53da91927b2f7dfa233554d5`。

结果保存在 `CALIBRATION.json`，原图/重构对照保存在 `calibration_montage.png`，已实际查看。

| source 重构指标 | 均值 | 最大误差 / 最小 IoU |
|---|---:|---:|
| 双侧几何读出成功 | 44 / 44 | 无失败 |
| object mask IoU | 0.82959 | 0.78608 |
| pusher mask IoU | 0.71721 | 0.47170 |
| pusher center 误差 | 1.50616 px | 3.49430 px |
| pusher radius 误差 | 0.40684 px | 1.12967 px |
| object centroid 误差 | 1.93092 px | 3.45246 px |
| 可见 gap 误差 | 0.92921 px | 4.17614 px |
| RGB MAE | 3.91086 / 255 | 4.16340 / 255 |

固定邻接分箱保留 41 / 44；三次变化均为 ambiguous→near，发生在 T0 step0/1 与 L0 step1。没有明确 near↔separated 翻转。但类别不均衡，41/44 不是通用事件分类准确率。示例图中轮廓有膨胀/模糊，因此不能把 2–4 px 间隙差解释为可信接触事件差。原始 decoder 归一化输出范围为 [-0.53792,1.11916]，逆归一化后的越界通道比例均值为 1.6565%；导出图像 clipping 前的值仍保存在封存文件中。

当前判断：decoder 并非不可用，足以继续检查**大尺度视觉分离**；细接触时机读出仍受像素误差与 0.5s 时间步限制，必须与 held branch 真值重构误差一起判断。

## 本轮完成边界

本报告完成接口/既有证据核查及源帧重构校准。主代理后续评估负责 branch truth 与实际选择。校准不是 F0 交互失败证据；结果应优先决定读出是否可用，再决定事件假说是否受到支持，禁止把读出问题自动上升成 JEPA 的结构性能力缺陷。
