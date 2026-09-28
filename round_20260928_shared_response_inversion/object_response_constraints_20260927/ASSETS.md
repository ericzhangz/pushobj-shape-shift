# 已完成视觉经历的离线先验资产审计

2026-09-27。本审计只用 CPU，未调用模型、训练、环境或本地 skills。读取字段限定为 `visual`、`proprio`、`normalized_model_actions`。当前 T/L sample 0/1 的 chunk 2 及之后只列文件名与 NPZ 字段名，没有解码数组。其他 episode 作为离线先验候选，与当前四个 query episode 分离；整个数据池已经用于过往研究，不称独立盲测。

## 可直接加载的数据

根目录：`<ADAJEPA_REPO>/artifacts/round_6_shared_revision/results/`。

相对路径：`donor_{T,L}_seed101_n12_capture/real_evidence/s{sample}_executed_mpc{chunk}.npz`。

| 字段 | 布局 | 用途 |
|---|---|---|
| visual | uint8 [6,224,224,3] RGB | 5 个已完成低层动作的 6 个观测端点 |
| proprio | float32 [6,4] | 已观测执行器位置/速度；不是物体真值 |
| normalized_model_actions | float32 [1,1,10] | reshape [5,2]；已归一化动作，不是像素位移 |

完整 dtype 以逐文件 ASSETS.json（历史本地产物，未纳入本次上传） 为准。档案还包含 `states` 等其他字段；本审计没有读取这些数组。跨 chunk 拼接必须检查 `visual[-1] == next.visual[0]` 与 proprio 同样精确一致，再删除重复边界帧。逐文件验证全部通过。

| 集合 | episodes | chunks | 不重复 RGB | 低层 transitions | 完整三帧原生历史且有下一步标签 | 含原生启动阶段的 stride-5 标签窗口 |
|---|---:|---:|---:|---:|---:|---:|
| 在线 D：T/L0/1，chunk0..1 | 4 | 8 | 44 | 40 | 0 | 24 |
| 离线 Π：T/L2..11，所有完成 chunks | 20 | 97 | 505 | 485 | 205 | 405 |

后两列不同：N 个低层 transitions 的完整历史窗口为 max(0,N−14)，含短历史启动窗口为 max(0,N−4)。这些相位窗口互相重叠，不能当独立样本；每条相位链应调用原有 `phase_chains`，不能将五个相位串成一条历史。

## 每条离线经历的时长

原有资产的低层采样为 10Hz，原生 stride 为 5，因此一 chunk 为 0.5 秒。

| episodes | chunks | 低层 steps | 不重复帧 | 时长 |
|---|---:|---:|---:|---:|
| T2,T3,T4,T5,T6,T7,T9,T10,T11 | 5，编号0..4 | 25 | 26 | 2.5s |
| T8 | 4，编号0..3 | 20 | 21 | 2.0s |
| L2,L3,L5,L6,L7,L8,L9,L10,L11 | 5，编号0..4 | 25 | 26 | 2.5s |
| L4 | 3，编号0..2 | 15 | 16 | 1.5s |

当前 T/L0/1 仅读取 chunk0..1，因此 D 长度均为 1 秒。不把它们之后的已执行数据重新定义成查询前历史。

**离线所有经历都不足以直接形成“3 个原生历史观测 + 5 个未来转移”的完整路径训练片段。** 三个历史观测跨度 1s，五个未来转移跨度 2.5s，合计需要至少3.5s/36低层帧；如果所称3history是3个历史转移，则需要4s/41帧。跨 episode 拼接不构成有效的时间连续轨迹。短窗局部响应仍可研究；行为系统/Hankel 的持久激励和公共线性关系并未由这些数量证明。

## 已有缓存和新生成的观测缓存

1. `<ADAJEPA_REPO>/artifacts/shared_revision_autoresearch_20260925/rgb_geometry_probe_audited/movement.csv`：T/L0/1 ×20微步=80行，含起止 gap、参考杠杆 r、推杆位移、物体配准平移/旋转、IoU/Chamfer/可见边界等。`step>=10` 超过当前 mpc2 的在线 cutoff。**不能整体导入当前 D。** 旧 `rgb_geometry_probe/movement.csv` 是相同旧测量的较少元数据版本，不是额外数据。
2. 同目录 `representation.csv`：64 stride-5 窗口×4表示臂=256行。刚体变换与推杆像素取自已观测目标，属于回顾性表示检查，不是新动作预测。
3. `.../contact_geometry_support_audit/{T0,T1,L0,L1}/COMPLETED_GEOMETRY.npz`：各自仅11帧 D，含 object_mask/unknown_mask [224,224]、reference [2]、poses [11,3]、pusher_pixels/observed_pusher_pixels [11,2]、pusher_radius、object_rms_radius 和 pixel_matrix [3,2]。生产者 `contact_geometry_support_probe.py` 明确只读 chunks0..1，未读候选真值或物体 state，当前可用。
4. 本轮生成 RGB_READOUT.csv（历史本地产物，未纳入本次上传）：549帧的物体可见形心、推杆中心/半径、可见面积。固定 `observe_rgb` 在全部505离线帧和44在线帧均成功，没有选择性删除失败帧。**形心位移不是完整刚体运动**，遮挡、旋转会改变可见形心；不得直接当物理质心速度。
5. 20条离线 episode 没有发现现成的同类型刚体配准缓存。若要拟合物体有限运动标签，应从这些 episode 已完成的 RGB 对调用既有 `register_pair`，保留可见性与配准残差。此操作读取已完成离线响应属于合法先验监督。

## RGB 几何能支持什么

代码 `research/reframe_v3/rgb_rigid_observer.py` 通过固定的 PushObj 调色规则分离物体、推杆和目标颜色，不读物体尺寸、形状模板、目标姿态或 simulator state。它是当前渲染域的可执行观察器，不是泛用视觉感知。

- `observe_rgb` 提供可见物体 mask、可见形心、推杆中心与面积等价半径。推杆 proprio 是单独可用的自运动感知，两者不可混成“全部从 RGB 读取”。
- `register_pair(source,target)` 在两个已完成观测的可见轮廓上估计 SE(2)。四个旋转起点、对称边界匹配、10% trimming；会报告未裁剪残差、覆盖、IoU、次优起点差与收敛状态。不能把有限残差或优化器收敛当姿态唯一性保证。正角度为图像坐标下顺时针。
- `fit_pusher_pixel_map` 只用已完成 RGB/proprio 拟合仿射像素映射，要求 rank3；更强的 proper-similarity 版本另加等尺度/无反射无切变先验。不能因为存在已知 renderer 比例就直接代入。
- `fuse_completed_object_assets` 将先前实际可见纹理运输到当前图；只在当前推杆遮挡区补历史观测，保留 unknown/conflict masks。它不能恢复此前从未看到的形状，参考点仍是可见形心，不是质心或摩擦中心。
- `RgbSceneRenderer` 从当前图、上述合法模板、已拟合像素映射及**预测**物体姿态/推杆位置生成图像，可以通过原冻结 encoder 回到视觉 token。它提供可计算表示，不提供动作到物体运动的定律。`repaint_pair` 则读取真实目标推杆像素，不能用于未知 query future。

旧四案的表示验证：64个 stride-5 配准 IoU 均值0.9606、最小0.9372；80个微步配准未裁剪 Chamfer 最大1.691px。源图背景刚体重绘 token MSE约0.001079，而实际任务代价 MAE约0.019019，说明这个表示有可测残差，不是无损桥接。该结论不能直接外推到新增20条离线 episode 的运动配准。

## 立即可执行且不泄漏的比较边界

离线 Π 可以同时向响应约束候选和普通 latent residual/context 参照开放20条 episode 的 RGB/proprio/actions。在线更新仅接收当前案 chunks0..1。查询时可以使用候选动作、共享模型产生的执行器路径与自产历史；不能使用查询实际推杆位移、物体姿态、真值 future RGB 或以真实后果逐分支改参。只预测真实观测起点上的一步响应不等于自由滚动验证。

旧四案与离线20案仍属反复使用的开发池。当前分离能建立此次训练数据边界，不能追溯性地将研究选择变成盲测，也不能解决跨场景一般化的证据要求。

可复现入口：inventory_assets.py（历史本地产物，未纳入本次上传），解释器 `<PYTHON_ENV>/python`。脚本拒绝覆盖 ASSETS.json 与 RGB_READOUT.csv，无旧数据写操作。
