# 交叉响应形成实验：冻结数据协议

本轮新增的是有限交叉响应数据，不是新环境、独立物理配置或已验证的新机制。全程不使用本地 skills。配置在任何新分支执行前写入 `FROZEN_DATA_CONFIG.json`。

## 范围与分割

- 复用 T/L 两形状的 sample 2..11，各形状 sample 2..7 训练、8..9 验证、10..11 测试。所有同 episode 的历史及分支共同属于同一 split。
- 训练候选锚点为 mpc2/3/4；验证和测试为 mpc2/4。不存在的历史在采集前逐条写入配置，不根据后果删除。实际训练34锚点/12 episodes、验证7锚点/4 episodes、测试8锚点/4 episodes。
- 两形状共享拟合机制是本实验的假设。不给模型显式 shape 标签；形状在合法 RGB 中可见。
- 这些原始 episode 曾用于开发。新动作结果是本轮新采集，但不能称为完全新环境的研究盲测。

## 动作与标签

环境微动作字典固定为 `[0,0]`、`[+.12,0]`、`[-.12,0]`、`[0,+.12]`、`[0,-.12]`。每个宏动作按时间顺序重复5微步，复用原 preprocessor 归一化为10维模型动作，绝不将模型数值零误称为环境零动作。

各 split 的每个锚点采集全部25个二步词。测试另外采集固定20个五步词：5个恒定词、5个正向循环、5个反向循环、5个交替词；三步测试只截取五步记录，不重复采集。验证仅二步，不用五步测试选择模型、rank、正则化或训练轮次。

## 主路径与完整性

`collect_responses.py` 直接调用既有 `PushObjReplayOracle._replay`，每条分支均从原始初态和原 seed 重置后重放完整事实前缀。没有复制环境 rollout，没有以不完整 public state 做中途 reset，也没有修改宿主源码。

每个锚点检查事实回放的 RGB、可见 proprio 和控制器 public state 精确相等；随后检查所有共享动作前缀的 RGB/proprio 精确相等，并重放一个完整分支作重复性检查。任何不一致立即中止。控制器 state 只用于门检查，不进入公开数据。

编码固定使用既有 `_load_runtime`、preprocessor、`model.encode_obs` 及其冻结 checkpoint。输出为全部384维视觉编码。原始四维 proprio 是执行器位置/速度，不是物体状态；数据同时提供官方 proprio_mean/std 供一致归一化。没有 predictor 前向、训练或参数更新。

## 文件接口

根目录保存 `train.npz`、`validation.npz`、`test_h2.npz`、`test_h5.npz`。每个文件字段如下：

| 字段 | 形状/含义 |
|---|---|
| history_visual | `[N,3,384]`，合法最后三帧编码 |
| history_proprio | `[N,3,4]`，原始可见执行器信息 |
| history_proprio_latent | `[N,3,10]`，供原生基线使用的既有编码 |
| history_actions | `[N,2,10]`，三帧之间的两个已完成动作 |
| words | `[K,H]`，字典动作编号 |
| macro_actions | `[5,10]`，归一化后的共同动作字典 |
| future_visual | `[N,K,H,384]`，真实分支逐步视觉编码 |
| future_proprio | `[N,K,H,4]`，真实分支可见信息 |
| future_proprio_latent | `[N,K,H,10]`，仅供原生10维输出的独立诊断 |
| root_ids / episode_ids | `[N]`，分组身份，不作为预测输入 |
| goal_visual / goal_proprio / goal_proprio_latent | 已知任务目标；不是动作执行结果 |
| proprio_mean / proprio_std | `[4]`，原 preprocessor 参数 |

`test_public_h2.npz` 与 `test_public_h5.npz` 严格剔除所有 `future_` 字段，供封存预测。测试真值文件仅由评分入口打开。原始 RGB 留在 `raw_observations/`，没有隐藏物理状态；不得用原始未来图片绕过训练/测试分割。

`*_COLLECTION.json` 记录数据哈希、精确门、数组布局及计数。编码器调用次数和编码帧数分开统计，重复图像不声称为独立经历。`TECHNICAL_PROBE.json` 是批量采集前的一次技术验证，其3次回放和50微步应另加到本轮总预算。

冻结配置的源码哈希是冻结时的快照；配置冻结后唯一实现补充是保存已经计算出的未来10维 proprio 编码字段，未改动作、分割、模型或测试词。最终运行源码哈希在 `COLLECTION_COMPLETE.json`。
