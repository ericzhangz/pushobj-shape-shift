# 本轮实验结果

只收录 2026-09-26 这轮结果；实验含义和结论另见 [分析报告](../reports/REPORT.md)。

## 原始结果入口

| 阶段 | 汇总 | 逐例/逐步数值 |
| --- | --- | --- |
| F0 粗边界匹配 | [RUN_SUMMARY](formation_run/RUN_SUMMARY.json)、[GATES](formation_run/GATES.csv) | [HELD_FIDELITY](formation_run/HELD_FIDELITY.csv)，各 case 的 TRAINING.csv |
| 前半形成与后半留出 | [RUN_SUMMARY](prefix_formation_run/RUN_SUMMARY.json)、[GATES](prefix_formation_run/GATES.csv) | [FACTUAL_TRANSFER](prefix_formation_run/FACTUAL_TRANSFER.csv)、[TIMINGS](prefix_formation_run/TIMINGS.csv) |
| 封存后零训练归因 | [RUN_SUMMARY](post_seal_attribution/RUN_SUMMARY.json) | [REPLACEMENT_UPDATE_NUMERICS](post_seal_attribution/REPLACEMENT_UPDATE_NUMERICS.csv) |

本目录包含 30 个 CSV、17 个 JSON 和 44 个 PT，共约 39.7 MiB。PT 包括学生权重、teacher 查询和封存预测，不包含原生预训练 checkpoint 或原始 RGB。各 case 的 PREFIX_MODELS_SEALED.json 记录保存模型早于后半标签打开的程序边界。

## 主结果

以下是四开发 case 的 RMS 算术平均；每个 case 度量为 sqrt(视觉 MSE + proprio MSE)。它不是跨 case pooled RMS。

| 读数 | 原生 F0 | PREFIX-FLOW0 | PREFIX-FLOW-D |
| --- | ---: | ---: | ---: |
| 支持 0→5 | 0.07871 | 0.09103 | 0.02226 |
| 纯留后 TF 5→10 | 0.09797 | 0.29441 | 0.29323 |
| 自产历史 H2 0→10 | 0.13575 | 0.28171 | 0.26793 |

F0 查询保真门为 4/4；独立 prefix lift 亦 4/4；D 形成门为 0/4。D 对自身 V0 的支持/TF/H2 改善为 4/4、2/4、3/4，两个主留后读数均 0/4 胜 F0。不能将失败门改写成“没有任何迁移”。

同实际 held TF 输入上，未适配 FLOW0 与 F0 的平均输出差为 0.29725；同字段 RK2 双分辨率差为 0.000846。数值细化不是此次较大差距的充分解释。

没有新动作 K_D、Delta E_D、regret、几何成功率或闭环收益结果。四个重叠留后窗口不是独立样本；已用开发锚点也不构成新的盲确认。

JSON 保留实际运行时的本机 checkpoint/source 路径字符串用于记录来源，它们不是其他机器可直接使用的配置。公开结果数字和张量未做修改；日志路径脱敏另见发布清单。不存在 SHA 或其他哈希校验步骤。
