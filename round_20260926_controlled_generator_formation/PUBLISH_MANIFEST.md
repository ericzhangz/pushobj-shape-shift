# 本轮发布边界

本次提交只增加 `round_20260926_controlled_generator_formation/` 与根目录 `EXPERIMENT_INDEX.md`，不改写 Round 2–5、原模型主文件或先前报告。未夹带其他本地轮次的结果。

## 包含

- 原始 30 CSV、17 JSON、44 PT：41,594,741 字节；PT 是本轮学生/teacher 查询/预测，不是预训练 checkpoint。
- 当次执行的 24 份源码快照，含核心、必要研究 helper、模型模块和配置 package marker。
- 执行前合同、结果报告、公开审查摘要、冻结主问题摘要及清晰导航。
- 三份实际日志的路径脱敏副本：只替换 Windows 用户目录与 checkout 路径；数值、状态、warning 和执行顺序不改。本地原始日志不改。

## 不包含

私人审查附件、用户目录内容、API 凭据、原生预训练 checkpoint、DINO cache、完整 donor/RGB 证据、skills、其他未提交研究或旧轮实验产物。

## 与本地的差异

公开 REPORT 的本机来源链接改为 [审查摘要](reports/REVIEW_CONTEXT.md)，proposal 入口改为 [研究背景摘要](reports/RESEARCH_CONTEXT.md)，其余链接适配此发布目录。数值和研究裁定不变。

JSON 中实际 checkpoint/source 路径仍作为原始执行元数据保留；不是通用配置。源代码中的本机 DINO cache 默认值亦保留以忠实记录执行条件，跨机器运行限制见代码说明。日志仅做路径脱敏，不影响数值复核。

校验使用逐字节比较、明确的文件路径/大小、CSV/JSON 解析、回归测试和 git 路径范围审查，没有 SHA 或其他哈希检查。逐文件清单见 `PUBLISH_FILES.csv`；该清单记录大小和分区，不记录散列值。
