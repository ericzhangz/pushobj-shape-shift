# 第二宿主迁移检查：NOT_RUN_ASSETS_UNAVAILABLE

## 判定

**NOT_RUN_ASSETS_UNAVAILABLE。**

按 V3 协议检查本机 `D:\EV-TTT` 后，没有找到可执行的 LeWM 代码目录、原生 PushT 配置或官方 checkpoint。顶层现有目录均属于 AdaJEPA、PushObj 资产、运行缓存、发布/验证副本或 legacy 资料；名称中没有 LeWM 路径。

全文检索只在 5 份研究备忘录中找到 `LeWM` 字样，没有命中可运行代码或配置。可见模型 checkpoint 属于当前 AdaJEPA PushObj 宿主及其 legacy 副本，不构成第二宿主。

因此：

- 没有下载外部资产；
- 没有用随机 MLP、从零训练模型或 AdaJEPA 的 H/K 配置伪装 LeWM；
- 第二宿主产生 0 次环境调用、0 次模型调用；
- 本轮结果不能支持跨宿主一般性。
