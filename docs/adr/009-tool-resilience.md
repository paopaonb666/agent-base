# ADR-009：工具调用韧性层（重试 / 黑名单 / 结构化参数反馈）

- 状态：已采纳（0.3.x）
- 日期：2026-11-01
- 相关：ADR-005 审批门、`scripts/quality_report.py`（`classify_error` 的离线模式表）、
  `core/llm.py` `ResilientLLM`（退避先例）、`extensions/scheduler.py`（指数冷却先例）

## 背景

工具调用模块核查（2026-11）结论：工具执行路径一次失败即终止，没有重试/退避；
注册表启动后冻结，运行时坏掉的工具永远暴露给模型反复撞墙；失败计数
（`tool_executions_total`、审计表）纯被动——没有任何代码用观测数据改变行为；
参数校验错误的 per-field 细节被 `handle_tool_error` 压平成 300 字符单字符串；
修复回路只有全局 `AGENT_RECURSION_LIMIT` 的粗边界，无法对"同一工具反复传错参"
定向止损。

仓库里已有两处退避先例（`ResilientLLM`：memory 管线 LLM 调用；scheduler：cron
任务），但工具执行路径——恰恰是面对外部依赖最多的路径——没有。

## 决策

1. **异常三分类**（`core/tool_resilience.py::classify_tool_error`）：
   - `schema`（pydantic `ValidationError` / langgraph `ToolInvocationError`）——
     模型的错，可修复：走结构化反馈 + 修复预算；
   - `transient`（网络族类型/MRO 名字/消息模式：429、rate limit、connection…）
     ——基础设施的错：指数退避重试；
   - `permanent`（其余）——重试无益：直接反馈模型自救。
   不 import httpx/openai（非基座直接依赖），按 MRO 名字匹配两家的网络错误族。
2. **重试保守化**：仅 transient 重试；默认 `TOOL_RETRY_MAX_ATTEMPTS=2`（含首次）、
   退避 `0.5s * 2^n` 封顶 8s 加抖动；**超时默认不重试**（`TOOL_RETRY_ON_TIMEOUT=false`）
   ——同步路径超时后底层线程仍在执行，盲目重试会造成重复副作用。
3. **短时黑名单**（`ToolHealthRegistry`）：per-tool 连续失败 ≥ 3 → 冷却 60s，
   指数翻倍封顶 3600s；冷却内调用短路返回可读原因（不执行、不计失败）；成功即
   全量清零。黑名单只统计**执行**失败——schema 失败是模型的错，不能算工具坏。
4. **全局失败预算**（防雪崩）：60s 滚动窗口内跨工具失败 ≥ 10 → 熔断态：暂停
   **一切**工具重试 + critical 日志 + `/health` degraded；窗口滑过自动恢复。
   熔断只降级重试、不禁用工具——可用性优先，雪崩靠少打无效请求抑制。
5. **结构化参数反馈**（`ToolErrorReporter`）：schema 错误按字段列出
   （`field "limit": ... (got "十")`），上限放宽到 1000 字符（多字段不被 300 截断）；
   其余错误维持 URL 脱敏 + 300 截断（阶段 3 语义逐字节不变）。
6. **修复预算**：`(thread, tool)` 连续 schema 失败 ≥ 3 次（`TOOL_ARG_REPAIR_MAX_ATTEMPTS`）
   给终止性反馈（"STOP calling this tool"），成功调用清零——定向止损，不再依赖
   全局 recursion_limit。
7. **单点挂载**：韧性逻辑全部收口在 `_TimeoutTool`（黑名单短路/重试/落账）与
   `ToolErrorReporter.handle`（三个错误收口：ToolNode 回调、审批门、planner 执行器，
   统一 `(exc, tool="")` 签名）——审批门在其上拦截，互不可见、互不干扰。
8. metrics 封闭枚举新增 `blocked`（工具未被执行，与 error 区分）；`quality_report`
   的成功率把它计入失败侧（它确实意味着用户请求没被工具满足）。

## 被否方案

- **重试一切错误**：permanent 重试纯烧钱烧延迟；schema 重试更糟——参数不会自己变对。
- **默认重试超时**：同步路径线程无法杀死，重试 = 并发双跑同一工具（写副作用翻倍）。
- **黑名单持久化**：失败是**会话级/依赖级**状态，进程重启即应翻篇；持久化会因一次
  夜间抖动把工具关到明天。冷却本身有指数封顶，无需持久层。
- **全局熔断时禁用工具**：把"依赖抖动"升级成"服务自残"——模型失去全部工具比
  反复失败更糟。
- **把 schema 失败计入黑名单**：会把"模型传错参"误判成"工具坏了"，把好工具
  拉黑——所以黑名单与修复预算是两本账。
- **handle_tool_error原地改造**：保留无状态版作兼容基线（测试直调、旧接线），
  有状态版按装配注入——避免强耦合也让"零韧性层"路径可回归。

## 后果

正面：

- 核查点 1（兜底三件套）与点 3（校验/反馈/有限修复）闭环：瞬时错误自愈、坏工具
  冷却、参数错误可定位可止损；
- 全部机制集中在两个收口，模块/审批/planner 路径自动同规；
- 默认开启但保守（重试 1 次临时错误），无需配置即获得兜底。

代价（明示）：

- `_TimeoutTool` 执行体变复杂（重试循环 + 短路分支），由 `test_tool_resilience.py`
  全路径覆盖；
- blocked 计入审计与 metrics 的新 status 值——消费方按"非 ok"聚合的语义不变；
- 冷却判定进程内单机口径（多 worker 部署各算各的，与既有 metrics 口径一致）。
