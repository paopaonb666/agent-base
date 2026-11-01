# ADR-005：敏感工具审批门（Human-in-the-Loop）

- 状态：已采纳（0.3.x）
- 日期：2026-11-01
- 相关：ADR-002 模块契约、`docs/agent-dev/agent-dev-knowledge.md` §人类监督与专家介入、
  `core/tools/spec.py`（`ToolSpec.safety` 预留注释）、`docs/agent-dev/integration-review.md` §3.2

## 背景

循环工程方法论要求：Agent 在执行敏感操作前应有人类确认点（approve/reject/modify）。
本项目在工具注册表里早已为此预留了元数据——`ToolSpec.safety`（read/network/exec）
字段的注释明说"为将来的审批门预留的元数据；本阶段只登记不强制"——但审批机制
本身从未实现：全仓无 `interrupt`/approval 逻辑，任何工具（含 MCP 外部工具）都
直接执行。

工具执行的真实风险面：MCP 工具是外部代码；`exec` 类工具（python repl、
LocalShellBackend 形态的命令执行）可产生任意副作用。目前唯一的防线是超时 +
审计（事后可见），没有事前拦截。

## 决策

1. **名单制审批，而非按 safety 全拦**。配置三项（`ToolsSettings`，默认全关）：
   - `TOOL_APPROVAL_ENABLED`（默认 false）；
   - `TOOL_APPROVAL_TOOLS`：CSV 工具名精确匹配（覆盖模块工具、MCP 前缀名、
     记忆包工具、TOOLKIT 工具）；
   - `TOOL_APPROVAL_SAFETY_LEVELS`：CSV safety 级（如 `exec`），作用于 TOOLKIT
     注册表中已声明该级的内置工具。
   两名单并集即"需审批工具集"。
2. **gate 节点 + `langgraph.types.interrupt()`**：审批启用时，共享图工厂
   `build_single_agent_graph` 在 `agent → tools` 边之间插入 gate 节点——
   检查待执行 AI 消息的 tool_calls，命中需审批集则对每个命中调用
   `interrupt({tool_call_id, tool, args})`；恢复后按决策三分支（照外部方法论）：
   - `approve`：放行原参数执行；
   - `reject`：不执行，注入拒绝 ToolMessage（内容含审批人决策），Agent 继续；
   - `modify`：以修改后的参数执行。
   未命中名单的 tool_calls 同轮直通，不整轮阻断。
3. **SSE 契约扩展**：`extensions/events.py` 新增 `ApprovalRequestEvent`
   （`tool_call_id` / `tool` / `args` / `thread_id`），纳入 `AgentEvent` 联合体与
   `_decode_custom_event` 白名单（封闭契约前提：未校验数据不得透给前端）。
4. **恢复端点**：`POST /v1/agents/{module}/threads/{thread_id}/resume`，body
   `{decision: approve|reject|modify, tool_call_id, modified_args?}`；
   经 `graph.astream(Command(resume=...), config)` 续走同一 SSE 事件流；
   端点复用既有鉴权与线程属主校验。
5. **持久化免费获得**：interrupt 状态由 checkpointer 保存（sqlite/mysql 均支持），
   服务重启后线程仍可 resume——这是选 LangGraph 原生 interrupt 而非自研
   "待审批表"的主要原因。
6. **审计闭环**：审批决策（谁、何时、对哪个 call_id、什么决策）写入结构化日志
   并随 `ToolCallRecord` 关联（同一 request_id 贯穿）。

## 被否方案

- **`interrupt_before=["tools"]` 整轮拦截**：粒度过粗——一轮内未命中的安全工具
  也被拦，且需要额外逻辑判断哪些轮次该拦；gate 节点按 tool_call 级精准控制。
- **HTTP 层前置拦截**（invoke 时检查消息内容）：无法预知模型会不会调用敏感工具，
  且脱离图状态，无法持久化恢复。
- **只审计不拦截**：审计已存在（`tool_call_records`），不解决事前风险。
- **自研待审批表 + 轮询**：重复实现 interrupt 的持久化/恢复语义，且 SSE 流
  需要另造挂起协议。
- **默认按 `safety=exec` 全部强制审批**：对未配置审批流（无 resume 调用方）的
  部署等于砍掉 exec 工具——违反"默认行为零变化"约束，故默认全关。

## 后果

正面：

- `ToolSpec.safety` 预留语义兑现；外部方法论 HITL 三分支完整落地；
- 名单制让审批粒度精确到工具（MCP 工具名可精确到 `{server}_{tool}`）；
- 中断态随 checkpointer 持久化，服务重启不丢待审批请求。

代价（明示）：

- 图拓扑增加一个条件节点，所有启用审批的模块图变深一层（未启用时零变化）；
- SSE 契约新增事件类型，前端消费方需同步处理（契约版本内向后兼容——未知事件
  本就要求丢弃）；
- modify 分支的 `modified_args` 由调用方自行构造，基座不做参数 schema 校验
  （工具自身 schema 校验兜底）；
- supervisor 模板（多子图）下的审批需各子图共享同一 gate 逻辑，首期只在
  共享图工厂生效，supervisor 子图审批列 P2。
