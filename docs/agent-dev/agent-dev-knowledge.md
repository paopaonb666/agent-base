# Agent 开发知识整理（腾讯文档「大模型应用开发8期」）

> **来源**：https://docs.qq.com/s/joml2m7Jrp2uDZhd1elKca（分享包「大模型应用开发8期」，共 7 个文件）
> **提取时间**：2026-11-01
> **本文用途**：沉淀分享包中与 Agent 开发直接相关的全部内容，作为 agent-base 项目改造的知识输入。
>
> **文件提取情况**：
>
> | 文件 | 与 Agent 开发的相关性 | 处理 |
> | --- | --- | --- |
> | The Art of Loop Engineering（循环工程的艺术） | ★★★ 核心方法论 | 全文提取（含全部代码），整理为第一部分；完整原文见 [reference-loop-engineering-full.md](reference-loop-engineering-full.md) |
> | 课堂笔记 | ★★★ 课程实战主线（智能体问答/RAG/记忆/NL2SQL/客服） | 全文提取，整理为第二部分 |
> | 但问智能数据分析系统 —— 技术架构与功能全景文档 | ★★☆ 课程综合落地案例（企业级 NL2SQL 平台） | 全文提取，整理为第三部分 |
> | 环境部署 | ★☆☆ 开发环境清单 | 全文提取，整理为第四部分 |
> | 杭州但问智能科技有限公司 — 公司介绍 | 无关（公司宣传） | 未提取 |
> | 空白流程图 | 无关（空流程图，且需登录查看） | 跳过 |
> | AI大模型应用开发畅所欲言反馈表 | 无关（反馈表单） | 跳过 |
>
> 注：原笔记中包含的 API Key、服务器账号密码等凭据一律未收录进本文。

---

## 目录

- [第一部分：循环工程 —— Agent 工程方法论（核心）](#第一部分循环工程--agent-工程方法论核心)
- [第二部分：课程实战笔记 —— DeepAgents 智能体开发主线](#第二部分课程实战笔记--deepagents-智能体开发主线)
- [第三部分：参考架构 —— 企业级 NL2SQL 平台（课程综合案例）](#第三部分参考架构--企业级-nl2sql-平台课程综合案例)
- [第四部分：开发环境清单](#第四部分开发环境清单)
- [第五部分：对 agent-base 项目的改造映射建议](#第五部分对-agent-base-项目的改造映射建议)

---

## 第一部分：循环工程 —— Agent 工程方法论（核心）

> 原文：Sydney Runkle《The Art of Loop Engineering》，2026-06-16，灵感来自 Swyx 的
> "loopcraft: the art of stacking loops"。原文链接：
> https://www.langchain.com/blog/the-art-of-loop-engineering
> 中文完整解读（含 4 套完整可运行案例代码）见 [reference-loop-engineering-full.md](reference-loop-engineering-full.md)。

### 1.1 核心论点

构建可靠的 Agent 不仅需要一个好模型，更需要一个精心设计的、适配特定任务的
**"循环骨架"（harness）**。Agent 的核心算法很简单——让 LLM 在循环中调用工具直到完成。
但这不是唯一的循环，**真正的威力在于堆叠和扩展多层循环**。

四层循环：

```
Loop 1  Agent 循环        —— 自动化工作
Loop 2  验证循环          —— 保证工作质量和正确性
Loop 3  事件驱动循环      —— 规模化自动工作
Loop 4  爬山循环          —— harness 持续改进（自动化"改进"本身）
```

### 1.2 Loop 1：Agent 循环（The Agent Loop）

**架构**：`模型 → 调用工具 → 获取结果 → 模型 → 调用工具 → ... → 完成`

**要点**：
- 本质：最基本的循环——模型在循环中调用工具，直到任务完成。
- LangChain 实现：`create_agent`——选择任意模型，接入工具，即可获得一个可工作的 Agent。
- **工具（Tools）** 是 Agent 在真实世界中采取行动的能力来源。
- 课程实例（LangChain 内部 Docs Agent）：接收文档改进请求 → 模型规划并起草修改 →
  使用工具克隆仓库、读取文件、撰写文档、发起 Pull Request。

**Docs Agent 工具集**（完整代码见参考文件 §Loop 1）：

| 工具 | 职责 | 关键实现点 |
| --- | --- | --- |
| `clone_repo` | 克隆 GitHub 仓库到临时目录 | `subprocess` + 超时控制；Pydantic `args_schema` 描述参数 |
| `read_file` | 按行区间读文件 | `start_line/end_line=-1` 语义 |
| `write_file` | 覆盖写文件 | `os.makedirs(dirname, exist_ok=True)` |
| `create_pr` | 建分支→commit→push→`gh pr create` | 串联多个 shell 步骤，逐步检查返回码 |

组装要点：

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = create_agent(
    model=llm,                      # ChatOpenAI(model="gpt-4o", temperature=0)
    tools=[clone_repo, read_file, write_file, create_pr],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,            # 带记忆：按 thread_id 隔离会话
)
# 调用：agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": ...}})
# 取最终响应：倒序遍历 result["messages"]，取第一条无 tool_calls 的消息
```

系统提示词结构（Docs Agent）：**职责清单 → 工作流程（先克隆→读→写→提PR）→ 质量约束
（链接可解析、代码可运行、遵循风格约定）**。

### 1.3 Loop 2：验证循环（Verification Loop）

**架构**：`Agent 输出 → 评分器（Grader）检查 → 不合格？→ 反馈给模型 → 重新生成 → 再次检查 ✅`

**要点**：
- 动机：Agent 循环能完成工作，但不一定一次就正确或一致。**当一致性很重要时，需要验证循环包裹它**。
- 核心机制：添加一个评分器（Grader），按标准（rubric）检查 Agent 的输出，不合格则带反馈信息返回模型重试。
- 评分器两类：
  - **确定性评分器（Deterministic）**：链接检查、CI 检查、代码块语法检查、diff 范围检查；
  - **代理式评分器（Agentic）**："LLM as a judge"。
- LangChain 实现：`RubricMiddleware` 或 `create_agent` 的 `after_agent` 钩子。
- **Trade-off**：增加验证会增加延迟和成本，但当质量优先于速度时（大多数生产场景），这是值得的。

**评分器实现要点**（完整代码见参考文件 §Loop 2）：

| 评分器 | 检查内容 | 实现方式 |
| --- | --- | --- |
| `LinkChecker` | 文档中所有链接可解析（HEAD 请求，<400 为通过） | 正则提取 markdown 链接 + 裸 URL；限制检查数量；返回 `GradingResult(grade, score, feedback, details)` |
| `CodeBlockValidator` | 代码块语法正确 | 提取 ``` 块，Python 用 `compile()` 捕获 `SyntaxError` |
| `DiffScopeChecker` | diff 只改动预期文件 | 正则提取 `+++ b/...`，与 expected_files 比对 |
| `LLMJudge` | LLM 按五维 rubric 打分（清晰度/完整性/准确性/代码质量/风格一致性） | 判官提示词强制 JSON 输出 `{各维度score+comment, overall_score, verdict: PASS\|FAIL\|NEEDS_REVISION, feedback}`；正则提取 JSON 解析 |

**统一评分结果契约**：

```python
class Grade(str, Enum):
    PASS = "PASS"; FAIL = "FAIL"; NEEDS_REVISION = "NEEDS_REVISION"

@dataclass
class GradingResult:
    grade: Grade
    score: float            # 0.0 - 1.0
    feedback: str           # 不合格时回传给模型的改进依据
    details: Dict[str, Any]
```

**VerificationLoop 引擎**（包裹任意 `agent_fn`，完整代码见参考文件）：
1. Agent 执行任务（若有上轮 feedback，拼入输入要求改进）；
2. 所有评分器打分 → 加权聚合（`pass_threshold=0.8` 判 PASS，≥0.5 判 NEEDS_REVISION；
   可配置 `require_all_pass`）；
3. 通过 → 返回结果；失败 → 构建反馈消息让 Agent 重试；
4. 超过 `max_retries` → **返回历史最佳结果**（best_result 跟踪）。

**LangChain 原生两种接法**：

```python
# 方式A：RubricMiddleware（中间件形态）
from langchain.agents.middleware import RubricMiddleware
rubric = RubricMiddleware(rubric="评估标准（1-5分）：...", model=llm, max_retries=3)
agent = create_agent(model=llm, tools=[...], middleware=[rubric])

# 方式B：after_agent hook（手动实现）
def after_agent_hook(state: dict) -> dict:   # 检查最后一条 AI 消息
    ...                                      # 不合格则写入 verification_feedback / needs_retry
agent = create_agent(model=llm, tools=[...], after_agent=after_agent_hook)
```

### 1.4 Loop 3：事件驱动循环（Event-Driven Loop）

**架构**：`事件触发（新文档/Slack消息/定时任务/Webhook）→ Agent 自动运行 → 更新系统`

**要点**：
- 动机：**Agent 开发中最关键的部分之一是集成层**——将 Agent 连接到生态系统中，使其能在后台持续运行。
- 本质：Agent 不再是手动调用的东西，而是**持续运行在更大系统中的组件**。
- 触发方式：新文档到达、定时任务（Cron）、Webhook、Slack 消息（如 #docs-plz 频道）。
- LangChain 实现：LangSmith Deployment（cron/webhook）、Fleet（channels/schedules）、openclaw 的 "heartbeats"。

**事件驱动引擎设计**（完整代码见参考文件 §Loop 3）：
- `EventType` 枚举：SLACK_MESSAGE / CRON_SCHEDULE / WEBHOOK / FILE_WATCHER / GITHUB_EVENT；
- `EventTrigger` 抽象基类：`on_event(callback)` 注册回调 + `_emit(payload)` 发事件；
- 三种触发器实现：`SlackTrigger`（监听频道）、`CronTrigger`（间隔轮询，如每小时链接健康检查）、
  `WebhookTrigger`（aiohttp 挂 `POST /webhook/...`，处理 GitHub 事件）；
- `EventDrivenAgent` 引擎：触发器注册 → 事件入 `asyncio.Queue` → N 个 worker 消费 →
  按事件类型构造 task_input → `agent.invoke` → 记录状态（pending/running/completed/failed）。

**LangSmith Deployment 生产配置形态**（声明式触发器，可直接映射为 YAML/JSON）：

```json
{
  "triggers": [
    {"type": "cron", "schedule": "0 */6 * * *", "task": "检查所有文档链接是否有效，修复失效链接"},
    {"type": "cron", "schedule": "0 9 * * 1",   "task": "扫描文档仓库，生成文档健康报告"},
    {"type": "webhook", "path": "/github-webhook", "events": ["pull_request.opened"], "task": "..."},
    {"type": "fleet_channel", "channel": "#docs-plz", "platform": "slack", "task": "..."}
  ],
  "retry": {"max_attempts": 3, "backoff": "exponential"}
}
```

### 1.5 Loop 4：爬山循环（Hill Climbing Loop）

**架构**：`生产运行 → 产生 Trace → 分析 Agent 检查 Trace → 发现问题 → 自动改进配置（prompt/工具/评分器）→ 回到 Loop 1`

**要点**：
- 动机：前三个循环自动化工作，**第四个自动化改进**！
- 核心机制：每次 Agent 运行产生 Trace → 分析 Agent 扫描 Trace → 根据发现重写 harness 配置
  （prompt、工具、评分器调整）。
- LangChain 实现：LangSmith Engine（Trace 分析 Agent）。
- **关键洞察**：返回箭头深入到内部直接更新 Agent 循环本身，**外层循环每次迭代让内层循环更有效**。
- 展望：可接入 RL 微调（开源模型）、改进辅助上下文（记忆和检索技能）。

**爬山循环引擎设计**（完整代码见参考文件 §Loop 4）：

| 组件 | 职责 | 关键设计 |
| --- | --- | --- |
| `AgentTrace` / `ToolCallTrace` / `VerificationTrace` | Trace 数据结构 | 状态枚举 SUCCESS/FAILED/TIMEOUT/VERIFICATION_FAILED；记录工具调用、验证结果、耗时、token、成本、prompt_version |
| `TraceStore` | 存储 + 统计 | `get_stats()`：成功率/失败率/验证失败率/平均耗时/token/成本、**工具使用频率与错误率**、**评分器失败分布**、Top-K 常见失败模式（关键词归因） |
| `TraceAnalyzer`（分析 Agent） | LLM 分析 Trace 找问题 | 五个分析维度：失败模式/工具使用/验证问题/性能瓶颈/Prompt 问题；输出 JSON：issues（severity/category/evidence/frequency/impact）+ recommendations（target/action/expected_improvement） |
| `ConfigOptimizer`（优化 Agent） | 依据分析结果改配置 | 可优化维度：Prompt 内容、工具描述与参数、评分器标准、重试策略、模型选择；输出新 system_prompt + tool/grader 改进 + config_changes + change_log |
| `HillClimbingLoop` | 循环引擎 | 查询近 N 小时 Trace（不足 10 条跳过）→ 分析 → 优化 → **人工审核（可选）** → 应用新配置 → prompt_version 递增（v1→v2）→ 记录 config_history |

**LangSmith Engine 生产配置形态**（检测规则驱动）：

```json
{
  "analysis": {"schedule": "daily", "lookback_hours": 24, "min_traces": 10},
  "detection_rules": [
    {"name": "success_rate_drop",        "condition": "success_rate < 0.8",         "severity": "critical", "action": "file_issue"},
    {"name": "latency_spike",            "condition": "avg_duration_ms > 30000",    "severity": "high",     "action": "file_issue"},
    {"name": "tool_error_rate",          "condition": "tool_errors > 5",            "severity": "medium",   "action": "file_issue"},
    {"name": "verification_fail_spike",  "condition": "verification_fail_rate > 0.3","severity": "high",    "action": "file_issue"}
  ],
  "auto_improvements": ["prompt_optimization", "tool_description_improvement", "grader_threshold_adjustment"],
  "human_review_required": ["prompt_optimization", "grader_threshold_adjustment"],
  "notifications": {"slack_channel": "#agent-health"}
}
```

### 1.6 人类监督与专家介入（Human-in-the-Loop）

文章强调：**自动化不意味着把人排除在循环之外**。每一层都有自然的人类介入点：

| 循环层级 | 人类介入方式 | 实现方式 |
| --- | --- | --- |
| Agent 循环 | 在执行敏感操作/工具调用前要求人类确认 | `interrupt_before` 钩子 / `langgraph.types.interrupt` |
| 验证循环 | 对于敏感工作流，人类可以充当评分器 | 人工 Grader（输出发审核频道，等待 Approve/Reject/Modify） |
| 应用/事件循环 | 人类在输出返回给最终用户前审批 | 审批中间件（after_agent 中 `interrupt`） |
| 爬山循环 | Harness 改进在部署前经过人类审查 | 人工审核流程（GitHub Issue/PR + 审批） |

实现要点（完整代码见参考文件 §人类监督）：
- 敏感工具清单（如 `create_pr/write_file/delete_file`）+ `before_tool` 钩子：调用前
  `interrupt({...options: [approve, reject, modify]})`，reject 抛异常、modify 二次 interrupt 收集新输入；
- `HumanGrader`：把输出发到审核频道（Slack），人工点按钮回传结果作为 `GradingResult`；
- `HumanReviewGate`：爬山循环的改进建议先建 Issue/PR 分配审查者，审批通过才 apply。

### 1.7 四层循环总览表

| 循环 | 做什么 | 影响 | LangChain 原语 | 关键代码 |
| --- | --- | --- | --- | --- |
| 1. Agent 循环 | 模型反复调用工具直到任务完成 | 自动化工作 | `create_agent`，任意 LangChain 支持的模型 | `create_agent(model=llm, tools=[...])` |
| 2. 验证循环 | Agent 运行，输出按标准评分，不合格则带反馈重试 | 保证工作质量和正确性 | `RubricMiddleware` | `RubricMiddleware(rubric="...", max_retries=3)` |
| 3. 事件驱动循环 | 事件触发 Agent 运行并更新真实系统 | 规模化自动工作 | LangSmith Deployment（cron/webhook）或 Fleet Channels | SlackTrigger, CronTrigger, WebhookTrigger |
| 4. 爬山循环 | 生产 Trace 输入分析 Agent，改进 harness 配置 | Harness 持续改进 | LangSmith Engine | TraceAnalyzer + ConfigOptimizer |

### 1.8 核心结论与成熟度模型

1. "AI 领导者如 Steipete、Boris、Andrej 都得出同一结论：**Agent 的潜力在于你围绕它们构建的循环**。"
2. Loop 1 和 Loop 2 已经被广泛思考。**焦点应该转向 Loop 3 和 Loop 4——价值通过将 Agent
   嵌入你的生态系统并持续根据你的标准改进而复合增长**。
3. Satya 的组织层面观点："早期构建学习循环的公司——人类判断和 token 资本共同复合增长——将建立难以复制的优势。"
4. 循环是模式，优化什么取决于你：Prompt 和工具配置是最容易改进的，但也可以扩展到
   RL 微调（开源模型）、记忆和检索技能优化等。

**Agent 工程成熟度模型**：

| 等级 | 能力 |
| --- | --- |
| Level 1 | 你能让 Agent 干活（基本能力） |
| Level 2 | 你能保证 Agent 干对活（质量保证） |
| Level 3 | 你能让 Agent 自动干活（规模化） |
| Level 4 | 你能让 Agent 越干越好（持续进化） |

> **最关键的洞见**：真正拉开差距的不是模型本身，而是 Loop 3（集成）和 Loop 4（自我改进）
> ——这两个循环让 Agent 从"一次性工具"变成"持续进化的系统组件"，价值随时间复合增长。

### 1.9 参考资源

- https://docs.langchain.com/oss/python/deepagents/quickstart
- https://docs.langchain.com/oss/python/langchain/agents
- https://docs.langchain.com/oss/python/langchain/middleware
- https://docs.langchain.com/langsmith/deployments
- https://docs.langchain.com/langsmith/engine
- https://docs.langchain.com/langsmith/fleet
- 中文解读：https://tosea.ai/blog/loop-engineering-ai-agents-complete-guide-2026
- 示例仓库：https://github.com/cobusgreyling/loop-engineering
- Rubric 中间件：https://docs.langchain.com/oss/python/deepagents/rubric

---

## 第二部分：课程实战笔记 —— DeepAgents 智能体开发主线

> 来源：《课堂笔记》（2026-03-31 至 2026-07-11，共 15 次课）。
> 课程主线：环境部署 → 智能体对话 → 中间件与文件解析 → 智能问答系统 → Memory →
> 记忆体系统 → llm-wiki/hermes → NL2SQL → RAG → 知识图谱 → AG-UI → 智能客服 →
> 智能代码分析平台 → 技术栈总结 → LoopEngineering。

### 2.1 课程技术栈层级（重要）

```
DeepAgents（create_deep_agent）        ← 应用层框架
    └── langchain（create_agent）      ← Agent 组装层
            └── langgraph              ← 编排运行时
```

对应关系：`create_deep_agent → create_agent → langgraph`。
子智能体放到 task 工具中（本质也变成工具）；**Middleware 是拦截器**（代码逻辑可以是
调用大模型+工具，也可以是任何其他代码）。

### 2.2 Agent 的组成要素

```
Agent
 ├── 大模型（文字大模型 deepseek / 多模态大模型 doubao）
 ├── 工具（mcp tools + 函数工具 + cli）
 ├── Skills（指导工具和大模型完成相应任务的技能）
 ├── 中间件（hook / AgentMiddleware）
 ├── 记忆（短期会话记忆 + 长期记忆体系统）
 └── RAG（基于 mcp 提供接口进行访问）
```

DeepAgents 与 Hermes Agent 的共同结构（课程总结）：

```
大模型（文本、图片、视频、音频）
工具（自定义、MCP）
技能（Skills，本质 = 提示词指令 + 工具调用的引导）
RAG —— 基于 mcp 提供接口进行访问
```

### 2.3 消息类型（langchain 消息契约）

| 类型 | 用途 |
| --- | --- |
| `SystemMessage` | 角色、身份、限制性约束 |
| `HumanMessage` | 人类发送给智能体的消息（可携带 `additional_kwargs.attachments`，如 base64 PDF） |
| `AIMessage` | 大模型返回的消息 |
| `ToolMessage` | 工具产生的消息 |

### 2.4 运行 Agent 的三种方式

```python
# 1、脚本中直接调用 invoke
agent.invoke({"messages": [{"role": "user", "content": "..."}]})

# 2、借助 fastapi 将 agent 的输出作为 api 返回（灵活、自由）
# 3、借助 langgraph api 的能力直接对外提供服务（生产级，课程默认方式）
#    - start_server.py 启动 langgraph-api 服务
#    - graph.json 配置文件加载对应 agent 到 langgraph-api 服务
```

langgraph-api 两种模式：

| 模式 | 用途 |
| --- | --- |
| inmem（默认） | 开发环境 |
| postgres 持久化 | 生产环境（langgraph-api 默认单用户单线程，改造后支持多用户） |

> 课程结论：**改造运行时环境对智能体代码没有任何影响**——生产级环境用
> langgraph-api + postgres 持久化支持多用户并发对话。

### 2.5 中间件体系（Middleware = 拦截器）

课程中中间件是最常用的扩展点（拦截模型调用、注入上下文、动态改提示词）。

**可用的钩子装饰器**（langchain.agents.middleware）：

```python
from langchain.agents.middleware import (
    before_model, wrap_model_call,
    ModelRequest, ModelResponse,
    AgentMiddleware,
)

@before_model()
def check_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None: ...

@wrap_model_call
async def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> Any:
    ...
    return await handler(request)
```

**类形式中间件**（继承 `AgentMiddleware`，重写 `awrap_model_call`）：

```python
class PDFParserMiddleware(AgentMiddleware):
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        # 拦截：可在此改写 request（注入 PDF 文本到 system_prompt 等）
        print(request)
        return await handler(request)

agent = create_agent(
    model=model,
    tools=[read_local_file],
    middleware=[PDFParserMiddleware()],
    system_prompt="You are a helpful assistant",
)
```

**课程实战：PDF 解析中间件**（图文并茂文档对话的实现思路）：

```
前端上传 PDF（转 base64 放进 HumanMessage.attachments）
  → PDF 中间件（wrap_model_call）拦截：把 base64 解码保存到磁盘
  → pymupdf4llm 提取文本
  → 将文本注入 system_prompt 作为对话上下文（或交给工具）
```

- 多模态模型示例：`ChatOpenAI(model="doubao-seed-2-0-lite-260215", base_url="https://ark.cn-beijing.volces.com/api/v3")`
- 文档解析组件选型：`pymupdf4llm`（https://docs.langchain.com/oss/python/integrations/document_loaders/pymupdf4llm）
- 中间件自定义文档：https://docs.langchain.com/oss/python/langchain/middleware/custom#wrap-style-hooks
- **动态提示词中间件**：解决前端数据动态变化问题——基于中间件在每轮调用前更新系统提示词。

### 2.6 工具体系

| 类别 | 说明 |
| --- | --- |
| 自定义函数工具 | `@tool` 装饰器（可配 Pydantic `args_schema`），如 `read_local_file`、`pdf_paraser` |
| MCP 工具 | 外部 mcp server 提供的工具（llm_wiki、graphiti、lightrag、数据库、业务接口都可封装成 MCP） |
| CLI 执行 | **LocalShellBackend**（推荐）：`execute` 执行任何命令 |
| 子智能体即工具 | 子智能体包成 tool，实现"多智能体协作"，本质仍是 单Agent+tools |

**文件解析工具演进**（课程作业路线）：`read_local_file` 只支持文本 → 扩展支持
图片 / pdf / 文本 / word 多格式；URL 地址也能解析；解析器独立成模块。

### 2.7 Skills 技能体系

- 本质：**提示词指令 + 工具调用的引导**，指导 agent+tools 完成一系列任务。
- 课程引入的技能：web-search 技能（Tavily，https://app.tavily.com/home ）、
  ppt 技能、docx 技能、pdf 技能——组合完成"编写研究报告并保存为 word 和 ppt"。
- Skills 可以实现多智能体协作能力（单 Agent 形态下）。
- Skills + MCP 的组合价值：**不绑定某个 agent 框架**（同一套 skills+mcp 可被
  claude / hermes / cursor / cherrystdio 等客户端使用）。

### 2.8 记忆体系统构建（★ 与本项目 M6 直接相关）

**记忆的目标**：Agent 越来越聪明（更懂你的业务）。问答对存储到记忆系统，
借助嵌入模型转换成向量。

**三种检索范式**：

| 检索方式 | 特点 |
| --- | --- |
| 关键字检索（全文检索） | 精确匹配 |
| 向量检索（相似性检索） | 语义相似 |
| 知识图谱检索 | 全局、关系 |

**检索管线**：`100 份数据 → 检索 top_k=30 → rerank → 10 份`
（先宽召回再精排；嵌入模型课程用 `qwen3-embedding:0.6b`，ollama 部署，
`OLLAMA_HOST=0.0.0.0:11434`）。

**langgraph Store 用法**（课程核心代码）：

```python
from langchain_ollama import OllamaEmbeddings
from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore   # 生产用 DB-backed store（PostgresStore）

embeddings = OllamaEmbeddings(base_url="...", model="qwen3-embedding:0.6b")
store = InMemoryStore(index=IndexConfig(embed=embeddings, dims=1024))

# namespace 三段约定： (workspace_id, "memories", user_id) —— 一个工作组共用一个空间
namespace = ("workspace_001", "memories", "user001")
store.put(namespace, "1", {"text": ["..."], "date": "2026-04-25"})

# 向量相似性 + 元数据过滤检索
items = store.search(namespace, filter={"date": "2026-04-25"}, query="...", limit=1)
```

**记忆读取工具**（暴露给 Agent，注意 `ToolRuntime` 注入 store 与 context）：

```python
@tool
def get_memory_info(query: str, runtime: ToolRuntime[AgentContext]) -> str:
    """从长期记忆存储中查询与用户问题相关的记忆内容。当用户询问与历史信息、
    过往记录相关的问题时调用此工具。"""
    workspace_id = runtime.context.workspace_id
    user_id = runtime.context.user_id
    namespace = (workspace_id, "memories", user_id)
    items = runtime.store.search(namespace, query=query, limit=5)
    # 拼接 value["text"]（列表），带 [日期: ...] 前缀，"\n\n---\n\n" 分隔
```

**记忆更新智能体**（跨线程记忆形成的关键设计）：

> 不定时分析**所有的对话记录（跨线程）**，将关键信息存入记忆库（文件、向量数据库、
> 关系数据库）。**记忆库是实时变化的**。

**数据库选型作业路线**：PostgresStore 构建 DeepAgents 用户记忆体（封装记忆存储/读取
两个工具）→ milvus 部署 → mem0 接入 deepagents 记忆系统（https://github.com/mem0ai/mem0）。

**记忆图谱方向**（进阶）：
- **graphiti**（https://github.com/getzep/graphiti ）：时序知识图谱记忆——把每次对话、
  每个实体、每个关系存成带时间戳的节点，能回答"上周用户问了什么""用户偏好如何演变"；
  NL2SQL 项目中：问答后自动将问答对注入记忆图谱，下次查询时注入 Prompt
  （实现参考 `memory/graph_memory.py` 的 `GraphMemoryTool`：自动保存查询历史、
  成功/失败模式、用户摘要）。
- 部署：graphiti 提供 mcp_server（https://github.com/getzep/graphiti/tree/main/mcp_server ），
  在 `create_agent` 中以 mcp tools 方式引入即可增删改查。

### 2.9 RAG 与知识库体系

**RAG 核心价值（课程原话）：过滤——尽可能覆盖必要的上下文信息，但又不能过多（可以多，但不能少）。**

**RAG 标准流程**：

```
文档 → 分块（如 100 块）→ 嵌入模型 → 向量 → 存入向量数据库
用户提问 → 嵌入模型 → 向量 → 向量库相似性查询 top_k=10 → 取回文本块 → 问题+上下文 → 大模型
```

**数据库类型与检索范式**：

| 类型 | 检索 |
| --- | --- |
| 关系数据库 | 关系检索、条件检索 |
| NoSQL（如 ES） | 全文检索（关键词） |
| 向量数据库 | 相似性查询、语义查询 |

**混合检索 = 关键字查询 + 语义查询**。

**基于 langchain 构建 RAG 系统要点**：
1. 准备文档（如 llm_course.pdf）——注意是否包含图片/流程图/思维导图等多模态信息，是否需要解析；
2. 加载器：`PyMuPDF4LLMLoader`、`JsonLoader` 等。

**知识库系统选型**（课程落地）：

| 系统 | 用途 | 接入方式 |
| --- | --- | --- |
| llm_wiki（https://github.com/nashsu/llm_wiki ） | 上传文件构建 wiki 知识库；摄入+查询 | 封装成 MCP 服务（llm-wiki-compiler：https://github.com/atomicmemory/llm-wiki-compiler ，重点工具 `ingest_source`、`compile_wiki`） |
| LightRAG（https://github.com/HKUDS/LightRAG ） | 文本 RAG；验证多模态文件处理（图片/流程图/思维导图）能力 | docker 部署；配置文本模型/视觉模型/嵌入模型 |
| raganything | 多模态 RAG | — |
| anything-chat-rag | 独立企业级 RAG 系统（检索：RAG+知识图谱+关键字） | 界面直接测试召回能力；封装 RAG MCP server（查询、嵌入两个工具）供第三方调用 |

**知识库分类管理**（NL2SQL 知识库设计）：
1. 问答对；2. 业务知识；3. 业务字段信息。
接入方式两案：①每类知识一个工作空间；②统一工作空间 + 上传文档时打元数据标签做分类召回。

### 2.10 NL2SQL / Text2SQL

**基本流程**：

```
用户问题（"这个月销售情况如何？"）+ 所有表结构（表名/表描述/字段/字段描述）
  → 大模型 → 生成 SQL → SQL 执行 → 生成报告或图表
```

**两个关键点**：
1. 如何**精确**将用户问题与**相关**表信息发给大模型（上下文过滤）；
2. 大模型能力要足够强（私有化部署 32B 以上）。

**规模化难题与 RAG 解法**：

```
1000 张表全发 → 模型眼花缭乱
RAG 过滤：用户问题 → 向量检索相关表 → 只发 10 张表及字段信息 → 大模型 → SQL
```

**高质量 Text2SQL 系统 = 论文（SQL-of-Thought）+ deepagents 框架**：
- `create_deep_agent`（skills + mcp + 子智能体）；
- 子智能体示例：工具1（子智能体1，create_agent + skills + 系统提示词）、工具2（子智能体2）…
- 缺件自建：①数据库访问组件（自研 MCP server：`db_mcp_server.run_sql(sql)`）；
  ②表结构知识库（RAG/知识图谱/llm_wiki：存表结构、表关系、样本数据、正确 SQL 问答对、文本描述）；
  ③创建智能体对接上述 MCP 工具（skills 做指导方案）。
- QueryWeaver（https://github.com/FalkorDB/QueryWeaver ）：图数据库驱动的 Text2SQL
  智能体（FalkorDB 出品），独立多智能体 + 自定义流式问答输出 + 表结构/字段获取能力。
  改造思路：把 QueryWeaver 的能力集成进 DeepAgents 框架——对话入口只有
  `create_deep_agent`，通过 **工具 + mcp + 中间件（动态提示词）+ 自定义 Skills** 融合。

**优化输入源**：借鉴 spring-ai-alibaba/dataagent（https://github.com/spring-ai-alibaba/dataagent ）
的"知识联动工作流（Workflow Integration）"，设计成 skills / system_prompt / memory。

**闭环飞轮**：对话界面加点赞反馈机制——点赞后把当前**问答对嵌入 llm_wiki**（指导智能体
把核心分析思路及问答结果更新到知识库），越用越准。

### 2.11 图数据库与知识图谱

| 组件 | 说明 | 部署 |
| --- | --- | --- |
| Neo4j | 通用图数据库 | `docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -v /data/neo4j/data:/data -e NEO4J_PLUGINS=["apoc"] -e NEO4J_AUTH=neo4j/<密码> neo4j` |
| **FalkorDB** | 基于 Redis 的图数据库（课程首选，部署简单） | `docker run -p 6379:6379 -p 3000:3000 -itd -v ./data:/var/lib/falkordb/data falkordb/falkordb` |
| QueryWeaver | FalkorDB 的 Text2SQL 入口 | https://github.com/FalkorDB/QueryWeaver |
| **graphiti** | 时序知识图谱记忆 | https://github.com/getzep/graphiti （含 mcp_server） |

多智能体协作的复杂度说明（swarm……）：课程结论——**单智能体控制多个子智能体**更实用：

```
一个 agent
  └── tools
       ├── 编写文章的智能体（接入大模型）—— 子智能体 sub1 = create_agent(tools=[]); sub1.invoke()
       └── 评审智能体 ...
本质：单个 Agent + tools（mcp + 自定义工具）；Skills 指导 agent+tools 完成一系列任务
```

### 2.12 智能客服平台设计（综合案例）

```
智能客服
 ├── 核心模块：问答
 │    ├── 问：语音对话（语音转文字，借鉴 openwebui 的 STT 能力）
 │    │       意图识别（核心）智能体
 │    └── 答：路由设计
 │            ├── graphiti 知识库
 │            ├── RAG 知识库
 │            ├── 对接系统接口（业务数据）
 │            ├── 互联网搜索
 │            └── ...
 ├── 知识接口：两大 MCP server
 │    ├── graphiti 时序知识库（单独部署 mcp server）
 │    └── lightrag RAG 知识库（anythingchatrag 部署 → 建工作空间上传材料 → 封装 RAG mcp server → 绑定到 DeepAgents）
 ├── 行为接口：业务数据接口封装成 MCP server
 │    （例：订单查询/取消/退款/追踪 API → fastmcp 框架封装成 mcp 工具；
 │      本地服务接口则直接调用工具而非 mcp server）
 ├── 互联网接口：内部/公网搜索引擎（直接用搜索引擎 mcp 服务或封装 tools）
 ├── 记忆：graphiti（mcp）+ RAG（lightrag）
 ├── 智能体框架：langchain（create_agent）/ DeepAgents
 ├── 前端框架：CopilotKit / DeepAgents-ui / OpenWebUI
 └── 认证系统（多租户）：集成企业内部认证；fastapi（自行扩展）或 langgraph-api 服务（推荐）
```

前端接入形态（CopilotKit）：

```
前端 --/api/copilotkit--> CopilotRuntime（服务）--> agent
后端两种实现：① langgraph-api 提供服务（默认）；② fastapi+agent（灵活自由）
```

OpenWebUI 一体化方案：前后端综合智能体平台——hermes agent（后端）+ OpenWebUI（前端对话界面），
支持语音（多 STT：Local Whisper/OpenAI/Deepgram/Azure；多 TTS：Azure/ElevenLabs/OpenAI 等）。

### 2.13 通用协议层（课程总结）

| 协议 | 解决的问题 |
| --- | --- |
| **MCP** | 任何智能体都可以调用同一个工具 |
| **Skills** | 通用提示词（通用的业务绑定） |
| **A2A** | 不同框架智能体之间的通信 |
| **AG-UI** | 前端与后端智能体的隔离 |

智能体开发框架对比提及：crewai、autogen、pydantic ai、llamaindex、langchain、claude sdk。

### 2.14 开发工作流方法论（用 AI 做项目的流程）

```
大模型+Agent 开发软件两种方式：
  低代码：coze、dify
  编写代码：DeepAgents（课程主线）

借助工具实现应用软件的流程：
  1. 方案设计（需求分析）——最重要环节
     工具：hermes agent（首选）/ OpenClaw / DeepSeek chat / Kimi Chat
  2. 人工评审环节：需求及方案的讨论（技术与业务的锲合度）
  3. 编码实现：选择合适的 AI 开发工具
     有条件：Augment + Claude；无条件：Kimi Code、GLM 等
  4. 自测：智能体项目看【准确度】和【业务的锲合度】
```

技术实现环节的经验：
1. 开发工具：善于使用——代码层面实现逻辑（做设计效果会打折扣——最关键）；
2. 前期设计和分析：借助 hermes agent（ https://github.com/NousResearch/hermes-agent ；
   自带 `hermes chat` 命令行；可接 openwebui / hermes-web-ui
   https://github.com/EKKOLearnAI/hermes-web-ui/blob/main/README_zh.md ）；
3. 项目落地：Agent + Tools（业务对接和扩展）+ Skills（技能指导，建议先在 hermes agent 中验证）。

### 2.15 课程作业清单（可当 feature checklist 检查自己的实现）

- [ ] 前后端环境部署，基于图片正常对话；上传 PDF（deepseek）
- [ ] PDF 解析中间件（wrap_model_call）+ pymupdf4llm 文本提取 + 注入 system_prompt
- [ ] 消息携带文件全路径即可解析；扩展到任意格式（图片/pdf/文本/word）+ URL
- [ ] 联网查询（Tavily MCP）
- [ ] 上传图片支持（中间件）+ 模型动态切换
- [ ] web-search/ppt/docx/pdf 技能组合：写研究报告并存 word+ppt（需 LocalShellBackend）
- [ ] 短期/长期记忆案例（脚本运行）
- [ ] mem0 接入 deepagents 记忆系统（可选）
- [ ] PostgresStore 记忆体 + 记忆存储/读取两个工具封装
- [ ] 部署 llm_wiki 并封装 MCP；部署 milvus；部署 qwen3 嵌入模型
- [ ] hermes + hermes-web-ui 部署
- [ ] NL2SQL：对接 postgres/mysql；优化 llm-wiki-compiler 的 MCP（ingest_source/compile_wiki）
- [ ] 点赞反馈 → 问答对回写知识库
- [ ] dataagent 式管理前端（对话嵌入 testing-deepagents-ui）
- [ ] nl2sql 支持 claude/hermes/cursor/cherrystdio 等客户端（skills+mcp 解耦）
- [ ] 部署 lightrag 验证多模态；raganything 多模态
- [ ] NL2SQL 终版：falkordb 图数据库 + graphiti 记忆系统全面升级
- [ ] graphiti mcp docker 部署（create_agent 引入 mcp tools 增删改查）
- [ ] ollama 部署嵌入模型 + 重排模型
- [ ] CopilotKit 两套后端（fastapi+agent / langgraph-api+agent）
- [ ] 完整认证体系（langgraph server api + 登录注册界面）
- [ ] 智能客服：语音、意图识别智能体、路由设计
- [ ] 业务接口封装 fastmcp（openapi.json → mcp server）
- [ ] 智能代码分析平台（codebase-memory-mcp 等选型对比；管理界面 fastapi 8100 + 对话服务 langgraph-api 2026 + postgres）
- [ ] hermes 生成子智能体案例（同步 and 异步）

### 2.16 Docker 部署命令速查（课程环境）

```bash
# PostgreSQL
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=<密码> -e POSTGRES_USER=postgres -e POSTGRES_DB=postgres \
  -p 5432:5432 -v /data/postgres/data:/var/lib/postgresql/data --restart always postgres:16

# Neo4j（带 apoc 插件）
docker run -d --name neo4j --publish=7474:7474 --publish=7687:7687 \
  --volume=/data/neo4j/data:/data -e NEO4J_PLUGINS=["apoc"] -e NEO4J_AUTH=neo4j/<密码> neo4j

# Redis（带密码 + AOF）
docker run -d --name redis -p 6379:6379 -v /data/redis/data:/data \
  redis redis-server --requirepass <密码> --appendonly yes

# Milvus standalone
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start

# FalkorDB
docker run -p 6379:6379 -p 3000:3000 --name falkordb -itd falkordb/falkordb:latest
```

### 2.17 智能代码分析平台（课程末段项目）

选型对比的开源项目：
- https://github.com/DeusData/codebase-memory-mcp
- https://github.com/Egonex-AI/Understand-Anything
- https://github.com/colbymchenry/codegraph
- https://github.com/abhigyanpatwari/GitNexus

目标形态：协助代码编写、审计、评审、测试的智能代码分析平台。

项目结构（前后端）：

```
前端：
  管理界面 —— 0到1完全定制
  对话界面 —— 基于 DeepAgents-ui 深度优化
  cd ui && npm run dev（开发）/ npm run start（生产）
后端：
  管理服务：fastapi（.env → API_PORT=8100），启动 start_api.py
  对话服务：langgraph-api（.env → LANGGRAPH_PORT=2026）
    inmem（开发）/ 持久化（生产），启动 start_server_inmem.py
  数据库：postgres（.env → DATABASE_URL=...）
项目管理：对话方式（新建、索引、重新索引）+ 手工添加方式（新建、索引、重新索引）
```

---

## 第三部分：参考架构 —— 企业级 NL2SQL 平台（课程综合案例）

> 来源：《但问智能数据分析系统 —— 技术架构与功能全景文档》v0.5.0。
> 定位：基于 LLM 与知识图谱的企业级 NL2SQL 平台——用"大白话"问数据库问题，
> 系统自动翻译成 SQL 并执行返回结果；能"记住"用户偏好、学习业务术语、
> 把数据库结构画成可视化知识图谱。

### 3.1 整体架构

| 层 | 技术 | 作用 |
| --- | --- | --- |
| 前端 | React + TypeScript | 界面、按钮、输入框、图谱渲染 |
| 前端 | Vite | 构建（比 Webpack 快 10 倍+），dev server 3000，`/api` 代理到后端 8065 |
| 前端 | Ant Design | 企业级 UI 零件库 |
| 后端 | FastAPI | 接收请求、调用 AI、返回结果 |
| 后端 | FalkorDB | 图数据库：存表结构和记忆（既能查表格又能找关系） |
| 后端 | LLM | "翻译官"：人话 ↔ SQL ↔ 结果人话 |

### 3.2 后端目录结构（结构参考价值高）

```
backend/src/app/
├── main.py                 # FastAPI 入口：路由、CORS、lifespan（启动时初始化 DB/FalkorDB/Embedding）
├── core/
│   ├── settings.py         # 配置管理（LLM API Key、模型名称）
│   ├── llms.py             # 大模型封装（DeepSeek / OpenAI 兼容）
│   └── query_engine.py     # NL2SQL 查询引擎（系统的心脏）
├── api/
│   ├── connections.py      # 数据库连接 CRUD + 连通性测试
│   ├── graphs.py           # 图数据管理 + NL2SQL 流式查询接口（StreamingResponse）
│   ├── schema.py           # Schema 发现与同步
│   ├── hybrid_qa.py        # 问答库（Few-shot 学习）
│   ├── settings.py         # 系统设置（API Key 验证）
│   ├── embeddings.py       # Embedding 服务（OpenAI / Ollama）
│   └── graph/
│       ├── falkordb_client.py   # FalkorDB 客户端
│       ├── graph_loader.py      # 将数据库 Schema 加载为图
│       ├── retrieval.py         # 向量+图混合 Schema 检索
│       └── search_tools.py      # 图谱搜索工具
├── db/
│   ├── session.py          # SQLAlchemy 异步会话管理
│   ├── models/database.py  # ORM（DBConnection / QAPair / BusinessMetric）
│   ├── sql_runner/         # SQL 执行抽象层
│   └── engine/             # 各数据库驱动（MySQL/PostgreSQL/...）
├── graphdb/
│   ├── client.py           # FalkorDB 连接池
│   └── schema_graph.py     # Schema 图谱 CRUD
├── memory/
│   └── graph_memory.py     # 基于 Graphiti 的时序知识图谱记忆
├── loaders/                # Schema 加载器（base/mysql/postgres/sqlite/... 共 10+ 种）
├── models/database.py
└── agents/
    ├── SYSTEM_PROMPT.md    # 系统提示词
    └── schema_tools.py
```

### 3.3 核心技术选型与解决的问题（问答式经验）

| 技术 | 痛点 | 方案 | 项目内应用 |
| --- | --- | --- | --- |
| React 18 | jQuery 式散装代码牵一发动全身 | 组件化，像搭积木拼装 | App.tsx 用 `<Routes>` 切 7 大页面 |
| TypeScript | 弱类型运行时才暴露错误 | 编译期类型保险 | `types/api.ts` 定义 20+ 接口，前后端严丝合缝 |
| Vite | CRA/Webpack 冷启动 10~30s | 浏览器原生 ES Module，毫秒级冷启动 | vite.config.ts 代理 `/api` → 8065 解决跨域 |
| Tailwind CSS | 自定义类名冲突/覆盖混乱 | 预定义原子类直接拼 | tailwind.config.cjs 定制主题色，与 antd 暗色融合 |
| Ant Design + Ant Design X | 自写组件费时难一致 | 60+ 高质量组件 + 对话式 UI 组件 | 导航菜单、Card/Table/Modal |
| Sigma.js + Graphology | 几十表几百字段表格展示看不懂 | WebGL 加速渲染上万节点/边 | GraphVisualizationPage：表=星球、外键=连线 |
| Zustand | useState 跨组件要 prop drilling | 极简全局仓库（比 Redux 轻 90%） | 用户配置、全局连接状态 |
| SWR | 手写 fetch 样板代码多 | `useSWR(key, fetcher)` 自动缓存/重验证/重试 | 图谱列表、连接列表，切页秒开 |
| FastAPI | Flask/Django 阻塞式并发弱 | async/await 单机扛几千并发 | main.py lifespan 管理启动初始化，全 async def 路由 |
| SQLAlchemy 2.0 + aiosqlite | 手写 SQL 有注入风险、换库要改代码 | ORM 类映射 + async | 异步引擎 + 连接池 |
| FalkorDB | 关系库查关联要多次 JOIN；纯向量库表达不出"表A外键指向表B" | 节点存表/字段、边存外键关系，Cypher 一句话查关联；**内置向量索引**（`CALL db.idx.vector.queryNodes`）语义找表 | graph_loader.py 建图；retrieval.py "向量搜索+图遍历"找相关 Schema（NL2SQL 准确率根基） |
| Graphiti 记忆 | 聊天机器人每次"失忆" | 时序知识图谱：对话/实体/关系存成带时间戳节点，能回答偏好演变 | `graph_memory.py` 的 GraphMemoryTool 自动保存查询历史、成功/失败模式、用户摘要，下次查询注入 Prompt |

### 3.4 SQL-of-Thought 查询引擎（NL2SQL 的心脏，六阶段）

痛点：直接让 LLM"根据这 100 张表生成 SQL"会眼花缭乱（上下文太长）、经常幻觉（编造不存在的表名）。

| 阶段 | 模块 | 作用 |
| --- | --- | --- |
| Phase 0 | Memory | 唤起记忆：用户以前问过类似问题吗？偏好什么格式？ |
| Phase 1 | Retrieval | Schema 检索：从几百张表里精准挑出相关的 3~5 张 |
| Phase 2 | SQL Generation | SQL 生成：LLM 基于精简 Schema 写高质量 SQL |
| Phase 3 | Safety Check | 安全检查：INSERT/UPDATE/DELETE 需要用户二次确认 |
| Phase 4 | Execution | SQL 执行：统一抽象层对接 10+ 种数据库 |
| Phase 5 | Formatting | 结果润色：把 JSON 数组翻译成自然语言报告 |

理论基础：SQL-of-Thought 论文（Multi-agentic Text-to-SQL with Guided Error Correction）。

### 3.5 多数据库引擎抽象层

`db/sql_runner/base.py` 定义抽象接口 `SqlRunner`，每种数据库实现一个 `sql_runner.py`，
上层代码完全无感知。支持：MySQL、PostgreSQL、SQLite、DuckDB、BigQuery、Snowflake、
ClickHouse、SQL Server (MSSQL)、Oracle、Presto/Trino（共 10 种）。

### 3.6 核心功能模块

- **智能查询（Smart Chat）**：前端 `pages/SmartChat/` 基于 LangChain LangGraph SDK，
  支持多轮对话、文件上传、线程历史；后端 `api/graphs.py` 的 `query_database` 与
  `confirm_query` 两个端点返回 StreamingResponse。
- **知识图谱可视化**：Sigma.js 渲染。
- **数据模型（Schema Management）**：字段语义配置（`cust_id = 客户编号`），配置后向量
  索引更新，问"客户"即可精准定位。
- **问答库（Hybrid QA）**：Few-shot 沉淀——团队存 100 个高频问题正确 SQL 后，
  新问题准确率接近 95%（越用越准）。

### 3.7 FAQ 选型洞见

- **为什么 FalkorDB 而非 Neo4j**：FalkorDB 基于 Redis，部署更简单（单容器），
  且原生支持向量索引（vecf32），"语义找表"不需要外挂向量数据库。
- **LLM 生成 SQL 安全吗**：三层防护——①Prompt 明确禁止危险操作；②代码层检测
  INSERT/UPDATE/DELETE/DROP 关键字强制弹窗确认；③生产环境数据库账号只读权限。
- **为什么 antd 和 Tailwind 同时用**：antd 提供复杂业务组件，Tailwind 负责快速微调
  和原子化布局；Radix UI 提供无头组件，Tailwind 负责皮肤。
- **没有 GPU 能跑吗**：可以。LLM 走云端 API，本地只需 FastAPI + FalkorDB + SQLite。

---

## 第四部分：开发环境清单

> 来源：《环境部署》文档。

| 项 | 要求 |
| --- | --- |
| 电脑配置 | 内存 16G 以上 |
| IDE | PyCharm 最新版（专业版）与 VSCode 最新版（建议都装） |
| Python | 3.13 |
| Node.js | v22.22.1 |
| 包管理 | **uv**（`uv add <包>` / `uv pip install`；`uv sync` 按 pyproject.toml 安装/卸载依赖） |
| 模型切换工具 | cc-switch（https://github.com/farion1231/cc-switch ） |
| AI 开发工具 | Claude Code（可接 claude/glm/kimi 模型）、Augment Code（Claude opus/sonnet/haiku，推荐）、Kimi Code、codex、cursor |
| 模型建议 | 国外大模型 + 国内大模型（智谱 GLM、Kimi）同时安装两个 |

---

## 第五部分：对 agent-base 项目的改造映射建议

> agent-base 定位：LangGraph 之上的生产治理基座（长期记忆 M6 + 成本治理 T4 + 事件可观测 SSE）。
> 分享包内容与之高度互补——课程解决"怎么把 Agent 能力建起来"，本项目解决"怎么敢上生产"。
> 以下映射按 [P0] 可直接落地 / [P1] 增强现有能力 / [P2] 远期方向 分级。

### 5.1 对照表：课程概念 ↔ agent-base 现有实现

| 课程内容 | agent-base 对应 | 状态 |
| --- | --- | --- |
| `create_agent` + checkpointer（MemorySaver/thread_id） | 核心装配层 + checkpointer（阶段3） | ✅ 已有 |
| 记忆体系统（mem0 两阶段/namespace/向量检索） | `packages/agent-base-memory`：混合检索（向量+BM25+时间半衰期+显著度）、mem0 式形成管线、Letta 式记忆块、Memobase 式画像 | ✅ 已有且更完整 |
| 记忆更新智能体（跨线程分析对话→写入记忆库） | 后台 LLM 抽取候选事实 → ADD/UPDATE/DELETE/NOOP 整合 + `memory_ops` 审计 | ✅ 已有 |
| 文档知识库（PDF→切块→向量化→检索工具） | `doc_chunks` + `knowledge_search` 工具 + M7 预览/切片可视化 | ✅ 已有 |
| 检索管线 top_k=30 → rerank → 10 | 混合检索（余弦+BM25+衰减），**未见 rerank 精排层** | ⚠️ 可补（见 5.2-A） |
| MCP 工具接入 | 共享工具池 + MCP Server 即插为外部工具来源 | ✅ 已有 |
| 中间件/hook（before_model/wrap_model_call/动态提示词） | 基座级上下文引擎（core/context.py 注入去重 + token 预算修剪） | ✅ 类似机制，可对齐 API 命名 |
| langgraph-api inmem→postgres 多用户 | FastAPI + SSE 服务入口（阶段4） | ✅ 已有 |
| Loop 1 Agent 循环 | 编排交给 LangGraph | ✅ 已有 |
| Loop 2 验证循环（Grader/Rubric） | **未见**——成本治理管"花多少钱"，不管"干得好不好" | ❌ 改造机会（见 5.2-B） |
| Loop 3 事件驱动循环（cron/webhook 触发 Agent） | **未见**定时/事件触发器 | ❌ 改造机会（见 5.2-C） |
| Loop 4 爬山循环（Trace 分析→自动改进 harness） | 有全量 LLM 计量 + `/metrics` + memory_ops 审计（**数据已备**），缺分析→改进闭环 | ❌ 改造机会（见 5.2-D） |
| Human-in-the-Loop（敏感工具 interrupt 确认） | 工具池有超时+审计，**未见审批门** | ⚠️ 可补（见 5.2-E） |
| graphiti 时序知识图谱记忆 | MemoryPort 协议依赖倒置，**实现可整体替换**——graphiti 可作第二种实现 | P2（见 5.3） |

### 5.2 改造方向

**[P0] A. 记忆检索补 rerank 精排层**（来源：课程记忆体系统 + anything-chat-rag）
现有混合检索输出直接进上下文；按课程管线补一层精排：`混合召回 top_k=30 → rerank（bge-reranker 或
LLM listwise）→ top 10`。落点：`agent-base-memory` 的检索管线，rerank 模型配置进 settings，
不可用时降级为现状。ollama 部署 rerank 模型（课程作业同款做法）。

**[P0] B. 验证循环作为一等公民（Loop 2 → 基座能力）**（来源：循环工程）
把 `GradingResult(grade/score/feedback)` 契约与 `VerificationLoop`（max_retries、pass_threshold、
确定性/代理式评分器、历史最佳结果回退）抽象成基座组件，挂在 MemoryPort 工具与模块输出上：
- 记忆形成管线（抽取/整合）用确定性评分器把关（如：抽取事实必须可归因到对话原文）；
- 模块可选注册 rubric，超时与审计逻辑复用工具池现有设施。
这与成本治理互补：**T4 管"花多少钱"，Loop 2 管"值不值"**。

**[P1] C. 事件驱动触发器（Loop 3 → 模块接入形态）**（来源：循环工程 + 课堂笔记）
参照 `CronTrigger/WebhookTrigger` 设计，以"模块"形态接入（不碰基座代码，符合 ADR-002）：
- cron：定期跑记忆健康检查（失效记忆清理、画像刷新）、知识库重索引；
- webhook：外部事件（PR 创建、工单到达）触发指定模块的 agent 运行；
- 事件经现有 SSE 事件契约对外可观测，任务状态机（pending/running/completed/failed）复用。

**[P1] D. 爬山循环的数据底座（Loop 4 → 用现有审计数据）**（来源：循环工程）
`memory_ops` 审计 + 全量 usage 计量 + `/metrics` 已经是 TraceStore 的雏形。补三件事：
1. Trace 归因字段：`prompt_version`（记忆注入模板版本）、工具错误分类（timeout/permission/not found…）；
2. 周期性分析 Agent：读 metrics+审计 → 输出 issues/recommendations（JSON 契约照搬参考实现）；
3. 改进建议先落 GitHub Issue（`human_review_required` 默认开启），人工批准后才改
   system_prompt/检索参数——**自动化改进，人来把关**。

**[P1] E. 敏感工具审批门（Human-in-the-Loop）**（来源：循环工程 §人类监督）
在共享工具池加"敏感工具清单"：清单内工具调用前发 SSE `approval_required` 事件，
等待 approve/reject/modify（langgraph `interrupt` 语义）；reject 抛异常进审计。

**[P2] F. graphiti 作为 MemoryPort 的第二种实现**（来源：课堂笔记 NL2SQL/客服案例）
MemoryPort 协议已经依赖倒置；可新增 `memory-graphiti` 实现包（时序知识图谱记忆，
回答"偏好如何演变"类问题），与现有关系/向量记忆形成对照选型。graphiti 提供 mcp_server，
先以 MCP 工具形态接入验证价值，再决定是否做成完整 Port 实现。

**[P2] G. 通用协议对齐**（来源：课程 §通用协议）
- Skills 体系：基座已有工具治理；可增加"技能包"约定（提示词指令 + 工具调用引导），
  使同一模块的能力可被 claude code / hermes / cursor 等外部客户端经 MCP 复用；
- A2A/AG-UI：前端对话若要解耦，参考 AG-UI（前端↔后端智能体隔离协议）与 CopilotKit
  的 `CopilotRuntime` 模式；当前 SSE 事件契约（step/delta/plan/tool）已具备等价能力，
  保持自有契约即可，仅需在文档中注明与 AG-UI 的映射关系。

### 5.3 明确不采纳的部分（避免过度设计）

| 课程内容 | 不采纳理由 |
| --- | --- |
| NL2SQL/FalkorDB/QueryWeaver 全套 | 属垂直业务系统，不属基座范围红线（ADR-002）；其六阶段查询引擎思路可在业务模块中参考 |
| 多数据库 SqlRunner 抽象（10 种库） | 基座存储跟随 CHECKPOINTER_BACKEND（sqlite/mysql），无多库诉求 |
| Sigma.js 图谱可视化、CopilotKit 前端 | 前端仓库另行决策；现有 SSE 契约已解耦 |
| swarm 多智能体框架 | 课程结论与本项目一致：单 Agent + tools（子智能体即工具）即可，无需引入 swarm |
| 低代码（coze/dify）路线 | 与"编写代码方式（DeepAgents/langgraph）"路线互斥，基座已选后者 |

---

## 附录

### A. 原始提取文件位置

- 完整原文（含全部代码）：`docs/agent-dev/reference-loop-engineering-full.md`（本目录）
- 临时提取工作区（含课堂笔记/环境部署/架构文档原始文本，含课程示例凭据，**阅后可删**）：
  `.qqdoc_extract/`（notes_doc.txt / env_doc_clean.txt / arch_doc.txt / loop_doc_full.txt）

### B. 提取方法备注（复现用）

腾讯文档正文为 canvas 渲染，DOM 无文本。可行路径：浏览器打开文档 → 画布内
`Ctrl+A` 全选 → `Ctrl+C` 复制 → 读取系统剪贴板（需 STA 模式读取 UnicodeText，
避免 GBK 乱码）。分享包内的文件通过列表行「⋯ → 新窗口打开」获得真实 doc URL
（`docs.qq.com/doc/<ID>`）：

| 文件 | doc ID |
| --- | --- |
| The Art of Loop Engineering | `DVGt6S1dzdkdxbWpZ` |
| 环境部署 | `DVFFNSEVsdGduZWF1` |
| 课堂笔记 | `DVEJMS0NiVHZaV3hF` |
| 但问智能数据分析系统技术架构 | `DVEdQRkVJQVVHQnFY` |
| 公司介绍（未提取） | `DVEdZQ25ScFhqUWRF` |
| 空白流程图（需登录） | `DVGZidHVCRkZtQmx4` |
