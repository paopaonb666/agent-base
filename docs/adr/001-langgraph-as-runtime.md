# ADR-001: LangGraph 作为基座运行时

- 状态：已接受（2026-09-04）
- 影响范围：整个基座的编排层、依赖策略、CI 安全门

## 背景

agent-base 需要一个成熟、运行稳定、社区活跃的开源 Agent 框架作为编排核心，
基座只在其上做薄封装（装配 / 注册 / 配置 / 规范），不重新发明框架能力。
候选对比（2026-09-04 时点联网核实）：

| 维度 | LangGraph | MS Agent Framework | PydanticAI | CrewAI | OpenAI Agents SDK |
| --- | --- | --- | --- | --- | --- |
| 稳定性 | 2025-10 GA，现 1.2.6，生产案例最多（Uber/LinkedIn/Klarna/Replit/Elastic） | 2026-04 GA，生态年轻 | v2.0 稳定（2026-06），单 agent 定位 | 生产可用，黑盒感强 | 绑定 OpenAI 生态 |
| 文档完善度 | 最全，中文资料丰富 | 微软规范，Python 资料薄 | 清晰，案例少 | 友好但深度不足 | 简洁 |
| 扩展能力 | 图原语最底层：checkpointer/Store/interrupt/subgraph/supervisor 一等公民 | 图 + MCP/A2A 原生，抽象层厚 | 类型安全最强，多 agent 编排需自行组装 | 角色模型固定 | 定制受限 |
| 生态支持 | LangChain 全家桶 + LangSmith + MCP adapters | Azure/Foundry 深度绑定 | 增长快体量小 | 大但向托管平台倾斜 | OpenAI 系 |

## 决策

1. 采用 **LangGraph** 作为唯一编排运行时，依赖下限锁定 **>= 1.2.6**：
   2026 年公开披露的 CVE-2025-68664（反序列化注入，CVSS 9.3）、
   CVE-2025-67644（SQLite checkpointer SQL 注入）、CVE-2026-34070（路径穿越）
   均已在 1.2.6 修复；CI 常驻 `pip-audit` 持续监控依赖漏洞。
2. 基座保持薄封装：业务模块面向 `AgentModule` 契约（阶段 1 落地），
   不直接 import LangChain 生态中易变或已弃用的命名空间
   （如 `langgraph.prebuilt`，1.0 起迁移至 `langchain.agents` 等价物）。
3. 三个扩展点直接使用原生能力，基座不二次抽象：
   - 工具调用 → `ToolNode` / function calling
   - 对话状态（短期） → checkpointer
   - 多 Agent 协作 → 1.1+ 原生 `create_supervisor` + subgraph

## 后果

**正面**
- checkpointer / interrupt / supervisor / 流式契约原生可用，基座零自研
- LangSmith 观测生态即插即用
- chat-agent 已有 LangGraph agent 模块，未来迁移路径平滑

**负面（及缓解）**
- 学习曲线陡（节点/边/状态/条件路由）→ 基座把这些概念封装进模块契约，
  业务模块只面对 `AgentModule` 协议
- 生态历史 API 变动频繁 → 版本锁定 + CI 兼容测试 + 模块不 import 易变命名空间

## 否决项

- **MS Agent Framework**：Azure/.NET 倾向，Python 侧资料积累薄
- **PydanticAI**：多 agent 编排弱；未来可在个别模块内部组合使用（类型安全边界）
- **CrewAI**：角色抽象限制控制力，排障黑盒，平台化锁定风险
- **OpenAI Agents SDK**：生态绑定 OpenAI
