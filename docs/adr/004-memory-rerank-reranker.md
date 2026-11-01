# ADR-004：记忆混合检索引入 rerank 精排层

- 状态：已采纳（0.3.x）
- 日期：2026-11-01
- 相关：ADR-003 记忆包拆分、`docs/agent-dev/agent-dev-knowledge.md` §记忆体系统、
  `docs/agent-dev/integration-review.md` §3.1

## 背景

`agent_base_memory/retrieval.py` 的混合检索按四分量线性加权（vector 0.40 /
keyword 0.30 / recency 0.15 / salience 0.15）打分后**直接截断 top_k**。线性融合
对"语义相似但表述不同"的候选区分度有限：BM25 与余弦各自归一后的分数噪声会把
真正相关的记忆挤出 top_k（注入路径 top_k 默认仅 6，容错极小）。

外部方法论（循环工程课程记忆体系统）的标准做法是两段式：**宽召回
（top_k=30）→ rerank 精排 → 10 份**。检索没有精排层，是记忆质量方向
评审（`docs/agent-dev/integration-review.md` §3.1）确认的首要缺口。

## 决策

1. **新增 `Reranker` 协议**（`agent_base_memory/rerank.py`），形态照抄
   `EmbeddingClient`：`async rerank(query, documents) -> list[float] | None`——
   **返回 None 即降级信号**（与嵌入不可用同语义），失败进入冷却期，冷却期内
   直接返回 None 不发请求。
2. **默认实现走 openai 兼容 `/rerank` 端点**（硅基流动 bge-reranker 系列同款
   请求/响应契约），HTTP 客户端用 `httpx`（openai SDK 已传递引入，显式声明
   无新增供应链面）。另提供 `NullReranker`（恒 None，未启用时装配）。
3. **插入点在 `MemoryService.search`**（工具 / HTTP 搜索 / 上下文注入三条路径
   的汇合处），而非 `recall_memories` 内部：reranker 激活时以
   `fetch_k = top_k × fetch_multiplier`（默认 4）宽召回，精排后截回 `top_k`。
4. **排序语义：rerank 分为主排序键、混合分为次级排序键**。rerank 相关度分与
   混合分不同分域，不做加权融合；rerank 分写入 `ScoredMemory.components["rerank"]`
   （既有 dict 分量，零 schema 破坏），消费方（工具展示、HTTP 响应）优先展示它。
5. **宽召回阶段 min_score 减半**：精排前宁多勿漏（与无向量降级路径的既有注释
   同一哲学）；精排后不再做二次混合分门槛。
6. **整合阶段旁路 rerank**：`pipeline.py` consolidate 的相似检索（top_k=3）
   保持现状——整合是后台任务且高频，精排收益小于延迟与成本代价。
   `search` 增加 `use_rerank: bool = True` 参数承载该语义。
7. **知识库路径同款接入**：`knowledge.py:search_knowledge` 在排序后、截断前
   应用同一 reranker。
8. **配置挂 `MemorySettings`**（平铺 env `MEMORY_RERANK_*`）：`ENABLED(false)` /
   `BASE_URL` / `API_KEY` / `MODEL` / `FETCH_MULTIPLIER(4)` / `TIMEOUT_SECONDS(5)` /
   `COOLDOWN_SECONDS(60)`。默认关闭；启用但 base_url/model/api_key 缺失 →
   启动 fail-fast（启用即承诺）。

## 被否方案

- **LLM listwise 精排**（把候选列表交给 LLM 排序）：延迟与成本高一个数量级，
  且不可缓存；作为 P2 备选，仅在 rerank 模型质量不足时评估。
- **本地 bge-reranker 推理**：引入 PyTorch/ONNX 运行时依赖，违背依赖地板约束。
- **在 `recall_memories` 内部插入精排**：函数签名需拆 fetch_k/final_k，且该函数
  被多处直接调用（含测试），改动面大；`MemoryService.search` 已是三条路径的
  汇合点，改动面最小。
- **rerank 分与混合分加权融合**：两分域量纲不同，权重不可解释；主次键排序
  语义更清晰。

## 后果

正面：

- 注入与检索路径的 top 命中质量提升（语义改写、同义替换的候选可被救回）；
- 降级路径完备：reranker 不可用时行为与现状完全一致（None 降级）；
- 配置默认关闭，既有部署零变化。

代价（明示）：

- 检索路径增加一次网络往返（宽召回阶段 min_score 减半也会多带候选），
  端到端延迟上升；冷却机制保证故障时快速回落；
- `fetch_multiplier` 过大时会放大候选文本量（精排请求体积随候选数线性增长），
  需在容量契约（候选上限 2000）内使用；
- `httpx` 成为记忆包的直接依赖（此前仅为传递依赖）。
