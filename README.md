# agent-base

Agent 软件基座——所有 agent 模块扩展的统一起点。

**基座 = LangGraph 运行时（不自研）+ 薄封装层（模块注册 / 配置 / 装配）+ 扩展点预留。**

> **范围红线**：RAG、长期记忆、知识库、业务鉴权、业务存储等**不属于基座**——
> 它们未来以"模块"形式接入（见 `docs/adr/`）。基座只提供运行时、注册规范、
> 配置管理、可运行入口与扩展点。

## 当前状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | 工程地基（脚手架 / 工具链 / CI 四道门 / ADR） | ✅ 已完成 |
| 1 | 核心装配层（config / contracts / registry / llm / bootstrap）+ chat 样板模块 + CLI 入口 | ✅ 已完成 |
| 2 | 可观测性（request_id 贯穿 / 结构化日志） | ⬚ |
| 3 | 工具扩展点 + 对话状态（checkpointer） | ⬚ |
| 4 | FastAPI + SSE 服务入口 + supervisor 多 Agent 模板 | ⬚ |
| 5 | 收尾 + 文档闭环（module-guide 五步接入实操） | ⬚ |

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows（Linux/macOS: source .venv/bin/activate）
pip install -e ".[dev]"
cp .env.example .env           # 填入 LLM_API_KEY（阶段 1 起由 config.py 消费）
pytest
```

运行一个对话（阶段 1 起可用）：

```bash
python -m agent_base --message "你好"      # 单轮对话（需 .env 已配 LLM_API_KEY）
python -m agent_base --module chat         # 交互式多轮对话
python -m agent_base --version
```

## 质量门（四道，本地与 CI 完全一致）

```bash
ruff check src tests && ruff format --check src tests   # 1. lint + format
mypy                                                     # 2. 类型检查（strict）
pytest                                                   # 3. 测试
pip-audit                                                # 4. 依赖安全审计
```

## 项目结构

```
agent-base/
├── pyproject.toml            # 依赖 + 工具链配置（ruff / mypy / pytest）
├── .env.example             # 配置契约（由 core/config.py 消费）
├── src/agent_base/
│   ├── core/                # config / contracts / registry / llm / bootstrap
│   ├── modules/chat/        # 样板模块（graph + module + tools，兼作接入模板）
│   └── entrypoints/cli.py   # CLI 入口（--module / --message / --version）
├── tests/                   # config / registry / chat / cli / smoke
├── docs/
│   ├── adr/                 # 架构决策记录
│   └── module-guide.md      # 五步接入手册（新模块接入不改基座）
└── .github/workflows/ci.yml # CI：lint → typecheck → test → audit
```

## 设计决策

- [ADR-001：LangGraph 作为基座运行时](docs/adr/001-langgraph-as-runtime.md)
- [ADR-002：显式清单模块注册](docs/adr/002-module-contract.md)
- [module-guide：五步接入新模块](docs/module-guide.md)
