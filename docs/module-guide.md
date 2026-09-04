# Module 接入手册（五步）

新增一个 agent 模块，**不改基座一行代码**。本文以 `hello` 模块为例，逐步走通。

## 前置约定

- 模块是 `src/agent_base/modules/<name>/` 下的一个 Python 包。
- 该包必须在 `__init__.py` 里暴露一个 `module` 对象，它实现 `AgentModule` 契约
  （见 `src/agent_base/core/contracts.py`）。
- `AGENT_MODULES` 环境变量是**唯一启用开关**（ADR-002）：模块代码存在 ≠ 运行时启用。

模块契约（`AgentModule`）四个成员：

| 成员 | 说明 |
| --- | --- |
| `name` | 稳定标识，必须与目录名、`AGENT_MODULES` 条目一致（registry 强制校验） |
| `description` | 非空描述，供发现 / supervisor 编排使用 |
| `build_graph(ctx)` | 构造并返回编译后的 LangGraph 图；`ctx` 提供 `settings` / `llm` / `checkpointer` |
| `get_tools()` | 返回模块贡献到全局工具池的工具列表（阶段 3 起被执行；现在可返回空） |

## 五步

### 第 1 步：建目录

```text
src/agent_base/modules/hello/
```

### 第 2 步：写 `graph.py`（简单模块可省）

```python
from langgraph.graph import END, START, MessagesState, StateGraph
from agent_base.core.contracts import ModuleContext

def build_hello_graph(ctx: ModuleContext):
    llm = ctx.llm

    def reply(state: MessagesState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("reply", reply)
    g.add_edge(START, "reply")
    g.add_edge("reply", END)
    return g.compile()
```

### 第 3 步：写 `module.py`（实现契约）

```python
from agent_base.core.contracts import ModuleContext
from agent_base.modules.hello.graph import build_hello_graph

class HelloModule:
    def __init__(self):
        self.name = "hello"
        self.description = "A hello-world sample module."

    def build_graph(self, ctx: ModuleContext):
        return build_hello_graph(ctx)

    def get_tools(self):
        return []
```

### 第 4 步：写 `tools.py`（可空）与 `__init__.py`

`tools.py`（阶段 3 前可返回空列表）：

```python
from langchain_core.tools import BaseTool

def get_tools() -> list[BaseTool]:
    return []
```

`__init__.py`（暴露 `module` 对象——registry 靠它发现模块）：

```python
from agent_base.modules.hello.module import HelloModule

module = HelloModule()
```

### 第 5 步：启用

在 `.env` 里把 `hello` 加进清单（顺序即装配顺序）：

```text
AGENT_MODULES=chat,hello
```

然后运行：

```bash
python -m agent_base --module hello --message "hi"
```

## 失败即报错（不静默跳过）

registry 在启动时对每个名字做四道校验，任何一道不过都**立即失败**：

1. 名字必须匹配 `^[a-z][a-z0-9_]*$`（防路径穿越 / 任意导入）
2. 包必须能 import（未知名 = 配置错误）
3. 包必须暴露 `module` 对象
4. `module.name` 必须与清单条目一致，且实现 `description` / `build_graph` / `get_tools`

## 不做什么（范围红线）

模块里可以放任何业务逻辑，但**基座不含** RAG、长期记忆、知识库、业务鉴权/存储——
这些能力未来以模块形式接入，或走 `docs/future/` 的路线图。
