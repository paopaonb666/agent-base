# 模块接入手册

本文是"新增一个 agent 模块、不改基座代码"的操作手册。示例与
`src/agent_base/modules/hello/` 逐行对齐——该模块就是按本手册接入的，
并由 `tests/test_hello_module.py` 验收。

## 接入五步

### 第 1 步：建包

在 `src/agent_base/modules/` 下新建目录，包名必须是

- 小写标识符，匹配 `^[a-z][a-z0-9_]*$`；
- 与模块的 `name` 完全一致；
- 不能叫 `supervisor`（保留入口名）。

最小文件集：

```
src/agent_base/modules/<name>/
├── __init__.py   # 暴露 module 对象（registry 只认这个属性）
├── module.py     # AgentModule 实现
├── graph.py      # 图构建
└── tools.py      # 工具贡献（可空）
```

### 第 2 步：实现 `AgentModule`

```python
# src/agent_base/modules/hello/module.py
from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.modules.hello.graph import build_hello_graph


class HelloModule:
    def __init__(self) -> None:
        self.name = "hello"
        self.description = "Hello-world sample module from the module-guide five-step recipe."

    def build_graph(self, ctx: ModuleContext) -> Graph:
        return build_hello_graph(ctx)

    def get_tools(self) -> list[BaseTool]:
        return []
```

`__init__.py` 必须暴露 `module`：

```python
from agent_base.modules.hello.module import HelloModule

module = HelloModule()
```

契约只有四个成员：`name` / `description` / `build_graph(ctx)` / `get_tools()`。
运行时服务一律从 `ctx` 取（`ctx.llm`、`ctx.checkpointer`、`ctx.tools`、
`ctx.memory`），模块不自行创建 LLM、数据库连接或工具池。

### 第 3 步：构图

绝大多数模块是"单 LLM 节点 + 可选工具循环"，直接用共享工厂：

```python
# src/agent_base/modules/hello/graph.py
from agent_base.core.contracts import Graph, ModuleContext
from agent_base.core.graphs import build_single_agent_graph

MODULE_NAME = "hello"


def build_hello_graph(ctx: ModuleContext) -> Graph:
    return build_single_agent_graph(ctx, name=MODULE_NAME)
```

工厂会自动：

- 用 `ctx.tools`（共享工具池）绑定模型，或在无工具时省掉 `ToolNode`；
- 对模型输入做上下文工程（注入去重 + token 预算修剪）；
- 用 `ctx.checkpointer` 编译图，并以 `name=<模块名>` 命名——supervisor 依赖这个名字。

需要自定义拓扑（多节点、条件边、子图）时，直接使用 `core.context` 与
LangGraph 原生 API 组装，**仍然必须**：

1. 用 `ctx.checkpointer` 编译（否则重启丢失会话）；
2. 以 `name=<模块名>` 编译（否则 supervisor 无法编排）；
3. 返回 `core.contracts.Graph` 类型。

### 第 4 步：贡献工具（可选）

`get_tools()` 返回的 `BaseTool` 会进入**共享工具池**：超时包装、调用审计、
指标、重名检查由基座统一施加，模块不需要重复实现。

```python
# src/agent_base/modules/hello/tools.py
from langchain_core.tools import BaseTool


def get_tools() -> list[BaseTool]:
    return []
```

规则：

- 工具名在**所有模块 + 基座工具库**之间必须唯一，重名在装配期快速失败；
- 工具失败不必自行 try/except——`handle_tool_error` 会归一成 `ToolMessage` 反馈给模型；
- 工具内部不要做长阻塞；单次执行有 `TOOL_TIMEOUT_SECONDS` 上限。

### 第 5 步：配置与测试

启用模块（`AGENT_MODULES` 是模块清单；模块不在基座自带包里时，还要把包前缀
加进 `AGENT_MODULE_PACKAGES`，见下一节）：

```bash
# .env
AGENT_MODULES=chat,writer,hello
AGENT_MODULE_PACKAGES=agent_base.modules   # 默认值；独立包追加在后面
```

至少覆盖四类测试：

| 测什么 | 参考 | 断言要点 |
| --- | --- | --- |
| 契约 | `test_hello_module_contract` | `name` / 非空 `description` / `get_tools()` 形态 |
| registry | `test_registry_loads_hello` | `load_modules(["hello"])` 返回顺序与名字一致 |
| 图可运行 | `test_hello_graph_responds` | 用 `ScriptedChatModel` + 最小 `ModuleContext` 跑通一轮 |
| 完整装配 | `test_hello_assembles_in_full_runtime` | `create_runtime(Settings(agent_modules="chat,writer,hello"))` 能取到图 |

为外部依赖注入 fake（LLM、checkpointer、记忆端口），不要连真实服务；
覆盖率门要求分支 ≥85%，纯声明型代码也要有装配路径测试。

## 把模块发布成独立包

模块不必住在基座仓库里。registry 按 `AGENT_MODULE_PACKAGES` 给出的导入前缀
依次查找 `<包>.<模块名>`，所以一个独立发行包只要满足「导入路径以模块名结尾、
`<模块名>/__init__.py` 暴露 `module`」就能被装配，基座无需改代码。

```
packages/acme-agent-modules/
├── pyproject.toml          # name = "acme-agent-modules", 依赖 agent-base>=0.3,<0.4
└── src/acme_modules/
    └── report/
        ├── __init__.py     # from .module import module; __all__ = ["module"]
        ├── module.py
        └── graph.py
```

消费方部署时登记包前缀：

```bash
# .env
AGENT_MODULE_PACKAGES=agent_base.modules,acme_modules
AGENT_MODULES=chat,report
```

规则：

- 前缀按书写顺序尝试，**同名以靠前的包为准**（`agent_base.modules` 在前时，
  基座自带的同名模块会遮住外部包，便于本地覆盖调试）；
- 前缀必须是小写点分标识符（`^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$`），
  否则装配期报 `invalid module package`；
- 列表为空直接报 `no module packages configured`——不存在“隐式默认”之外的兜底；
- 包内导入失败（不是包本身缺失）会以 `failed to import from '<前缀>'` 上报，
  不会被静默跳过。

测试独立包模块时把前缀传进去，不必改动全局配置：

```python
modules = load_modules(["report"], packages=["acme_modules"])
assert list(modules) == ["report"]
```

发布这类包不需要基座配合发版：只要遵守 `AgentModule` 契约与本节规则，
就可以独立版本化、独立进 CI。

## 接入检查清单

- [ ] 包名 = 模块 `name`，匹配 `^[a-z][a-z0-9_]*$`，且不是 `supervisor`
- [ ] `__init__.py` 暴露 `module` 对象
- [ ] `description` 非空，能一句话说明模块能力
- [ ] `build_graph` 用 `ctx.checkpointer` 编译、`name=<模块名>`
- [ ] 需要模型时用 `ctx.llm`，不自行构造客户端
- [ ] 工具名全局唯一；长任务遵守超时约定
- [ ] `AGENT_MODULES` 默认值与部署 `.env` 均已登记；模块在独立发行包里时，
      包前缀同时登记到 `AGENT_MODULE_PACKAGES`
- [ ] 契约 / registry / 图 / 完整装配四类测试通过
- [ ] `ruff check`、`ruff format --check`、`mypy`、`pytest` 全绿

## 常见失败与定位

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 启动即 `unknown module 'x'` | 所有 `AGENT_MODULE_PACKAGES` 里都没有这个包名 | 确认包已安装，且前缀已登记 |
| `invalid module package 'X'` | 前缀不是小写点分标识符 | 改成 `acme_modules` 这类形式 |
| `module 'x' does not expose a 'module' object` | `__init__.py` 没导出 `module` | 按第 2 步补上 |
| `module loaded as 'x' but declares name 'y'` | 包名与 `name` 不一致 | 改 `name` 或改目录名 |
| `duplicate tool name ...` | 与其它模块/工具库重名 | 给工具加模块前缀 |
| `supervisor` 相关报错 | 用了保留名 | 换一个模块名 |
| 重启后会话丢失 | 图没用 `ctx.checkpointer` 编译 | 按第 3 步修正 |
