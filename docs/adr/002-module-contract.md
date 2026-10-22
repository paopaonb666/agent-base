# ADR-002：显式清单 + fail-fast 的模块接入契约

- 状态：已采纳（Accepted）
- 日期：2026-09-20
- 决策者：agent-base 维护者
- 相关：ADR-001（运行时）、`src/agent_base/core/contracts.py`、`src/agent_base/core/registry.py`

## 背景

基座承诺"业务以模块接入、不改基座代码"。要让这条承诺可验证，必须回答：

1. **哪些模块处于激活状态**——由代码自动发现、配置文件声明，还是两者混合？
2. **模块坏了怎么办**——启动时报错，还是静默跳过并留下一个看似正常的服务？
3. **契约怎么校验**——鸭子类型、运行时 Protocol 检查，还是显式字段校验？

常见的"插件自动发现"（扫描目录 / entry points）看似方便，但会把
**加载顺序、同名冲突、半成品模块**变成运行期的隐式行为：服务能起来，
却少了某个模块，错误只在第一条用户请求时暴露。对自托管、单进程为主的
部署形态，这类静默降级比启动失败更昂贵。

## 决策

### 1. `AGENT_MODULES` 是唯一事实来源

- 逗号分隔（也接受 JSON 数组），**顺序即装配顺序**；默认 `chat,writer`。
- 只有清单内的模块会被导入。未列入的包即使存在也不会被扫描。
- `supervisor` 是保留入口名（把所有已注册模块编排为 sub-agent），
  不允许作为普通模块名。
- 清单本身属于配置契约（`.env.example`），不是代码常量——部署时改环境变量
  即可切换能力组合，无需改代码或重打包。

### 2. 显式清单 + 严格校验

`load_modules(names, *, prefix="agent_base.modules")` 按清单顺序执行：

| 步骤 | 规则 | 失败结果 |
| --- | --- | --- |
| 名字校验 | `^[a-z][a-z0-9_]*$`（小写标识符，无路径分隔符） | `RegistryError` |
| 重复校验 | 同一名字出现两次即拒绝 | `RegistryError` |
| 导入 | `import {prefix}.{name}`；模块自身的 `ModuleNotFoundError` 与"未知模块"区分报错 | `RegistryError` |
| 暴露对象 | 包的 `__init__` 必须暴露 `module` 属性 | `RegistryError` |
| 契约字段 | `name` 必须等于请求名；`description` 非空字符串；`build_graph` / `get_tools` 可调用 | `RegistryError` |

`prefix` 可注入：测试可以把 registry 指向 fixture 包，证明"显式清单机制"
与基座自身的模块目录解耦。

### 3. fail-fast，不静默跳过

任何未命中（未知名字、导入异常、契约不符）都以 `RegistryError` 中止启动。
入口层（CLI / uvicorn）把它转成可行动的报错并退出，而不是带着残缺模块继续服务。
"启用即承诺"是同一原则的另一面：`MCP_ENABLED=true` 但 extras 未安装、
planner 被请求但未启用（503），都遵循"要么可用、要么明确失败"。

### 4. 契约保持最小

`AgentModule` 只有四个成员：`name` / `description` / `build_graph(ctx)` /
`get_tools()`。运行时服务（LLM、checkpointer、工具池、记忆端口、计划端口）
经 `ModuleContext` 注入，模块不自行构造基础设施——这是"不改基座"的结构性保证。

## 后果

**正面**

- 启动成本与模块数线性相关，无目录扫描、无 entry-point 元数据解析。
- 配置即文档：看 `AGENT_MODULES` 就知道线上跑哪些能力。
- 测试可注入 `prefix` 与最小 `ModuleContext`，模块可在无网络、无数据库下验证。
- 失败模式单一：坏模块 = 起不来，不会出现"服务正常但没有记忆/没有工具"。

**约束与代价**

- **新增模块必须同时改配置**：建包之外还要把名字加进 `AGENT_MODULES`
  （默认值与生产部署的 `.env` 各一处），否则不会生效。
- **没有第三方自动发现**：外部包不能靠安装即插即用，需要显式登记；
  这与"可审计的装配"目标一致，但牺牲了一点便利。
- **顺序有语义**：模块按清单顺序装配；工具重名在装配期快速失败
  （共享池不允许跨模块同名工具），因此模块命名需要命名空间意识。
- **契约校验是结构性的**：`runtime_checkable` 只验证成员存在，不验证行为；
  行为正确性由各模块测试负责。

## 参考

- `src/agent_base/core/registry.py`：实现与错误信息。
- `src/agent_base/core/contracts.py`：`AgentModule` / `ModuleContext`。
- `docs/module-guide.md`：五步接入手册与检查清单。
- `tests/test_hello_module.py`：按手册逐字接入的验收测试。
