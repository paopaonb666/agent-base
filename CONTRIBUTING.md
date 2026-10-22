# 贡献指南

感谢你愿意改进 agent-base。本仓库的定位是"编排之上的治理基座"：契约、安全与
可观测性优先于功能数量。提交前请确认改动符合定位，并跑通下面四道门。

## 环境

- Python ≥ 3.10（开发与 CI 主版本为 3.12）
- 可选：Docker（仅验证容器构建时需要）
- 可选：MySQL ≥ 8.0.19（仅运行 integration 标记的测试时需要）

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,doc,search,mcp]"
cp .env.example .env                 # 跑需要 LLM 的示例时填 LLM_API_KEY
```

## 四道质量门（本地必须与 CI 一致）

```bash
ruff check src tests examples
ruff format --check src tests examples
mypy
pytest
pip-audit --skip-editable
```

- `pytest` 默认跑核心集（排除 `integration` 标记），分支覆盖率门槛 85%；
- MySQL 集成测试带 `@pytest.mark.integration`，连接参数经
  `MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD` 传入，
  连接不可用时自动 skip；
- 容器改动请额外跑 `docker compose config --quiet`（先 `cp .env.example .env`）。

## 契约优先

- 新能力以**模块**接入：实现 `AgentModule`，不要改 `core/` 的公共契约。
  五步流程与检查清单见 [docs/module-guide.md](docs/module-guide.md)。
- 修改 `AgentModule` / `ModuleContext` / `MemoryPort` 属于破坏性变更，
  必须附 ADR（`docs/adr/`）并在 `CHANGELOG.md` 标注。
- `AGENT_MODULES` 是唯一开关；不要引入目录扫描或 entry-point 自动发现。
- 失败必须显式：新增配置项要有校验与可行动的报错，不做静默降级。

## 安全地板

- 不引入弱默认密钥；`ENV=production` 下的快速失败校验不得放宽。
- 不降低 `langgraph>=1.2.6` 等 CVE 版本地板；依赖升级必须同时通过
  `pip-audit --skip-editable`。
- 用户作用域端点必须经 `AuthBackend` 解析身份，不得直接信任裸头。
- 日志、指标、错误信息中不得出现 API key、密码或完整 prompt。
- 发现漏洞请不要开公开 issue，按 [SECURITY.md](SECURITY.md) 私下报告。

## 文档与测试

- 行为变化必须同步更新 README 与相关 `docs/` 页面；新增配置项要进
  `.env.example`（配置契约）。
- 新模块至少覆盖：契约、registry 装配、图可运行、完整运行时四类测试。
- 测试用 fake 注入外部依赖（LLM / checkpointer / 记忆端口），不连真实服务。
- 中文文档与注释可以保留；标识符、日志键、指标名保持英文。
- 变更记录写入 `CHANGELOG.md` 的 `Unreleased` 节（发布时归档到版本号）。

## 提交流程

1. Fork 仓库并从 `main` 切分支：`feat/<topic>`、`fix/<topic>`、`docs/<topic>`；
2. 一个 commit 只做一件事，消息用英文祈使句（如
   `add docker compose deployment`），正文说明"为什么"而非"做了什么"；
3. 推送分支并开 PR，描述里写清：动机、行为影响、验证命令与结果；
4. CI 四道门 + docker 构建全绿后由维护者合并；不要求签署 CLA，
   贡献按 [Apache-2.0](LICENSE) 授权。

## 报告问题

- 功能缺陷 / 文档错误：GitHub Issues；
- 安全漏洞：GitHub Security Advisories（见 [SECURITY.md](SECURITY.md)）；
- 设计讨论：先开 Issue 说明场景与约束，达成一致后再写代码。
