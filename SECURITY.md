# 安全策略

## 支持版本

| 版本 | 支持状态 |
| --- | --- |
| 0.2.x | ✅ 接收安全修复 |
| 0.1.x | ❌ 已停止支持，请升级 |

安全修复只进入最新的次版本；升级说明见 [CHANGELOG.md](CHANGELOG.md)。

## 私下报告

**不要为安全漏洞开公开 issue。**

请通过 GitHub Security Advisories 私下报告：

<https://github.com/paopaonb666/agent-base/security/advisories/new>

报告请尽量包含：

- 受影响的版本 / commit；
- 复现步骤或最小 PoC；
- 影响面（数据泄露、越权、拒绝服务、供应链等）；
- 你希望的披露时间（默认 90 天）。

我们会在 3 个工作日内确认收到，并在修复发布后致谢（除非你要求匿名）。
在修复可用之前，请不要公开细节。

## 已有防线

- **生产快速失败**：`ENV=production` 时缺 `LLM_API_KEY`、记忆开启却缺
  `MEMORY_AUTH_SECRET`、`CORS_ORIGINS="*"`、MySQL 连接项不全，都会拒绝启动；
- **身份与防重放**：用户作用域端点经可插拔 `AuthBackend`；
  默认 `HmacHeaderAuth` 校验 `X-User-Sig = HMAC-SHA256(user_id\ntimestamp\nnonce, secret)`，
  时间戳窗口默认 300s，窗口内 nonce 重放返回 401；
- **依赖地板**：`langgraph>=1.2.6` 修复 CVE-2025-68664（CVSS 9.3）、
  CVE-2025-67644、CVE-2026-34070；CI 常驻 `pip-audit --skip-editable`；
- **工具治理**：共享工具池统一超时、审计与重名快速失败；
  `python_repl` 为 exec 级能力，默认关闭；
- **输入约束**：上传图片按 magic bytes 校验，文档解析有输入/输出上限，
  用户 ID 限定 `[A-Za-z0-9._-]{1,64}`；
- **凭据卫生**：仓库不含 `.env`、本地数据库或密钥；日志不打印明文密钥。

## 已知边界（不属于漏洞）

- nonce 缓存与成本账本为**单进程口径**：多 worker 部署需网关侧防重放与
  采集侧聚合（见 [docs/design-review.md](docs/design-review.md) H3）；
- 记忆检索单用户候选上限 2000 条，超出部分不参与召回（H2）；
- `memory` 后端的成本账本重启清零，方向安全但会低估已花费额（H6）；
- 基座不提供业务鉴权：公网多用户部署应注入真实 `AuthBackend`。

这些是明确记录的设计取舍，请勿作为漏洞报告；如果它们在你的场景下造成
实际安全影响，欢迎在报告里说明具体利用路径。
