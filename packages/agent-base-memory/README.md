# agent-base-memory

`agent-base` 的长期记忆子系统（0.1.0）。

它把"记得住"和"知道得准"从基座里分出来，作为可选发行包提供：
回忆与画像、会话记忆形成管线、文档知识索引（切片 + 向量）、以及
`/v1/memory` 管理 API。基座只保留契约（`agent_base.core.contracts.MemoryServicePort`）
与一个 entry point 发现点，源码不 import 本包。

## 安装

本包**尚未发布到 PyPI**，目前从基座仓库安装：

`bash
# 在基座仓库根：基座 + 本包（可编辑安装，开发与 CI 的方式）
pip install -e . -e packages/agent-base-memory

# 或者用 extra（等价于上面）
pip install -e ".[memory]"

# 只装本包：自动带上匹配版本的 agent-base
pip install -e packages/agent-base-memory
`

只装基座（不需要长期记忆）时，必须同时设 @@MEMORY_ENABLED=false@@；
默认值是 @@true@@，启用即承诺——缺包会启动失败并提示安装命令。
发布到 PyPI 后再改为 @@pip install agent-base-memory@@ 形式。

## 装配

基座通过 `agent_base.memory` entry point 发现两样东西：

- `service` → `agent_base_memory.plugin:build`：装配 `MemoryService` 与记忆工具；
- `routes` → `agent_base_memory.api:routers`：`/v1/memory` 管理路由。

`MEMORY_ENABLED=true`（默认）时这些必须可发现——启用记忆却缺包会在启动
失败并提示安装；`MEMORY_ENABLED=false` 时基座完全不加载本包。

## 迁移

记忆表有独立 Alembic 链（`alembic_version_memory`），与基座迁移分开：

`bash
# 仓库根
python -m alembic -c packages/agent-base-memory/alembic.ini upgrade head

# 容器内（镜像里这套资产在 /app/memory-alembic/）
python -m alembic -c memory-alembic/alembic.ini upgrade head

# 回滚：删掉六张记忆表，基座表不动
python -m alembic -c packages/agent-base-memory/alembic.ini downgrade base
`

连接参数与基座一致：`AGENT_BASE_DB_*` 环境变量 > `.env` 中的
`CHECKPOINTER_MYSQL_*`（见仓库根的 `config.py` / `.env.example`）。

首迁移是幂等的（`CREATE TABLE IF NOT EXISTS`，索引内联在建表语句里），
从 0.2.0 的单包版本升级时**无需** @@stamp@@：老库里表已存在，迁移只把
版本写进 @@alembic_version_memory@@。详见基座仓库的 @@docs/memory-package-split.md@@。
