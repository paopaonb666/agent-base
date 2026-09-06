"""基线：langgraph MySQL checkpointer 表。

Revision ID: 0001
Revises:
Create Date: 2026-09-06

说明
----
本迁移不重复声明 DDL，而是**委托 langgraph-checkpoint-mysql 库自身的
``MIGRATIONS`` 序列**逐条执行，并同步 ``checkpoint_migrations`` 版本记录。
这样：

- 迁移与库内建表逻辑永不脱节（langgraph 升级时此处自动跟随）；
- ``AIOMySQLSaver.setup()`` 在读取到 ``checkpoint_migrations`` 已到最新
  版本后变为幂等 no-op，二者不冲突；
- 已有库（如本机已由 setup() 建表）可通过 ``alembic stamp head`` 直接对齐，
  dump / downgrade 仍可控。

幂等性：通过逐条读取 ``checkpoint_migrations`` 的当前最大版本，仅执行尚未
应用的迁移（与库内 setup() 一致的判断方式），因此重复运行是安全的。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _migrations() -> list[str]:
    """从 langgraph 库获取完整的 MIGRATIONS 序列（惰性导入）。"""
    from langgraph.checkpoint.mysql.base import MIGRATIONS

    return list(MIGRATIONS)


def _current_version(bind: sa.Connection) -> int:
    """读取 checkpoint_migrations 的最大版本；表不存在时视为 -1。"""
    if not sa.inspect(bind).has_table("checkpoint_migrations"):
        return -1
    result = bind.execute(sa.text("SELECT MAX(v) AS v FROM checkpoint_migrations"))
    row = result.fetchone()
    return -1 if row is None or row[0] is None else int(row[0])


def upgrade() -> None:
    """升级 schema：按序执行 langgraph 尚未应用的迁移。"""
    migrations = _migrations()
    bind = op.get_bind()
    current = _current_version(bind)
    if current >= len(migrations) - 1:
        # 已是最新（例如库已由 setup() 建好）；无需重复执行。
        return
    # MIGRATIONS[0] 建立 checkpoint_migrations 表本身；若该表尚不存在，
    # _current_version 已返回 -1，因此循环会从 0 开始正确建表。
    for v in range(current + 1, len(migrations)):
        stmt = migrations[v]
        if not stmt.strip():
            continue
        bind.execute(sa.text(stmt))
        bind.execute(sa.text("INSERT INTO checkpoint_migrations (v) VALUES (:v)").bindparams(v=v))


def downgrade() -> None:
    """降级 schema：删除 langgraph checkpointer 相关表。"""
    for table in (
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoints",
        "checkpoint_migrations",
    ):
        op.execute(sa.text(f"DROP TABLE IF EXISTS `{table}`"))
