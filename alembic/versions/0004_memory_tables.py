"""记忆系统五张表（已移交 agent-base-memory，本迁移保留为空操作）。

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-12

说明
----
P1 拆分（agent-base 0.3.0）后，记忆子系统的 schema 归 agent-base-memory
所有：建表 SQL 见该包的 alembic/versions/0001_memory_tables.py，版本记录
写在独立的 alembic_version_memory 表里。

本文件保留 revision 标识，只把升级/降级改为空操作，原因：

- 旧库（升级前建过 alembic_version = 0004）继续 upgrade head 时，Alembic
  需要能沿 0003 → 0004 → 0005 → … 找到链路，删文件会报
  "Can't locate revision identified by '0004'"；
- 新库由记忆包自己的迁移建表，基座不再重复执行 DDL，避免两处 DDL 漂移。

因此这里的空操作是兼容垫片，不是「表格被删除」：已存在的表不会被本迁移
触碰，旧库也不需要 alembic stamp。
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """空操作：建表已移交 agent-base-memory 的迁移链。"""


def downgrade() -> None:
    """空操作：删表同样由 agent-base-memory 的迁移链负责。"""
