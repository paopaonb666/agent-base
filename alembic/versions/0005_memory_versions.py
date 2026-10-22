"""记忆内容版本史表（已移交 agent-base-memory，本迁移保留为空操作）。

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-13

说明
----
memory_versions 随其余记忆表一起移交 agent-base-memory（见该包
alembic/versions/0001_memory_tables.py）。本文件只保留 revision 链，
供旧库继续 upgrade head 时解析历史；升级/降级均为空操作。
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """空操作：建表已移交 agent-base-memory 的迁移链。"""


def downgrade() -> None:
    """空操作：删表同样由 agent-base-memory 的迁移链负责。"""
