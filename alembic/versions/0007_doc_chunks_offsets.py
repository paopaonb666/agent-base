"""doc_chunks 切片偏移列（已移交 agent-base-memory，本迁移保留为空操作）。

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-13

说明
----
doc_chunks 属于记忆子系统，offsets_json 列直接建在 agent-base-memory 的
建表语句里（该包 alembic/versions/0001_memory_tables.py 中 doc_chunks 已含
该列），不再需要单独的 ALTER 迁移。本文件保留 revision 链，升级/降级均为
空操作。
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """空操作：列已并入 agent-base-memory 的建表语句。"""


def downgrade() -> None:
    """空操作：删列同样由 agent-base-memory 的迁移链负责。"""
