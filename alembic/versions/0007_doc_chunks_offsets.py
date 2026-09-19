"""doc_chunks 切片偏移列（切片可视化：原文高亮边界的数据源）。

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-13

说明
----
``doc_chunks`` 新增 ``offsets_json``：各段在原文（extracted_text）中的
字符区间列表（如 ``[[0,103],[105,208]]``）。段落打包块是多区间，固定
窗口切片是单区间；旧行为 NULL，前端降级为纯卡片视图。写入方是
``ingest_document``（chunk_text 返回 ChunkSpan）。sqlite 侧无 alembic，
由运行时自迁移（PRAGMA 检查 + ALTER TABLE，见 ``memory/store.py``）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """升级 schema：doc_chunks 补 offsets_json 列。"""
    op.execute(sa.text("ALTER TABLE `doc_chunks` ADD COLUMN `offsets_json` LONGTEXT NULL"))


def downgrade() -> None:
    """降级 schema：删除 offsets_json 列。"""
    op.execute(sa.text("ALTER TABLE `doc_chunks` DROP COLUMN `offsets_json`"))
