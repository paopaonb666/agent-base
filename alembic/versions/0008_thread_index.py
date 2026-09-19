"""thread_index 会话索引表（S1 线程作用域的数据面）。

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-20

说明
----
会话元数据自管：invoke 时 upsert（属主/模块/标题/更新时间），线程端点
的列表与属主校验改查该表——列表不再全表扫描 checkpointer，任意用户
也无法再枚举/读取/删除他人的会话。历史线程（无索引行）视为不可访问，
与审查报告 S1 的迁移策略一致。DDL 与 ``memory/store/ddl.py`` 同步维护。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """升级 schema：新建 thread_index 表与属主索引。"""
    op.execute(
        sa.text(
            """
            CREATE TABLE IF NOT EXISTS `thread_index` (
              `thread_id` VARCHAR(190) NOT NULL,
              `user_id` VARCHAR(64) NOT NULL,
              `module` VARCHAR(64) NOT NULL DEFAULT '',
              `title` LONGTEXT NOT NULL,
              `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (`thread_id`),
              KEY `idx_thread_index_user` (`user_id`, `module`, `updated_at`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
    )


def downgrade() -> None:
    """降级 schema：删除 thread_index 表。"""
    op.execute(sa.text("DROP TABLE IF EXISTS `thread_index`"))
