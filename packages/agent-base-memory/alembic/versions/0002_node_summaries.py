"""node_summaries：会话消息树的叶子节点滚动摘要（ADR-013）。

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-04

说明
----
树模式下滚动摘要按**节点** memoize："root → 该节点的路径摘要"。
分支共享祖先节点的摘要（写一次，兄弟分支直接继承），各自分岔后独立
续写——这是 SillyTavern 摘要扩展跨分支错乱（issue #3945）的反面设计。
线程口径的 session_summaries 保留（非树模式 / 旧数据仍走它）。

CREATE TABLE IF NOT EXISTS：全新库与旧库都安全；sqlite 无 alembic，
运行时按 ``store/ddl.py`` 自举。降级删除本表（节点摘要可由线程摘要
重建，代价可接受）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """升级 schema：新建 node_summaries 表。"""
    op.execute(
        sa.text(
            """
            CREATE TABLE IF NOT EXISTS `node_summaries` (
              `user_id` VARCHAR(64) NOT NULL,
              `node_id` VARCHAR(32) NOT NULL,
              `summary` LONGTEXT NOT NULL,
              `agent_id` VARCHAR(64) NOT NULL DEFAULT '',
              `covered_message_count` INT NOT NULL DEFAULT 0,
              `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (`user_id`, `node_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
    )


def downgrade() -> None:
    """降级 schema：删除 node_summaries（可由 session_summaries 重建）。"""
    op.drop_table("node_summaries")
