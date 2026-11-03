"""message_nodes + tree_pointers 会话消息树（ADR-013）。

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-04

说明
----
会话从线性历史升级为消息树：任意消息节点可分出岔路（从此分岔 / 编辑
重发 / 重新生成都是"在指定节点下 append 子节点"的特例），前端侧边
树面板可视化整棵树并对子树做维护（重命名 / 星标 / 归档 / 软删恢复 /
导出）。``tree_pointers`` 持久化每个会话的活跃叶子指针——活跃分支
指针必须落库（HF chat-ui 的教训：只存内存，刷新即回第一分支）。

DDL 与 ``core/storage/ddl.py`` 单一来源同步维护；sqlite 无 alembic，
运行时按 DDL 自举（新表零迁移）。降级用 alembic 的表操作 API（不拼
原始 SQL）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import LONGTEXT as MYSQL_LONGTEXT

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | Sequence[str] | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_MESSAGE_NODES_PK = sa.PrimaryKeyConstraint("node_id", name="pk_message_nodes")
# 与 core/storage/ddl.py 对齐：TEXT 在 MySQL 侧用 LONGTEXT 承载长消息。
# 注意 LONGTEXT 必须显式导入——`sa.dialects.mysql` 不随 `import sqlalchemy`
# 自动加载（曾经在这里埋过运行时 AttributeError）。
_LONGTEXT = sa.Text().with_variant(MYSQL_LONGTEXT(), "mysql")


def upgrade() -> None:
    """升级 schema：新建消息树表与指针表。"""
    op.create_table(
        "message_nodes",
        sa.Column("node_id", sa.String(32), nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("thread_id", sa.String(190), nullable=False),
        sa.Column("parent_id", sa.String(32), nullable=True),
        sa.Column("msg_id", sa.String(64), nullable=False, server_default=sa.text("''")),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", _LONGTEXT, nullable=False),
        sa.Column("content_json", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("tool_calls_json", _LONGTEXT, nullable=False),
        sa.Column("tool_call_id", sa.String(64), nullable=False, server_default=sa.text("''")),
        sa.Column("name", sa.String(128), nullable=False, server_default=sa.text("''")),
        sa.Column("status", sa.String(16), nullable=False, server_default=sa.text("''")),
        sa.Column("attachments_json", _LONGTEXT, nullable=False),
        sa.Column("label", sa.String(120), nullable=False, server_default=sa.text("''")),
        sa.Column("starred", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("archived", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.TIMESTAMP(), nullable=True),
        _MESSAGE_NODES_PK,
        sa.Index("idx_nodes_thread", "thread_id", "seq"),
        sa.Index("idx_nodes_parent", "thread_id", "parent_id"),
    )
    op.create_table(
        "tree_pointers",
        sa.Column("thread_id", sa.String(190), nullable=False),
        sa.Column("current_node_id", sa.String(32), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("thread_id", name="pk_tree_pointers"),
    )


def downgrade() -> None:
    """降级 schema：删除消息树表与指针表（消息历史退回 checkpointer）。"""
    op.drop_table("tree_pointers")
    op.drop_table("message_nodes")
