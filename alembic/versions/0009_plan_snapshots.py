"""plan_snapshots 计划快照表（M10 跨轮延续的数据面）。

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-28

说明
----
planner 图在每个状态变更点（拆解/推进/重规划/完成）upsert 快照，plan
检查端点读表而非图状态——chat 轮穿插（chat 图的 checkpoint 只含
messages 通道）不再重置计划。tasks_json 是序列化后的任务清单。
DDL 与 ``memory/store/ddl.py`` 同步维护（sqlite 侧由运行时自举）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """升级 schema：新建 plan_snapshots 表。"""
    op.execute(
        sa.text(
            """
            CREATE TABLE IF NOT EXISTS `plan_snapshots` (
              `thread_id` VARCHAR(190) NOT NULL,
              `module` VARCHAR(64) NOT NULL DEFAULT '',
              `tasks_json` LONGTEXT NOT NULL,
              `cursor` BIGINT NOT NULL DEFAULT 0,
              `replans` BIGINT NOT NULL DEFAULT 0,
              `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (`thread_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
    )


def downgrade() -> None:
    """降级 schema：删除 plan_snapshots 表。"""
    op.execute(sa.text("DROP TABLE IF EXISTS `plan_snapshots`"))
