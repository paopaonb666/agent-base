"""记忆内容版本史表（锐评 #7：软删除审计 + 内容差分）。

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-13

说明
----
``memory_versions`` 为每条记忆保留内容快照：create（首版）/ update
（content 为新值、previous_content 为旧值）/ delete（墓碑，content 为
空串）/ restore。seq 自增列用于同秒稳定排序（sqlite 侧用 rowid）。
写入方是 ``MemoryService`` 的失败安全路径——版本史缺失不影响主写入。
建表 DDL 与运行时自举（``memory/store.py``）一致，幂等可重复执行。
存储后端跟随 ``CHECKPOINTER_BACKEND``：sqlite 在运行时自建同构表，
本迁移只服务 MySQL。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DDL = """
CREATE TABLE IF NOT EXISTS `memory_versions` (
  `version_id` VARCHAR(40) NOT NULL,
  `seq` BIGINT NOT NULL AUTO_INCREMENT,
  `memory_id` VARCHAR(40) NOT NULL,
  `user_id` VARCHAR(64) NOT NULL,
  `op` VARCHAR(16) NOT NULL,
  `content` LONGTEXT NOT NULL,
  `previous_content` LONGTEXT,
  `status` VARCHAR(16) NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`version_id`),
  KEY `idx_versions_memory` (`memory_id`, `seq`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def upgrade() -> None:
    """升级 schema：建记忆版本史表（幂等）。"""
    op.execute(sa.text(_DDL))


def downgrade() -> None:
    """降级 schema：删除记忆版本史表。"""
    op.execute(sa.text("DROP TABLE IF EXISTS `memory_versions`"))
