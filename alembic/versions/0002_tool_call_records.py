"""工具调用审计表（M5 可观测）。

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-12

说明
----
``tool_call_records`` 存每一次工具调用的审计记录：参数（JSON）、结果
文本、结局（ok / timeout / error）、错误信息与耗时——成功与失败 alike，
由池的 ``_TimeoutTool`` 收口写入（见 ``extensions/toollog.py``）。

幂等性：``CREATE TABLE IF NOT EXISTS``——运行时（``MySqlToolCallRecorder``
的首写自举）与迁移通道可以各自执行、互不冲突。存储后端跟随
``CHECKPOINTER_BACKEND``：sqlite 在运行时自建同构表，本迁移只服务 MySQL。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DDL = """
CREATE TABLE IF NOT EXISTS `tool_call_records` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `call_id` VARCHAR(32) NOT NULL DEFAULT '',
  `thread_id` VARCHAR(190) NOT NULL DEFAULT '',
  `module` VARCHAR(64) NOT NULL DEFAULT '',
  `request_id` VARCHAR(64) NOT NULL DEFAULT '',
  `tool` VARCHAR(128) NOT NULL,
  `args_json` TEXT,
  `result_text` TEXT,
  `status` VARCHAR(16) NOT NULL,
  `error_text` TEXT,
  `duration_ms` INT,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_tcr_thread` (`thread_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def upgrade() -> None:
    """升级 schema：建工具调用审计表（幂等）。"""
    op.execute(sa.text(_DDL))


def downgrade() -> None:
    """降级 schema：删除审计表。"""
    op.execute(sa.text("DROP TABLE IF EXISTS `tool_call_records`"))
