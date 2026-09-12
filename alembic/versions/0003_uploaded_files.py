"""上传附件表（附件对话功能）。

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-12

说明
----
``uploaded_files`` 存每一次上传的附件：原始字节（LONGBLOB，可重解析/
可扩展下载）与解析提取文本（注入对话上下文），thread_id 在附件被消息
引用时回填（上传先于线程生成）。建表 DDL 与运行时自举
（``extensions/filestore.py`` 首写 ``CREATE TABLE IF NOT EXISTS``）一致，
幂等可重复执行。存储后端跟随 ``CHECKPOINTER_BACKEND``：sqlite 在运行时
自建同构表，本迁移只服务 MySQL。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DDL = """
CREATE TABLE IF NOT EXISTS `uploaded_files` (
  `file_id` VARCHAR(32) PRIMARY KEY,
  `thread_id` VARCHAR(190) NOT NULL DEFAULT '',
  `module` VARCHAR(64) NOT NULL DEFAULT '',
  `filename` VARCHAR(255) NOT NULL,
  `format` VARCHAR(16) NOT NULL,
  `pages` INT NULL,
  `paragraphs` INT NULL,
  `truncated` SMALLINT NOT NULL DEFAULT 0,
  `text_len` INT NOT NULL DEFAULT 0,
  `warning` VARCHAR(500) NOT NULL DEFAULT '',
  `extracted_text` LONGTEXT,
  `content` LONGBLOB,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def upgrade() -> None:
    """升级 schema：建上传附件表（幂等）。"""
    op.execute(sa.text(_DDL))


def downgrade() -> None:
    """降级 schema：删除上传附件表。"""
    op.execute(sa.text("DROP TABLE IF EXISTS `uploaded_files`"))
