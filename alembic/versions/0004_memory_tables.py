"""记忆系统五张表（M6a：长期记忆/记忆块/会话摘要/文档分块/操作审计）。

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-12

说明
----
M6 记忆系统的存储层：

- ``memories``          —— 跨会话长期记忆（含 embedding BLOB、显著度、
                          时间有效性与访问记账）
- ``memory_blocks``     —— 常驻上下文的记忆块（persona/human/自定义）
- ``session_summaries`` —— 每线程滚动摘要（主键 (user_id, thread_id)：
                          模块名已含于完整 thread id，agent_id 仅作展示）
- ``doc_chunks``        —— 文档知识库分块（上传文件的文本切片 + embedding）
- ``memory_ops``        —— 记忆系统全量操作审计；seq 自增列用于同秒
                          稳定排序（sqlite 侧用 rowid，见
                          ``memory/store.py``）

建表 DDL 与运行时自举（``memory/store.py`` 首写 ``CREATE TABLE IF NOT
EXISTS``）一致，幂等可重复执行。存储后端跟随 ``CHECKPOINTER_BACKEND``：
sqlite 在运行时自建同构表，本迁移只服务 MySQL。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS `memories` (
      `memory_id` VARCHAR(40) PRIMARY KEY,
      `user_id` VARCHAR(64) NOT NULL,
      `agent_id` VARCHAR(64) NOT NULL DEFAULT '',
      `kind` VARCHAR(16) NOT NULL,
      `content` LONGTEXT NOT NULL,
      `tags_json` LONGTEXT NOT NULL,
      `embedding` BLOB,
      `embedding_dim` INT NULL,
      `salience` DOUBLE NOT NULL DEFAULT 0.5,
      `status` VARCHAR(16) NOT NULL DEFAULT 'active',
      `source_thread_id` VARCHAR(190) NOT NULL DEFAULT '',
      `source_refs_json` LONGTEXT NOT NULL,
      `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      `last_accessed_at` TIMESTAMP NULL,
      `access_count` INT NOT NULL DEFAULT 0,
      KEY `idx_mem_scope` (`user_id`, `status`),
      KEY `idx_mem_agent` (`user_id`, `agent_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS `memory_blocks` (
      `user_id` VARCHAR(64) NOT NULL,
      `agent_id` VARCHAR(64) NOT NULL,
      `label` VARCHAR(64) NOT NULL,
      `content` LONGTEXT NOT NULL,
      `char_limit` INT NOT NULL DEFAULT 2000,
      `version` INT NOT NULL DEFAULT 0,
      `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (`user_id`, `agent_id`, `label`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS `session_summaries` (
      `user_id` VARCHAR(64) NOT NULL,
      `thread_id` VARCHAR(190) NOT NULL,
      `summary` LONGTEXT NOT NULL,
      `agent_id` VARCHAR(64) NOT NULL DEFAULT '',
      `covered_message_count` INT NOT NULL DEFAULT 0,
      `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (`user_id`, `thread_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS `doc_chunks` (
      `chunk_id` VARCHAR(40) PRIMARY KEY,
      `file_id` VARCHAR(32) NOT NULL,
      `thread_id` VARCHAR(190) NOT NULL DEFAULT '',
      `user_id` VARCHAR(64) NOT NULL,
      `agent_id` VARCHAR(64) NOT NULL DEFAULT '',
      `ordinal` INT NOT NULL DEFAULT 0,
      `text` LONGTEXT NOT NULL,
      `embedding` BLOB,
      `embedding_dim` INT NULL,
      `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      KEY `idx_chunks_user` (`user_id`),
      KEY `idx_chunks_file` (`file_id`),
      KEY `idx_chunks_thread` (`thread_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS `memory_ops` (
      `op_id` VARCHAR(40) NOT NULL,
      `seq` BIGINT NOT NULL AUTO_INCREMENT,
      `op` VARCHAR(32) NOT NULL,
      `user_id` VARCHAR(64) NOT NULL DEFAULT '',
      `agent_id` VARCHAR(64) NOT NULL DEFAULT '',
      `thread_id` VARCHAR(190) NOT NULL DEFAULT '',
      `detail_json` LONGTEXT NOT NULL,
      `status` VARCHAR(16) NOT NULL DEFAULT 'ok',
      `error_text` LONGTEXT,
      `duration_ms` INT NOT NULL DEFAULT 0,
      `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (`op_id`),
      KEY `idx_ops_seq` (`seq`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
)


def upgrade() -> None:
    """升级 schema：建记忆系统五张表（幂等）。"""
    for ddl in _DDL:
        op.execute(sa.text(ddl))


def downgrade() -> None:
    """降级 schema：按依赖逆序删除记忆系统五张表。"""
    for table in ("memory_ops", "doc_chunks", "session_summaries", "memory_blocks", "memories"):
        op.execute(sa.text(f"DROP TABLE IF EXISTS `{table}`"))
