"""记忆子系统六张表（agent-base-memory 首个也是当前的 head）。

Revision ID: 0001
Revises:
Create Date: 2026-10-22

说明
----
本迁移接管原先分散在基座 0004/0005/0007 中的建表：那三个 revision 在
agent-base 0.3.0 里保留为空操作（兼容旧库的版本链），建表权移交到这里。

六张表：memories / memory_blocks / session_summaries / doc_chunks /
memory_ops / memory_versions。全部 CREATE TABLE IF NOT EXISTS，因此：

- 全新库：直接建表；
- 旧库（基座已升到 0009、库里已有这六表）：迁移是空操作，旧库无需 stamp；
- doc_chunks 的 offsets_json 列已并入建表语句；极旧的库若表在但列不在，
  upgrade 末尾会用 inspector 检查后补列（MySQL 8 不支持 ADD COLUMN IF
  NOT EXISTS，所以显式判断）。

downgrade 会删除这六张表——记忆数据随之丢失，只在空库/测试环境使用。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
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
      `offsets_json` LONGTEXT NULL,
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
    """
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
    """,
)


def upgrade() -> None:
    """升级 schema：建记忆六表（幂等），并补齐旧库缺失的 offsets_json 列。"""
    for ddl in _DDL:
        op.execute(sa.text(ddl))

    try:
        columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("doc_chunks")}
    except sa.exc.NoInspectionAvailable:
        # offline 模式（--sql）只有 MockConnection，无法反射表结构；需要补列的
        # 极旧库请在线执行 upgrade，此处只是让 SQL 输出不因此中断。
        return
    if "offsets_json" not in columns:
        op.execute(sa.text("ALTER TABLE `doc_chunks` ADD COLUMN `offsets_json` LONGTEXT NULL"))


def downgrade() -> None:
    """降级 schema：按依赖逆序删除六张表（数据不可恢复）。"""
    for table in (
        "memory_versions",
        "memory_ops",
        "doc_chunks",
        "session_summaries",
        "memory_blocks",
        "memories",
    ):
        op.execute(sa.text(f"DROP TABLE IF EXISTS `{table}`"))
