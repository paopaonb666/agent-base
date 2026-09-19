"""记忆表的 DDL 单一来源（S3 收敛）：sqlite 与 MySQL 逐表对应。

MySQL 5.7 兼容约束：TEXT 不设默认值、索引用内联 KEY、显式
ENGINE/CHARSET（与 filestore/toollog 一致）。alembic 迁移链（alembic/
versions/）面向已受管的生产库；sqlite 无 alembic，运行时按本文件自举
（幂等建表 + 旧库补列）。新增表/列时：先改这里，再补一条 alembic 迁移
保持两边一致——不允许出现两份各自漂移的 DDL。
"""

# ─────────────────────────────── DDL ───────────────────────────────
# sqlite 与 MySQL 的建表语句逐表对应；MySQL 5.7 兼容约束：TEXT 不设默认
# 值、索引用内联 KEY、显式 ENGINE/CHARSET（与 filestore/toollog 一致）。

_SQLITE_DDLS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS memories (
      memory_id VARCHAR(40) PRIMARY KEY,
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      kind VARCHAR(16) NOT NULL,
      content TEXT NOT NULL,
      tags_json TEXT NOT NULL DEFAULT '[]',
      embedding BLOB,
      embedding_dim INTEGER,
      salience REAL NOT NULL DEFAULT 0.5,
      status VARCHAR(16) NOT NULL DEFAULT 'active',
      source_thread_id VARCHAR(190) NOT NULL DEFAULT '',
      source_refs_json TEXT NOT NULL DEFAULT '[]',
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      last_accessed_at TIMESTAMP,
      access_count INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_blocks (
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL,
      label VARCHAR(64) NOT NULL,
      content TEXT NOT NULL,
      char_limit INTEGER NOT NULL DEFAULT 2000,
      version INTEGER NOT NULL DEFAULT 0,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (user_id, agent_id, label)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_summaries (
      user_id VARCHAR(64) NOT NULL,
      thread_id VARCHAR(190) NOT NULL,
      summary TEXT NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      covered_message_count INTEGER NOT NULL DEFAULT 0,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (user_id, thread_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS doc_chunks (
      chunk_id VARCHAR(40) PRIMARY KEY,
      file_id VARCHAR(32) NOT NULL,
      thread_id VARCHAR(190) NOT NULL DEFAULT '',
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      ordinal INTEGER NOT NULL DEFAULT 0,
      text TEXT NOT NULL,
      offsets_json TEXT,
      embedding BLOB,
      embedding_dim INTEGER,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_ops (
      op_id VARCHAR(40) PRIMARY KEY,
      op VARCHAR(32) NOT NULL,
      user_id VARCHAR(64) NOT NULL DEFAULT '',
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      thread_id VARCHAR(190) NOT NULL DEFAULT '',
      detail_json TEXT NOT NULL DEFAULT '{}',
      status VARCHAR(16) NOT NULL DEFAULT 'ok',
      error_text TEXT,
      duration_ms INTEGER NOT NULL DEFAULT 0,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_versions (
      version_id VARCHAR(40) PRIMARY KEY,
      memory_id VARCHAR(40) NOT NULL,
      user_id VARCHAR(64) NOT NULL,
      op VARCHAR(16) NOT NULL,
      content TEXT NOT NULL,
      previous_content TEXT,
      status VARCHAR(16) NOT NULL DEFAULT 'active',
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
)

_SQLITE_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories (user_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories (user_id, agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_user ON doc_chunks (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_file ON doc_chunks (file_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_thread ON doc_chunks (thread_id)",
    "CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions (memory_id, created_at)",
)

_MYSQL_DDLS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS memories (
      memory_id VARCHAR(40) PRIMARY KEY,
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      kind VARCHAR(16) NOT NULL,
      content LONGTEXT NOT NULL,
      tags_json LONGTEXT NOT NULL,
      embedding BLOB,
      embedding_dim INT NULL,
      salience DOUBLE NOT NULL DEFAULT 0.5,
      status VARCHAR(16) NOT NULL DEFAULT 'active',
      source_thread_id VARCHAR(190) NOT NULL DEFAULT '',
      source_refs_json LONGTEXT NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      last_accessed_at TIMESTAMP NULL,
      access_count INT NOT NULL DEFAULT 0,
      KEY idx_mem_scope (user_id, status),
      KEY idx_mem_agent (user_id, agent_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_blocks (
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL,
      label VARCHAR(64) NOT NULL,
      content LONGTEXT NOT NULL,
      char_limit INT NOT NULL DEFAULT 2000,
      version INT NOT NULL DEFAULT 0,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (user_id, agent_id, label)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS session_summaries (
      user_id VARCHAR(64) NOT NULL,
      thread_id VARCHAR(190) NOT NULL,
      summary LONGTEXT NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      covered_message_count INT NOT NULL DEFAULT 0,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (user_id, thread_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS doc_chunks (
      chunk_id VARCHAR(40) PRIMARY KEY,
      file_id VARCHAR(32) NOT NULL,
      thread_id VARCHAR(190) NOT NULL DEFAULT '',
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      ordinal INT NOT NULL DEFAULT 0,
      text LONGTEXT NOT NULL,
      offsets_json LONGTEXT,
      embedding BLOB,
      embedding_dim INT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      KEY idx_chunks_user (user_id),
      KEY idx_chunks_file (file_id),
      KEY idx_chunks_thread (thread_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_ops (
      op_id VARCHAR(40) NOT NULL,
      seq BIGINT NOT NULL AUTO_INCREMENT,
      op VARCHAR(32) NOT NULL,
      user_id VARCHAR(64) NOT NULL DEFAULT '',
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      thread_id VARCHAR(190) NOT NULL DEFAULT '',
      detail_json LONGTEXT NOT NULL,
      status VARCHAR(16) NOT NULL DEFAULT 'ok',
      error_text LONGTEXT,
      duration_ms INT NOT NULL DEFAULT 0,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (op_id),
      KEY idx_ops_seq (seq)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_versions (
      version_id VARCHAR(40) NOT NULL,
      seq BIGINT NOT NULL AUTO_INCREMENT,
      memory_id VARCHAR(40) NOT NULL,
      user_id VARCHAR(64) NOT NULL,
      op VARCHAR(16) NOT NULL,
      content LONGTEXT NOT NULL,
      previous_content LONGTEXT,
      status VARCHAR(16) NOT NULL DEFAULT 'active',
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (version_id),
      KEY idx_versions_memory (memory_id, seq)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
)
# 会话索引（S1 线程作用域）：属主/模块/标题/更新时间。invoke 时 upsert，
# 线程端点的列表与属主校验改查该表。alembic 0008 与本节同步。
_THREAD_INDEX_SQLITE = """
CREATE TABLE IF NOT EXISTS thread_index (
  thread_id VARCHAR(190) PRIMARY KEY,
  user_id VARCHAR(64) NOT NULL,
  module VARCHAR(64) NOT NULL DEFAULT '',
  title TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_THREAD_INDEX_MYSQL = """
CREATE TABLE IF NOT EXISTS thread_index (
  thread_id VARCHAR(190) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  module VARCHAR(64) NOT NULL DEFAULT '',
  title LONGTEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (thread_id),
  KEY idx_thread_index_user (user_id, module, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# 追加到各自后端的建表序列（保持单一来源：两个后端都从这里拿）。
_SQLITE_DDLS = (*_SQLITE_DDLS, _THREAD_INDEX_SQLITE)
_MYSQL_DDLS = (*_MYSQL_DDLS, _THREAD_INDEX_MYSQL)

_SQLITE_INDEXES = (
    *_SQLITE_INDEXES,
    "CREATE INDEX IF NOT EXISTS idx_thread_index_user"
    " ON thread_index (user_id, module, updated_at)",
)
