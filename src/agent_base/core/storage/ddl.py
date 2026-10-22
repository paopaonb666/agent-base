"""基座表 DDL（sqlite / MySQL 单一来源）。

与 ``src/agent_base/core/storage/*`` 的实现同步；MySQL 侧由运行时惰性
建表，sqlite 侧由 ``create`` 自举。alembic 0008/0009 与本节内容对应。
"""

from __future__ import annotations

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

_PLAN_SNAPSHOTS_SQLITE = """
CREATE TABLE IF NOT EXISTS plan_snapshots (
  thread_id VARCHAR(190) PRIMARY KEY,
  module VARCHAR(64) NOT NULL DEFAULT '',
  tasks_json TEXT NOT NULL DEFAULT '[]',
  cursor INTEGER NOT NULL DEFAULT 0,
  replans INTEGER NOT NULL DEFAULT 0,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_PLAN_SNAPSHOTS_MYSQL = """
CREATE TABLE IF NOT EXISTS plan_snapshots (
  thread_id VARCHAR(190) NOT NULL,
  module VARCHAR(64) NOT NULL DEFAULT '',
  tasks_json LONGTEXT NOT NULL,
  cursor BIGINT NOT NULL DEFAULT 0,
  replans BIGINT NOT NULL DEFAULT 0,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (thread_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_SQLITE_DDLS = (_THREAD_INDEX_SQLITE, _PLAN_SNAPSHOTS_SQLITE)
_MYSQL_DDLS = (_THREAD_INDEX_MYSQL, _PLAN_SNAPSHOTS_MYSQL)

_SQLITE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_thread_index_user"
    " ON thread_index (user_id, module, updated_at)",
)
