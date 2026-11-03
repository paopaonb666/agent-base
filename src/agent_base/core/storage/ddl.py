"""基座表 DDL（sqlite / MySQL 单一来源）。

与 ``src/agent_base/core/storage/*`` 的实现同步；MySQL 侧由运行时惰性
建表，sqlite 侧由 ``create`` 自举。alembic 0008/0009/0010 与本节内容
对应。
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

_MESSAGE_NODES_SQLITE = """
CREATE TABLE IF NOT EXISTS message_nodes (
  node_id VARCHAR(32) PRIMARY KEY,
  seq INTEGER NOT NULL DEFAULT 0,
  thread_id VARCHAR(190) NOT NULL,
  parent_id VARCHAR(32),
  msg_id VARCHAR(64) NOT NULL DEFAULT '',
  role VARCHAR(16) NOT NULL,
  content TEXT NOT NULL DEFAULT '',
  content_json INTEGER NOT NULL DEFAULT 0,
  tool_calls_json TEXT NOT NULL DEFAULT '[]',
  tool_call_id VARCHAR(64) NOT NULL DEFAULT '',
  name VARCHAR(128) NOT NULL DEFAULT '',
  status VARCHAR(16) NOT NULL DEFAULT '',
  attachments_json TEXT NOT NULL DEFAULT '[]',
  label VARCHAR(120) NOT NULL DEFAULT '',
  starred INTEGER NOT NULL DEFAULT 0,
  archived INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted_at TIMESTAMP
)
"""

_MESSAGE_NODES_MYSQL = """
CREATE TABLE IF NOT EXISTS message_nodes (
  node_id VARCHAR(32) NOT NULL,
  seq BIGINT NOT NULL DEFAULT 0,
  thread_id VARCHAR(190) NOT NULL,
  parent_id VARCHAR(32) NULL,
  msg_id VARCHAR(64) NOT NULL DEFAULT '',
  role VARCHAR(16) NOT NULL,
  content LONGTEXT NOT NULL,
  content_json INT NOT NULL DEFAULT 0,
  tool_calls_json LONGTEXT NOT NULL,
  tool_call_id VARCHAR(64) NOT NULL DEFAULT '',
  name VARCHAR(128) NOT NULL DEFAULT '',
  status VARCHAR(16) NOT NULL DEFAULT '',
  attachments_json LONGTEXT NOT NULL,
  label VARCHAR(120) NOT NULL DEFAULT '',
  starred INT NOT NULL DEFAULT 0,
  archived INT NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted_at TIMESTAMP NULL,
  PRIMARY KEY (node_id),
  KEY idx_nodes_thread (thread_id, seq),
  KEY idx_nodes_parent (thread_id, parent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# 会话消息树的活跃叶子指针（ADR-013）：独立小表而不是给 thread_index 加
# 列——指针是树层的独立状态，新表走 CREATE IF NOT EXISTS，旧库零迁移；
# 指针必须持久化（HF chat-ui 的教训：活跃分支只存内存，刷新即回第一支）。
_TREE_POINTERS_SQLITE = """
CREATE TABLE IF NOT EXISTS tree_pointers (
  thread_id VARCHAR(190) PRIMARY KEY,
  current_node_id VARCHAR(32) NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_TREE_POINTERS_MYSQL = """
CREATE TABLE IF NOT EXISTS tree_pointers (
  thread_id VARCHAR(190) NOT NULL,
  current_node_id VARCHAR(32) NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (thread_id)
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

_SQLITE_DDLS = (
    _THREAD_INDEX_SQLITE,
    _MESSAGE_NODES_SQLITE,
    _TREE_POINTERS_SQLITE,
    _PLAN_SNAPSHOTS_SQLITE,
)
_MYSQL_DDLS = (
    _THREAD_INDEX_MYSQL,
    _MESSAGE_NODES_MYSQL,
    _TREE_POINTERS_MYSQL,
    _PLAN_SNAPSHOTS_MYSQL,
)

_SQLITE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_thread_index_user"
    " ON thread_index (user_id, module, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_thread ON message_nodes (thread_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_parent ON message_nodes (thread_id, parent_id)",
)
