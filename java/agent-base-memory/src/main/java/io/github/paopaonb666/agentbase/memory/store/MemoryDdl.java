package io.github.paopaonb666.agentbase.memory.store;

import java.util.List;

/**
 * 记忆表的 DDL 单一来源（S3 收敛）：sqlite 与 MySQL 逐表对应。
 *
 * <p>MySQL 5.7 兼容约束：TEXT 不设默认值、索引用内联 KEY、显式
 * ENGINE/CHARSET（与 filestore/toollog 一致）。alembic 迁移链
 * （packages/agent-base-memory/alembic/versions/）面向已受管的生产库；
 * sqlite 无 alembic，运行时按本文件自举（幂等建表 + 旧库补列，见
 * {@link SqliteMemoryStore}）。新增表/列时：先改这里，再补一条 alembic
 * 迁移保持两边一致——不允许出现两份各自漂移的 DDL。
 *
 * <p>六张表逐条对应 Python 侧 {@code agent_base_memory/store/ddl.py} 的
 * {@code _SQLITE_DDLS} / {@code _SQLITE_INDEXES} / {@code _MYSQL_DDLS}
 * （仅排版缩进因 Java 文本块而异，SQL 语句逐字一致）。
 */
public final class MemoryDdl {

    private MemoryDdl() {}

    /** sqlite 建表语句（与对话状态同一文件；六张表：memories / memory_blocks / session_summaries / doc_chunks / memory_ops / memory_versions）。 */
    public static final List<String> SQLITE_DDLS = List.of(
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
            """);

    /**
     * sqlite 索引（与 Python _SQLITE_INDEXES 逐条一致，外加 #159 的检索
     * 索引）：memories 的检索主路径按 {@code WHERE user_id=? … ORDER BY
     * updated_at DESC} 查询，没有 (user_id, updated_at) 索引万级记忆即
     * 全表排序劣化。IF NOT EXISTS 幂等——存量库下次打开自动补齐。
     */
    public static final List<String> SQLITE_INDEXES = List.of(
            "CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories (user_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories (user_id, agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_mem_user_updated ON memories (user_id, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_user ON doc_chunks (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_file ON doc_chunks (file_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_thread ON doc_chunks (thread_id)",
            "CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions (memory_id, created_at)");

    /**
     * sqlite 声明式迁移（#275）：经 SchemaMigrations 按序应用并记录到
     * {@code memory_schema_versions}。001 即 #159 的检索索引（语句幂等，
     * 存量库无论走索引循环还是迁移路径都收敛到同一结果）；后续列新增/
     * 索引变更/数据回填类演进一律追加到本清单，不再散点硬编码。
     */
    public static final List<io.github.paopaonb666.agentbase.core.storage.SchemaMigrations.Migration>
            SQLITE_MIGRATIONS = List.of(
            new io.github.paopaonb666.agentbase.core.storage.SchemaMigrations.Migration(
                    "001_idx_mem_user_updated",
                    "memories 检索主路径索引 (#159)",
                    List.of("CREATE INDEX IF NOT EXISTS idx_mem_user_updated "
                            + "ON memories (user_id, updated_at)")));

    /**
     * MySQL 侧为存量表补 #159 索引的守卫 ALTER（内联 KEY 只对新建表
     * 生效——CREATE TABLE IF NOT EXISTS 不会给已存在的表补索引；MySQL
     * 8.0 又没有 CREATE INDEX IF NOT EXISTS）。列定义与 memories 的
     * 内联 KEY 一致。
     */
    public static final String MYSQL_INDEX_MEM_USER_UPDATED =
            "ALTER TABLE memories ADD KEY idx_mem_user_updated (user_id, updated_at)";

    /**
     * MySQL 建表语句（与 Python _MYSQL_DDLS 逐条一致 + #159 检索索引）：
     * LONGTEXT、内联 KEY、显式 ENGINE/CHARSET；memory_ops /
     * memory_versions 的 seq 为 AUTO_INCREMENT（同秒并列时用 seq 而非
     * rowid 做稳定次序）。
     */
    public static final List<String> MYSQL_DDLS = List.of(
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
              KEY idx_mem_agent (user_id, agent_id),
              KEY idx_mem_user_updated (user_id, updated_at)
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
              -- MySQL 要求 AUTO_INCREMENT 列自身带索引；复合索引里 seq 不是首列不算。
              KEY idx_versions_seq (seq),
              KEY idx_versions_memory (memory_id, seq)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """);
}
