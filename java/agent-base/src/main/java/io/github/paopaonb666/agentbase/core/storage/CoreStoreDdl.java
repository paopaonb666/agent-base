package io.github.paopaonb666.agentbase.core.storage;

import java.util.List;

/**
 * 基座表 DDL（sqlite / MySQL 单一来源，对应 Python 侧 {@code ddl.py}）。
 *
 * <p>MySQL 侧逐条复制 alembic 0008（thread_index）/ 0009（plan_snapshots）；
 * sqlite 侧由 {@link SqliteCoreStore} 构造期自举。DDL 字面量与
 * {@code alembic/versions/0008_thread_index.py}、{@code 0009_plan_snapshots.py}
 * 保持逐字符一致（含反引号与 ENGINE/CHARSET 尾缀），便于对照评审。
 */
final class CoreStoreDdl {

    private CoreStoreDdl() {}

    /** thread_index 的 sqlite DDL（TIMESTAMP 仅为亲和类型，值由写入方提供）。 */
    static final String SQLITE_THREAD_INDEX = """
            CREATE TABLE IF NOT EXISTS thread_index (
              thread_id VARCHAR(190) PRIMARY KEY,
              user_id VARCHAR(64) NOT NULL,
              module VARCHAR(64) NOT NULL DEFAULT '',
              title TEXT NOT NULL DEFAULT '',
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""";

    /** plan_snapshots 的 sqlite DDL。 */
    static final String SQLITE_PLAN_SNAPSHOTS = """
            CREATE TABLE IF NOT EXISTS plan_snapshots (
              thread_id VARCHAR(190) PRIMARY KEY,
              module VARCHAR(64) NOT NULL DEFAULT '',
              tasks_json TEXT NOT NULL DEFAULT '[]',
              cursor INTEGER NOT NULL DEFAULT 0,
              replans INTEGER NOT NULL DEFAULT 0,
              updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""";

    /** sqlite 建表自举顺序。 */
    static final List<String> SQLITE_DDLS = List.of(SQLITE_THREAD_INDEX, SQLITE_PLAN_SNAPSHOTS);

    /** 属主索引：线程列表按 (user_id, module, updated_at) 走索引。 */
    static final List<String> SQLITE_INDEXES = List.of(
            "CREATE INDEX IF NOT EXISTS idx_thread_index_user"
                    + " ON thread_index (user_id, module, updated_at)");

    /** thread_index 的 MySQL DDL（逐条复制 alembic 0008）。 */
    static final String MYSQL_THREAD_INDEX = """
            CREATE TABLE IF NOT EXISTS `thread_index` (
              `thread_id` VARCHAR(190) NOT NULL,
              `user_id` VARCHAR(64) NOT NULL,
              `module` VARCHAR(64) NOT NULL DEFAULT '',
              `title` LONGTEXT NOT NULL,
              `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (`thread_id`),
              KEY `idx_thread_index_user` (`user_id`, `module`, `updated_at`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""";

    /** plan_snapshots 的 MySQL DDL（逐条复制 alembic 0009）。 */
    static final String MYSQL_PLAN_SNAPSHOTS = """
            CREATE TABLE IF NOT EXISTS `plan_snapshots` (
              `thread_id` VARCHAR(190) NOT NULL,
              `module` VARCHAR(64) NOT NULL DEFAULT '',
              `tasks_json` LONGTEXT NOT NULL,
              `cursor` BIGINT NOT NULL DEFAULT 0,
              `replans` BIGINT NOT NULL DEFAULT 0,
              `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (`thread_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""";

    /** MySQL 建表顺序（运行时惰性执行，{@code CREATE TABLE IF NOT EXISTS} 幂等）。 */
    static final List<String> MYSQL_DDLS = List.of(MYSQL_THREAD_INDEX, MYSQL_PLAN_SNAPSHOTS);
}
