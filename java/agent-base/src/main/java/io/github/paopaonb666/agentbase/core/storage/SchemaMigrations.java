package io.github.paopaonb666.agentbase.core.storage;

import org.slf4j.Logger;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 轻量 schema 迁移机制（#275）：{@code <component>_schema_versions} 表 +
 * 声明式迁移列表（每条 = 幂等 SQL 语句集），构造期按序执行、已应用的
 * 跳过。对应 Python 侧 alembic 双链的 Java 对应物——列新增/索引变更/
 * 数据回填三类演进不再靠「构造期 PRAGMA 补列」散点硬编码。
 *
 * <p>失败语义：<b>fail-fast</b>——迁移执行失败抛
 * {@link IllegalStateException}（含迁移 id 与修复指向），拒绝带着缺失
 * 列/索引静默启动。语句本身要求幂等（CREATE INDEX IF NOT EXISTS /
 * 带存在性检查的 ALTER），保证「执行了但版本记录失败」的重入安全。
 */
public final class SchemaMigrations {

    /** 一条声明式迁移：稳定 id（应用后不再重跑）+ 幂等语句集。 */
    public record Migration(String id, String description, List<String> statements) {
        public Migration {
            statements = List.copyOf(statements);
        }
    }

    /** 目标库的执行通道（sqlite/mysql 方言差异由调用方适配）。 */
    public interface Executor {

        /** 幂等创建版本表。 */
        void ensureVersionsTable(String table) throws Exception;

        /** 记录一条已应用的迁移 id（与业务语句同一事务语义由方言自决）。 */
        void recordVersion(String table, String id) throws Exception;

        /** 已应用的迁移 id 集合（表为空时返回空集）。 */
        List<String> appliedVersions(String table) throws Exception;

        /** 执行一条迁移语句。 */
        void execute(String sql) throws Exception;
    }

    private SchemaMigrations() {
    }

    /**
     * 按序应用未执行的迁移；版本表名 = {@code <component>_schema_versions}。
     * 全部语句成功后记版本——执行失败即抛（fail-fast，见类注释）。
     */
    public static void apply(String component, Executor executor,
                             List<Migration> migrations, Logger log) {
        String table = component + "_schema_versions";
        try {
            executor.ensureVersionsTable(table);
            Set<String> applied = new HashSet<>(executor.appliedVersions(table));
            for (Migration migration : migrations) {
                if (applied.contains(migration.id())) {
                    continue;
                }
                try {
                    for (String sql : migration.statements()) {
                        executor.execute(sql);
                    }
                    executor.recordVersion(table, migration.id());
                    log.info("schema migration applied: {} ({})", migration.id(),
                            migration.description());
                } catch (Exception migrationError) {
                    // fail-fast：带迁移 id 与修复指向，拒绝带缺失列/索引静默启动
                    throw new IllegalStateException(
                            "schema migration '" + migration.id() + "'（" + migration.description()
                                    + "）失败：" + migrationError.getMessage()
                                    + "；请检查 " + table + " 表与库状态后重试", migrationError);
                }
            }
        } catch (Exception e) {
            if (e instanceof IllegalStateException stateError) {
                throw stateError;
            }
            throw new IllegalStateException(
                    "schema migration 失败（component=" + component + "）：" + e.getMessage()
                            + "；请检查 " + table + " 表与库状态后重试", e);
        }
    }
}
