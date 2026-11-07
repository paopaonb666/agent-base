package io.github.paopaonb666.agentbase.core.storage;

import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 轻量 schema 迁移机制（#275）：按序应用、已应用跳过、版本记录、
 * 失败 fail-fast 且不记版本（重入安全依赖语句幂等）。
 */
class SchemaMigrationsTest {

    /** 记录全部动作的内存执行通道。 */
    private static final class FakeExecutor implements SchemaMigrations.Executor {
        final List<String> appliedSql = new ArrayList<>();
        final Set<String> versions = new LinkedHashSet<>();
        private String failOnSql = null;

        void failNextTimeOn(String sqlFragment) {
            failOnSql = sqlFragment;
        }

        @Override
        public void ensureVersionsTable(String table) {
            appliedSql.add("ensure:" + table);
        }

        @Override
        public void recordVersion(String table, String id) {
            versions.add(id);
        }

        @Override
        public List<String> appliedVersions(String table) {
            return List.copyOf(versions);
        }

        @Override
        public void execute(String sql) {
            if (failOnSql != null && sql.contains(failOnSql)) {
                throw new RuntimeException("boom: " + sql);
            }
            appliedSql.add(sql);
        }
    }

    private static final List<SchemaMigrations.Migration> MIGRATIONS = List.of(
            new SchemaMigrations.Migration("001_a", "第一版",
                    List.of("CREATE INDEX IF NOT EXISTS idx_a ON t (a)")),
            new SchemaMigrations.Migration("002_b", "第二版",
                    List.of("ALTER TABLE t ADD COLUMN b TEXT", "CREATE INDEX IF NOT EXISTS idx_b ON t (b)")));

    @Test
    void appliesPendingInOrderAndRecordsVersions() {
        FakeExecutor executor = new FakeExecutor();
        SchemaMigrations.apply("test", executor, MIGRATIONS,
                LoggerFactory.getLogger(SchemaMigrationsTest.class));
        assertEquals(List.of("ensure:test_schema_versions",
                "CREATE INDEX IF NOT EXISTS idx_a ON t (a)",
                "ALTER TABLE t ADD COLUMN b TEXT",
                "CREATE INDEX IF NOT EXISTS idx_b ON t (b)"), executor.appliedSql);
        assertEquals(Set.of("001_a", "002_b"), executor.versions);
    }

    @Test
    void alreadyAppliedMigrationsAreSkipped() {
        FakeExecutor executor = new FakeExecutor();
        executor.versions.add("001_a");
        SchemaMigrations.apply("test", executor, MIGRATIONS,
                LoggerFactory.getLogger(SchemaMigrationsTest.class));
        // 只剩 002 的两条语句（ensure 行除外）
        List<String> statements = statementsWithoutEnsure(executor);
        assertEquals(List.of("ALTER TABLE t ADD COLUMN b TEXT",
                "CREATE INDEX IF NOT EXISTS idx_b ON t (b)"), statements);
    }

    @Test
    void rerunIsANoOp() {
        FakeExecutor executor = new FakeExecutor();
        SchemaMigrations.apply("test", executor, MIGRATIONS,
                LoggerFactory.getLogger(SchemaMigrationsTest.class));
        SchemaMigrations.apply("test", executor, MIGRATIONS,
                LoggerFactory.getLogger(SchemaMigrationsTest.class));
        assertEquals(3, statementsWithoutEnsure(executor).size()); // 重跑不执行语句
    }

    private static List<String> statementsWithoutEnsure(FakeExecutor executor) {
        return executor.appliedSql.stream()
                .filter(sql -> !sql.startsWith("ensure:")).toList();
    }

    @Test
    void failingMigrationAbortsAndDoesNotRecordVersion() {
        FakeExecutor executor = new FakeExecutor();
        executor.failNextTimeOn("ADD COLUMN b");
        List<SchemaMigrations.Migration> two = List.of(
                MIGRATIONS.get(0), MIGRATIONS.get(1));
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> SchemaMigrations.apply("test", executor, two,
                        LoggerFactory.getLogger(SchemaMigrationsTest.class)));
        assertTrue(error.getMessage().contains("002_b"));
        assertTrue(error.getMessage().contains("test_schema_versions"));
        // 001 成功已记版本；002 失败未记——修复后重跑只补 002
        assertEquals(Set.of("001_a"), executor.versions);
    }
}
