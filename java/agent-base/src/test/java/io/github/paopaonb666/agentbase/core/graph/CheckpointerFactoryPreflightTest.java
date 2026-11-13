package io.github.paopaonb666.agentbase.core.graph;

import io.github.paopaonb666.agentbase.core.config.Settings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A-#305 CheckpointerFactory 预检测试：memory/sqlite(可写与不可写路径)/
 * 未知后端的装配分支——sqlite 的「路径可写性预检」在启动期给出可读配置
 * 错误（坏路径不会以存储原始异常在会话中途爆出）；mysql 无活体服务器，
 * 启动自检的快速失败语义由「坏 host 建连失败」间接锁定。
 */
class CheckpointerFactoryPreflightTest {

    @Test
    void memoryBackendBuildsInMemoryCheckpointer() {
        Settings settings = Settings.load(Map.of("ENV", "development",
                "CHECKPOINTER_BACKEND", "memory"), null);
        assertInstanceOf(InMemoryCheckpointer.class, CheckpointerFactory.build(settings));
    }

    @Test
    void sqliteBackendRejectsUnwritablePathAtStartup() {
        Settings settings = Settings.load(Map.of("ENV", "development",
                "CHECKPOINTER_BACKEND", "sqlite",
                "CHECKPOINTER_SQLITE_PATH", "Z:\\does\\not\\exist\\state.db"), null);
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                () -> CheckpointerFactory.build(settings));
        assertTrue(error.getMessage().contains("is not writable"), error.getMessage());
    }

    @Test
    void sqliteBackendBuildsAtWritablePath(@TempDir Path tempDir) {
        Settings settings = Settings.load(Map.of("ENV", "development",
                "CHECKPOINTER_BACKEND", "sqlite",
                "CHECKPOINTER_SQLITE_PATH", tempDir.resolve("state.db").toString()), null);
        Checkpointer checkpointer = CheckpointerFactory.build(settings);
        assertInstanceOf(SqliteCheckpointer.class, checkpointer);
        checkpointer.close();
    }

    @Test
    void unknownBackendFailsFastWithReadableError() {
        // 合法值集合校验在 Settings 层（#103 前例）；工厂层的 default 分支
        // 防御未知值（反射/手工构造绕过校验时仍可读报错）。
        Settings settings = Settings.load(Map.of("ENV", "development",
                "CHECKPOINTER_BACKEND", "memory"), null);
        Settings.CheckpointerSettings bad = settings.checkpointer;
        String original = bad.backend;
        bad.backend = "oracle";
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                () -> CheckpointerFactory.build(settings));
        assertTrue(error.getMessage().contains("oracle"), error.getMessage());
        bad.backend = original;
    }

    @Test
    void mysqlBackendFailsFastAtStartupWithoutLiveServer() {
        // mysql 启动自检（init 建连+建表）：无活体服务器 → 快速失败（启动
        // 期 IllegalStateException，不等到会话中途）。逻辑审形态——无 MySQL
        // 活体（与 MySqlPools 系同口径）。
        Settings settings = Settings.load(Map.of("ENV", "development",
                "CHECKPOINTER_BACKEND", "mysql",
                "CHECKPOINTER_MYSQL_HOST", "127.0.0.1",
                "CHECKPOINTER_MYSQL_PORT", "1"), null);
        assertThrows(IllegalStateException.class, () -> CheckpointerFactory.build(settings));
    }
}
