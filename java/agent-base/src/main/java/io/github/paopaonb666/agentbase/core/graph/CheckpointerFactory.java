package io.github.paopaonb666.agentbase.core.graph;

import io.github.paopaonb666.agentbase.core.config.Settings;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 对话状态装配：按 {@code CHECKPOINTER_BACKEND} 构建 checkpointer（对应
 * Python 侧 {@code extensions/memory.py} 的 {@code build_checkpointer}）。
 *
 * <p>后端：{@code memory}（零依赖，仅进程内）/ {@code sqlite}（默认，
 * 单文件持久化）/ {@code mysql}（MySQL 持久化）。未知后端抛
 * {@link Settings.SettingsError}，文案与 Python 保持一致。
 *
 * <p>sqlite / mysql 后端在构造期完成建连与建表（对应 Python 侧构建期的
 * {@code connect + saver.setup()}）：坏配置在启动时快速失败，而不是会话
 * 中途以原始的 SQLException 爆出。MySQL 长驻连接的 keepalive 概念在
 * Java 侧不适用——{@link MysqlCheckpointer} 用每操作短连接，见其类注释。
 */
public final class CheckpointerFactory {

    private CheckpointerFactory() {}

    /** 装配由 {@code settings.checkpointer.backend} 选定的 checkpointer。 */
    public static Checkpointer build(Settings settings) {
        String backend = settings.checkpointer.backend;
        return switch (backend) {
            case "memory" -> new InMemoryCheckpointer();
            case "sqlite" -> buildSqlite(settings.checkpointer.sqlitePath);
            case "mysql" -> buildMysql(settings.checkpointer);
            default -> throw new Settings.SettingsError(
                    "unknown CHECKPOINTER_BACKEND '%s'".formatted(backend));
        };
    }

    private static Checkpointer buildSqlite(String path) {
        // 先预检路径可写性：否则坏路径会以原始的 aiosqlite.OperationalError
        // 在会话中途爆出，而不是启动时一条可读的配置错误（移植 Python
        // _build_sqlite_checkpointer 的同一道预检）。
        Path file = Path.of(path);
        Path parent = file.getParent() == null ? Path.of(".") : file.getParent();
        if (!Files.isDirectory(parent) || !Files.isWritable(parent)) {
            throw new Settings.SettingsError(
                    "CHECKPOINTER_SQLITE_PATH '%s' is not writable: the parent directory "
                            + "does not exist or denies write access".formatted(path));
        }
        return new SqliteCheckpointer(path);
    }

    private static Checkpointer buildMysql(Settings.CheckpointerSettings s) {
        MysqlCheckpointer checkpointer = new MysqlCheckpointer(s);
        checkpointer.init(); // 建连 + 建表：启动期失败快速失败（对应 saver.setup()）
        return checkpointer;
    }
}
