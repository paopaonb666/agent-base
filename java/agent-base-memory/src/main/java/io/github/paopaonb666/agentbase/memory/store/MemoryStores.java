package io.github.paopaonb666.agentbase.memory.store;

import io.github.paopaonb666.agentbase.core.config.Settings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 记忆存储的装配（对应 Python 侧 {@code agent_base_memory/store/__init__.py}
 * 的 {@code build_memory_store}）：跟随 checkpointer 后端选型
 * （memory / sqlite / mysql 各一套实现，记忆六表与对话状态同库）。
 *
 * <p>与 Python 一致：未知后端不抛错，记 WARNING 后返回 {@code null}
 * （memory store 禁用）；sqlite/mysql 建库失败原样抛出（由调用方决定
 * 是否降级），不做静默吞异常。
 */
public final class MemoryStores {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryStores.class);

    private MemoryStores() {}

    /** 按 {@code settings.checkpointer.backend} 装配记忆存储；未知后端返回 null。 */
    public static MemoryStore build(Settings settings) {
        String backend = settings.checkpointer.backend;
        switch (backend) {
            case "memory":
                return new InMemoryMemoryStore();
            case "sqlite":
                return new SqliteMemoryStore(settings.checkpointer.sqlitePath);
            case "mysql":
                return new MysqlMemoryStore(settings);
            default:
                LOG.warn("memory: unknown checkpointer backend '{}'; memory store disabled",
                        backend);
                return null;
        }
    }
}
