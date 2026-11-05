package io.github.paopaonb666.agentbase.core.storage;

import io.github.paopaonb666.agentbase.core.config.Settings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 基座自有存储的装配（对应 Python 侧 {@code core/storage/__init__.py} 的
 * {@code build_core_store}）：跟随 checkpointer 后端选型（memory / sqlite /
 * mysql 各一套实现，两张业务表与状态表同库）。
 *
 * <p><b>降级语义</b>：sqlite 建库/建表失败、mysql 建连失败只记 WARNING 并
 * 返回 null——线程列表与计划检查端点按"存储不可用"处理，基座其余功能
 * 不受影响（对应 Python 侧返回 {@code None} 的形态）；未知后端同样记
 * WARNING 后返回 null。
 */
public final class CoreStores {

    private static final Logger LOG = LoggerFactory.getLogger(CoreStores.class);

    private CoreStores() {}

    /** 按 {@code settings.checkpointer.backend} 装配基座存储；失败返回 null（降级）。 */
    public static CoreStore build(Settings settings) {
        String backend = settings.checkpointer.backend;
        try {
            return switch (backend) {
                case "memory" -> new MemoryCoreStore();
                case "sqlite" -> new SqliteCoreStore(settings.checkpointer.sqlitePath);
                case "mysql" -> new MysqlCoreStore(settings.checkpointer);
                default -> {
                    // 与 Python 一致：未知后端不抛错，记 WARNING，存储禁用。
                    LOG.warn("core store: unknown checkpointer backend '{}'; storage disabled", backend);
                    yield null;
                }
            };
        } catch (RuntimeException e) {
            LOG.warn("core store: 后端 {} 装配失败，线程索引/计划快照降级为不可用: {}",
                    backend, e.getMessage());
            return null;
        }
    }
}
