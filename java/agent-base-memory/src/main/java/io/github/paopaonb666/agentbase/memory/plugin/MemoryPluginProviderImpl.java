package io.github.paopaonb666.agentbase.memory.plugin;

import java.util.List;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.core.memory.MemoryPlugins;
import io.github.paopaonb666.agentbase.memory.api.MemoryApiController;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import io.github.paopaonb666.agentbase.memory.tools.MemoryTools;

/**
 * 记忆发行包的 SPI 实现（对应 Python 侧 {@code agent_base_memory/plugin.py}
 * 的 entry point {@code plugin:build} / {@code api:routers}）。
 *
 * <p>经 {@code META-INF/services} 注册：MEMORY_ENABLED=true 时基座经
 * ServiceLoader 发现本类（启用即承诺——store 不可用直接抛错，不静默
 * 降级）；false 时基座零发现零加载。
 */
public final class MemoryPluginProviderImpl implements MemoryPlugins.MemoryPluginProvider {

    @Override
    public MemoryPlugins.MemoryPlugin build(Settings settings, ModelRuntime llm,
                                            AgentModule.ChatInvokerPort fastLlm) {
        MemoryService service = MemoryService.build(settings, llm, fastLlm);
        if (service == null) {
            throw new IllegalStateException(
                    "memory: 记忆服务装配失败（存储后端不可用）——启用即承诺，拒绝静默降级");
        }
        return new MemoryPlugins.MemoryPlugin(service,
                MemoryTools.build(service, settings.memory.sensitiveContentFilter));
    }

    @Override
    public List<Class<?>> routeControllers() {
        return List.of(MemoryApiController.class);
    }
}
