package io.github.paopaonb666.agentbase.core.memory;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;

import java.util.List;
import java.util.ServiceLoader;

/**
 * 记忆子系统的装配点：ServiceLoader SPI 发现（对应 Python 侧 0.3.0 拆分
 * 的 entry point 机制 {@code agent_base.memory}）。
 *
 * <p>基座源码<b>不</b> import 记忆包。记忆实现通过发行包在
 * {@code META-INF/services/io.github.paopaonb666.agentbase.core.memory.MemoryPluginProvider}
 * 注册自己（对应 Python 的 entry point group）。
 *
 * <p>语义：MEMORY_ENABLED=true 即"启用即承诺"——找不到实现就报错并提示
 * 安装，而不是静默降级；false 时零发现零加载，与没有记忆包时等价。
 */
public final class MemoryPlugins {

    private MemoryPlugins() {}

    /** 记忆子系统已启用，但实现包缺失或装配失败。 */
    public static final class MemoryPluginError extends RuntimeException {
        public MemoryPluginError(String message) { super(message); }
    }

    /** 记忆实现返回给组合根的装配结果。 */
    public record MemoryPlugin(AgentModule.MemoryServicePort service, List<AgentTool> tools) {

        public MemoryPlugin {
            tools = tools == null ? List.of() : List.copyOf(tools);
        }
    }

    /**
     * SPI：记忆发行包实现本接口并经 {@code META-INF/services} 注册
     * （对应 Python entry point {@code service = "agent_base_memory.plugin:build"}）。
     */
    public interface MemoryPluginProvider {

        /** 装配记忆服务；未启用由组合根拦截，实现方总是返回非 null。 */
        MemoryPlugin build(Settings settings, ModelRuntime llm, AgentModule.ChatInvokerPort fastLlm);

        /**
         * 记忆管理 API 的控制器类清单（对应 Python entry point
         * {@code routes}）：由服务入口经 Spring 注册。返回带
         * {@code @RestController} 注解的类（构造器注入
         * {@link AgentModule.MemoryServicePort} 与运行时依赖由入口装配）。
         */
        List<Class<?>> routeControllers();
    }

    private static MemoryPluginError missing() {
        return new MemoryPluginError(
                "MEMORY_ENABLED=true but no memory implementation is installed. "
                        + "Install the optional package (agent-base-memory jar on the classpath) "
                        + "or set MEMORY_ENABLED=false.");
    }

    /** 装配记忆服务；未启用返回 null，启用但实现缺失抛 {@link MemoryPluginError}。 */
    public static MemoryPlugin build(Settings settings, ModelRuntime llm, AgentModule.ChatInvokerPort fastLlm) {
        if (!settings.memory.enabled) {
            return null;
        }
        MemoryPluginProvider provider = select();
        if (provider == null) {
            throw missing();
        }
        MemoryPlugin plugin = provider.build(settings, llm, fastLlm);
        if (plugin == null) {
            throw new MemoryPluginError("memory plugin provider returned no MemoryPlugin");
        }
        return plugin;
    }

    /** 记忆管理 API 的控制器（未启用时应由调用方跳过；缺失抛错）。 */
    public static List<Class<?>> loadRouteControllers() {
        MemoryPluginProvider provider = select();
        if (provider == null) {
            throw missing();
        }
        return provider.routeControllers();
    }

    private static MemoryPluginProvider select() {
        ServiceLoader<MemoryPluginProvider> loader =
                ServiceLoader.load(MemoryPluginProvider.class, MemoryPlugins.class.getClassLoader());
        MemoryPluginProvider first = null;
        int count = 0;
        for (MemoryPluginProvider provider : loader) {
            count++;
            if (first == null) {
                first = provider;
            }
        }
        if (count > 1) {
            org.slf4j.LoggerFactory.getLogger(MemoryPlugins.class)
                    .warn("memory plugin: {} providers registered; using the first", count);
        }
        return first;
    }
}
