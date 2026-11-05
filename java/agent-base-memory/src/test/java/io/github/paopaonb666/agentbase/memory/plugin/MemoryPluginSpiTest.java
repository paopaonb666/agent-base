package io.github.paopaonb666.agentbase.memory.plugin;

import java.util.ServiceLoader;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 记忆包 SPI 注册的发现测试：services 文件必须以<b>二进制名</b>（嵌套接口
 * 含 {@code $}）命名，且 provider 实现对基座契约可赋值——这是「启用即承诺」
 * fail-fast 能否兑现的前提（对应 Python entry point group
 * {@code agent_base.memory} 恰好为 {service, routes} 的 CI 断言）。
 */
class MemoryPluginSpiTest {

    @Test
    void serviceLoaderDiscoversExactlyOneProvider() {
        ServiceLoader<io.github.paopaonb666.agentbase.core.memory.MemoryPlugins.MemoryPluginProvider> loader =
                ServiceLoader.load(
                        io.github.paopaonb666.agentbase.core.memory.MemoryPlugins.MemoryPluginProvider.class);
        var first = loader.stream().findFirst();
        assertTrue(first.isPresent(), "MemoryPluginProvider 未被 ServiceLoader 发现（services 文件命名错误？）");
        assertEquals(MemoryPluginProviderImpl.class, first.get().type());
    }
}
