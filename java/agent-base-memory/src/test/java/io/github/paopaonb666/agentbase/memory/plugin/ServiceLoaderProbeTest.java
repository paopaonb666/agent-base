package io.github.paopaonb666.agentbase.memory.plugin;

import java.util.ServiceLoader;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** ServiceLoader 发现探针（并入 MemoryPluginSpiTest 的发现断言）。 */
class ServiceLoaderProbeTest {

    @Test
    void probe() {
        var loader = ServiceLoader.load(
                io.github.paopaonb666.agentbase.core.memory.MemoryPlugins.MemoryPluginProvider.class);
        var first = loader.stream().findFirst();
        assertTrue(first.isPresent(), "MemoryPluginProvider 未被发现（services 文件名须用二进制名含 $）");
        System.out.println("PROBE FOUND " + first.get().type().getName());
    }
}
