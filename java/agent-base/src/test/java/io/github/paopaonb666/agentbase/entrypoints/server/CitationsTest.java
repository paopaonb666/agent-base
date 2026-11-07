package io.github.paopaonb666.agentbase.entrypoints.server;

import io.github.paopaonb666.agentbase.extensions.events.Events;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 回复引用解析契约（#338）：[n] 锚解析、登记簿去重、解析回 sources，
 * 容错语义（无锚/越界 → 空表 → 调用方不 emit）锁定。
 */
class CitationsTest {

    @Test
    void parsesCitationAnchorsInOrderWithoutDuplication() {
        assertEquals(List.of(1, 3, 2), Citations.parse("见 [1] 与 [3]，另 [2]；重复 [1] 无效"));
        assertEquals(List.of(12), Citations.parse("多位的 [12] 也要解析"));
    }

    @Test
    void nonCitationBracketsDoNotMatch() {
        // Markdown 链接、[0]、代码下标 arr[1]：都不算引用锚
        assertTrue(Citations.parse("[拉康](https://example.com) 的镜像是 [0] 号").isEmpty());
        assertTrue(Citations.parse("arr[1] 取下标").isEmpty());
        assertTrue(Citations.parse("x[12] 同样是代码下标").isEmpty());
        assertTrue(Citations.parse("").isEmpty());
        assertTrue(Citations.parse(null) == null || Citations.parse(null).isEmpty());
        // 中文紧邻（无空格）仍是引用
        assertEquals(List.of(1), Citations.parse("镜像是[1]号的证明"));
    }

    @Test
    void dedupKeepsFirstOccurrenceOrder() {
        List<Events.Source> deduped = Citations.dedup(List.of(
                new Events.Source("A", "u1"),
                new Events.Source("B", null),
                new Events.Source("A", "u1")));
        assertEquals(2, deduped.size());
        assertEquals("A", deduped.get(0).title());
        assertEquals("B", deduped.get(1).title());
    }

    @Test
    void resolveMapsAnchorsToRegistryEntries() {
        List<Events.Source> registry = Citations.dedup(List.of(
                new Events.Source("Alpha", "https://a"),
                new Events.Source("Beta", "https://b"),
                new Events.Source("文件 f1 第 2 段", null)));
        List<Events.Source> cited = Citations.resolve(
                "结论 [1] 和 [3] 支持这一点，[2] 反对。", registry);
        assertEquals(List.of(registry.get(0), registry.get(2), registry.get(1)), cited);
    }

    @Test
    void resolveIsTolerantToOutOfRangeAndEmptyRegistry() {
        List<Events.Source> registry = List.of(new Events.Source("A", null));
        // 越界 [5] 跳过，只回 [1]
        assertEquals(List.of(new Events.Source("A", null)),
                Citations.resolve("[1] 和 [5]", registry));
        // 登记簿空（本轮无来源事件）→ 空 → 调用方不 emit
        assertTrue(Citations.resolve("引用 [1]", List.of()).isEmpty());
        // 无锚 → 空
        assertTrue(Citations.resolve("普通回答没有引用", registry).isEmpty());
    }
}
