package io.github.paopaonb666.agentbase.extensions;

import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.EventSink;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 事件丢弃计数（roadmap #210）：断连后节点的 emit 是面板缺口——计入
 * agent_events_dropped_total{reason="cancelled"}；SSE 心跳 ping 满队丢弃
 * 计入 {reason="ping_full"}。封闭枚举防基数爆炸。进程级单例：断言只看
 * 本测试造成的增量。
 */
class EventDropMetricsTest {

    @Test
    void enumIsClosed() {
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> Metrics.events().observeDrop("banana"));
        assertTrue(unknown.getMessage().contains("cancelled, ping_full"));
    }

    @Test
    void cancelledSinkDropsAreCountedPerGraphRun() {
        // 取消的 sink：节点 emit 静默丢弃（原本连日志都没有），现在计数。
        long before = Metrics.events().dropped("cancelled");
        AtomicBoolean cancelled = new AtomicBoolean(false);

        CompiledGraph graph = GraphBuilder.singleNode("emit", (state, nodeCtx) -> {
            cancelled.set(true); // 模拟客户端先断连
            nodeCtx.emit(new Events.DeltaEvent("面板收不到这段"));
            nodeCtx.emit(new Events.DeltaEvent("这段也是"));
            return Map.of();
        }, new InMemoryCheckpointer());

        // 图照常跑完（emit 丢弃不中断执行），计数恰好 +2。
        io.github.paopaonb666.agentbase.core.graph.EventSink sink = new EventSink() {
            @Override public void accept(Object event) { }
            @Override public boolean cancelled() { return cancelled.get(); }
        };
        graph.stream(new GraphState(
                java.util.List.of(new dev.langchain4j.data.message.UserMessage("hi")), Map.of()),
                new RunConfig("drop-probe:" + System.nanoTime(), "u", 10), sink);
        assertEquals(2, Metrics.events().dropped("cancelled") - before);
    }

    @Test
    void healthySinkDoesNotCountDrops() {
        long before = Metrics.events().dropped("cancelled");
        CompiledGraph graph = GraphBuilder.singleNode("emit", (state, nodeCtx) -> {
            nodeCtx.emit(new Events.DeltaEvent("正常投递"));
            return Map.of();
        }, new InMemoryCheckpointer());
        graph.stream(new GraphState(
                java.util.List.of(new dev.langchain4j.data.message.UserMessage("hi")), Map.of()),
                new RunConfig("drop-clean:" + System.nanoTime(), "u", 10),
                new EventSink() {
                    @Override public void accept(Object event) { }
                });
        assertEquals(before, Metrics.events().dropped("cancelled"));
    }

    @Test
    void rendersIntoMetricsSection() {
        Metrics.events().observeDrop("ping_full");
        String body = Metrics.events().renderEventDropMetrics();
        assertTrue(body.matches("(?s).*agent_events_dropped_total\\{reason=\"ping_full\"\\} \\d+.*"), body);
        // 空渲染为空串（无尾换行），与其他段一致。
        assertEquals("", new Metrics.EventDropMetrics().renderEventDropMetrics());
    }
}
