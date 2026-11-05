package io.github.paopaonb666.agentbase.extensions.events;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** SSE 事件契约的线上格式测试：字段名/顺序/null 序列化与 Python 侧逐字节对齐。 */
class EventsTest {

    @Test
    void pingEncodesExactly() {
        assertEquals("event: ping\ndata: {\"type\":\"ping\"}\n\n", Events.encodeSse(new Events.PingEvent()));
    }

    @Test
    void deltaEncodesExactly() {
        assertEquals("event: delta\ndata: {\"type\":\"delta\",\"content\":\"he\"}\n\n",
                Events.encodeSse(new Events.DeltaEvent("he")));
    }

    @Test
    void stepIncludesNullDetail() {
        // Pydantic model_dump_json 保留 null 字段。
        assertEquals("event: step\ndata: {\"type\":\"step\",\"name\":\"agent\",\"status\":\"running\",\"detail\":null}\n\n",
                Events.encodeSse(Events.StepEvent.running("agent")));
    }

    @Test
    void doneCarriesThreadId() {
        assertEquals("event: done\ndata: {\"type\":\"done\",\"thread_id\":\"chat:abc\"}\n\n",
                Events.encodeSse(new Events.DoneEvent("chat:abc")));
        assertEquals("event: done\ndata: {\"type\":\"done\",\"thread_id\":null}\n\n",
                Events.encodeSse(new Events.DoneEvent(null)));
    }

    @Test
    void toolCallUsesSnakeCaseFields() {
        Events.ToolCallEvent event = new Events.ToolCallEvent(
                "call1", "echo", "start", Map.of("text", "hi"), null, null, null, null);
        assertEquals("event: tool_call\ndata: {\"type\":\"tool_call\",\"call_id\":\"call1\",\"name\":\"echo\","
                        + "\"phase\":\"start\",\"args\":{\"text\":\"hi\"},\"result\":null,\"status\":null,"
                        + "\"duration_ms\":null,\"error\":null}\n\n",
                Events.encodeSse(event));
    }

    @Test
    void planCarriesTaskList() {
        Events.PlanEvent event = new Events.PlanEvent("created",
                List.of(new Events.PlanTask(1, "查天气", "pending")), "拆解出 1 个子任务");
        String json = Events.encodeSse(event);
        assertEquals("event: plan\ndata: {\"type\":\"plan\",\"status\":\"created\","
                        + "\"plan\":[{\"id\":1,\"goal\":\"查天气\",\"status\":\"pending\"}],"
                        + "\"detail\":\"拆解出 1 个子任务\"}\n\n",
                json);
    }

    @Test
    void errorCarriesMessage() {
        assertEquals("event: error\ndata: {\"type\":\"error\",\"message\":\"出错了\"}\n\n",
                Events.encodeSse(new Events.ErrorEvent("出错了")));
    }

    @Test
    void sourcesReserved() {
        assertEquals("event: sources\ndata: {\"type\":\"sources\",\"sources\":[]}\n\n",
                Events.encodeSse(new Events.SourcesEvent(List.of())));
    }
}
