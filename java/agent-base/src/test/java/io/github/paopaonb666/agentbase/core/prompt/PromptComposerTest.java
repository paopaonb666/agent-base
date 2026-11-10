package io.github.paopaonb666.agentbase.core.prompt;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.extensions.collab.SupervisorGraphs;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * PromptComposer（条目 13）锁定测试：<b>最终 prompt 逐字节不变</b>是硬要求
 * ——persona 与 {@code SystemMessage.from(text)} 值相等（langchain4j equals
 * 按 text 逐字节）、assemble 与收口前 Graphs agent 节点的临时拼装逐字节
 * 等价（null 直通 = 同一引用；有 persona = [system, ...input]）、层 id 是
 * 7 层封闭集、层 id/版本/token 只进日志（零 SSE 面变化）。
 */
class PromptComposerTest {

    /** 多行 + CJK + 制表符样本：任何换行/缩进漂移在此即红。 */
    private static final String SAMPLE_TEXT =
            "你是执行总结者。下面是一次多步执行的子任务清单与各自结果。\n"
                    + "综合它们给用户一个连贯的最终答复；如实标注未完成/失败的子任务，不要虚构。\n"
                    + "子任务清单：\n#1 [done] 目标A：已完成\t（含 tab 与空行）\n\n";

    @Test
    void personaReturnsByteIdenticalSystemMessage() {
        PromptComposer.Layer layer = PromptComposer.Layer.plannerSynth(SAMPLE_TEXT);
        // 文本引用直传不拷贝不改写（"引用原常量/调用方传入文本"的锁定面）。
        assertSame(SAMPLE_TEXT, layer.text());
        SystemMessage message = PromptComposer.persona(layer);
        // langchain4j SystemMessage equals 按 text 逐字节——与收口前形态相等。
        assertEquals(SystemMessage.from(SAMPLE_TEXT), message);
        assertEquals(SAMPLE_TEXT, message.text());

        // 无人格形态：null 层 / null 文本 → null（chat/hello 直通）。
        assertNull(PromptComposer.persona(null));
        assertNull(PromptComposer.persona(PromptComposer.Layer.modulePersona(null)));
    }

    @Test
    void assembleWithoutPersonaIsPassThrough() {
        List<ChatMessage> input = List.of(
                SystemMessage.from("【长期记忆上下文·固定】冻结段"),
                UserMessage.from("human 消息"));
        // null persona / null 文本 persona：与收口前 Graphs 的 null 提示形态
        // 逐字节等价——同一引用直通，模型输入一个字节不多。
        assertSame(input, PromptComposer.assemble(null, input));
        assertSame(input, PromptComposer.assemble(PromptComposer.Layer.modulePersona(null), input));
        assertSame(input, PromptComposer.assemble(PromptComposer.Layer.modulePersona(null), "chat", input));
    }

    @Test
    void assemblePrependsPersonaByteExactly() {
        List<ChatMessage> modelInput = List.of(
                SystemMessage.from("【长期记忆上下文·固定】冻结段"),
                UserMessage.from("human 消息"));
        List<ChatMessage> out = PromptComposer.assemble(
                PromptComposer.Layer.modulePersona(SAMPLE_TEXT), "writer", modelInput);
        // 形状：[persona SystemMessage, ...原输入]。
        assertEquals(modelInput.size() + 1, out.size());
        assertEquals(SystemMessage.from(SAMPLE_TEXT), out.get(0));
        assertEquals(SAMPLE_TEXT, ((SystemMessage) out.get(0)).text());
        // 原输入逐元素保持（同引用、同序）。
        assertSame(modelInput.get(0), out.get(1));
        assertSame(modelInput.get(1), out.get(2));
        // 与收口前的手工拼装逐字节等价（Graphs agent 节点旧形态对照）。
        List<ChatMessage> expected = new ArrayList<>();
        expected.add(SystemMessage.from(SAMPLE_TEXT));
        expected.addAll(modelInput);
        assertEquals(expected, out);
    }

    @Test
    void layerCatalogIsClosedSevenIdSet() {
        assertEquals(Set.of("module.persona", "planner.execute", "planner.synth",
                        "collab.supervisor", "memory.frozen", "memory.recall", "input.attachment"),
                PromptComposer.knownLayerIds());
        // 每个工厂：id 命中封闭集、版本 = 1（版本只进日志，不进 prompt 文本）。
        record Expected(String id, PromptComposer.Layer layer) {}
        List<Expected> all = List.of(
                new Expected("module.persona", PromptComposer.Layer.modulePersona("t")),
                new Expected("planner.execute", PromptComposer.Layer.plannerExecute("t")),
                new Expected("planner.synth", PromptComposer.Layer.plannerSynth("t")),
                new Expected("collab.supervisor", PromptComposer.Layer.collabSupervisor("t")),
                new Expected("memory.frozen", PromptComposer.Layer.memoryFrozen("t")),
                new Expected("memory.recall", PromptComposer.Layer.memoryRecall("t")),
                new Expected("input.attachment", PromptComposer.Layer.inputAttachment("t")));
        for (Expected e : all) {
            assertEquals(e.id(), e.layer().id(), e.id());
            assertEquals(1, e.layer().version(), e.id());
            assertEquals("t", e.layer().text(), e.id());
        }
    }

    @Test
    void assemblyLogsLayerVersionTokensAndGraph() {
        Logger composerLogger = (Logger) LoggerFactory.getLogger(PromptComposer.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        composerLogger.addAppender(appender);
        try {
            // assemble：一行层日志（graph 名来自调用点）。
            PromptComposer.assemble(
                    PromptComposer.Layer.collabSupervisor(SupervisorGraphs.DEFAULT_PROMPT),
                    SupervisorGraphs.MODULE_NAME, List.of(UserMessage.from("q")));
            List<ILoggingEvent> assembleEvents = events(appender, "prompt: layer=collab.supervisor#v1");
            assertEquals(1, assembleEvents.size());
            assertEquals(Level.INFO, assembleEvents.get(0).getLevel());
            String message = assembleEvents.get(0).getFormattedMessage();
            assertTrue(message.contains("graph=supervisor"), message);
            // token 计量与 ContextEngine.countMessageTokens(SystemMessage) 同口径。
            int expectedTokens = ContextEngine.countMessageTokens(
                    SystemMessage.from(SupervisorGraphs.DEFAULT_PROMPT));
            assertTrue(message.contains("tokens=" + expectedTokens), message);

            // logLayer（server 注入段专用）：memory.frozen / memory.recall 标注。
            PromptComposer.logLayer(PromptComposer.Layer.memoryFrozen("冻结段文本"), "chat");
            PromptComposer.logLayer(PromptComposer.Layer.memoryRecall("召回段文本"), "chat");
            assertEquals(1, events(appender, "prompt: layer=memory.frozen#v1").size());
            assertEquals(1, events(appender, "prompt: layer=memory.recall#v1").size());

            // 静默形态：null 层 / null / 空文本不打日志（与注入段静默降级一致）。
            int before = appender.list.size();
            PromptComposer.logLayer(null, "chat");
            PromptComposer.logLayer(PromptComposer.Layer.memoryFrozen(null), "chat");
            PromptComposer.logLayer(PromptComposer.Layer.memoryRecall(""), "chat");
            assertEquals(before, appender.list.size());
        } finally {
            composerLogger.detachAppender(appender);
            appender.stop();
        }
    }

    private static List<ILoggingEvent> events(ListAppender<ILoggingEvent> appender, String marker) {
        return appender.list.stream()
                .filter(e -> e.getFormattedMessage().contains(marker))
                .toList();
    }
}
