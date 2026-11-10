package io.github.paopaonb666.agentbase.memory.tools;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import io.github.paopaonb666.agentbase.core.MemoryScope;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * knowledge_search 的引用锚与来源事件（#254/#338）：输出带 [n] 序号与
 * 文件/段位元数据，同序 emit SourcesEvent（title=文件+段位、url=null）；
 * 图外调用（nodeContext null）不发事件也不报错。
 */
class KnowledgeSearchCitationTest {

    private RecordingNodeContext nodeCtx;
    private MemoryStore store;

    @BeforeEach
    void setUp() {
        nodeCtx = new RecordingNodeContext();
        MemoryScope.set("u1", "chat:cite");
    }

    @AfterEach
    void tearDown() {
        MemoryScope.clear();
        if (store != null) {
            store.close();
        }
    }

    /** 固定向量的嵌入替身（2 维同向：关键词+向量路径都可用）。 */
    private static EmbeddingClient stubEmbedder() {
        return new EmbeddingClient() {
            @Override public Integer dims() { return 2; }
            @Override public List<List<Float>> embed(List<String> texts) {
                return texts.stream().map(t -> List.of(0.5f, 0.5f)).toList();
            }
            @Override public boolean probe() { return true; }
        };
    }

    private AgentTool knowledgeTool(String docText) throws Exception {
        store = new InMemoryMemoryStore();
        MemoryService service = new MemoryService(store, stubEmbedder(),
                Settings.load(Map.of("ENV", "development")), null, null);
        service.ingestDocument("f1", "u1", "chat", docText, "chat:cite");
        for (AgentTool tool : MemoryTools.build(service)) {
            if (tool.name().equals("knowledge_search")) {
                return tool;
            }
        }
        throw new IllegalStateException("knowledge_search 工具未注册");
    }

    @Test
    void knowledgeSearchNumbersResultsAndEmitsSources() throws Exception {
        AgentTool tool = knowledgeTool(
                "拉康的镜像阶段理论认为自我经由镜像认同而建构。第二段内容关于想象界与象征界的区分。");
        String out = tool.call("{\"query\": \"镜像阶段\"}",
                new AgentTool.ToolContext(nodeCtx.config(), nodeCtx));

        // 输出：[n] 锚 + 文件/段位元数据 + 引用约定行
        assertTrue(out.contains("请用 [n] 标注来源序号"));
        assertTrue(out.contains("[1]（相关度 "));
        assertTrue(out.contains("文件 f1 第 1 段"));

        // 同序 sources 事件
        assertEquals(1, nodeCtx.events.size());
        Events.SourcesEvent sources = (Events.SourcesEvent) nodeCtx.events.get(0);
        assertEquals(1, sources.sources().size());
        assertEquals("文件 f1 第 1 段", sources.sources().get(0).title());
        assertEquals(null, sources.sources().get(0).url());
    }

    @Test
    void knowledgeSearchOutsideGraphSkipsEmission() throws Exception {
        AgentTool tool = knowledgeTool("关于镜像阶段的一段文本。");
        String out = tool.call("{\"query\": \"镜像阶段\"}",
                new AgentTool.ToolContext(nodeCtx.config(), null));
        assertTrue(out.contains("[1]（相关度 "));
        assertTrue(nodeCtx.events.isEmpty());
    }

    @Test
    void knowledgeSearchMergesAdjacentHitsIntoParentSection() throws Exception {
        // #50：相邻命中（ordinal 连续）合一条——段号范围「第 1-2 段」、
        // 文本取父块（两段全文），sources 事件同序同标签；[n] 按父块条目计数。
        // 直插相邻分块（短文档经 ingestDocument 只产 1 块，构不成相邻对）。
        store = new InMemoryMemoryStore();
        MemoryService service = new MemoryService(store, stubEmbedder(),
                Settings.load(Map.of("ENV", "development")), null, null);
        store.putChunks(List.of(
                new io.github.paopaonb666.agentbase.memory.store.Models.DocChunk(
                        "k1", "f1", "chat:cite", "u1", "chat", 0,
                        "镜像阶段理论认为自我经由镜像认同而建构。",
                        List.of(new long[]{0, 18}),
                        io.github.paopaonb666.agentbase.memory.store.Models.encodeEmbedding(List.of(0.5f, 0.5f)), 2,
                        io.github.paopaonb666.agentbase.memory.store.Models.now()),
                new io.github.paopaonb666.agentbase.memory.store.Models.DocChunk(
                        "k2", "f1", "chat:cite", "u1", "chat", 1,
                        "第二段讲镜像阶段之后的想象界与象征界的区分。",
                        List.of(new long[]{19, 36}),
                        io.github.paopaonb666.agentbase.memory.store.Models.encodeEmbedding(List.of(0.5f, 0.5f)), 2,
                        io.github.paopaonb666.agentbase.memory.store.Models.now())));
        AgentTool tool = MemoryTools.build(service).stream()
                .filter(t -> t.name().equals("knowledge_search")).findFirst()
                .orElseThrow(() -> new IllegalStateException("knowledge_search 工具未注册"));
        String out = tool.call("{\"query\": \"镜像\", \"limit\": 5}",
                new AgentTool.ToolContext(nodeCtx.config(), nodeCtx));

        assertTrue(out.contains("第 1-2 段"), out);
        assertTrue(out.contains("想象界与象征界"), "父块文本应含第二段全文：" + out);
        assertEquals(1, nodeCtx.events.size());
        Events.SourcesEvent sources = (Events.SourcesEvent) nodeCtx.events.get(0);
        assertEquals(1, sources.sources().size());
        assertEquals("文件 f1 第 1-2 段", sources.sources().get(0).title());
    }

    @Test
    void knowledgeSearchSingleHitKeepsLegacyRendering() throws Exception {
        // 独立命中（无相邻块）→ 渲染保持旧形态（第 N 段单数）。
        AgentTool tool = knowledgeTool("唯一段落谈镜像阶段。另一段谈不相关的事。");
        String out = tool.call("{\"query\": \"镜像阶段\"}",
                new AgentTool.ToolContext(nodeCtx.config(), nodeCtx));
        assertTrue(out.contains("文件 f1 第 1 段"), out);
    }

    @Test
    void knowledgeSearchEmptyResultHasNoCitations() throws Exception {
        // 空知识库：searchKnowledge 返回空表 → 无锚无事件
        store = new InMemoryMemoryStore();
        MemoryService service = new MemoryService(store, stubEmbedder(),
                Settings.load(Map.of("ENV", "development")), null, null);
        AgentTool tool = MemoryTools.build(service).stream()
                .filter(t -> t.name().equals("knowledge_search")).findFirst()
                .orElseThrow(() -> new IllegalStateException("knowledge_search 工具未注册"));
        String out = tool.call("{\"query\": \"随便什么查询\"}",
                new AgentTool.ToolContext(nodeCtx.config(), nodeCtx));
        assertEquals("（知识库中没有相关内容）", out);
        assertTrue(nodeCtx.events.isEmpty());
    }

    /** 录制事件的 NodeContext 替身（本模块无既有桩，局部自包含）。 */
    private static final class RecordingNodeContext
            implements io.github.paopaonb666.agentbase.core.graph.NodeContext {
        final List<Object> events = new ArrayList<>();
        private final RunConfig config = RunConfig.of("chat:cite", "u1");

        @Override
        public RunConfig config() {
            return config;
        }

        @Override
        public void emit(Object event) {
            events.add(event);
        }

        @Override
        public String callTool(String name, String argsJson) {
            return "";
        }

        @Override
        public io.github.paopaonb666.agentbase.core.tools.ToolPool tools() {
            return null;
        }

        @Override
        public boolean cancelled() {
            return false;
        }
    }
}
