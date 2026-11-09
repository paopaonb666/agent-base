package io.github.paopaonb666.agentbase.memory.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.MemoryScope;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import io.github.paopaonb666.agentbase.memory.tools.BlockLimits;
import io.github.paopaonb666.agentbase.memory.tools.MemoryTools;

/**
 * 记忆块 API 契约（对应 Python api.py put_memory_block）。
 *
 * <p>回归：{@code putBlock} 曾在写入前把 content 截断到 char_limit——
 * Python 原样入库（char_limit 只是声明容量，注入组装时才按预算裁剪），
 * 截断会导致管理端写入的数据永久丢字、版本号照常递增。
 *
 * <p>#178：三路径上限一致性矩阵——API（1..8000）与工具（1..4000）两条
 * 写入路径经同一服务层落库，服务层永不截断；常量收口于
 * {@link BlockLimits}（矩阵语义见其 javadoc）。
 */
class MemoryServiceBlockTest {

    /** 恒不向量化：MemoryService 各路径对 null 嵌入的降级分支。 */
    static EmbeddingClient neverEmbedder() {
        return new EmbeddingClient() {
            @Override public Integer dims() { return null; }
            @Override public List<List<Float>> embed(List<String> texts) { return null; }
            @Override public boolean probe() { return false; }
        };
    }

    private static MemoryService service(MemoryStore store) {
        return new MemoryService(store, neverEmbedder(), Settings.load(Map.of(), null),
                null, null);
    }

    @AfterEach
    void clearScope() {
        MemoryScope.clear();
    }

    @Test
    void putBlockStoresContentVerbatim() {
        try (MemoryStore store = new InMemoryMemoryStore()) {
            MemoryService service = service(store);
            String longContent = "字".repeat(5000);
            var block = service.putBlock("u1", "chat", "persona", longContent, 2000);
            assertEquals(5000, block.content().length(),
                    "putBlock 必须原样存储 content（截断只发生在注入组装）");
            assertEquals(5000, store.getBlock("u1", "chat", "persona").content().length());
        }
    }

    /** AgentTool.call 抛受检异常：测试统一经此收口（照 MemoryToolsTest）。 */
    private static String callTool(AgentTool tool, String argsJson) {
        try {
            return tool.call(argsJson, null);
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void threePathLimitMatrixAlignsOnSameContent() {
        // 常量矩阵自检（#178 收口）：API 8000 / 工具 4000 / char_limit
        // 50..8000 缺省 2000；工具侧比 API 侧保守。
        assertEquals(8000, BlockLimits.CONTENT_MAX_API);
        assertEquals(4000, BlockLimits.CONTENT_MAX_TOOL);
        assertEquals(2000, BlockLimits.CHAR_LIMIT_DEFAULT);
        assertEquals(50, BlockLimits.CHAR_LIMIT_MIN);
        assertEquals(8000, BlockLimits.CHAR_LIMIT_MAX);
        assertTrue(BlockLimits.CONTENT_MAX_TOOL < BlockLimits.CONTENT_MAX_API,
                "工具入参上限低于 API 上限（模型写入更保守）");

        try (MemoryStore store = new InMemoryMemoryStore()) {
            MemoryService service = service(store);

            // API 路径（控制器校验 1..CONTENT_MAX_API 后委托 putBlock）：
            // 上限边界 8000 字原样入库，服务层不截断。
            service.putBlock("u1", "chat", "api_block",
                    "字".repeat(BlockLimits.CONTENT_MAX_API), BlockLimits.CHAR_LIMIT_DEFAULT);
            assertEquals(BlockLimits.CONTENT_MAX_API,
                    store.getBlock("u1", "chat", "api_block").content().length());

            // 工具路径：入参上限 CONTENT_MAX_TOOL 边界值成功（块声明容量
            // 预置为 CHAR_LIMIT_MAX，避免保尾截断干扰边界对照）。
            store.upsertBlock(new Models.MemoryBlock("u1", "chat", "tool_block",
                    "seed", BlockLimits.CHAR_LIMIT_MAX, 1, Models.now()));
            MemoryScope.set("u1", "chat:t1");
            AgentTool updateBlock = MemoryTools.build(service).stream()
                    .filter(t -> t.name().equals("memory_update_block")).findFirst().orElseThrow();
            String receipt = callTool(updateBlock, "{\"label\": \"tool_block\", \"content\": \""
                    + "字".repeat(BlockLimits.CONTENT_MAX_TOOL) + "\", \"operation\": \"replace\"}");
            assertTrue(receipt.contains("已更新（v2，" + BlockLimits.CONTENT_MAX_TOOL
                    + "/" + BlockLimits.CHAR_LIMIT_MAX + " 字符）"), receipt);
            assertEquals(BlockLimits.CONTENT_MAX_TOOL,
                    store.getBlock("u1", "chat", "tool_block").content().length());

            // 语义对照：同一份 4001 字内容——工具路径拒绝（入参校验抛错），
            // API 路径放行且原样入库（服务层永不截断）。
            String boundary = "字".repeat(BlockLimits.CONTENT_MAX_TOOL + 1);
            IllegalArgumentException toolRejected = assertThrows(IllegalArgumentException.class,
                    () -> callTool(updateBlock, "{\"label\": \"tool_block\", \"content\": \""
                            + boundary + "\"}"));
            assertEquals("content 必须是 1..4000 字符", toolRejected.getMessage());
            var viaApiBoundary = service.putBlock("u1", "chat", "boundary", boundary,
                    BlockLimits.CHAR_LIMIT_DEFAULT);
            assertEquals(boundary.length(), viaApiBoundary.content().length(),
                    "API 路径对工具超限的同一内容放行且原样入库");
        }
    }
}
