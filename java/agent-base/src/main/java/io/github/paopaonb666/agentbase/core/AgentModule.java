package io.github.paopaonb666.agentbase.core;

import dev.langchain4j.data.message.ChatMessage;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;

import java.util.List;
import java.util.Map;

/**
 * 每个 agent 模块都要实现的唯一扩展点。基座从不导入业务逻辑；
 * 模块实现这个接口，并由 registry（{@code core/ModuleRegistry}）
 * 把它们接入运行时。
 *
 * <p>契约刻意保持最小化：{@code name} / {@code description} /
 * {@code buildGraph} / {@code getTools}。
 *
 * <p>对应 Python 侧 {@code core/contracts.py} 的 {@code AgentModule} 协议。
 */
public interface AgentModule {

    /** 稳定标识符；必须与 AGENT_MODULES 条目一致（registry 强制执行）。 */
    String name();

    /** 用于发现 / supervisor 的人类可读摘要。 */
    String description();

    /**
     * 能力自述（#69）：handoff 工具描述的附加段——supervisor 据此判断
     * 「什么场景该交给这个模块」。缺省空串 = 不拼接，handoff 描述与
     * {@code description()} 单行逐字节一致（向后兼容；既有模块零强制）。
     * Python 侧模块类无此契约字段（差异登记簿承载）。
     */
    default String capabilities() {
        return "";
    }

    /**
     * 构建该模块编译后的图；以模块名命名，supervisor 据此把图作为
     * sub-agent 编排。
     */
    CompiledGraph buildGraph(ModuleContext ctx);

    /** 该模块贡献给共享池的工具（可为空列表）。 */
    List<AgentTool> getTools();

    /**
     * 基座交给每个模块的运行时服务（对应 Python 侧 {@code ModuleContext}
     * dataclass）。
     *
     * @param settings     已校验的配置
     * @param llm          装配好的对话模型（openai 兼容；流式/非流式统一面）
     * @param checkpointer 对话状态存储器；仅手动构造 context（测试）时为 null
     * @param tools        共享工具池：各模块 getTools() 的贡献 + 超时包装
     * @param memory       记忆服务端口；null = 未启用
     * @param planStore    计划快照端口；null = 存储不可用
     */
    record ModuleContext(
            io.github.paopaonb666.agentbase.core.config.Settings settings,
            ModelRuntime llm,
            io.github.paopaonb666.agentbase.core.graph.Checkpointer checkpointer,
            List<AgentTool> tools,
            MemoryPort memory,
            PlanStorePort planStore,
            io.github.paopaonb666.agentbase.core.tools.ToolPool toolPool) {

        public ModuleContext {
            tools = tools == null ? List.of() : List.copyOf(tools);
        }

        /** 六参便利构造（toolPool=null：测试直构/未接池的模块）。 */
        public ModuleContext(io.github.paopaonb666.agentbase.core.config.Settings settings,
                             ModelRuntime llm,
                             io.github.paopaonb666.agentbase.core.graph.Checkpointer checkpointer,
                             List<AgentTool> tools, MemoryPort memory, PlanStorePort planStore) {
            this(settings, llm, checkpointer, tools, memory, planStore, null);
        }
    }

    /**
     * 记忆子系统暴露给模块的最小面（依赖倒置）。core 契约只认这个端口，
     * 不 import 记忆包的具体类——替换记忆实现时，模块与基座契约都不用动。
     *
     * <p>跨边界的载荷一律是 {@code Map}/{@code List}（依赖倒置：基座源码
     * 中不出现 {@code agent_base_memory}）。
     */
    interface MemoryPort {

        /** 用户级混合检索；返回记忆 meta 列表（含 score/content/memory_id）。 */
        List<Map<String, Object>> search(String userId, String agentId, String query, Integer topK);

        /** 组装注入上下文（冻结段+召回段四合一）；无内容时返回 null。 */
        String composeContext(String userId, String agentId, String threadId, String query);

        /**
         * 每轮对话形成钩子：抽取候选 + 整合 + 画像 + 摘要（后台异步语义
         * 由实现方负责）；messages 为该轮快照的全部消息。
         */
        Map<String, Object> captureTurn(String userId, String agentId, String threadId,
                                        List<ChatMessage> messages);

        /** 释放底层资源（连接池等）。 */
        void close();
    }

    /**
     * 入口层需要的记忆服务完整面（0.3.0 拆分后由记忆包实现）。
     * {@code MemoryPort} 是业务模块看到的最小面；HTTP 层还需要知识库
     * 入库/检索、线程级联清理与健康探针。
     */
    interface MemoryServicePort extends MemoryPort {

        /** 记忆形成管线是否可用（未配置 LLM 时不可用）。 */
        boolean captureReady();

        /** 用户级逐轮召回注入段（贴尾注入）。 */
        String recallContext(String userId, String query);

        /** 线程级联清理（摘要/分块/快照；跨会话记忆保留）。 */
        void purgeThread(String threadId);

        /**
         * #172 可选参数变体：{@code purgeMemories=true} 时连跨会话记忆一起删
         * （按 source_thread_id 清理）——删除线程默认保留长期记忆，只有显式
         * 彻底清除才置 true。默认忽略标记委托旧签名（保留记忆），装饰器/
         * 替身无需逐个跟进（与下方 searchKnowledge 五参 default 同模式）。
         */
        default void purgeThread(String threadId, boolean purgeMemories) {
            purgeThread(threadId);
        }

        /** 文档入库：切块 + 向量化；返回写入的分块数。 */
        int ingestDocument(String fileId, String userId, String agentId, String text, String threadId);

        /** 按文件撤销知识分块；返回删除的分块数。 */
        int revokeDocument(String fileId, String userId);

        /** 知识库混合检索；返回切片视图列表。 */
        List<Map<String, Object>> searchKnowledge(String userId, String agentId, String query, Integer topK);

        /**
         * 知识库混合检索（#184：file_id 下推）：{@code fileId} 非空时候选
         * 集在存储层收窄到该文件——单文件检索不再对全库 2000 候选白算。
         * 默认忽略 fileId 委托 4 参版（装饰器/替身无需逐个跟进）。
         */
        default List<Map<String, Object>> searchKnowledge(String userId, String agentId,
                                                          String query, Integer topK, String fileId) {
            return searchKnowledge(userId, agentId, query, topK);
        }

        /** 文件切片视图列表（M7 可视化）。 */
        List<Map<String, Object>> listFileChunks(String userId, String fileId, int limit);

        /** file_id -> 分块计数（M8 文件列表）。 */
        Map<String, Integer> chunkCountsByFile(String userId);

        /** 健康探针：store 可读 + embedding 探活；返回 ok/degraded。 */
        String healthProbe();
    }

    /**
     * 后台管线用到的快档模型最小面（0.3.0 拆分引入）。快档是
     * ResilientLlm——它不进图执行路径，只做后台单次 invoke。
     */
    interface ChatInvokerPort {
        String invoke(List<ChatMessage> messages);
    }

    /**
     * 计划快照存储的最小端口（M10 跨轮延续，依赖倒置同 MemoryPort）。
     * planner 图在每个状态变更点 upsert 快照；plan 检查端点读表而非图
     * 状态。载荷刻意用 Map 进出：core 契约不 import 存储层的数据类。
     */
    interface PlanStorePort {

        /** upsert 计划快照（tasks 为任务 dict 列表）。 */
        void savePlan(String threadId, String module, List<Map<String, Object>> tasks,
                      int cursor, int replans);

        /** 读取快照：{@code {thread_id, module, tasks, cursor, replans}}（与
         * Python get_plan 逐键一致）；缺席返回 null。 */
        Map<String, Object> getPlan(String threadId);

        /** 删除快照。 */
        void deletePlan(String threadId);
    }
}
