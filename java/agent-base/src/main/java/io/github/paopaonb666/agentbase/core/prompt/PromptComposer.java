package io.github.paopaonb666.agentbase.core.prompt;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * PromptComposer（roadmap 13）：对话输入 system 层的分层组装 + 层 id/版本
 * 结构化标注。<b>纯结构化重构——最终发给模型的 prompt 逐字节不变</b>是硬
 * 要求：层文本一律引用调用方传入的原文/原常量（WRITER_PROMPT、
 * DEFAULT_PROMPT、PlannerPrompts 运行期渲染结果），本类从不重写（text block
 * 重写的换行/缩进漂移 = 模型行为 + 前缀缓存双变化）。
 *
 * <p><b>收口范围 = 对话输入的 system 层</b>（5 处 SystemMessage 形态拼装点）：
 *
 * <ul>
 *   <li>{@code module.persona}：单 LLM 节点模块人格（Graphs 工厂经参传入；
 *       writer 经 buildSingleAgentGraph 传入 WRITER_PROMPT，无独立拼装点；
 *       chat/hello 传 null = 无人格层直通）；</li>
 *   <li>{@code planner.execute} / {@code planner.synth}：planner 两个
 *       SystemMessage 形态节点（ExecuteNode.subtaskInput / CheckNodes.makeSynthesize）；</li>
 *   <li>{@code collab.supervisor}：supervisor 模板的 DEFAULT_PROMPT。</li>
 * </ul>
 *
 * <p>server 层的记忆注入（{@code memory.frozen} 冻结段 / {@code memory.recall}
 * 召回段）与附件段（{@code input.attachment}）<b>只在日志侧标注</b>——冻结段
 * 文本经 MemoryService 线程级 LRU 快照缓存已逐字节冻结，
 * composeContext/recallContext 返回值与 {@code SystemMessage.from(block)}
 * 消息构造不归本类触碰。
 *
 * <p><b>不收口</b>（登记）：MemoryPipeline 的 4 个 prompt（后台管线、
 * UserMessage 形态、文案是管线行为契约）；planner 的 plan/replan/verdict
 * 任务提示（UserMessage 单条调用形态，PlanNode/CheckNodes.invokeText）。
 * 「base 工程约定层」与「环境信息层」本期不建——现状无对应文本，建层 =
 * 加字节 = 行为变化。「prompt 文本外置为资源文件热加载」登记不做：prompt
 * 是行为契约，外置热加载改变部署语义。
 *
 * <p>层版本（{@link Layer#version()}）只进日志/可观测面，<b>不进 prompt
 * 文本</b>。装配可见性本期走日志通道（INFO 一行/次装配：层 id#version +
 * {@link ContextEngine#countMessageTokens} 同口径 token 数——零 SSE 面
 * 变化零引擎改动）；step 事件 detail 补层信息登记后置（CompiledGraph 是
 * 热区）。#14 system-reminder 未来同包。对应 Python 侧无组件——#307 登记
 * 「PromptComposer 为 Java 单边增强，Python 侧 prompt 组装点与文本不动」。
 */
public final class PromptComposer {

    private static final Logger LOG = LoggerFactory.getLogger(PromptComposer.class);

    /** 层版本（只进日志）：模块人格层。本期全部 v1。 */
    public static final int VERSION_MODULE_PERSONA = 1;
    /** 层版本（只进日志）：planner execute 节点层。 */
    public static final int VERSION_PLANNER_EXECUTE = 1;
    /** 层版本（只进日志）：planner synthesize 节点层。 */
    public static final int VERSION_PLANNER_SYNTH = 1;
    /** 层版本（只进日志）：supervisor 模板层。 */
    public static final int VERSION_COLLAB_SUPERVISOR = 1;
    /** 层版本（只进日志）：记忆冻结段（仅日志标注）。 */
    public static final int VERSION_MEMORY_FROZEN = 1;
    /** 层版本（只进日志）：记忆召回段（仅日志标注）。 */
    public static final int VERSION_MEMORY_RECALL = 1;
    /** 层版本（只进日志）：附件上下文段（仅日志标注）。 */
    public static final int VERSION_INPUT_ATTACHMENT = 1;

    private PromptComposer() {}

    /**
     * 一层 prompt：稳定 id + 版本 + 文本。层 id 是封闭常量集——构造期白名单
     * 校验（未知 id 抛 {@link IllegalArgumentException}，防拼错 id 静默绕过
     * 封闭集；规范构造器访问级别不得低于 record 本身，故封闭性走运行期
     * 校验而非私有构造器）。text 引用直传不拷贝不改写（逐字节契约）。
     */
    public record Layer(String id, int version, String text) {

        public Layer {
            if (!knownLayerIds().contains(id)) {
                throw new IllegalArgumentException(
                        "未知 prompt 层 id（封闭集见 PromptComposer.knownLayerIds）：" + id);
            }
        }

        /** 模块人格层（Graphs 工厂经参传入；chat/hello 传 null = 无人格层）。 */
        public static Layer modulePersona(String text) {
            return new Layer("module.persona", VERSION_MODULE_PERSONA, text);
        }

        /** planner execute 节点层（PlannerPrompts.execute 运行期渲染结果）。 */
        public static Layer plannerExecute(String text) {
            return new Layer("planner.execute", VERSION_PLANNER_EXECUTE, text);
        }

        /** planner synthesize 节点层（PlannerPrompts.synth 运行期渲染结果）。 */
        public static Layer plannerSynth(String text) {
            return new Layer("planner.synth", VERSION_PLANNER_SYNTH, text);
        }

        /** supervisor 模板层（SupervisorGraphs.DEFAULT_PROMPT）。 */
        public static Layer collabSupervisor(String text) {
            return new Layer("collab.supervisor", VERSION_COLLAB_SUPERVISOR, text);
        }

        /** 记忆冻结段（仅日志标注——文本由 MemoryService.composeFrozen 冻结）。 */
        public static Layer memoryFrozen(String text) {
            return new Layer("memory.frozen", VERSION_MEMORY_FROZEN, text);
        }

        /** 记忆召回段（仅日志标注——文本由 MemoryContext.composeRecallContext 组装）。 */
        public static Layer memoryRecall(String text) {
            return new Layer("memory.recall", VERSION_MEMORY_RECALL, text);
        }

        /** 附件上下文段（仅日志标注——ContextEngine.ATTACHMENT_CONTEXT_PREFIX 契约）。 */
        public static Layer inputAttachment(String text) {
            return new Layer("input.attachment", VERSION_INPUT_ATTACHMENT, text);
        }
    }

    /** 层 id 封闭集（观测/测试面：7 层；新增层须随 roadmap 条目登记）。 */
    public static Set<String> knownLayerIds() {
        return Set.of("module.persona", "planner.execute", "planner.synth",
                "collab.supervisor", "memory.frozen", "memory.recall", "input.attachment");
    }

    /**
     * 人格层消息构造：{@code SystemMessage.from(layer.text())}（与收口前各
     * 拼装点逐字节 equals）+ 一行结构化日志（层 id#version + token 计量）。
     * {@code layer} 为 null 或 text 为 null 时返回 null 不打日志（chat/hello
     * 的无人格形态）。graph 名仅进日志，可传 null。
     */
    public static SystemMessage persona(Layer layer) {
        return persona(layer, null);
    }

    /** 同 {@link #persona(Layer)}，日志带 graph 名。 */
    public static SystemMessage persona(Layer layer, String graph) {
        if (layer == null || layer.text() == null) {
            return null;
        }
        SystemMessage message = SystemMessage.from(layer.text());
        logLayer(layer, graph);
        return message;
    }

    /**
     * 注入段日志标注（memory.frozen / memory.recall / input.attachment 专用）：
     * 只观测不构造消息——注入段的消息构造仍是调用方的
     * {@code SystemMessage.from(block)} 原样，冻结段逐字节冻结语义不受影响。
     * null/空文本静默跳过（与注入段的静默降级一致）。
     */
    public static void logLayer(Layer layer, String graph) {
        if (layer == null || layer.text() == null || layer.text().isEmpty()) {
            return;
        }
        LOG.info("prompt: layer={}#v{} tokens={} graph={}",
                layer.id(), layer.version(),
                ContextEngine.countMessageTokens(SystemMessage.from(layer.text())),
                graph);
    }

    /**
     * 组装：人格层消息加在模型输入<b>最前</b>（与收口前 Graphs agent 节点的
     * 临时拼装逐字节等价）。{@code persona} 为 null 或 text 为 null 时原样
     * 直通（同一引用——chat/hello 的无人格形态，模型输入一个字节不多）。
     * 装配即打一行层日志（经 {@link #persona(Layer, String)}）。
     */
    public static List<ChatMessage> assemble(Layer persona, List<ChatMessage> modelInput) {
        return assemble(persona, null, modelInput);
    }

    /** 同 {@link #assemble(Layer, List)}，日志带 graph 名。 */
    public static List<ChatMessage> assemble(Layer persona, String graph, List<ChatMessage> modelInput) {
        if (persona == null || persona.text() == null) {
            return modelInput;
        }
        List<ChatMessage> withPrompt = new ArrayList<>();
        withPrompt.add(persona(persona, graph));
        withPrompt.addAll(modelInput);
        return withPrompt;
    }
}
