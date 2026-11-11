package io.github.paopaonb666.agentbase.toolkit;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;

import io.github.paopaonb666.agentbase.core.text.CharMetrics;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileInfo;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore;

/**
 * {@code read_tool_result} 工具（条目 6/23）：按需分页读回外置的超大工具
 * 结果——ContextEngine.spillIfNeeded 摘要块给出 file_id 引用，模型用它
 * 取回后续窗口，避免「一次读回又撑爆上下文」的自伤循环。
 *
 * <p><b>装配方式</b>：不走 TOOLKIT_ENABLED 注册表（它随外置开关存在——
 * store 未装配或阈值 0 时整个机制停用，无独立启用语义），由 AgentRuntime
 * 在 fileStore 就绪且阈值 &gt; 0 时直装进共享池。
 *
 * <p><b>属主校验自带（store 层 getMany 无属主校验，不能依赖）</b>：取行后
 * 按 userId + threadId 双等校验，不匹配或不存在统一回「引用不存在或无权
 * 访问」（404 掩蔽风格——跨用户与跨线程 file_id 枚举一次拦死）。
 *
 * <p><b>分页约定示范（条目 23）</b>：limit/offset 整数页参，不用 opaque
 * cursor——被读对象是静态文本串（存库后不变），整数寻址天然适配、模型好
 * 推理（「首屏 M 字符 → 下次 offset=M」）。越界报错不钳制（既有文案先例）；
 * offset 超出文本长度是合法空屏（返回「已到底」）。只读（readOnly 默认
 * true，#336 并发语义），纯读不产生副作用。
 */
public final class ReadToolResultTool implements AgentTool {

    /** 与 {@link ToolResultLimits#READ_TOOL_NAME} 单源（改名只动常量一处）。 */
    public static final String NAME = ToolResultLimits.READ_TOOL_NAME;

    /** 不存在/非属主的统一掩蔽文案（404 掩蔽风格）。 */
    static final String DENIED_TEXT = "引用不存在或无权访问";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final UploadedFileStore fileStore;

    /** 构造：store 由 AgentRuntime 装配期传入（null store 不会走到直装分支）。 */
    public ReadToolResultTool(UploadedFileStore fileStore) {
        this.fileStore = fileStore;
    }

    /** 给模型看的说明书（三段式规范；本工具同时是条目 23 分页约定的示范）。 */
    private static final String DESCRIPTION = """
            读取此前被外置的超大工具结果：当工具结果的 ToolMessage 以
            「【结果过大已外置】」摘要块开头（含 file_id 引用）时，用本工具
            按页取回后续原文，不必一次读回全文。

            何时不用：结果没有外置（ToolMessage 里就是全文）时不要调用；
            首屏已足够回答问题时也不必继续翻页。

            Args:
                file_id: 必填——摘要块给出的引用 id。
                offset: 起始码点偏移，从 0 起；默认 0。负数报错不钳制；
                    超出文本长度时合法返回空屏并标注「已到底」。
                limit: 单屏长度，默认 2000，上限 8000；越界（<1 或 >8000）
                    报错不钳制。返回首行是「共 N 字符 / 本屏 offset=X
                    limit=L / next_offset=Y 或 已到底」元信息，其后为本屏
                    内容；下一页传 offset=Y。
            """.stripTrailing();

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public String description() {
        return DESCRIPTION;
    }

    @Override
    public ToolSpecification specification() {
        return ToolSpecification.builder()
                .name(NAME)
                .description(DESCRIPTION)
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("file_id", "摘要块给出的引用 id。")
                        .addIntegerProperty("offset",
                                "起始码点偏移（从 0 起）；默认 0，超出文本长度返回空屏（已到底）。")
                        .addIntegerProperty("limit",
                                "单屏长度；默认 2000，上限 8000，越界报错不钳制。")
                        .required("file_id")
                        .build())
                .build();
    }

    @Override
    public String call(String argsJson, ToolContext ctx) throws Exception {
        Map<String, Object> args = parseArgs(argsJson);
        String fileId = strOf(args, "file_id");
        if (fileId.isEmpty()) {
            throw new IllegalArgumentException("file_id 必填");
        }
        int offset = intOf(args, "offset", 0);
        if (offset < 0) {
            throw new IllegalArgumentException("offset 必须 >= 0（越界报错不钳制）");
        }
        int limit = intOf(args, "limit", ToolResultLimits.READ_PAGE_DEFAULT);
        if (limit < 1 || limit > ToolResultLimits.READ_PAGE_MAX) {
            throw new IllegalArgumentException(
                    "limit 必须在 1.." + ToolResultLimits.READ_PAGE_MAX + " 之间");
        }
        // 属主校验（store 层 getMany 无校验）：userId + threadId 双等，
        // 身份源是 RunConfig（权威、无 MDC 依赖）。空 threadId（孤儿行）同样拒绝。
        String userId = ctx != null && ctx.runConfig() != null && ctx.runConfig().userId() != null
                ? ctx.runConfig().userId() : "";
        String threadId = ctx != null && ctx.runConfig() != null
                ? ctx.runConfig().threadId() : "";
        List<UploadedFileInfo> found = fileStore.getMany(List.of(fileId));
        if (found.isEmpty()
                || !java.util.Objects.equals(found.get(0).userId(), userId)
                || !java.util.Objects.equals(found.get(0).threadId(), threadId)) {
            return DENIED_TEXT;
        }
        String text = found.get(0).extractedText();
        int total = CharMetrics.len(text);
        if (offset >= total) {
            // offset 超出文本长度：合法空屏 + 已到底（不报错——游标走完的
            // 自然终点，模型据此停止翻页）。
            return "共 " + total + " 字符 / 本屏 offset=" + offset
                    + " limit=" + limit + " / 已到底";
        }
        int end = Math.min(offset + limit, total);
        String window = slice(text, offset, end);
        String cursor = end < total
                ? "next_offset=" + end
                : "已到底";
        return "共 " + total + " 字符 / 本屏 offset=" + offset
                + " limit=" + limit + " / " + cursor + "\n" + window;
    }

    /** 参数解析：非法 JSON / 非对象直接报错（池归一成模型可读文本）。 */
    private static Map<String, Object> parseArgs(String argsJson) throws Exception {
        if (argsJson == null || argsJson.strip().isEmpty()) {
            return Map.of();
        }
        return MAPPER.readValue(argsJson, new TypeReference<Map<String, Object>>() {});
    }

    private static String strOf(Map<String, Object> args, String key) {
        Object value = args.get(key);
        return value == null ? "" : String.valueOf(value);
    }

    private static int intOf(Map<String, Object> args, String key, int fallback) {
        Object value = args.get(key);
        if (value == null) {
            return fallback;
        }
        if (value instanceof Number number) {
            return number.intValue();
        }
        throw new IllegalArgumentException(key + " 必须是整数");
    }

    /** 按码点切窗口（Python 切片语义：代理对不劈半）。 */
    private static String slice(String text, int startCodePoint, int endCodePoint) {
        int startUtf16 = text.offsetByCodePoints(0, startCodePoint);
        int endUtf16 = text.offsetByCodePoints(startUtf16, endCodePoint - startCodePoint);
        return text.substring(startUtf16, endUtf16);
    }
}
