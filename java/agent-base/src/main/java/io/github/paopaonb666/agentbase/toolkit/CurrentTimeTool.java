package io.github.paopaonb666.agentbase.toolkit;

import java.time.DateTimeException;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.Locale;
import java.util.TimeZone;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;

import io.github.paopaonb666.agentbase.core.tools.AgentTool;

/**
 * {@code current_time} 工具：获取当前日期时间（零依赖基础工具之一，
 * 对应 Python 侧 {@code tools/basic.py} 的 {@code current_time}）。
 *
 * <p>时区处理对齐：{@code local}（或空白）用服务器本地时区的带时区
 * 时间；IANA 名称（{@code Asia/Shanghai} 等）经 {@link ZoneId} 解析；
 * 时区数据来自 JDK 自带的 tzdata（对应 Python 的 {@code tzdata}/
 * {@code zoneinfo}，均为 IANA 库）。
 *
 * <p><b>逐字对齐说明（含已知的宽松语义）</b>：Python 的 {@code utc}
 * 分支用的是不带时区的 {@code datetime.now()}——即服务器本地挂钟
 * 时间——只是把后缀标成 {@code (UTC)}。本机非 UTC 时该输出会被误标；
 * 为了行为一致这里原样移植（{@link LocalDateTime#now()}），若要修正
 * 应 Python/Java 两侧同步修正。
 *
 * <p>已声明的平台差异：时区短名（{@code tzname()}）在 Python 下来自
 * 操作系统（Windows 上是 "China Standard Time" 一类注册表名，Linux 上
 * 是 "CST" 一类缩写），Java 侧取 {@link TimeZone} 的英文短显示名
 * （如 "GMT+08:00"）。该名字本就平台相关，格式保持一致。
 */
public final class CurrentTimeTool implements AgentTool {

    public static final String NAME = "current_time";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** 给模型看的说明书（逐字复制 Python 版 docstring，docstring 即提示词）。 */
    private static final String DESCRIPTION = """
            获取当前日期时间。返回 ISO 8601 格式（含时区偏移），例如
            2026-09-12T14:30:00+08:00。

            Args:
                tz: 时区。"local"（服务器本地时区，默认）、"UTC" 或 IANA 名称
                    （如 "Asia/Shanghai"、"America/New_York"）。
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
        // tz 可选（默认 "local"），对应 Python 签名 tz: str = "local"。
        return ToolSpecification.builder()
                .name(NAME)
                .description(DESCRIPTION)
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("tz",
                                "时区。\"local\"（服务器本地时区，默认）、\"UTC\" 或 IANA 名称"
                                        + "（如 \"Asia/Shanghai\"、\"America/New_York\"）。")
                        .build())
                .build();
    }

    @Override
    public String call(String argsJson, ToolContext ctx) {
        JsonNode node = argsJson == null || argsJson.isBlank()
                ? MAPPER.createObjectNode() : parseArgs(argsJson);
        JsonNode tzNode = node.get("tz");
        String tz = tzNode == null || tzNode.isNull() ? "local" : tzNode.asText();
        String norm = tz.strip().toLowerCase(Locale.ROOT);
        if (norm.isEmpty() || norm.equals("local")) {
            ZonedDateTime now = ZonedDateTime.now();
            return isoformat(now) + " (" + shortName(now.getZone()) + ")";
        }
        if (norm.equals("utc")) {
            // 逐字对齐 Python：本地挂钟时间 + "(UTC)" 后缀（见类注释）。
            return isoformat(LocalDateTime.now()) + " (UTC)";
        }
        // Python ZoneInfo 只接受 tz 数据库里的精确键名（区分大小写）：
        // "+08:00"/"GMT+8"/"Z" 这类偏移形式抛 ZoneInfoNotFoundError → 由
        // 工具转成"未知时区"提示。ZoneId.of 会全盘接受，这里先做键名
        // 成员检查对齐行为。
        String zoneName = tz.strip();
        if (!java.time.zone.ZoneRulesProvider.getAvailableZoneIds().contains(zoneName)) {
            throw new IllegalArgumentException(
                    "未知时区 '" + tz + "'：请使用 IANA 时区名（如 Asia/Shanghai）或 'local'/'UTC'");
        }
        ZoneId zone = ZoneId.of(zoneName);
        ZonedDateTime now = ZonedDateTime.now(zone);
        // Python: f"{now.isoformat()} ({zone.key}，{now.tzname()})" —— 区名与
        // 时区名之间是全角逗号。
        return isoformat(now) + " (" + zone.getId() + "，" + shortName(zone) + ")";
    }

    /** 近似 Python {@code datetime.isoformat()}：秒级基座 + 非零时补 6 位微秒。 */
    private static String isoformat(ZonedDateTime t) {
        String offset = t.getOffset().getId();
        // Java 用 "Z" 表示零偏移；Python 恒写 "+00:00"。
        return isoformat(t.toLocalDateTime()) + ("Z".equals(offset) ? "+00:00" : offset);
    }

    private static String isoformat(LocalDateTime t) {
        StringBuilder sb = new StringBuilder(32);
        sb.append(String.format("%04d-%02d-%02dT%02d:%02d:%02d",
                t.getYear(), t.getMonthValue(), t.getDayOfMonth(),
                t.getHour(), t.getMinute(), t.getSecond()));
        int micros = t.getNano() / 1000;
        if (micros != 0) {
            sb.append(String.format(".%06d", micros));
        }
        return sb.toString();
    }

    private static String shortName(ZoneId zone) {
        return TimeZone.getTimeZone(zone).getDisplayName(false, TimeZone.SHORT, Locale.US);
    }

    private static JsonNode parseArgs(String argsJson) {
        try {
            return MAPPER.readTree(argsJson);
        } catch (Exception e) {
            throw new IllegalArgumentException("工具参数不是合法 JSON：" + e.getMessage());
        }
    }
}
