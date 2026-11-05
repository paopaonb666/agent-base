package io.github.paopaonb666.agentbase.extensions;

import org.slf4j.MDC;

import java.util.Map;
import java.util.UUID;

/**
 * 可观测性基础：请求 ID（request_id）传播 + 结构化日志。
 *
 * <p>请求 ID 存放在 SLF4J MDC（对应 Python 侧的 ContextVar）中，每一条
 * 日志渲染都会补充上该 ID，格式化器渲染为人类可读的行（开发环境）或
 * JSON（生产环境，{@code LOG_JSON=true}）。这里没有任何内容与特定入口
 * 耦合——CLI 与 HTTP 服务都会调用本模块。
 *
 * <p>对应 Python 侧 {@code extensions/observability.py}。
 */
public final class Observability {

    private Observability() {}

    /** MDC 键（对应 Python 的 ContextVar 名 request_id）。 */
    public static final String REQUEST_ID_KEY = "request_id";

    private static final String CONFIG_LOGGER = "agent_base.tracing";

    /** 本模块配置的 appender 名：重复调用 setup 时是替换而非复制。 */
    private static final String HANDLER_NAME = "agent_base.console";

    /** 返回当前线程的请求 ID（未设置时返回 ""）。 */
    public static String requestId() {
        String rid = MDC.get(REQUEST_ID_KEY);
        return rid == null ? "" : rid;
    }

    /** 一个简短、唯一、可排序的请求 ID（12 位 hex）。 */
    public static String newRequestId() {
        return UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    /** 在某个请求 ID 下运行一个代码块；未提供时自动生成一个。 */
    public static void withRequestId(String rid, Runnable body) {
        String previous = MDC.get(REQUEST_ID_KEY);
        MDC.put(REQUEST_ID_KEY, rid == null || rid.isEmpty() ? newRequestId() : rid);
        try {
            body.run();
        } finally {
            if (previous == null) {
                MDC.remove(REQUEST_ID_KEY);
            } else {
                MDC.put(REQUEST_ID_KEY, previous);
            }
        }
    }

    /**
     * 捕获当前 MDC 上下文的 Runnable 包装：用于把 request_id 带进线程池
     * （MDC 是 ThreadLocal 语义，executor 线程不会自动继承）。对应
     * Python ContextVar 在 asyncio.to_thread 里的自动复制语义。
     */
    public static Runnable wrap(Runnable body) {
        Map<String, String> context = MDC.getCopyOfContextMap();
        return () -> {
            Map<String, String> previous = MDC.getCopyOfContextMap();
            if (context != null) {
                MDC.setContextMap(context);
            } else {
                MDC.clear();
            }
            try {
                body.run();
            } finally {
                if (previous != null) {
                    MDC.setContextMap(previous);
                } else {
                    MDC.clear();
                }
            }
        };
    }

    /**
     * 配置一次 root logger：request-id 注入 + 控制台格式。
     * {@code jsonLines} 选择结构化的 JSON 输出；默认人类可读格式面向本地
     * 开发。幂等：重复调用复用具名 appender（对应 Python 的具名 handler）。
     */
    public static synchronized void setupLogging(boolean jsonLines) {
        try {
            ch.qos.logback.classic.LoggerContext ctx =
                    (ch.qos.logback.classic.LoggerContext) org.slf4j.LoggerFactory.getILoggerFactory();
            ch.qos.logback.classic.Logger root =
                    ctx.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
            ch.qos.logback.core.ConsoleAppender<?> existing = null;
            java.util.Iterator<ch.qos.logback.core.Appender<ch.qos.logback.classic.spi.ILoggingEvent>> it =
                    root.iteratorForAppenders();
            while (it.hasNext()) {
                ch.qos.logback.core.Appender<ch.qos.logback.classic.spi.ILoggingEvent> appender = it.next();
                if (HANDLER_NAME.equals(appender.getName())) {
                    existing = (ch.qos.logback.core.ConsoleAppender<ch.qos.logback.classic.spi.ILoggingEvent>)
                            (ch.qos.logback.core.ConsoleAppender<?>) appender;
                    break;
                }
            }
            ch.qos.logback.core.ConsoleAppender<ch.qos.logback.classic.spi.ILoggingEvent> appender =
                    existing != null
                            ? (ch.qos.logback.core.ConsoleAppender<ch.qos.logback.classic.spi.ILoggingEvent>) existing
                            : new ch.qos.logback.core.ConsoleAppender<>();
            appender.setName(HANDLER_NAME);
            ch.qos.logback.core.encoder.LayoutWrappingEncoder<ch.qos.logback.classic.spi.ILoggingEvent> encoder =
                    new ch.qos.logback.core.encoder.LayoutWrappingEncoder<>();
            encoder.setLayout(jsonLines ? new JsonLineLayout() : humanLayout());
            appender.setEncoder(encoder);
            if (existing == null) {
                appender.setContext(ctx);
                appender.start();
                root.addAppender(appender);
            } else {
                appender.stop();
                appender.start();
            }
            root.setLevel(ch.qos.logback.classic.Level.INFO);
        } catch (LinkageError e) {
            // logback 不在类路径（裸 JUnit 环境）时静默跳过：日志格式是
            // 可观测性优化，不是功能契约。
        }
    }

    private static ch.qos.logback.classic.PatternLayout humanLayout() {
        ch.qos.logback.classic.LoggerContext ctx =
                (ch.qos.logback.classic.LoggerContext) org.slf4j.LoggerFactory.getILoggerFactory();
        ch.qos.logback.classic.PatternLayout layout = new ch.qos.logback.classic.PatternLayout();
        // 对应 Python HumanFormatter："%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"。
        // 注意：logback 的 %d{} 用逗号分隔"日期格式,时区"——毫秒用点号连接
        // （Python asctime 的逗号分隔在 logback 里会把 SSS 当成时区解析）。
        layout.setPattern("%d{yyyy-MM-dd HH:mm:ss.SSS} %level [%X{request_id:-}] %logger: %msg%n");
        layout.setContext(ctx);
        layout.start();
        return layout;
    }

    /**
     * 每行一个 JSON 对象的机器可解析日志（对应 Python JsonFormatter 的
     * 字段集：ts/level/logger/request_id/message）。
     */
    public static final class JsonLineLayout extends ch.qos.logback.core.LayoutBase<ch.qos.logback.classic.spi.ILoggingEvent> {

        @Override
        public String doLayout(ch.qos.logback.classic.spi.ILoggingEvent event) {
            StringBuilder sb = new StringBuilder(256);
            // 对应 Python formatTime(record, "%Y-%m-%dT%H:%M:%S%z")。
            String ts = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssXX")
                    .withZone(java.time.ZoneId.systemDefault())
                    .format(java.time.Instant.ofEpochMilli(event.getTimeStamp()));
            sb.append("{\"ts\":\"").append(ts).append('"');
            sb.append(",\"level\":\"").append(event.getLevel()).append('"');
            sb.append(",\"logger\":\"").append(escape(event.getLoggerName())).append('"');
            String rid = event.getMDCPropertyMap().getOrDefault(REQUEST_ID_KEY, "-");
            sb.append(",\"request_id\":\"").append(escape(rid)).append('"');
            sb.append(",\"message\":\"").append(escape(event.getFormattedMessage())).append('"');
            if (event.getThrowableProxy() != null) {
                sb.append(",\"exc_info\":\"").append(escape(
                        ch.qos.logback.classic.spi.ThrowableProxyUtil.asString(event.getThrowableProxy()))).append('"');
            }
            sb.append("}\n");
            return sb.toString();
        }

        private static String escape(String value) {
            if (value == null) {
                return "";
            }
            StringBuilder sb = new StringBuilder(value.length() + 8);
            for (int i = 0; i < value.length(); i++) {
                char c = value.charAt(i);
                switch (c) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    case '\r' -> sb.append("\\r");
                    case '\t' -> sb.append("\\t");
                    default -> {
                        if (c < 0x20) {
                            sb.append(String.format("\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                    }
                }
            }
            return sb.toString();
        }
    }

    /**
     * 报告环境启用了哪些追踪 provider（由环境开关控制）。基座不导入任何
     * 厂商 SDK——它只把环境已经决定好的状态暴露出来。
     */
    public static void logTracingConfig() {
        org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(CONFIG_LOGGER);
        String langsmithTracing = envOrNull("LANGSMITH_TRACING");
        boolean langsmith = isTruthy(langsmithTracing) || !envOrNull("LANGSMITH_API_KEY").isEmpty();
        boolean langfuse = !envOrNull("LANGFUSE_PUBLIC_KEY").isEmpty()
                && !envOrNull("LANGFUSE_SECRET_KEY").isEmpty();
        logger.info("tracing: langsmith={} langfuse={} （设置 LANGSMITH_TRACING=true / LANGFUSE_* 密钥以启用）",
                langsmith ? "on" : "off", langfuse ? "on" : "off");
    }

    private static String envOrNull(String key) {
        String value = System.getenv(key);
        return value == null ? "" : value;
    }

    private static boolean isTruthy(String value) {
        return switch ((value == null ? "" : value).strip().toLowerCase(java.util.Locale.ROOT)) {
            case "1", "true", "yes", "on" -> true;
            default -> false;
        };
    }
}
