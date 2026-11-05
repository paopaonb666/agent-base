package io.github.paopaonb666.agentbase.core;

/**
 * 线程标识值对象：把 {@code {module}:{user_thread_id}} 的隐式字符串协议显式化。
 *
 * <p>一处定义、处处引用：core 的工具执行上下文、memory 的作用域解析与
 * server 的线程路由都经 {@code ThreadId} 拼接/解析，分隔符约定不再散落
 * 在各处。raw 部分含分隔符在构造时即被拒绝（静默错位是这类字符串协议
 * 最隐蔽的故障形态——module 拆错、作用域串号都不会报错）。
 *
 * <p>对应 Python 侧 {@code core/threads.py}。
 */
public record ThreadId(String module, String raw) {

    /** 分隔符（对应 Python 侧 SEPARATOR）。 */
    public static final String SEPARATOR = ":";

    /** 非法的线程标识（空组成部分 / 含分隔符 / 解析输入缺分隔符）。 */
    public static final class ThreadIdError extends IllegalArgumentException {
        public ThreadIdError(String message) {
            super(message);
        }
    }

    public ThreadId {
        if (module == null || module.isEmpty()) {
            throw new ThreadIdError("ThreadId.module 不能为空");
        }
        if (raw == null || raw.isEmpty()) {
            throw new ThreadIdError("ThreadId.raw 不能为空");
        }
        if (module.contains(SEPARATOR)) {
            throw new ThreadIdError("ThreadId.module 不能包含分隔符 '%s': '%s'".formatted(SEPARATOR, module));
        }
        if (raw.contains(SEPARATOR)) {
            throw new ThreadIdError("ThreadId.raw 不能包含分隔符 '%s': '%s'".formatted(SEPARATOR, raw));
        }
    }

    /** 产出存储与 configurable 使用的 {@code module:raw} 形态。 */
    @Override
    public String toString() {
        return module + SEPARATOR + raw;
    }

    /** 从存储/checkpointer 里的完整字符串解析；格式非法抛 {@link ThreadIdError}。 */
    public static ThreadId parse(String value) {
        int idx = value == null ? -1 : value.indexOf(SEPARATOR);
        if (idx <= 0 || idx == value.length() - 1) {
            throw new ThreadIdError(
                    "不是命名空间线程标识（应为 module%sraw）: '%s'".formatted(SEPARATOR, value));
        }
        return new ThreadId(value.substring(0, idx), value.substring(idx + 1));
    }

    /** 宽松解析：空串 / 无分隔符 / 格式非法返回 null（调用方自行给默认值）。 */
    public static ThreadId tryParse(String value) {
        if (value == null || value.isEmpty()) {
            return null;
        }
        try {
            return parse(value);
        } catch (ThreadIdError e) {
            return null;
        }
    }
}
