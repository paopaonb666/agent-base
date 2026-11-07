package io.github.paopaonb666.agentbase.core.config;

import java.util.Objects;

/**
 * 敏感配置值（对应 Python 侧 pydantic {@code SecretStr} 语义，#106）：
 * {@link #toString()} 恒为掩码，堵住调试器/日志/意外序列化的泄露面；
 * 真值只经 {@link #value()} 在送入客户端、签名等消费点显式取出。
 *
 * <p>刻意不用 record——record 自动生成的 toString 会把字段真值打出来。</p>
 */
public final class Secret {

    private final String value;

    private Secret(String value) {
        this.value = value == null ? "" : value;
    }

    public static Secret of(String value) {
        return new Secret(value);
    }

    /** 取真值（仅在消费点调用；不要进日志）。 */
    public String value() {
        return value;
    }

    /** 与 {@code String.isBlank()} 同语义的判空便捷方法。 */
    public boolean isBlank() {
        return value.isBlank();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Secret other && value.equals(other.value);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(value);
    }

    @Override
    public String toString() {
        return "***";
    }
}
