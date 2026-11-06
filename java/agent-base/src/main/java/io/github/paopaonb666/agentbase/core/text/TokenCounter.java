package io.github.paopaonb666.agentbase.core.text;

/**
 * token 计量抽象（#339/旧 #2）：上下文预算、压缩触发、成本估算共同
 * 依赖的口径。默认实现是真实 tokenizer（{@link JTokkitTokenCounter}），
 * 字符系数估算保留为降级实现（{@link EstimatingTokenCounter}）。
 */
public interface TokenCounter {

    /** 计一段文本的 token 数；null/空文本返回 0。 */
    int countTokens(String text);
}
