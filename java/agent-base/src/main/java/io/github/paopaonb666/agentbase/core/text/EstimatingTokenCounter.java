package io.github.paopaonb666.agentbase.core.text;

/**
 * 字符系数估算（降级实现，原 {@code ContextEngine.estimateTokens}）：
 * DeepSeek 系 tokenizer 的经验值——中文约 0.6 token/字，ASCII 约
 * 0.25 token/字符（~4 字符/token）。估算宁大勿小——宁可早修剪也不要
 * 静默撑爆上下文。真实 tokenizer 可用时的误差仍达 ±30%（#339），
 * 仅作为 jtokkit 不可用时的兜底与测试的确定性口径。
 */
public final class EstimatingTokenCounter implements TokenCounter {

    public static final EstimatingTokenCounter INSTANCE = new EstimatingTokenCounter();

    private static final double CJK_TOKEN_PER_CHAR = 0.6;
    private static final double ASCII_TOKEN_PER_CHAR = 0.25;

    private EstimatingTokenCounter() {}

    @Override
    public int countTokens(String text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        int cjk = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c >= '\u4e00' && c <= '\u9fff') {
                cjk++;
            }
        }
        return (int) (cjk * CJK_TOKEN_PER_CHAR + (text.length() - cjk) * ASCII_TOKEN_PER_CHAR);
    }
}
