package io.github.paopaonb666.agentbase.core.text;

import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * TokenCounter 计量口径测试（roadmap #339/旧 #2）：真实 tokenizer
 * （jtokkit cl100k）为默认预算口径，字符系数估算保留为降级实现；
 * ContextEngine 的预算路径走可注入的计量器。
 */
class TokenCounterTest {

    @Test
    void jtokkitCountsRealTokensNotCharacters() {
        // cl100k 基准值（tiktoken 对齐）："hello world" = 2 tokens；
        // 空输入 0；null 0。
        assertEquals(2, JTokkitTokenCounter.CL100K.countTokens("hello world"));
        assertEquals(0, JTokkitTokenCounter.CL100K.countTokens(""));
        assertEquals(0, JTokkitTokenCounter.CL100K.countTokens(null));
        // 真实 tokenizer 下 CJK 通常 1 字 1 token 上下，远高于估算系数。
        assertTrue(JTokkitTokenCounter.CL100K.countTokens("你好世界") >= 2,
                "真实 tokenizer 对 CJK 的计数应明显高于 0.6 系数估算");
    }

    @Test
    void estimatingCounterKeepsTheLegacyHeuristic() {
        // 降级口径锁定（原 ContextEngine.estimateTokens 行为逐值不变）。
        assertEquals(0, EstimatingTokenCounter.INSTANCE.countTokens(null));
        assertEquals(0, EstimatingTokenCounter.INSTANCE.countTokens(""));
        assertEquals(1, EstimatingTokenCounter.INSTANCE.countTokens("你好"));
        assertEquals(1, EstimatingTokenCounter.INSTANCE.countTokens("abcd"));
        assertEquals(1, EstimatingTokenCounter.INSTANCE.countTokens("你abcd"));
    }

    @Test
    void contextEngineDefaultsToJtokkitAndAcceptsInjection() {
        assertEquals(JTokkitTokenCounter.CL100K, ContextEngine.tokenCounter());
        ContextEngine.setTokenCounter(EstimatingTokenCounter.INSTANCE);
        assertEquals(EstimatingTokenCounter.INSTANCE, ContextEngine.tokenCounter());
        // null 回落默认；结束恢复默认（不影响其他测试的预算口径）。
        ContextEngine.setTokenCounter(null);
        assertEquals(JTokkitTokenCounter.CL100K, ContextEngine.tokenCounter());
    }
}
