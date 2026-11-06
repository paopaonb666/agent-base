package io.github.paopaonb666.agentbase.core.text;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.EncodingType;

/**
 * 真实 tokenizer 计量（#339/旧 #2）：jtokkit（tiktoken 的纯 Java 实现）。
 * 默认编码 cl100k_base；国产模型（DeepSeek/GLM 系）的厂商 tokenizer
 * 校准是后续条目（#340）——cl100k 对它们只是近似，但误差远小于字符
 * 系数估算的 ±30%。Encoding 实例线程安全，全局复用一个。
 */
public final class JTokkitTokenCounter implements TokenCounter {

    /** 默认编码：cl100k_base（GPT-3.5/4 系与多数兼容端点的最大公约数）。 */
    public static final JTokkitTokenCounter CL100K =
            new JTokkitTokenCounter(EncodingType.CL100K_BASE);

    private final Encoding encoding;

    public JTokkitTokenCounter(EncodingType type) {
        EncodingRegistry registry = Encodings.newDefaultEncodingRegistry();
        this.encoding = registry.getEncoding(type);
    }

    @Override
    public int countTokens(String text) {
        return text == null || text.isEmpty() ? 0 : encoding.countTokens(text);
    }
}
