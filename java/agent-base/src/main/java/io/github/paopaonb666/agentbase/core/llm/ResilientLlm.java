package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.Semaphore;

/**
 * 快档包装：并发限流 + 瞬时错误重试 + 主力回退（对应 Python core/llm.py
 * 的 {@code ResilientLLM}，鸭子类型包装的 Java 契约版）。
 *
 * <p>面向后台管线（记忆形成：抽取/画像/摘要等单消息列表直答），实现
 * {@link AgentModule.ChatInvokerPort}——刻意不进图执行路径（图执行走
 * {@link ModelRuntime#stream} / {@link ModelRuntime#invoke}，不经本类）。
 *
 * <ul>
 *   <li><b>并发限流</b>：{@link Semaphore}，尊重免费档并发配额（免费档
 *       普遍按并发数限速；组合根传 {@code LLM_FAST_CONCURRENCY}，Python
 *       默认 2）。信号量只包住快档主调用——主力回退在信号量之外（与
 *       Python {@code async with self._semaphore} 的作用域一致）。</li>
 *   <li><b>重试</b>：指数退避（{@code base_delay * 2^attempt}，Python 默认
 *       base_delay=1.0s、max_retries=2，即最多 3 次尝试）。与 Python 一致，
 *       捕获的是所有运行时异常而非仅 429/超时——Python 的 docstring 说
 *       "瞬时错误（429/超时）"，但实现是 {@code except Exception}，本移植
 *       以实现为准。等待退避期间响应线程中断（对应 asyncio.sleep 可被
 *       取消）：恢复中断标志并中止重试。</li>
 *   <li><b>主力回退</b>：重试耗尽后调用主力模型（结果原样返回，不再重试）；
 *       无回退（all_main）时原样抛出最后一次错误。</li>
 * </ul>
 *
 * <p>档位分派按 {@code MEMORY_PIPELINE_PROFILE}：{@code split}（默认）抽取/
 * 画像/摘要走快档、回落主力；{@code all_fast} 整条管线走快档、回落主力；
 * {@code all_main} 只走主力、不回退（对应 Python 组合根在 all_main 时干脆
 * 不装 ResilientLLM）。快档为 null 时全部流量走主力——快档是优化不是依赖。
 */
public final class ResilientLlm implements AgentModule.ChatInvokerPort {

    /** Python ResilientLLM 构造参数默认值：max_retries=2。 */
    public static final int DEFAULT_MAX_RETRIES = 2;
    /** Python ResilientLLM 构造参数默认值：base_delay=1.0 秒。 */
    public static final double DEFAULT_BASE_DELAY_SECONDS = 1.0;

    private static final Logger LOG = LoggerFactory.getLogger(ResilientLlm.class);

    /** 快档运行时；null = 未配置（LLM_FAST_* 缺席），全部流量走主力。 */
    private final ModelRuntime fast;
    /** 主力运行时（回退目标；all_main 档的唯一直连）。 */
    private final ModelRuntime main;
    private final String pipelineProfile;
    private final int maxRetries;
    private final long baseDelayMillis;
    private final Semaphore semaphore;

    /**
     * 组合根用的构造（Python 组合根等价物：
     * {@code ResilientLLM(primary=build_llm(fast), fallback=llm,
     * max_concurrency=llm_fast.concurrency)}）：默认 split 分派、Python
     * 默认重试参数。快档未配置传 null 即可，全部流量走主力。
     */
    public ResilientLlm(ModelRuntime fast, ModelRuntime main, int maxConcurrency) {
        this(fast, main, maxConcurrency, "split", DEFAULT_MAX_RETRIES, DEFAULT_BASE_DELAY_SECONDS);
    }

    /** 全参构造：显式指定管线档位与重试参数（测试与非默认档位用）。 */
    public ResilientLlm(ModelRuntime fast, ModelRuntime main, int maxConcurrency,
                        String pipelineProfile, int maxRetries, double baseDelaySeconds) {
        if (maxConcurrency < 1) {
            throw new IllegalArgumentException("max_concurrency must be >= 1, got " + maxConcurrency);
        }
        if (maxRetries < 0) {
            throw new IllegalArgumentException("max_retries must be >= 0, got " + maxRetries);
        }
        if (baseDelaySeconds < 0) {
            throw new IllegalArgumentException("base_delay must be >= 0, got " + baseDelaySeconds);
        }
        if (!Settings.KNOWN_PIPELINE_PROFILES.contains(pipelineProfile)) {
            throw new IllegalArgumentException("unknown pipeline profile '%s'; expected one of %s"
                    .formatted(pipelineProfile, Settings.KNOWN_PIPELINE_PROFILES.stream().sorted().toList()));
        }
        this.fast = fast;
        this.main = Objects.requireNonNull(main, "main runtime is required");
        this.pipelineProfile = pipelineProfile;
        this.maxRetries = maxRetries;
        this.baseDelayMillis = (long) (baseDelaySeconds * 1000);
        this.semaphore = new Semaphore(maxConcurrency);
    }

    /**
     * 单次调用（不进图执行路径）：按管线档位分派，限流、重试、耗尽后
     * 回退主力。异常语义见类 docstring。
     */
    @Override
    public String invoke(List<ChatMessage> messages) {
        ModelRuntime primary;
        ModelRuntime fallback;
        if ("all_main".equals(pipelineProfile)) {
            primary = main;
            fallback = null;
        } else {
            // split（默认）与 all_fast：快档优先；未配置则主力本身就是快档
            // 的替身（对应 Python build_llm("fast") 回落主力配置的语义）。
            primary = fast != null ? fast : main;
            fallback = main;
        }

        int attempts = maxRetries + 1;
        RuntimeException lastError = null;
        for (int attempt = 0; attempt < attempts; attempt++) {
            RuntimeException failure = null;
            try {
                semaphore.acquire();
                try {
                    return invokeText(primary, messages);
                } finally {
                    semaphore.release();
                }
            } catch (InterruptedException interrupted) {
                // 信号量等待被中断：恢复标志并中止（对应 asyncio 取消传播）。
                Thread.currentThread().interrupt();
                throw interruptedWait(interrupted, lastError);
            } catch (RuntimeException exc) {
                failure = exc;
                lastError = exc;
            }
            // 退避等待在信号量之外（对应 Python：async with 块随异常传播
            // 先退出，asyncio.sleep 在许可已归还后执行）——重试背压不占用
            // 并发额度，否则 LLM_FAST_CONCURRENCY=1 时后续调用会被
            // 睡眠中的线程白白串行化。
            if (attempt < attempts - 1) {
                long delayMillis = baseDelayMillis * (1L << attempt);
                LOG.warn("llm: 快速档调用失败（第 {}/{} 次），{}s 后重试: {}",
                        attempt + 1, attempts, delayMillis / 1000.0, failure.toString());
                sleepBeforeRetry(delayMillis, lastError);
            }
        }
        if (fallback != null) {
            // 主力回退在信号量之外（Python 语义）：结果原样返回，异常原样
            // 上抛——回退调用本身不再重试。
            LOG.warn("llm: 快速档重试耗尽，回退主力模型");
            return invokeText(fallback, messages);
        }
        throw lastError;
    }

    /**
     * {@link ModelRuntime#invoke} 的 AiMessage → 文本投影（ChatInvokerPort
     * 契约返回 String）。工具调用轮回复没有文本时按空串返回——后台管线
     * 按文本判断产出（Python 侧 ainvoke 返回 AIMessage 后取 content 的等价物）。
     */
    private static String invokeText(ModelRuntime runtime, List<ChatMessage> messages) {
        AiMessage reply = runtime.invoke(messages);
        return reply.text() == null ? "" : reply.text();
    }

    /** 退避等待；被中断时恢复标志、携带最后错误中止重试。 */
    private void sleepBeforeRetry(long delayMillis, RuntimeException lastError) {
        try {
            Thread.sleep(delayMillis);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            RuntimeException abort = new IllegalStateException("llm 重试等待被中断", interrupted);
            if (lastError != null) {
                abort.addSuppressed(lastError);
            }
            throw abort;
        }
    }

    private static RuntimeException interruptedWait(InterruptedException interrupted,
                                                    RuntimeException lastError) {
        RuntimeException abort = new IllegalStateException("llm 调用等待被中断", interrupted);
        if (lastError != null) {
            abort.addSuppressed(lastError);
        }
        return abort;
    }
}
