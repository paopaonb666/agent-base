package io.github.paopaonb666.agentbase.memory.embeddings;

import java.util.List;

/**
 * embedding 客户端的最小接口。
 *
 * <p>{@code embed} <b>永不抛异常</b>：返回 {@code null} 即"当前不可用"
 * （网络失败/冷却期/未配置），调用方据此走无向量（关键词降级）路径；
 * 空输入返回空列表。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/embeddings.py} 的
 * {@code EmbeddingClient} Protocol。
 */
public interface EmbeddingClient {

    /** 已配置/已知的向量维度；null 表示不向量化（NullEmbedding）。 */
    Integer dims();

    /**
     * 向量化一组文本；失败返回 null（含冷却期内的快速返回），
     * 永不抛异常。
     */
    List<List<Float>> embed(List<String> texts);

    /**
     * 轻量可达性探测（/health 用）：任何 HTTP 应答都算 ok（网络可达即证明），
     * 配额/鉴权不属于健康判定。结果缓存 60s。
     */
    boolean probe();
}
