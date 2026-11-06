package io.github.paopaonb666.agentbase.memory.retrieval;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 向量索引下推抽象（#324/旧 #35）：把「候选集里取向量 top-K」从全量
 * 内存余弦解耦成可替换的后端——检索容量天花板（2000 候选全量打分）的
 * 结构性出口。
 *
 * <ul>
 *   <li><b>默认实现</b> {@link BruteForceVectorIndex}：内存 Map + 全量
 *       余弦——与现实现语义等价（下推改造的基线）。</li>
 *   <li><b>可选实现</b>：sqlite-vec（#323：Maven Central 无 JDBC 构件，
 *       延后）/ 外置向量库（Qdrant/pgvector）。</li>
 * </ul>
 *
 * <p>合并门槛（#325）：任何实现的 golden 召回回归
 * （RetrievalGoldenEvalTest）不得劣于基线。实现必须线程安全。
 */
public interface VectorIndexPort {

    /** 一条向量命中：记忆 id + 余弦相似度。 */
    record VectorHit(String id, double similarity) {}

    /** upsert 一条向量（写路径随 addMemory/updateMemory 调用）。 */
    void upsert(String id, List<Float> vector);

    /** 删除一条向量（记忆删除时调用；不存在为无操作）。 */
    void delete(String id);

    /** 余弦 top-K；查询向量为 null 时返回空表。 */
    List<VectorHit> search(List<Float> queryVector, int topK);

    /** 默认实现：内存暴力余弦（与现全量打分语义等价的基线）。 */
    static VectorIndexPort bruteForce() {
        return new BruteForceVectorIndex();
    }

    /** 内存暴力实现：ConcurrentHashMap + 全量余弦排序。 */
    final class BruteForceVectorIndex implements VectorIndexPort {

        private final Map<String, List<Float>> vectors = new ConcurrentHashMap<>();

        @Override
        public void upsert(String id, List<Float> vector) {
            if (id == null || vector == null || vector.isEmpty()) {
                return;
            }
            vectors.put(id, List.copyOf(vector));
        }

        @Override
        public void delete(String id) {
            if (id != null) {
                vectors.remove(id);
            }
        }

        @Override
        public List<VectorHit> search(List<Float> queryVector, int topK) {
            if (queryVector == null || queryVector.isEmpty() || topK <= 0) {
                return List.of();
            }
            record Hit(String id, double sim) {}
            List<Hit> all = new java.util.ArrayList<>(vectors.size());
            for (Map.Entry<String, List<Float>> e : vectors.entrySet()) {
                Double sim = Retrieval.cosineSimilarity(queryVector, e.getValue());
                if (sim != null) {
                    all.add(new Hit(e.getKey(), sim));
                }
            }
            all.sort((a, b) -> Double.compare(b.sim(), a.sim()));
            List<VectorHit> out = new java.util.ArrayList<>(Math.min(topK, all.size()));
            for (Hit hit : all.subList(0, Math.min(topK, all.size()))) {
                out.add(new VectorHit(hit.id(), hit.sim()));
            }
            return out;
        }
    }
}
